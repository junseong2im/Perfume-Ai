"""
Complete CAS→SMILES Resolution
================================
1. goodscents/cas_to_cid.json → CAS→CID
2. All Pyrfume molecules → CID→SMILES
3. keller_2016 molecules (481 CAS) → match to CID via name
4. Combine all sources → match IFRA 453 CAS
"""
import json, os, csv, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

def build_complete_cas_smiles():
    """Build CAS→SMILES from ALL available sources"""
    cas_to_smi = {}  # Final: CAS → SMILES
    cid_to_smi = {}  # CID → SMILES
    cas_to_cid = {}  # CAS → CID
    
    # Step 1: Load goodscents cas_to_cid.json
    gs_cas_cid = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'goodscents', 'cas_to_cid.json')
    if os.path.exists(gs_cas_cid):
        with open(gs_cas_cid, 'r') as f:
            gs_map = json.load(f)
        # Format might be {CAS: CID} or {CAS: [CIDs]}
        for cas, cid_val in gs_map.items():
            if isinstance(cid_val, list):
                for c in cid_val:
                    cas_to_cid[cas.strip()] = str(c)
            else:
                cas_to_cid[cas.strip()] = str(cid_val)
        print(f"  goodscents cas_to_cid: {len(cas_to_cid)} entries")
    
    # Step 2: Build CID→SMILES from ALL molecule files
    pdir = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all')
    for sub in os.listdir(pdir):
        mp = os.path.join(pdir, sub, 'molecules.csv')
        if os.path.exists(mp):
            try:
                with open(mp, 'r', encoding='utf-8', errors='replace') as f:
                    for row in csv.DictReader(f):
                        cid = str(row.get('CID', '')).strip()
                        smi = row.get('IsomericSMILES', '').strip()
                        cas = row.get('CAS', '').strip()
                        if cid and smi:
                            cid_to_smi[cid] = smi
                        if cas and smi:
                            cas_to_smi[cas] = smi
            except:
                pass
    print(f"  CID→SMILES: {len(cid_to_smi)} entries")
    print(f"  Direct CAS→SMILES: {len(cas_to_smi)} entries")
    
    # Step 3: Bridge CAS→CID→SMILES
    bridged = 0
    for cas, cid in cas_to_cid.items():
        if cas not in cas_to_smi and cid in cid_to_smi:
            cas_to_smi[cas] = cid_to_smi[cid]
            bridged += 1
    print(f"  Bridged CAS→CID→SMILES: {bridged}")
    
    # Step 4: Hardcoded IFRA SMILES
    try:
        from scripts.ifra_hardcoded import CAS_SMILES
        for cas, smi in CAS_SMILES.items():
            if smi and cas not in cas_to_smi:
                cas_to_smi[cas] = smi
        print(f"  After hardcoded: {len(cas_to_smi)} total CAS→SMILES")
    except:
        pass
    
    print(f"\n  TOTAL CAS→SMILES: {len(cas_to_smi)}")
    return cas_to_smi

def match_ifra_453():
    """Match all 453 IFRA to ingredients using complete CAS→SMILES"""
    cas_to_smi = build_complete_cas_smiles()
    
    # Load IFRA
    ifra_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
    with open(ifra_path, 'r', encoding='utf-8') as f:
        ifra = json.load(f)
    
    cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
    with open(cat4_path, 'r', encoding='utf-8') as f:
        cat4 = json.load(f)
    
    # Check how many IFRA CAS we can resolve
    resolvable = sum(1 for cas in ifra if cas in cas_to_smi)
    print(f"\n  IFRA CAS resolvable: {resolvable}/{len(ifra)}")
    
    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    print(f"  Ingredients: {len(ingredients)}")
    
    try:
        from rdkit import Chem
        def canonical(s):
            try:
                m = Chem.MolFromSmiles(s)
                return Chem.MolToSmiles(m) if m else s
            except:
                return s
    except:
        canonical = lambda x: x
    
    can_to_idx = {}
    for i, ing in enumerate(ingredients):
        smi = ing.get('smiles', '').strip()
        if smi:
            can = canonical(smi)
            can_to_idx[can] = i
    
    # Clear old tags
    for ing in ingredients:
        for k in ['ifra_cas','ifra_type','ifra_restricted','ifra_prohibited','ifra_limit_pct','ifra_name']:
            ing.pop(k, None)
    
    # Match
    matched = 0
    p_list, r_list = [], []
    unresolvable = 0
    resolved_no_match = 0
    
    for cas, entry in ifra.items():
        ifra_type = entry.get('type', '')
        name = entry.get('name', '')
        categories = entry.get('categories', {})
        
        smi = cas_to_smi.get(cas)
        if not smi:
            unresolvable += 1
            continue
        
        can = canonical(smi)
        if can not in can_to_idx:
            resolved_no_match += 1
            continue
        
        idx = can_to_idx[can]
        is_prohibited = ifra_type == 'P'
        is_restricted = ifra_type in ('R', 'RS')
        
        # Get Fine Fragrance limit
        ff_limit = None
        if cas in cat4:
            if cat4[cas].get('prohibited'):
                ff_limit = 0.0
            else:
                ff_limit = cat4[cas].get('max_pct_cat4')
        if ff_limit is None:
            for k, v in categories.items():
                if 'fine' in k.lower() or 'fragrance' in k.lower():
                    ff_limit = v
                    break
        
        ingredients[idx]['ifra_cas'] = cas
        ingredients[idx]['ifra_type'] = ifra_type
        ingredients[idx]['ifra_prohibited'] = is_prohibited
        ingredients[idx]['ifra_restricted'] = is_prohibited or is_restricted
        ingredients[idx]['ifra_name'] = name
        if is_prohibited:
            ingredients[idx]['ifra_limit_pct'] = 0.0
            p_list.append(name)
        elif ff_limit is not None:
            ingredients[idx]['ifra_limit_pct'] = ff_limit
            r_list.append(f"{name}: {ff_limit}%")
        else:
            ingredients[idx]['ifra_limit_pct'] = 100.0
        
        # Clamp max_pct
        if is_prohibited:
            ingredients[idx]['max_pct'] = 0.0
        elif ff_limit is not None and ff_limit < ingredients[idx].get('max_pct', 100):
            ingredients[idx]['max_pct'] = ff_limit
        
        matched += 1
    
    total_tagged = sum(1 for x in ingredients if x.get('ifra_cas'))
    total_p = sum(1 for x in ingredients if x.get('ifra_prohibited'))
    total_r = sum(1 for x in ingredients if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    
    print(f"\n  === FULL IFRA RESULTS ===")
    print(f"  CAS resolvable to SMILES: {resolvable}/453")
    print(f"  Resolved but not in DB: {resolved_no_match}")
    print(f"  Unresolvable CAS: {unresolvable}")
    print(f"  MATCHED to DB: {matched}/453 ({100*matched//453}%)")
    print(f"  Prohibited: {total_p}")
    print(f"  Restricted: {total_r}")
    
    if p_list:
        print(f"\n  Prohibited ({len(p_list)}):")
        for p in p_list[:15]:
            print(f"    x {p}")
        if len(p_list) > 15:
            print(f"    +{len(p_list)-15} more")
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Save enforcement
    enforcement = {
        'version': '51st Amendment FULL',
        'total': len(ifra),
        'resolvable': resolvable,
        'matched': matched,
        'prohibited': total_p,
        'restricted': total_r,
    }
    with open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_enforcement.json'), 'w') as f:
        json.dump(enforcement, f, indent=2)
    
    return matched, total_p, total_r

if __name__ == '__main__':
    match_ifra_453()
