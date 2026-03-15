"""
FULL DATA EXPANSION
====================
Fix ALL data subsetting issues found by audit:
1. IFRA: 59 → 453 (full IFRA 51st Amendment)
2. Ingredients: 5000 → add molecules from unused Pyrfume datasets
3. Update IFRA tags with all 453 entries
4. Verify
"""
import json, os, sys, csv, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

# ============================================================
# 1. IFRA: Use ALL 453 entries, not just cat4
# ============================================================
def expand_ifra_full():
    """Tag ingredients with ALL 453 IFRA entries, not just cat4 59"""
    print("=" * 60)
    print("  [1] IFRA: 59 → 453 Full Integration")
    print("=" * 60)
    
    # Load FULL IFRA
    ifra_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
    with open(ifra_path, 'r', encoding='utf-8') as f:
        ifra_full = json.load(f)
    print(f"  Full IFRA entries: {len(ifra_full)}")
    
    # Load cat4 (has max_pct_cat4 for Fine Fragrance)
    cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
    with open(cat4_path, 'r', encoding='utf-8') as f:
        cat4 = json.load(f)
    print(f"  Cat4 entries: {len(cat4)}")
    
    # Load full IFRA with categories
    full_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_full.json')
    ifra_full2 = {}
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            ifra_full2 = json.load(f)
        print(f"  ifra_51st_full.json: {len(ifra_full2)} entries")
    
    # Build comprehensive IFRA database keyed by CAS
    # Merge all sources
    all_ifra = {}
    for cas, entry in ifra_full.items():
        ifra_type = entry.get('type', '')
        name = entry.get('name', '')
        categories = entry.get('categories', {})
        
        # Get Fine Fragrance limit from categories or cat4
        ff_limit = None
        for k, v in categories.items():
            if 'fine' in k.lower() or 'fragrance' in k.lower():
                ff_limit = v
                break
        
        # Override with cat4 if available
        if cas in cat4:
            cat4_info = cat4[cas]
            if cat4_info.get('max_pct_cat4') is not None:
                ff_limit = cat4_info['max_pct_cat4']
            if cat4_info.get('prohibited'):
                ff_limit = 0.0
                ifra_type = 'P'
        
        all_ifra[cas] = {
            'name': name,
            'type': ifra_type,
            'prohibited': ifra_type == 'P',
            'restricted': ifra_type in ('R', 'RS'),
            'ff_limit': ff_limit if ff_limit is not None else (0.0 if ifra_type == 'P' else None),
            'categories': categories,
        }
    
    prohibited = sum(1 for v in all_ifra.values() if v['prohibited'])
    restricted = sum(1 for v in all_ifra.values() if v['restricted'])
    with_limit = sum(1 for v in all_ifra.values() if v['ff_limit'] is not None)
    
    print(f"\n  Consolidated: {len(all_ifra)} total")
    print(f"  Prohibited: {prohibited}")
    print(f"  Restricted: {restricted}")
    print(f"  With Fine Fragrance limit: {with_limit}")
    
    # Now build CAS→SMILES lookup from ALL our data sources
    # Source: IFRA 2019 Pyrfume dataset (has 1060 molecules with SMILES!)
    ifra2019_path = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'ifra_2019', 'molecules.csv')
    cas_to_smiles = {}
    if os.path.exists(ifra2019_path):
        with open(ifra2019_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cas = row.get('CAS', '').strip()
                smi = row.get('IsomericSMILES', '').strip()
                if cas and smi:
                    cas_to_smiles[cas] = smi
        print(f"  IFRA 2019 Pyrfume: {len(cas_to_smiles)} CAS→SMILES")
    
    # Also load behavior from IFRA 2019 (has restriction info)
    ifra2019_beh = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'ifra_2019', 'behavior.csv')
    if os.path.exists(ifra2019_beh):
        with open(ifra2019_beh, 'r', encoding='utf-8', errors='replace') as f:
            beh_rows = sum(1 for _ in f) - 1
        print(f"  IFRA 2019 behavior: {beh_rows} rows")
    
    # Source: all other Pyrfume molecule files
    pyrfume_dir = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all')
    for subdir in os.listdir(pyrfume_dir):
        mol_path = os.path.join(pyrfume_dir, subdir, 'molecules.csv')
        if os.path.exists(mol_path):
            try:
                with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cas = row.get('CAS', '').strip()
                        smi = row.get('IsomericSMILES', '').strip()
                        if cas and smi and cas not in cas_to_smiles:
                            cas_to_smiles[cas] = smi
            except:
                pass
    print(f"  Total CAS→SMILES: {len(cas_to_smiles)}")
    
    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    print(f"  Ingredients: {len(ingredients)}")
    
    # Build canonical SMILES index
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
    
    # Clear old IFRA tags
    for ing in ingredients:
        for k in ['ifra_cas','ifra_type','ifra_restricted','ifra_prohibited','ifra_limit_pct','ifra_name']:
            ing.pop(k, None)
    
    # Match ALL 453 IFRA to ingredients
    matched = 0
    p_list, r_list = [], []
    
    # Include hardcoded CAS→SMILES from previous script
    from scripts.ifra_hardcoded import CAS_SMILES as hardcoded
    
    for cas, info in all_ifra.items():
        smi = None
        
        # Try CAS→SMILES lookup
        if cas in cas_to_smiles:
            smi = cas_to_smiles[cas]
        elif cas in hardcoded and hardcoded[cas]:
            smi = hardcoded[cas]
        
        if not smi:
            continue
        
        can = canonical(smi)
        if can not in can_to_idx:
            continue
        
        idx = can_to_idx[can]
        ingredients[idx]['ifra_cas'] = cas
        ingredients[idx]['ifra_type'] = info['type']
        ingredients[idx]['ifra_prohibited'] = info['prohibited']
        ingredients[idx]['ifra_restricted'] = info['prohibited'] or info['restricted']
        ingredients[idx]['ifra_name'] = info['name']
        
        if info['prohibited']:
            ingredients[idx]['ifra_limit_pct'] = 0.0
            p_list.append(f"{info['name']} ({cas})")
        elif info['ff_limit'] is not None:
            ingredients[idx]['ifra_limit_pct'] = info['ff_limit']
            r_list.append(f"{info['name']}: max {info['ff_limit']}%")
        else:
            ingredients[idx]['ifra_limit_pct'] = 100.0
        
        matched += 1
    
    # Enforce max_pct clamping
    clamped = 0
    for ing in ingredients:
        if ing.get('ifra_restricted') and not ing.get('ifra_prohibited'):
            limit = ing.get('ifra_limit_pct', 100)
            max_pct = ing.get('max_pct', 15)
            if max_pct > limit:
                ing['max_pct'] = limit
                clamped += 1
    
    total_tagged = sum(1 for x in ingredients if x.get('ifra_cas'))
    total_p = sum(1 for x in ingredients if x.get('ifra_prohibited'))
    total_r = sum(1 for x in ingredients if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    
    print(f"\n  === IFRA Full Results ===")
    print(f"  Matched: {matched}/453")
    print(f"  Tagged: {total_tagged}")
    print(f"  Prohibited: {total_p}")
    print(f"  Restricted: {total_r}")
    print(f"  Clamped: {clamped}")
    
    if p_list:
        print(f"\n  Prohibited ({len(p_list)}):")
        for p in p_list[:10]:
            print(f"    ❌ {p}")
        if len(p_list) > 10:
            print(f"    ... +{len(p_list)-10} more")
    
    if r_list:
        print(f"\n  Restricted with FF limits ({len(r_list)}):")
        for r in r_list[:10]:
            print(f"    ⚠️ {r}")
        if len(r_list) > 10:
            print(f"    ... +{len(r_list)-10} more")
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Update enforcement module
    enforcement = {
        'version': '51st Amendment (FULL)',
        'total_ifra_substances': len(all_ifra),
        'cas_to_smiles_available': len(cas_to_smiles),
        'matched_to_db': matched,
        'prohibited_count': total_p,
        'restricted_count': total_r,
        'clamped_count': clamped,
        'prohibited_smiles': [],
    }
    # Add prohibited SMILES
    for cas, info in all_ifra.items():
        if info['prohibited']:
            smi = cas_to_smiles.get(cas) or hardcoded.get(cas)
            if smi:
                enforcement['prohibited_smiles'].append(smi)
    
    enf_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_enforcement.json')
    with open(enf_path, 'w', encoding='utf-8') as f:
        json.dump(enforcement, f, indent=2)
    
    return matched, total_p, total_r

# ============================================================
# 2. Add molecules from UNUSED Pyrfume datasets
# ============================================================
def expand_ingredients_from_pyrfume():
    """Add molecules from unused Pyrfume datasets to ingredients.json"""
    print(f"\n{'='*60}")
    print("  [2] Expand Ingredients from Unused Pyrfume")
    print(f"{'='*60}")
    
    # Load current ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    
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
    
    # Build existing SMILES set
    existing = set()
    for ing in ingredients:
        smi = ing.get('smiles', '').strip()
        if smi:
            existing.add(canonical(smi))
    
    print(f"  Existing: {len(existing)} unique SMILES")
    
    # Datasets to import from (with behavior data = has odor descriptions)
    datasets_with_behavior = [
        ('aromadb', 'data/pom_data/pyrfume_all/aromadb'),
        ('flavornet', 'data/pom_data/pyrfume_all/flavornet'),
        ('goodscents', 'data/pom_data/pyrfume_all/goodscents'),
        ('leffingwell', 'data/pom_data/pyrfume_all/leffingwell'),
        ('fragrancedb', 'data/pom_data/pyrfume_all/fragrancedb'),
    ]
    
    added_total = 0
    
    for name, path in datasets_with_behavior:
        mol_path = os.path.join(BASE, path, 'molecules.csv')
        beh_path = os.path.join(BASE, path, 'behavior.csv')
        
        if not os.path.exists(mol_path):
            continue
        
        # Load molecules
        molecules = {}
        with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                smi = row.get('IsomericSMILES', '').strip()
                mol_name = row.get('name', row.get('Name', '')).strip()
                cas = row.get('CAS', '').strip()
                cid = row.get('CID', '').strip()
                if smi:
                    molecules[smi] = {'name': mol_name, 'cas': cas, 'cid': cid}
        
        # Load behavior (odor descriptions)
        behavior = {}
        if os.path.exists(beh_path):
            with open(beh_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    smi = row.get('IsomericSMILES', row.get('SMILES', '')).strip()
                    desc = row.get('descriptor', row.get('Descriptor', row.get('labels', ''))).strip()
                    if smi and desc:
                        behavior[smi] = desc
        
        added = 0
        for smi, info in molecules.items():
            can = canonical(smi)
            if can in existing:
                continue
            
            desc = behavior.get(smi, '')
            new_ing = {
                'id': info['name'].lower().replace(' ', '_')[:50] if info['name'] else f'{name}_{added}',
                'name_en': info['name'],
                'smiles': smi,
                'category': 'aromatic' if 'aromatic' in desc.lower() else 'general',
                'source': f'pyrfume_{name}',
            }
            if info['cas']:
                new_ing['cas_number'] = info['cas']
            if desc:
                new_ing['descriptors'] = desc
            
            ingredients.append(new_ing)
            existing.add(can)
            added += 1
        
        added_total += added
        print(f"  {name}: +{added} molecules (from {len(molecules)} total)")
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    print(f"\n  Total added: {added_total}")
    print(f"  New DB size: {len(ingredients)}")
    return added_total, len(ingredients)

# ============================================================
# Main
# ============================================================
def main():
    ifra_matched, p, r = expand_ifra_full()
    added, db_size = expand_ingredients_from_pyrfume()
    
    # Re-run IFRA matching after DB expansion
    print(f"\n{'='*60}")
    print("  [3] Re-Match IFRA After DB Expansion")
    print(f"{'='*60}")
    ifra2_matched, p2, r2 = expand_ifra_full()
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  IFRA: {ifra2_matched}/453 matched ({100*ifra2_matched//453}%)")
    print(f"    Prohibited: {p2}, Restricted: {r2}")
    print(f"  DB: {db_size} entries")
    print(f"  Added from Pyrfume: {added}")

if __name__ == '__main__':
    main()
