"""
IFRA Integration v2: Use clean 59 CAS from cat4.json + full 453 from official.json
Strategy: 
1. PubChem CAS→SMILES for 59 clean CAS
2. RDKit canonical matching to 4983 ingredients
3. Tag all matches + build prohibition/restriction enforcement
"""
import json, os, time, sys, math
import urllib.request, urllib.parse, urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

CAS_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'pubchem_cas_smiles.json')

def pubchem_cas_to_smiles(cas):
    """Query PubChem: CAS → CID → SMILES"""
    # Method 1: Direct CAS via PUG REST synonym search
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES,IUPACName/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "POM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
            props = d.get('PropertyTable', {}).get('Properties', [])
            if props:
                return props[0].get('CanonicalSMILES', ''), props[0].get('IUPACName', '')
    except:
        pass
    
    # Method 2: Try via xref
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RN/{cas}/cids/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "POM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
            cids_info = d.get('InformationList', {}).get('Information', [])
            if cids_info:
                cid = cids_info[0].get('CID', [None])[0]
                if cid:
                    url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName/JSON"
                    req2 = urllib.request.Request(url2, headers={"User-Agent": "POM/1.0"})
                    with urllib.request.urlopen(req2, timeout=15) as r2:
                        d2 = json.loads(r2.read())
                        props = d2.get('PropertyTable', {}).get('Properties', [])
                        if props:
                            return props[0].get('CanonicalSMILES', ''), props[0].get('IUPACName', '')
    except:
        pass
    
    return None, None

def main():
    # Load clean CAS list
    cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
    with open(cat4_path, 'r', encoding='utf-8') as f:
        cat4 = json.load(f)
    print(f"Clean CAS entries: {len(cat4)}")
    
    # Also load full IFRA
    ifra_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
    with open(ifra_path, 'r', encoding='utf-8') as f:
        ifra_full = json.load(f)
    print(f"Full IFRA entries: {len(ifra_full)}")
    
    # Load cache
    cache = {}
    if os.path.exists(CAS_CACHE):
        with open(CAS_CACHE, 'r') as f:
            cache = json.load(f)
    
    # Resolve all CAS from cat4 (59 clean CAS)
    all_cas = list(cat4.keys())
    to_query = [cas for cas in all_cas if cas not in cache or cache[cas] is None]
    
    print(f"\nResolving {len(to_query)} CAS via PubChem (cached: {len(all_cas)-len(to_query)})...")
    success = 0
    for i, cas in enumerate(to_query):
        smi, name = pubchem_cas_to_smiles(cas)
        if smi:
            cache[cas] = smi
            success += 1
            print(f"  ✅ {cas} → {smi[:40]} ({name[:30] if name else ''})")
        else:
            cache[cas] = None
            print(f"  ❌ {cas}")
        
        time.sleep(0.5)  # Rate limit
        
        if (i + 1) % 20 == 0:
            with open(CAS_CACHE, 'w') as f:
                json.dump(cache, f, indent=2)
    
    # Save cache
    with open(CAS_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    resolved_count = sum(1 for cas in all_cas if cache.get(cas))
    print(f"\nResolved: {resolved_count}/{len(all_cas)}")
    
    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    print(f"Ingredients: {len(ingredients)}")
    
    # Build canonical SMILES index
    try:
        from rdkit import Chem
        def canonical(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol) if mol else smi
            except:
                return smi
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
        for key in ['ifra_type', 'ifra_restricted', 'ifra_prohibited', 'ifra_name', 'ifra_limit_pct', 'ifra_cas']:
            ing.pop(key, None)
    
    # Match cat4 CAS → SMILES → ingredients
    matched = 0
    prohibited_list = []
    restricted_list = []
    
    for cas, info in cat4.items():
        smi = cache.get(cas)
        if not smi:
            continue
        can = canonical(smi)
        if can not in can_to_idx:
            continue
        
        idx = can_to_idx[can]
        is_prohibited = info.get('prohibited', False)
        is_restricted = info.get('restricted', False)
        max_pct = info.get('max_pct_cat4', 100.0)
        
        # Get name from full IFRA
        ifra_name = ifra_full.get(cas, {}).get('name', cas)
        
        ingredients[idx]['ifra_cas'] = cas
        ingredients[idx]['ifra_prohibited'] = is_prohibited
        ingredients[idx]['ifra_restricted'] = is_prohibited or is_restricted
        ingredients[idx]['ifra_limit_pct'] = 0.0 if is_prohibited else max_pct
        ingredients[idx]['ifra_name'] = ifra_name
        matched += 1
        
        if is_prohibited:
            prohibited_list.append(f"{ifra_name} ({cas})")
        elif is_restricted:
            restricted_list.append(f"{ifra_name}: max {max_pct}% ({cas})")
    
    # Also check the full 453 IFRA for additional matches
    for cas, info in ifra_full.items():
        if cas in cat4:
            continue  # Already processed
        smi = cache.get(cas)
        if not smi:
            continue
        can = canonical(smi)
        if can not in can_to_idx:
            continue
        idx = can_to_idx[can]
        if ingredients[idx].get('ifra_cas'):
            continue  # Already tagged
        
        ifra_type = info.get('type', '')
        ingredients[idx]['ifra_cas'] = cas
        ingredients[idx]['ifra_prohibited'] = ifra_type == 'P'
        ingredients[idx]['ifra_restricted'] = ifra_type in ('P', 'R', 'RS')
        ingredients[idx]['ifra_limit_pct'] = 0.0 if ifra_type == 'P' else 100.0
        ingredients[idx]['ifra_name'] = info.get('name', cas)
        matched += 1
    
    total_tagged = sum(1 for ing in ingredients if ing.get('ifra_cas'))
    total_prohibited = sum(1 for ing in ingredients if ing.get('ifra_prohibited'))
    total_restricted = sum(1 for ing in ingredients if ing.get('ifra_restricted') and not ing.get('ifra_prohibited'))
    
    print(f"\n=== IFRA Integration Results ===")
    print(f"Matched to DB: {matched}")
    print(f"IFRA-tagged: {total_tagged}")
    print(f"Prohibited: {total_prohibited}")
    print(f"Restricted: {total_restricted}")
    
    if prohibited_list:
        print(f"\n❌ Prohibited ({len(prohibited_list)}):")
        for p in prohibited_list:
            print(f"  {p}")
    if restricted_list:
        print(f"\n⚠️ Restricted ({len(restricted_list)}):")
        for r in restricted_list:
            print(f"  {r}")
    
    # Save
    for path in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Build enforcement
    enforcement = {
        'version': '51st Amendment',
        'total_ifra_substances': len(ifra_full),
        'resolved_smiles': resolved_count,
        'matched_to_db': matched,
        'prohibited_count': total_prohibited,
        'restricted_count': total_restricted,
        'prohibited_smiles': [cache.get(cas) for cas in cat4 if cat4[cas].get('prohibited') and cache.get(cas)],
    }
    enf_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_enforcement.json')
    with open(enf_path, 'w', encoding='utf-8') as f:
        json.dump(enforcement, f, indent=2)
    
    print(f"\nSaved enforcement module")

if __name__ == '__main__':
    main()
