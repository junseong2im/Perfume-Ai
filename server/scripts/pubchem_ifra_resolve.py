"""
PubChem Batch CAS→SMILES Resolver + IFRA Full Integration
===========================================================
1. Query PubChem REST API for all 453 IFRA CAS numbers → get SMILES
2. Match resolved SMILES to our 4983 ingredient SMILES via RDKit canonical
3. Tag all matched ingredients with IFRA type/limit
4. Build enforcement module with full coverage
"""
import json, os, time, sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

IFRA_PATH = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
CAS_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'pubchem_cas_smiles.json')

def resolve_cas_to_smiles(cas_list, cache=None):
    """Batch resolve CAS numbers to SMILES via PubChem REST API"""
    if cache is None:
        cache = {}
    
    resolved = dict(cache)  # Start with cached results
    to_query = [cas for cas in cas_list if cas not in resolved]
    
    if not to_query:
        print(f"  All {len(cas_list)} CAS already cached")
        return resolved
    
    print(f"  Querying PubChem for {len(to_query)} CAS numbers...")
    success = 0
    failed = 0
    
    for i, cas in enumerate(to_query):
        try:
            # PubChem REST API: CAS → compound → canonical SMILES
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
            req = urllib.request.Request(url, headers={'User-Agent': 'POMEngine/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smi = props[0].get('CanonicalSMILES', '')
                    if smi:
                        resolved[cas] = smi
                        success += 1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                resolved[cas] = None  # Not found in PubChem
            failed += 1
        except Exception as e:
            failed += 1
        
        # Rate limiting: PubChem allows 5 requests/second
        if (i + 1) % 5 == 0:
            time.sleep(1.1)
        
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(to_query)}: {success} resolved, {failed} failed")
            # Save intermediate cache
            with open(CAS_CACHE, 'w') as f:
                json.dump(resolved, f, indent=2)
    
    print(f"  Final: {success} resolved, {failed} failed out of {len(to_query)}")
    
    # Save final cache
    with open(CAS_CACHE, 'w') as f:
        json.dump(resolved, f, indent=2)
    
    return resolved

def match_and_tag():
    """Match resolved SMILES to ingredient DB and tag with IFRA info"""
    # Load IFRA
    with open(IFRA_PATH, 'r', encoding='utf-8') as f:
        ifra_db = json.load(f)
    print(f"IFRA entries: {len(ifra_db)}")
    
    # Load or create CAS→SMILES cache
    cache = {}
    if os.path.exists(CAS_CACHE):
        with open(CAS_CACHE, 'r') as f:
            cache = json.load(f)
        print(f"CAS cache: {len(cache)} entries")
    
    # Resolve all IFRA CAS numbers
    cas_list = list(ifra_db.keys())
    resolved = resolve_cas_to_smiles(cas_list, cache)
    
    # Count resolved
    with_smiles = sum(1 for v in resolved.values() if v is not None and v)
    print(f"\n  Resolved SMILES: {with_smiles}/{len(cas_list)}")
    
    # Load ingredients
    ing_paths = [
        os.path.join(BASE, '..', 'data', 'ingredients.json'),
        os.path.join(BASE, 'data', 'ingredients.json'),
    ]
    ingredients = None
    ing_path_used = None
    for p in ing_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                ingredients = json.load(f)
            ing_path_used = p
            break
    
    if ingredients is None:
        print("ERROR: ingredients.json not found")
        return
    
    print(f"  Ingredients: {len(ingredients)}")
    
    # Build canonical SMILES → ingredient index
    try:
        from rdkit import Chem
        def canonical(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    return Chem.MolToSmiles(mol)
            except:
                pass
            return smi
        rdkit_ok = True
    except:
        canonical = lambda x: x
        rdkit_ok = False
    
    can_to_idx = {}
    for i, ing in enumerate(ingredients):
        smi = ing.get('smiles', '').strip()
        if smi:
            can = canonical(smi)
            if can:
                can_to_idx[can] = i
    
    print(f"  Canonical index: {len(can_to_idx)} entries (RDKit: {rdkit_ok})")
    
    # Clear old IFRA tags
    for ing in ingredients:
        for key in ['ifra_type', 'ifra_restricted', 'ifra_prohibited', 'ifra_name', 'ifra_limit_pct', 'ifra_cas']:
            ing.pop(key, None)
    
    # Match IFRA → ingredients via resolved SMILES
    matched = 0
    prohibited_list = []
    restricted_list = []
    
    for cas, ifra_entry in ifra_db.items():
        ifra_type = ifra_entry.get('type', '')
        ifra_name = ifra_entry.get('name', '')
        categories = ifra_entry.get('categories', {})
        
        # Get Fine Fragrance limit
        ff_limit = None
        for k, v in categories.items():
            if 'fine' in k.lower() or 'fragrance' in k.lower():
                ff_limit = v
                break
        if ff_limit is None and categories:
            ff_limit = list(categories.values())[0]  # Take first category limit
        
        # Try SMILES match
        smi = resolved.get(cas)
        if smi:
            can = canonical(smi)
            if can in can_to_idx:
                idx = can_to_idx[can]
                ingredients[idx]['ifra_cas'] = cas
                ingredients[idx]['ifra_type'] = ifra_type
                ingredients[idx]['ifra_restricted'] = ifra_type in ('P', 'R', 'RS')
                ingredients[idx]['ifra_prohibited'] = ifra_type == 'P'
                ingredients[idx]['ifra_name'] = ifra_name
                if ifra_type == 'P':
                    ingredients[idx]['ifra_limit_pct'] = 0.0
                    prohibited_list.append(f"{ifra_name} ({cas})")
                elif ff_limit is not None:
                    ingredients[idx]['ifra_limit_pct'] = ff_limit
                    restricted_list.append(f"{ifra_name}: {ff_limit}%")
                else:
                    ingredients[idx]['ifra_limit_pct'] = 100.0
                matched += 1
    
    total_tagged = sum(1 for ing in ingredients if ing.get('ifra_type'))
    total_prohibited = sum(1 for ing in ingredients if ing.get('ifra_prohibited'))
    total_restricted = sum(1 for ing in ingredients if ing.get('ifra_restricted') and not ing.get('ifra_prohibited'))
    
    print(f"\n  === Results ===")
    print(f"  SMILES-matched: {matched}")
    print(f"  Total IFRA-tagged: {total_tagged}")
    print(f"  Prohibited in DB: {total_prohibited}")
    print(f"  Restricted in DB: {total_restricted}")
    print(f"  IFRA Coverage: {total_tagged}/{len(ifra_db)} ({100*total_tagged/len(ifra_db):.0f}%)")
    
    if prohibited_list:
        print(f"\n  Prohibited substances found ({len(prohibited_list)}):")
        for p in prohibited_list[:15]:
            print(f"    ❌ {p}")
        if len(prohibited_list) > 15:
            print(f"    ... and {len(prohibited_list)-15} more")
    
    if restricted_list:
        print(f"\n  Restricted substances with limits ({len(restricted_list)}):")
        for r in restricted_list[:15]:
            print(f"    ⚠️ {r}")
        if len(restricted_list) > 15:
            print(f"    ... and {len(restricted_list)-15} more")
    
    # Save
    for path in ing_paths:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Update enforcement module
    enforcement = {
        'version': '51st Amendment',
        'total_substances': len(ifra_db),
        'resolved_smiles': with_smiles,
        'matched_to_db': matched,
        'prohibited_count': total_prohibited,
        'restricted_count': total_restricted,
        'prohibited_cas': [cas for cas, v in ifra_db.items() if v.get('type') == 'P'],
        'prohibited_smiles': [resolved.get(cas) for cas, v in ifra_db.items() 
                             if v.get('type') == 'P' and resolved.get(cas)],
    }
    enf_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_enforcement.json')
    with open(enf_path, 'w', encoding='utf-8') as f:
        json.dump(enforcement, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved enforcement module: {enf_path}")
    return total_tagged, total_prohibited, total_restricted

if __name__ == '__main__':
    match_and_tag()
