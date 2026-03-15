import json, urllib.request, urllib.error, time, traceback, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

ifra = json.load(open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json'), 'r', encoding='utf-8'))
cas_list = list(ifra.keys())
print(f"Total CAS: {len(cas_list)}")

results = {}
for i, cas in enumerate(cas_list):
    smi = None
    iupac = None
    
    # Method 1: direct CAS as name
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES,IUPACName/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "POM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode('utf-8'))
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                smi = props[0].get('CanonicalSMILES', '')
                iupac = props[0].get('IUPACName', '')
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  HTTP {e.code} for {cas}")
    except Exception as e:
        print(f"  ERR1 {cas}: {type(e).__name__}: {e}")
    
    # Method 2: xref if method 1 failed
    if not smi:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/xref/RN/{cas}/cids/JSON"
            req = urllib.request.Request(url, headers={"User-Agent": "POM/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode('utf-8'))
                info = data.get('InformationList', {}).get('Information', [])
                if info:
                    cid_list = info[0].get('CID', [])
                    if cid_list:
                        cid = cid_list[0]
                        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName/JSON"
                        req2 = urllib.request.Request(url2, headers={"User-Agent": "POM/1.0"})
                        with urllib.request.urlopen(req2, timeout=15) as r2:
                            data2 = json.loads(r2.read().decode('utf-8'))
                            props = data2.get('PropertyTable', {}).get('Properties', [])
                            if props:
                                smi = props[0].get('CanonicalSMILES', '')
                                iupac = props[0].get('IUPACName', '')
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"  HTTP {e.code} for {cas} (xref)")
        except Exception as e:
            print(f"  ERR2 {cas}: {type(e).__name__}: {e}")
    
    if smi:
        results[cas] = smi
        short_name = iupac[:25] if iupac else ''
        print(f"  OK {cas} -> {smi[:35]} ({short_name})")
    else:
        results[cas] = None
    
    time.sleep(0.3)
    
    if (i + 1) % 20 == 0:
        ok = sum(1 for v in results.values() if v)
        print(f"  Progress: {i+1}/{len(cas_list)}, resolved: {ok}")

# Save
cache_path = os.path.join(BASE, 'data', 'pom_upgrade', 'pubchem_cas_smiles.json')
with open(cache_path, 'w') as f:
    json.dump(results, f, indent=2)

ok = sum(1 for v in results.values() if v)
print(f"\nFINAL: {ok}/{len(cas_list)} resolved")

# Now match to ingredients
if ok > 0:
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
    
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    ings = json.load(open(ing_path, 'r', encoding='utf-8'))
    
    can_idx = {}
    for i, ing in enumerate(ings):
        s = ing.get('smiles', '')
        if s:
            can_idx[canonical(s)] = i
    
    # Clear old tags
    for ing in ings:
        for k in ['ifra_cas','ifra_type','ifra_restricted','ifra_prohibited','ifra_limit_pct','ifra_name']:
            ing.pop(k, None)
    
    matched = 0
    for cas, info in ifra.items():
        smi = results.get(cas)
        if not smi:
            continue
        can = canonical(smi)
        if can in can_idx:
            idx = can_idx[can]
            ings[idx]['ifra_cas'] = cas
            ings[idx]['ifra_prohibited'] = info.get('prohibited', False)
            ings[idx]['ifra_restricted'] = info.get('prohibited', False) or info.get('restricted', False)
            ings[idx]['ifra_limit_pct'] = 0.0 if info.get('prohibited') else info.get('max_pct_cat4', 100.0)
            matched += 1
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            json.dump(ings, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        except:
            pass
    
    total_p = sum(1 for x in ings if x.get('ifra_prohibited'))
    total_r = sum(1 for x in ings if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    print(f"Matched: {matched}, Prohibited: {total_p}, Restricted: {total_r}")
