"""
IFRA CAS→SMILES Direct Matcher
================================
Instead of matching CAS in ingredients (which have no CAS),
match IFRA CAS → PubChem SMILES → ingredient SMILES.
Uses IFRA full data which may have SMILES directly,
or builds a CAS→SMILES lookup from the molecule CSVs we already have.
"""
import json, os, csv
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')

def build_cas_smiles_lookup():
    """Build CAS → SMILES from all available molecule CSV files"""
    lookup = {}  # CAS → SMILES
    name_to_smiles = {}  # name → SMILES
    
    # Source 1: Pyrfume molecule files (have CID, CAS sometimes, SMILES)
    pyrfume_dirs = [
        'data/pom_data/pyrfume_all/abraham_2012',
        'data/pom_data/pyrfume_all/snitz_2013',
        'data/pom_data/pyrfume_all/bushdid_2014',
    ]
    for pdir in pyrfume_dirs:
        mol_path = os.path.join(BASE, pdir, 'molecules.csv')
        if os.path.exists(mol_path):
            try:
                with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
                    for row in csv.DictReader(f):
                        smi = row.get('IsomericSMILES', '').strip()
                        name = row.get('name', '').lower().strip()
                        cas = row.get('CAS', '').strip()
                        if smi:
                            if cas:
                                lookup[cas] = smi
                            if name:
                                name_to_smiles[name] = smi
            except:
                pass

    # Source 2: curated CSV (has SMILES + descriptors)
    curated = os.path.join(BASE, 'data', 'curated_GS_LF_merged_4983.csv')
    if os.path.exists(curated):
        with open(curated, 'r', encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if row and row[0] and row[0] != 'nonStereoSMILES':
                    smi = row[0].strip()
                    desc = row[1].strip().lower() if len(row) > 1 else ''
                    if desc:
                        for part in desc.split(','):
                            name_to_smiles[part.strip()] = smi

    # Source 3: GoodScents data if available
    gs_files = [
        'data/raw/pyrfume/snitz_2013_molecules.csv',
        'data/raw/pyrfume/bushdid_2014_molecules.csv',
    ]
    for gf in gs_files:
        fp = os.path.join(BASE, gf)
        if os.path.exists(fp):
            try:
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    for row in csv.DictReader(f):
                        smi = row.get('IsomericSMILES', row.get('SMILES', '')).strip()
                        cas = row.get('CAS', '').strip()
                        name = row.get('name', row.get('Name', '')).lower().strip()
                        if smi and cas:
                            lookup[cas] = smi
                        if smi and name:
                            name_to_smiles[name] = smi
            except:
                pass

    print(f"  CAS→SMILES: {len(lookup)} entries")
    print(f"  Name→SMILES: {len(name_to_smiles)} entries")
    return lookup, name_to_smiles

def match_ifra_to_ingredients():
    """Match IFRA entries to ingredients via SMILES"""
    # Load IFRA
    ifra_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
    with open(ifra_path, 'r', encoding='utf-8') as f:
        ifra_db = json.load(f)
    print(f"IFRA entries: {len(ifra_db)}")

    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    print(f"Ingredients: {len(ingredients)}")

    # Build SMILES index for ingredients
    smi_to_idx = {}
    name_to_idx = {}
    for i, ing in enumerate(ingredients):
        smi = ing.get('smiles', '').strip()
        if smi:
            smi_to_idx[smi] = i
        name = ing.get('id', '').lower().strip()
        if name:
            name_to_idx[name] = i
        name_en = ing.get('name_en', '').lower().strip()
        if name_en:
            name_to_idx[name_en] = i

    # Build CAS→SMILES lookup
    cas_lookup, name_lookup = build_cas_smiles_lookup()

    # Also try RDKit canonical SMILES matching
    try:
        from rdkit import Chem
        def canonical(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                return Chem.MolToSmiles(mol)
            return smi
        
        # Build canonical SMILES index
        can_to_idx = {}
        for smi, idx in smi_to_idx.items():
            can = canonical(smi)
            if can:
                can_to_idx[can] = idx
        print(f"  Canonical SMILES index: {len(can_to_idx)}")
        rdkit_available = True
    except:
        can_to_idx = smi_to_idx
        canonical = lambda x: x
        rdkit_available = False

    # Match IFRA → ingredients
    matched_cas_smi = 0
    matched_name = 0
    unmatched = 0
    prohibited_in_db = []

    for cas, ifra_entry in ifra_db.items():
        ifra_type = ifra_entry.get('type', '')
        ifra_name = ifra_entry.get('name', '')
        categories = ifra_entry.get('categories', {})
        matched = False

        # Method 1: CAS → SMILES → ingredient
        if cas in cas_lookup:
            smi = cas_lookup[cas]
            can = canonical(smi) if rdkit_available else smi
            if can in can_to_idx:
                idx = can_to_idx[can]
                ingredients[idx]['ifra_cas'] = cas
                ingredients[idx]['ifra_type'] = ifra_type
                ingredients[idx]['ifra_restricted'] = ifra_type in ('P', 'R', 'RS')
                ingredients[idx]['ifra_prohibited'] = ifra_type == 'P'
                ingredients[idx]['ifra_name'] = ifra_name
                ingredients[idx]['ifra_limit_pct'] = 0.0 if ifra_type == 'P' else categories.get('Fine Fragrance', 100.0)
                matched_cas_smi += 1
                matched = True
                if ifra_type == 'P':
                    prohibited_in_db.append(ifra_name)

        # Method 2: IFRA name → ingredient name
        if not matched:
            ifra_name_clean = ifra_name.lower().replace(' ', '_').replace('-', '_').replace(',', '')
            # Try direct
            if ifra_name_clean in name_to_idx:
                idx = name_to_idx[ifra_name_clean]
                ingredients[idx]['ifra_cas'] = cas
                ingredients[idx]['ifra_type'] = ifra_type
                ingredients[idx]['ifra_restricted'] = ifra_type in ('P', 'R', 'RS')
                ingredients[idx]['ifra_prohibited'] = ifra_type == 'P'
                ingredients[idx]['ifra_name'] = ifra_name
                ingredients[idx]['ifra_limit_pct'] = 0.0 if ifra_type == 'P' else categories.get('Fine Fragrance', 100.0)
                matched_name += 1
                matched = True
                if ifra_type == 'P':
                    prohibited_in_db.append(ifra_name)

        # Method 3: IFRA name contains common ingredient name
        if not matched:
            for ing_name, idx in name_to_idx.items():
                # Match if IFRA name contains ingredient name (>5 chars to avoid false positives)
                if len(ing_name) > 5 and ing_name in ifra_name_clean:
                    ingredients[idx]['ifra_cas'] = cas
                    ingredients[idx]['ifra_type'] = ifra_type
                    ingredients[idx]['ifra_restricted'] = ifra_type in ('P', 'R', 'RS')
                    ingredients[idx]['ifra_prohibited'] = ifra_type == 'P'
                    ingredients[idx]['ifra_name'] = ifra_name
                    ingredients[idx]['ifra_limit_pct'] = 0.0 if ifra_type == 'P' else categories.get('Fine Fragrance', 100.0)
                    matched_name += 1
                    matched = True
                    if ifra_type == 'P':
                        prohibited_in_db.append(ifra_name)
                    break

        if not matched:
            unmatched += 1

    total_tagged = sum(1 for ing in ingredients if ing.get('ifra_type'))
    total_prohibited = sum(1 for ing in ingredients if ing.get('ifra_prohibited'))
    total_restricted = sum(1 for ing in ingredients if ing.get('ifra_restricted') and not ing.get('ifra_prohibited'))

    print(f"\n  === IFRA Matching Results ===")
    print(f"  CAS→SMILES matched: {matched_cas_smi}")
    print(f"  Name matched: {matched_name}")
    print(f"  Unmatched: {unmatched}")
    print(f"  Total IFRA-tagged: {total_tagged}")
    print(f"  Prohibited in DB: {total_prohibited}")
    print(f"  Restricted in DB: {total_restricted}")
    
    if prohibited_in_db:
        print(f"\n  Prohibited substances found in DB:")
        for p in prohibited_in_db[:20]:
            print(f"    ❌ {p}")

    # Save
    with open(ing_path, 'w', encoding='utf-8') as f:
        json.dump(ingredients, f, indent=2, ensure_ascii=False)
    # Also update alternative path
    alt_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(alt_path, 'w', encoding='utf-8') as f:
        json.dump(ingredients, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {ing_path} and {alt_path}")

    return total_tagged, total_prohibited, total_restricted

if __name__ == '__main__':
    match_ifra_to_ingredients()
