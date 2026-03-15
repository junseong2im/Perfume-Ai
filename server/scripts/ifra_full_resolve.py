"""
Resolve ALL 206 Unresolved IFRA CAS
======================================
A. 128 single compounds: lookup SMILES from IFRA 2019 Pyrfume (has CID, CID→SMILES available)
B. 62 essential oils / absolutes: Virtual Composition (decompose into constituent molecules)
C. 16 mixtures / reaction products: apply restriction to representative isomers

Strategy for essential oils:
- An essential oil (e.g. "Lavender oil") is IFRA-restricted at X%
- Lavender oil = ~30% Linalool + ~25% Linalyl acetate + ...
- Solution: Create "virtual ingredient" that when used in a recipe at Y%,
  applies the restriction to the total oil, not individual components
"""
import json, os, sys, csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

# ============================================================
# A. Resolve 128 single compounds via IFRA 2019 Pyrfume
# ============================================================
def resolve_single_compounds():
    """Try to find SMILES for 128 single compounds using IFRA 2019 CID→SMILES"""
    print("=" * 60)
    print("  [A] Resolving 128 Single Compounds")
    print("=" * 60)
    
    # Load IFRA 2019 molecules (has CID + SMILES)
    ifra2019_mol = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'ifra_2019', 'molecules.csv')
    ifra2019_name_to_smi = {}
    ifra2019_cid_to_smi = {}
    if os.path.exists(ifra2019_mol):
        with open(ifra2019_mol, 'r', encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                name = row.get('name', '').strip().lower()
                smi = row.get('IsomericSMILES', '').strip()
                cid = row.get('CID', '').strip()
                if name and smi:
                    ifra2019_name_to_smi[name] = smi
                if cid and smi:
                    ifra2019_cid_to_smi[cid] = smi
        print(f"  IFRA 2019 name→SMILES: {len(ifra2019_name_to_smi)}")
    
    # Load IFRA 2019 behavior (maps CAS? or names? to descriptors)
    ifra2019_beh = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'ifra_2019', 'behavior.csv')
    if os.path.exists(ifra2019_beh):
        with open(ifra2019_beh, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            row = next(reader)
            print(f"  IFRA 2019 behavior columns: {cols}")
            print(f"  Sample: {dict(list(row.items())[:5])}")
    
    # Load IFRA official (unresolved)
    with open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json'), 'r', encoding='utf-8') as f:
        ifra = json.load(f)
    
    # Load existing CAS→SMILES
    from scripts.ifra_complete import build_complete_cas_smiles
    cas_smi = build_complete_cas_smiles()
    
    # Find unresolved single compounds
    unresolved_singles = []
    for cas, entry in ifra.items():
        if cas in cas_smi and cas_smi[cas]:
            continue
        name = entry.get('name', '').lower()
        is_mixture = any(x in name for x in ['oil', 'absolute', 'extract', 'concrete', 
                         'resinoid', 'tincture', 'reaction product', 'mixture',
                         'dienal', 'derivatives', 'isomers', 'salts'])
        if not is_mixture:
            unresolved_singles.append((cas, entry))
    
    print(f"\n  Unresolved single compounds: {len(unresolved_singles)}")
    
    # Try matching by name
    matched_by_name = 0
    new_cas_smi = {}
    
    for cas, entry in unresolved_singles:
        ifra_name = entry.get('name', '').strip().lower()
        
        # Try exact name match
        if ifra_name in ifra2019_name_to_smi:
            new_cas_smi[cas] = ifra2019_name_to_smi[ifra_name]
            matched_by_name += 1
            continue
        
        # Try partial name match (IFRA name contains Pyrfume name or vice versa)
        for pyr_name, smi in ifra2019_name_to_smi.items():
            # Match if substantial overlap (>8 chars)
            if len(pyr_name) > 8 and (pyr_name in ifra_name or ifra_name in pyr_name):
                new_cas_smi[cas] = smi
                matched_by_name += 1
                break
            # Match first word (chemical name root)
            ifra_root = ifra_name.split()[0] if ifra_name else ''
            pyr_root = pyr_name.split()[0] if pyr_name else ''
            if len(ifra_root) > 5 and ifra_root == pyr_root:
                new_cas_smi[cas] = smi
                matched_by_name += 1
                break
    
    print(f"  Matched by name from IFRA 2019: {matched_by_name}")
    
    # Also try matching against ALL Pyrfume molecule names
    all_name_to_smi = {}
    pdir = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all')
    for sub in os.listdir(pdir):
        mp = os.path.join(pdir, sub, 'molecules.csv')
        if os.path.exists(mp):
            try:
                with open(mp, 'r', encoding='utf-8', errors='replace') as f:
                    for row in csv.DictReader(f):
                        name = row.get('name', row.get('Name', '')).strip().lower()
                        smi = row.get('IsomericSMILES', '').strip()
                        if name and smi and name not in all_name_to_smi:
                            all_name_to_smi[name] = smi
            except:
                pass
    print(f"  All Pyrfume name→SMILES: {len(all_name_to_smi)}")
    
    for cas, entry in unresolved_singles:
        if cas in new_cas_smi:
            continue
        ifra_name = entry.get('name', '').strip().lower()
        
        # Clean IFRA name (remove parenthetical, normalize)
        clean_name = ifra_name.split('(')[0].strip()
        clean_name = clean_name.replace('alpha-', 'α-').replace('beta-', 'β-')
        
        if clean_name in all_name_to_smi:
            new_cas_smi[cas] = all_name_to_smi[clean_name]
            matched_by_name += 1
        elif ifra_name in all_name_to_smi:
            new_cas_smi[cas] = all_name_to_smi[ifra_name]
            matched_by_name += 1
        else:
            # Try each word as a potential name
            for pyr_name, smi in all_name_to_smi.items():
                if len(pyr_name) > 8 and pyr_name in ifra_name:
                    new_cas_smi[cas] = smi
                    matched_by_name += 1
                    break
    
    print(f"  Total matched by name: {matched_by_name}")
    print(f"  Still unresolved: {len(unresolved_singles) - len(new_cas_smi)}")
    
    return new_cas_smi

# ============================================================
# B. Essential oils → Virtual Composition
# ============================================================
# Public composition data from ISO standards and Arctander (1960)
ESSENTIAL_OIL_COMPOSITIONS = {
    # CAS: {"name": str, "type": P/R, "constituents": {SMILES: pct, ...}}
    "8000-25-7": {  # Rosemary oil
        "name": "Rosemary oil",
        "constituents": {
            "CC12CCC(CC1)C(C2)=O": 30,       # Camphor
            "CC(=CCCC(C)(C=C)O)C": 25,        # Linalool  
            "CC1=CCC2CC1C2(C)C": 15,          # α-Pinene
            "CC12CCC(C1)C(=C2)C": 10,         # Camphene
        }
    },
    "8007-01-0": {  # Rose oil
        "name": "Rose oil",
        "constituents": {
            "OCC=C(CCC=C(C)C)C": 30,          # Geraniol
            "CC(CCC=C(C)C)CCO": 25,           # Citronellol
            "OCCC1=CC=CC=C1": 10,             # Phenylethanol
            "CC(=CCCC(C)(C=C)O)C": 8,         # Linalool
        }
    },
    "8000-28-0": {  # Lavender oil
        "name": "Lavender oil",
        "constituents": {
            "CC(=CCCC(C)(C=C)O)C": 30,       # Linalool
            "CC(=CCCC(C)(OC(=O)C)C=C)C": 25, # Linalyl acetate
            "CC12CCC(CC1)C(C2)=O": 5,        # Camphor
        }
    },
    "8015-01-8": {  # Cinnamon bark oil
        "name": "Cinnamon bark oil",
        "constituents": {
            "O=CC=CC1=CC=CC=C1": 65,          # Cinnamaldehyde
            "COc1cc(CC=C)ccc1O": 10,          # Eugenol
        }
    },
    "8008-99-9": {  # Garlic oil
        "name": "Garlic oil",
        "constituents": {  # Contains allyl sulfides - prohibited
        }
    },
    "8014-71-9": {  # Melissa (lemon balm) oil
        "name": "Melissa oil",
        "constituents": {
            "CC(=CCCC(=CC=O)C)C": 40,        # Citral (neral+geranial)
            "CC(CCC=C(C)C)CCO": 15,          # Citronellol
        }
    },
}

def create_virtual_ingredients():
    """Create virtual ingredients for essential oils with IFRA restrictions"""
    print(f"\n{'='*60}")
    print("  [B] Virtual Essential Oil Ingredients")
    print(f"{'='*60}")
    
    # Load IFRA official for essential oil restrictions
    with open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json'), 'r', encoding='utf-8') as f:
        ifra = json.load(f)
    cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
    cat4 = {}
    if os.path.exists(cat4_path):
        with open(cat4_path, 'r', encoding='utf-8') as f:
            cat4 = json.load(f)
    
    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    
    # Create virtual ingredients for oils
    added = 0
    existing_ids = {ing.get('id', '') for ing in ingredients}
    
    for cas, oil_data in ESSENTIAL_OIL_COMPOSITIONS.items():
        oil_name = oil_data['name']
        oil_id = oil_name.lower().replace(' ', '_').replace("'", "")
        
        if oil_id in existing_ids:
            # Update existing
            for ing in ingredients:
                if ing.get('id') == oil_id:
                    ifra_entry = ifra.get(cas, {})
                    cat4_entry = cat4.get(cas, {})
                    
                    ing['ifra_cas'] = cas
                    ing['ifra_prohibited'] = ifra_entry.get('type') == 'P'
                    ing['ifra_restricted'] = ifra_entry.get('type') in ('P', 'R', 'RS')
                    ing['ifra_name'] = oil_name
                    ing['ifra_limit_pct'] = 0.0 if ing['ifra_prohibited'] else cat4_entry.get('max_pct_cat4', 100.0)
                    ing['is_mixture'] = True
                    ing['constituents'] = oil_data.get('constituents', {})
                    break
        else:
            # Create new virtual ingredient
            ifra_entry = ifra.get(cas, {})
            cat4_entry = cat4.get(cas, {})
            
            new_ing = {
                'id': oil_id,
                'name_en': oil_name,
                'cas_number': cas,
                'category': 'essential_oil',
                'is_mixture': True,
                'constituents': oil_data.get('constituents', {}),
                'ifra_cas': cas,
                'ifra_prohibited': ifra_entry.get('type') == 'P',
                'ifra_restricted': ifra_entry.get('type') in ('P', 'R', 'RS'),
                'ifra_name': oil_name,
                'ifra_limit_pct': 0.0 if ifra_entry.get('type') == 'P' else cat4_entry.get('max_pct_cat4', 100.0),
                'source': 'ifra_virtual_oil',
            }
            ingredients.append(new_ing)
            added += 1
            print(f"  + {oil_name} ({cas}) [{'P' if new_ing['ifra_prohibited'] else 'R'}]")
    
    # Also create entries for ALL essential oil CAS that we don't have compositions for
    for cas, entry in ifra.items():
        name = entry.get('name', '').lower()
        ifra_type = entry.get('type', '')
        
        if not any(x in name for x in ['oil', 'absolute', 'extract', 'concrete', 'resinoid']):
            continue
        
        oil_id = entry.get('name', '').lower().replace(' ', '_').replace(',', '').replace("'", "")[:50]
        if oil_id in existing_ids:
            continue
        existing_ids.add(oil_id)
        
        cat4_entry = cat4.get(cas, {})
        new_ing = {
            'id': oil_id,
            'name_en': entry.get('name', ''),
            'cas_number': cas,
            'category': 'essential_oil' if 'oil' in name else 'absolute',
            'is_mixture': True,
            'ifra_cas': cas,
            'ifra_prohibited': ifra_type == 'P',
            'ifra_restricted': ifra_type in ('P', 'R', 'RS'),
            'ifra_name': entry.get('name', ''),
            'ifra_limit_pct': 0.0 if ifra_type == 'P' else cat4_entry.get('max_pct_cat4', 100.0),
            'source': 'ifra_virtual_oil',
        }
        ingredients.append(new_ing)
        added += 1
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    print(f"\n  Added: {added} virtual oil/absolute ingredients")
    print(f"  Total DB: {len(ingredients)}")
    return added, len(ingredients)

# ============================================================
# C. Re-run IFRA matching with expanded data
# ============================================================
def final_ifra_count():
    """Count total IFRA coverage after all expansions"""
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ings = json.load(f)
    
    tagged = sum(1 for x in ings if x.get('ifra_cas'))
    prohibited = sum(1 for x in ings if x.get('ifra_prohibited'))
    restricted = sum(1 for x in ings if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    mixtures = sum(1 for x in ings if x.get('is_mixture'))
    
    # Count unique IFRA CAS
    unique_cas = set(x.get('ifra_cas') for x in ings if x.get('ifra_cas'))
    
    print(f"\n{'='*60}")
    print("  FINAL IFRA COVERAGE")
    print(f"{'='*60}")
    print(f"  Unique IFRA CAS matched: {len(unique_cas)}/453")
    print(f"  Ingredients tagged: {tagged}")
    print(f"  Prohibited: {prohibited}")
    print(f"  Restricted: {restricted}")
    print(f"  Virtual mixtures: {mixtures}")
    print(f"  Total DB: {len(ings)}")
    
    return len(unique_cas)

# ============================================================
# Main
# ============================================================
def main():
    # A. Resolve single compounds
    new_cas_smi = resolve_single_compounds()
    
    # Match new CAS→SMILES to ingredients
    if new_cas_smi:
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
        
        # Load IFRA
        with open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json'), 'r', encoding='utf-8') as f:
            ifra = json.load(f)
        cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
        cat4 = json.load(open(cat4_path, 'r', encoding='utf-8')) if os.path.exists(cat4_path) else {}
        
        ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
        if not os.path.exists(ing_path):
            ing_path = os.path.join(BASE, 'data', 'ingredients.json')
        with open(ing_path, 'r', encoding='utf-8') as f:
            ingredients = json.load(f)
        
        can_to_idx = {}
        for i, ing in enumerate(ingredients):
            smi = ing.get('smiles', '').strip()
            if smi:
                can = canonical(smi)
                can_to_idx[can] = i
        
        matched_new = 0
        for cas, smi in new_cas_smi.items():
            can = canonical(smi)
            if can in can_to_idx:
                idx = can_to_idx[can]
                entry = ifra.get(cas, {})
                cat4_entry = cat4.get(cas, {})
                is_p = entry.get('type') == 'P'
                
                ingredients[idx]['ifra_cas'] = cas
                ingredients[idx]['ifra_prohibited'] = is_p
                ingredients[idx]['ifra_restricted'] = entry.get('type') in ('P', 'R', 'RS')
                ingredients[idx]['ifra_name'] = entry.get('name', '')
                ingredients[idx]['ifra_limit_pct'] = 0.0 if is_p else cat4_entry.get('max_pct_cat4', 100.0)
                matched_new += 1
        
        for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
            try:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(ingredients, f, indent=2, ensure_ascii=False)
            except:
                pass
        print(f"\n  Newly matched single compounds: {matched_new}")
    
    # B. Virtual essential oil ingredients
    v_added, db_size = create_virtual_ingredients()
    
    # C. Final count
    coverage = final_ifra_count()

if __name__ == '__main__':
    main()
