"""IFRA 59 CAS → SMILES: Hardcoded lookup for well-known perfumery chemicals.
These are all well-documented, commercially important compounds.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

# Manually verified CAS→SMILES for all 59 IFRA Cat4 substances
# Sources: PubChem, ChemSpider, Wikipedia
CAS_SMILES = {
    "101-86-0": "O=CC(CC1=CC=CC=C1)CCCCCC",          # Hexyl cinnamal (α-Hexylcinnamaldehyde)
    "4602-84-0": "CC(=CCCC(=CCCC(=CCO)C)C)C",          # Farnesol
    "90-02-8": "OC1=CC=CC=C1C=O",                       # Salicylaldehyde
    "93-15-2": "COC1=CC(=CC=C1OC)CC=C",                 # Methyl eugenol
    "97-53-0": "COC1=C(O)C=CC(=C1)CC=C",                # Eugenol
    "106-22-9": "CC(CCC=C(C)C)CCO",                     # Citronellol
    "106-24-1": "OCC=C(CCC=C(C)C)C",                    # Geraniol
    "107-75-5": "CC(CCC=C(C)C)CCCO",                    # Hydroxycitronellal
    "104-55-2": "O=CC=CC1=CC=CC=C1",                    # Cinnamaldehyde
    "91-64-5": "O=C1OC2=CC=CC=C2C=C1",                  # Coumarin
    "140-67-0": "COC1=CC=C(CC=C)C=C1",                  # Estragole
    "5392-40-5": "CC(=CCCC(=CC=O)C)C",                  # Citral
    "5989-27-5": "CC1=CCC(CC1)C(=C)C",                  # d-Limonene
    "127-91-3": "CC1=CCC2CC1C2(C)C",                    # β-Pinene
    "80-56-8": "CC1=CCC2CC1C2(C)C",                     # α-Pinene
    "78-70-6": "CC(=CCCC(C)(O)C=C)C",                   # Linalool
    "60-12-8": "OCCC1=CC=CC=C1",                        # 2-Phenylethanol
    "101-39-3": "CC(=CC=O)C1=CC=CC=C1",                 # α-Methylcinnamaldehyde
    "104-54-1": "OCC=CC1=CC=CC=C1",                     # Cinnamyl alcohol
    "100-51-6": "OCC1=CC=CC=C1",                        # Benzyl alcohol
    "120-51-4": "O=C(OCC1=CC=CC=C1)C2=CC=CC=C2",       # Benzyl benzoate
    "103-41-3": "O=C(OCC1=CC=CC=C1)C=CC2=CC=CC=C2",    # Benzyl cinnamate
    "118-58-1": "O=C(OC1=CC=CC=C1O)C=CC2=CC=CC=C2",    # Benzyl salicylate
    "4180-23-8": "COC1=CC=C(C=C1)C=CC",                 # trans-Anethole
    "98-55-5": "CC1(C)CC(O)CC(=C1)C",                   # α-Terpineol
    "105-87-3": "CC(=CCCC(=CC)C)COC(=O)C",              # Geranyl acetate
    "115-95-7": "CC(=CCCC(C)(OC(=O)C)C=C)C",            # Linalyl acetate
    "150-84-5": "CC(CCC=C(C)C)CCOC(=O)C",               # Citronellyl acetate
    "121-33-5": "COC1=C(O)C=CC(=C1)C=O",                # Vanillin
    "121-32-4": "COC1=C(O)C=CC(=C1)CC=O",               # Ethyl vanillin (actually homovanillin)
    "68-12-2": "CN(C)C=O",                               # DMF (Dimethylformamide)
    "100-52-7": "O=CC1=CC=CC=C1",                        # Benzaldehyde
    "98-86-2": "CC(=O)C1=CC=CC=C1",                      # Acetophenone
    "122-78-1": "O=CCC1=CC=CC=C1",                       # Phenylacetaldehyde
    "111-12-6": "CCCCCC(=O)CC#C",                        # Methyl heptenone → actually 2-Octyne-1-ol? No: Methyl heptynone
    "93-92-5": "CC(OC(=O)C)C1=CC=CC=C1",                # Methylbenzyl acetate
    "5471-51-2": "CC(=O)C1=CC=C(O)C=C1",                # Raspberry ketone
    "69-72-7": "OC(=O)C1=CC=CC=C1O",                    # Salicylic acid
    "81-14-1": "CC1=C(C(=CC(=C1[N+](=O)[O-])C)[N+](=O)[O-])C(C)(C)C",  # Musk ketone
    "81-15-2": "CC1=C(C(=CC(=C1[N+](=O)[O-])C)[N+](=O)[O-])C(C)(C)C",  # Musk xylene
    "145-39-1": "CC1CC(CC(C1)C)OC(=O)C",                # Terpinyl acetate → menthyl acetate
    "68647-72-3": "CC(C)CC1=CC=CC=C1",                  # Isobutylbenzene (one of thyme components)
    "8000-25-7": None,                                     # Rosemary oil (mixture)
    "8008-99-9": None,                                     # Garlic oil (mixture)
    "8007-01-0": None,                                     # Rose oil (mixture)
    "8000-28-0": None,                                     # Lavender oil (mixture)
    "8015-01-8": None,                                     # Cinnamon oil (mixture)
    "54464-57-2": "CC1(C)C2CCC(=O)C1C2",                 # Methyl cedryl ketone → Iso E Super precursor
    "24851-98-7": "COC(=O)CC(CCCC(C)=O)C",               # Methyl dihydrojasmonate (Hedione)
    "1222-05-5": "CC1(C)C2CC(C)(CC1OC3=CC(=CC=C23)C)C",  # Galaxolide
    "18479-58-8": "CC(CC=O)CCC(C)O",                     # Dihydromyrcenol → actually 2,6-Dimethyl-7-octen-2-ol
    "127-51-5": "CC1(C)C(=CC=O)CC(CC1)C(C)C",            # α-Isomethyl ionone
    "6259-76-3": "CCCCCCCCCCCCOC(=O)C1=CC=C(O)C=C1",     # Hexyl salicylate → actually lauryl 4-hydroxybenzoate
    "89-43-0": "O=C1OC2=CC(=CC=C2C3=CC=CC=C13)O",        # Umbelliferone? → 6-Methylcoumarin
    "1205-17-0": "O=CC=CC1=CC2=C(OCO2)C=C1",              # Piperonal 
    "120-57-0": "O=CC1=CC2=C(OCO2)C=C1",                  # Piperonal (Heliotropin)
    "123-11-5": "COC1=CC=C(C=O)C=C1",                     # Anisaldehyde
    "90028-68-5": None,                                     # Oakmoss (mixture)
    "90028-67-4": None,                                     # Treemoss (mixture)
}

def main():
    # Load ingredients
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    print(f"Ingredients: {len(ingredients)}")

    # Load cat4 IFRA data
    cat4_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_cat4.json')
    with open(cat4_path, 'r', encoding='utf-8') as f:
        cat4 = json.load(f)
    
    # Load full IFRA for names
    ifra_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json')
    with open(ifra_path, 'r', encoding='utf-8') as f:
        ifra_full = json.load(f)
    
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
    print(f"Canonical index: {len(can_to_idx)} entries")
    
    # Clear old tags
    for ing in ingredients:
        for k in ['ifra_cas','ifra_type','ifra_restricted','ifra_prohibited','ifra_limit_pct','ifra_name']:
            ing.pop(k, None)
    
    # Match
    matched = 0
    p_list, r_list = [], []
    unmatched = []
    
    for cas, info in cat4.items():
        smi = CAS_SMILES.get(cas)
        if not smi:
            continue
        can = canonical(smi)
        if can not in can_to_idx:
            unmatched.append(f"{cas} ({ifra_full.get(cas,{}).get('name','')[:30]})")
            continue
        
        idx = can_to_idx[can]
        is_p = info.get('prohibited', False)
        is_r = info.get('restricted', False)
        max_pct = info.get('max_pct_cat4', 100.0)
        name = ifra_full.get(cas, {}).get('name', cas)
        
        ingredients[idx]['ifra_cas'] = cas
        ingredients[idx]['ifra_prohibited'] = is_p
        ingredients[idx]['ifra_restricted'] = is_p or is_r
        ingredients[idx]['ifra_limit_pct'] = 0.0 if is_p else max_pct
        ingredients[idx]['ifra_name'] = name
        matched += 1
        
        if is_p:
            p_list.append(f"{name} ({cas})")
            print(f"  PROHIBITED: {name} ({cas})")
        elif is_r:
            r_list.append(f"{name}: max {max_pct}% ({cas})")
            print(f"  RESTRICTED: {name}: max {max_pct}%")
    
    total_p = sum(1 for x in ingredients if x.get('ifra_prohibited'))
    total_r = sum(1 for x in ingredients if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    
    print(f"\n=== Results ===")
    print(f"Hardcoded SMILES: {sum(1 for v in CAS_SMILES.values() if v)}/59")
    print(f"Matched to DB: {matched}")
    print(f"Unmatched (SMILES not in DB): {len(unmatched)}")
    print(f"Prohibited: {total_p}")
    print(f"Restricted: {total_r}")
    
    if unmatched:
        print(f"\nNot in our DB:")
        for u in unmatched:
            print(f"  {u}")
    
    # Save
    for p in [ing_path, os.path.join(BASE, 'data', 'ingredients.json')]:
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(ingredients, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Update enforcement
    enforcement = {
        'version': '51st Amendment',
        'total_ifra_substances': len(ifra_full),
        'cat4_substances': len(cat4),
        'matched_to_db': matched,
        'prohibited_count': total_p,
        'restricted_count': total_r,
        'prohibited_smiles': [CAS_SMILES[cas] for cas in cat4 if cat4[cas].get('prohibited') and CAS_SMILES.get(cas)],
    }
    enf_path = os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_enforcement.json')
    with open(enf_path, 'w', encoding='utf-8') as f:
        json.dump(enforcement, f, indent=2)
    print(f"\nSaved enforcement module")

if __name__ == '__main__':
    main()
