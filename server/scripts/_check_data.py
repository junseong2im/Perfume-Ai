"""Check all available data sources for unified training"""
import sys, os, json, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')

print("=" * 60)
print("  Available Data Sources")
print("=" * 60)

# 1. GS+LF CSV
csv_path = os.path.join(DATA_DIR, 'curated_GS_LF_merged_4983.csv')
if os.path.exists(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    print(f"\n[1] GS+LF CSV: {len(rows)} molecules, {len(header)-1} descriptors")
    print(f"    Descriptors sample: {header[1:6]}")
else:
    print("\n[1] GS+LF CSV: NOT FOUND")

# 2. ChemBERTa cache
cache_path = os.path.join(WEIGHTS_DIR, 'chemberta_cache.npz')
if os.path.exists(cache_path):
    d = np.load(cache_path, allow_pickle=True)
    print(f"\n[2] ChemBERTa cache: {len(d['smiles'])} molecules, {d['hidden_size']}d")
else:
    print("\n[2] ChemBERTa cache: NOT FOUND")

# 3. molecules.json
mol_path = os.path.join(DATA_DIR, 'molecules.json')
if os.path.exists(mol_path):
    mols = json.load(open(mol_path, 'r', encoding='utf-8'))
    print(f"\n[3] molecules.json: {len(mols)} entries")
    if mols:
        print(f"    Keys: {list(mols[0].keys())}")
        has_labels = sum(1 for m in mols if m.get('odor_labels'))
        has_smiles = sum(1 for m in mols if m.get('smiles'))
        print(f"    With SMILES: {has_smiles}, With odor_labels: {has_labels}")
else:
    print("\n[3] molecules.json: NOT FOUND")

# 4. ingredient_smiles.json
ing_path = os.path.join(DATA_DIR, 'ingredient_smiles.json')
if os.path.exists(ing_path):
    ings = json.load(open(ing_path, 'r', encoding='utf-8'))
    has_smiles = sum(1 for v in ings.values() if v)
    print(f"\n[4] ingredient_smiles.json: {len(ings)} entries, {has_smiles} with SMILES")
else:
    print("\n[4] ingredient_smiles.json: NOT FOUND")

# 5. DB ingredients with odor_descriptors
import database as db_module
db_ings = db_module.get_all_ingredients()
has_descriptors = sum(1 for i in db_ings if i.get('odor_descriptors'))
print(f"\n[5] DB ingredients: {len(db_ings)} total, {has_descriptors} with odor_descriptors")
if has_descriptors > 0:
    sample = [i for i in db_ings if i.get('odor_descriptors')][:3]
    for s in sample:
        print(f"    {s['id']}: {s.get('odor_descriptors', [])[:5]}")

# 6. DB molecules
db_mols = db_module.get_all_molecules()
print(f"\n[6] DB molecules: {len(db_mols)} total")
has_both = sum(1 for m in db_mols if m.get('smiles') and m.get('odor_labels'))
print(f"    With SMILES+labels: {has_both}")

# 7. Chemprop data
chemprop_dirs = [
    os.path.join(DATA_DIR, 'chemprop'),
    os.path.join(os.path.dirname(__file__), '..', 'data', 'chemprop'),
]
for cp_dir in chemprop_dirs:
    if os.path.exists(cp_dir):
        files = os.listdir(cp_dir)
        print(f"\n[7] Chemprop dir: {cp_dir}")
        for f in files[:5]:
            fp = os.path.join(cp_dir, f)
            if f.endswith('.csv'):
                with open(fp, encoding='utf-8') as fh:
                    lines = sum(1 for _ in fh) - 1
                print(f"    {f}: {lines} rows")
        break
else:
    print("\n[7] Chemprop data: NOT FOUND")

print("\n" + "=" * 60)
