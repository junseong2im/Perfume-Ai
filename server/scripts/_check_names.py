"""Dump all unmapped ingredient IDs for manual SMILES mapping"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import database as db

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Load previous results
prev = {}
prev_path = os.path.join(DATA_DIR, 'ingredient_smiles.json')
if os.path.exists(prev_path):
    with open(prev_path, 'r', encoding='utf-8') as f:
        prev = json.load(f)

# Load all manual SMILES from smiles_mapper.py
from scripts.smiles_mapper import MANUAL_SMILES

# Existing molecules.json
mol_path = os.path.join(DATA_DIR, 'molecules.json')
existing = {}
if os.path.exists(mol_path):
    with open(mol_path, 'r', encoding='utf-8') as f:
        for mol in json.load(f):
            for src in mol.get('source_ingredients', []):
                existing[src] = mol['smiles']

db_ings = db.get_all_ingredients()
unmapped = []
mapped = 0

for ing in db_ings:
    ing_id = ing.get('id', '')
    if ing_id in MANUAL_SMILES or ing_id in existing or (ing_id in prev and prev[ing_id]):
        mapped += 1
    else:
        unmapped.append({
            'id': ing_id,
            'name_en': ing.get('name_en', ''),
            'category': ing.get('category', ''),
            'cas': ing.get('cas_number', ''),
        })

print(f"Mapped: {mapped}")
print(f"Unmapped: {len(unmapped)}")
print(f"\nAll unmapped ingredient IDs (by category):")

# Group by category
by_cat = {}
for u in unmapped:
    cat = u['category']
    if cat not in by_cat:
        by_cat[cat] = []
    by_cat[cat].append(u)

for cat in sorted(by_cat.keys()):
    items = by_cat[cat]
    print(f"\n=== {cat} ({len(items)}) ===")
    for u in items:
        cas_info = f" CAS={u['cas']}" if u['cas'] else ""
        print(f"  '{u['id']}': '',  # {u['name_en']}{cas_info}")
