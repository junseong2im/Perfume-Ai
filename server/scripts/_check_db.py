import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json, database as db

# Load both sources
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ingredients.json'), 'r', encoding='utf-8') as f:
    json_ings = {i['id']: i for i in json.load(f)}

db_ings_raw = db.get_all_ingredients()
db_ings = {i['id']: i for i in db_ings_raw if i.get('id')}

# Check accords ingredients differences
accords_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'accords.json')
with open(accords_path, 'r', encoding='utf-8') as f:
    accords = json.load(f)

print("=== Ingredient category differences (accords only) ===\n")
for accord in accords:
    for ing_id in accord['ingredients']:
        json_cat = json_ings.get(ing_id, {}).get('category', 'N/A')
        db_cat = db_ings.get(ing_id, {}).get('category', 'N/A')
        if json_cat != db_cat:
            print(f"  [{accord['id']}] {ing_id}: JSON={json_cat}, DB={db_cat}")

# Also check descriptor differences for accord ingredients
print("\n=== Descriptor differences ===\n")
for accord in accords:
    for ing_id in accord['ingredients']:
        json_desc = json_ings.get(ing_id, {}).get('descriptors', [])
        db_desc = db_ings.get(ing_id, {}).get('descriptors') or []
        if set(json_desc) != set(db_desc):
            print(f"  [{accord['id']}] {ing_id}:")
            print(f"    JSON: {json_desc}")
            print(f"    DB:   {db_desc}")
