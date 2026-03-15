"""
Recalibrate ALL recipe percentages using real formula data as ground truth.

Strategy:
1. Extract per-ingredient average percentages from the 12 real formulas
2. Build a "weight profile" for every ingredient in our DB
3. Re-assign percentages to ALL other recipes based on this profile
4. Normalize to 100%
"""
import json
import sys
import os
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database import get_all_ingredients

all_ings = get_all_ingredients()
DB_IDS = {i['id'] for i in all_ings}
ING_CAT = {i['id']: i.get('category', 'other') for i in all_ings}

# ============================================================
# STEP 1: Load data and separate real formulas from the rest
# ============================================================
with open('data/recipe_training_data.json', 'r', encoding='utf-8') as f:
    all_recipes = json.load(f)

REAL_SOURCES = {
    'iff_demo_formula', 'firmenich_demo_formula', 'classic_accord_formula',
    'perfumers_apprentice_formula', 'published_oud_formula',
    'olfactorian_reconstruction', 'classic_reconstruction',
    'good_scents_demo', 'diy_community_formula'
}

real = [r for r in all_recipes if r.get('source', '') in REAL_SOURCES]
variations = [r for r in all_recipes if r.get('source', '').startswith('variation_of_')]
others = [r for r in all_recipes if r.get('source', '') not in REAL_SOURCES and not r.get('source', '').startswith('variation_of_')]

print(f"Real formulas: {len(real)}")
print(f"Variations: {len(variations)}")
print(f"Others to recalibrate: {len(others)}")

# ============================================================
# STEP 2: Build ingredient weight profile from real formulas
# For each ingredient+note combo, collect actual percentages
# ============================================================
ing_note_pcts = defaultdict(list)   # {(ing_id, note): [pct, pct, ...]}
ing_pcts = defaultdict(list)        # {ing_id: [pct, pct, ...]}
cat_note_pcts = defaultdict(list)   # {(category, note): [pct, pct, ...]}

# Use both real formulas AND their variations (they're based on real data)
reference_recipes = real + variations

for r in reference_recipes:
    for ing in r['ingredients']:
        ing_note_pcts[(ing['id'], ing['note'])].append(ing['pct'])
        ing_pcts[ing['id']].append(ing['pct'])
        cat = ING_CAT.get(ing['id'], 'other')
        cat_note_pcts[(cat, ing['note'])].append(ing['pct'])

# Also build note-level averages (how much should all tops/mids/bases total?)
note_totals = defaultdict(list)  # {note: [total_pct_in_recipe, ...]}
for r in reference_recipes:
    note_sums = defaultdict(float)
    for ing in r['ingredients']:
        note_sums[ing['note']] += ing['pct']
    for note, total in note_sums.items():
        note_totals[note].append(total)

# Calculate statistics
print("\n=== Note-level totals from real formulas ===")
for note in ['top', 'middle', 'base']:
    vals = note_totals.get(note, [0])
    avg = sum(vals) / len(vals) if vals else 0
    mn = min(vals) if vals else 0
    mx = max(vals) if vals else 0
    print(f"  {note}: avg={avg:.1f}%, range=[{mn:.1f}, {mx:.1f}]")

print(f"\n=== Ingredient profiles built ===")
print(f"  Unique (ingredient, note) combos: {len(ing_note_pcts)}")
print(f"  Unique ingredients: {len(ing_pcts)}")
print(f"  Unique (category, note) combos: {len(cat_note_pcts)}")

# Show top ingredient profiles
print(f"\n=== Top 20 ingredient weight profiles ===")
sorted_ings = sorted(ing_pcts.items(), key=lambda x: -len(x[1]))[:20]
for ing_id, pcts in sorted_ings:
    avg = sum(pcts) / len(pcts)
    mn, mx = min(pcts), max(pcts)
    print(f"  {ing_id:20s}: avg={avg:5.1f}%, range=[{mn:.1f}, {mx:.1f}], n={len(pcts)}")


# ============================================================
# STEP 3: Assign realistic percentages to each ingredient
# Priority: (1) exact ingredient+note match -> (2) ingredient avg 
#           -> (3) category+note match -> (4) default by note
# ============================================================
def get_weight_for_ingredient(ing_id, note, n_in_note):
    """Get realistic weight for an ingredient given its note type and 
    how many other ingredients share the same note."""
    
    # Priority 1: exact ingredient + note match from real data
    key = (ing_id, note)
    if key in ing_note_pcts:
        vals = ing_note_pcts[key]
        avg = sum(vals) / len(vals)
        std = (sum((v - avg)**2 for v in vals) / max(len(vals), 1)) ** 0.5
        # Random sample around average
        weight = avg + random.gauss(0, max(std * 0.5, 0.5))
        return max(0.5, weight)
    
    # Priority 2: ingredient-level average (any note)
    if ing_id in ing_pcts:
        vals = ing_pcts[ing_id]
        avg = sum(vals) / len(vals)
        std = (sum((v - avg)**2 for v in vals) / max(len(vals), 1)) ** 0.5
        weight = avg + random.gauss(0, max(std * 0.5, 0.5))
        return max(0.5, weight)
    
    # Priority 3: category + note match
    cat = ING_CAT.get(ing_id, 'other')
    cat_key = (cat, note)
    if cat_key in cat_note_pcts:
        vals = cat_note_pcts[cat_key]
        avg = sum(vals) / len(vals)
        std = (sum((v - avg)**2 for v in vals) / max(len(vals), 1)) ** 0.5
        weight = avg + random.gauss(0, max(std * 0.3, 0.3))
        return max(0.3, weight)
    
    # Priority 4: default by note position
    defaults = {'top': 6.0, 'middle': 8.0, 'base': 7.0}
    base = defaults.get(note, 7.0)
    # Scale by how many share this note
    scale = max(1.0, n_in_note) ** 0.5
    weight = base / scale + random.gauss(0, 1.5)
    return max(0.3, weight)


def recalibrate_recipe(recipe):
    """Recalibrate a single recipe's percentages using real data profiles."""
    ings = recipe['ingredients']
    if not ings:
        return recipe
    
    # Count ingredients per note
    note_counts = defaultdict(int)
    for i in ings:
        note_counts[i['note']] += 1
    
    # Assign raw weights
    for i in ings:
        raw_weight = get_weight_for_ingredient(i['id'], i['note'], note_counts[i['note']])
        i['_raw_weight'] = raw_weight
    
    # Now we want note-level totals to match real formula patterns
    # From our real data: top ~23%, mid ~43%, base ~34% (averages)
    target_totals = {
        'top': sum(note_totals.get('top', [23])) / max(len(note_totals.get('top', [1])), 1),
        'middle': sum(note_totals.get('middle', [43])) / max(len(note_totals.get('middle', [1])), 1),
        'base': sum(note_totals.get('base', [34])) / max(len(note_totals.get('base', [1])), 1),
    }
    
    # Add some randomness to targets
    for note in target_totals:
        target_totals[note] *= (1 + random.uniform(-0.15, 0.15))
    
    # Normalize within each note group to match targets
    for note in ['top', 'middle', 'base']:
        note_ings = [i for i in ings if i['note'] == note]
        if not note_ings:
            continue
        
        raw_total = sum(i['_raw_weight'] for i in note_ings)
        if raw_total > 0:
            target = target_totals.get(note, 33.0)
            scale = target / raw_total
            for i in note_ings:
                i['pct'] = round(i['_raw_weight'] * scale, 1)
    
    # Final normalize to 100%
    total = sum(i['pct'] for i in ings)
    if total > 0:
        for i in ings:
            i['pct'] = round(i['pct'] / total * 100.0, 1)
    
    # Clean up temp field
    for i in ings:
        if '_raw_weight' in i:
            del i['_raw_weight']
    
    return recipe


# ============================================================
# STEP 4: Apply recalibration to all non-real recipes
# ============================================================
print(f"\nRecalibrating {len(others)} recipes...")

for r in others:
    recalibrate_recipe(r)

# Also recalibrate variations with slight adjustments
for r in variations:
    recalibrate_recipe(r)

print("Done!")

# ============================================================
# STEP 5: Save and verify
# ============================================================
final = others + real + variations
with open('data/recipe_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

# Verify quality
print(f"\n=== FINAL VERIFICATION ===")
print(f"Total recipes: {len(final)}")

# Check pct distribution
all_pcts = [i['pct'] for r in final for i in r['ingredients']]
print(f"Overall pct: min={min(all_pcts):.1f}, max={max(all_pcts):.1f}, avg={sum(all_pcts)/len(all_pcts):.1f}")

# Check unique patterns
pct_patterns = set()
for r in final:
    pattern = tuple(sorted([round(i['pct'], 0) for i in r['ingredients']]))
    pct_patterns.add(pattern)
print(f"Unique pct patterns: {len(pct_patterns)}/{len(final)} ({len(pct_patterns)*100//len(final)}%)")

# Check note-level totals match real data
note_check = defaultdict(list)
for r in others[:100]:  # Sample
    note_sums = defaultdict(float)
    for i in r['ingredients']:
        note_sums[i['note']] += i['pct']
    for note, total in note_sums.items():
        note_check[note].append(total)

print(f"\nRecalibrated note totals (sample of 100):")
for note in ['top', 'middle', 'base']:
    vals = note_check.get(note, [0])
    if vals:
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        print(f"  {note}: avg={avg:.1f}%, range=[{mn:.1f}, {mx:.1f}]")

# Show before/after comparison
print(f"\n=== BEFORE vs AFTER sample (first Fragrantica recipe) ===")
frag_sample = [r for r in others if r.get('source') == 'fragrantica'][:1]
if frag_sample:
    r = frag_sample[0]
    print(f"{r['name']}")
    for ing in r['ingredients']:
        print(f"  {ing['id']:20s} {ing['note']:8s} {ing['pct']:5.1f}%")

# Show ingredient-level consistency check
print(f"\n=== Bergamot percentage distribution across ALL recipes ===")
berg_pcts = [i['pct'] for r in final for i in r['ingredients'] if i['id'] == 'bergamot']
print(f"  n={len(berg_pcts)}, avg={sum(berg_pcts)/len(berg_pcts):.1f}%, range=[{min(berg_pcts):.1f}, {max(berg_pcts):.1f}]")

print(f"\n=== Musk percentage distribution ===")
musk_pcts = [i['pct'] for r in final for i in r['ingredients'] if i['id'] == 'musk']
print(f"  n={len(musk_pcts)}, avg={sum(musk_pcts)/len(musk_pcts):.1f}%, range=[{min(musk_pcts):.1f}, {max(musk_pcts):.1f}]")

print(f"\n=== Vanilla percentage distribution ===")
van_pcts = [i['pct'] for r in final for i in r['ingredients'] if i['id'] == 'vanilla']
print(f"  n={len(van_pcts)}, avg={sum(van_pcts)/len(van_pcts):.1f}%, range=[{min(van_pcts):.1f}, {max(van_pcts):.1f}]")

print(f"\n=== Jasmine percentage distribution ===")
jas_pcts = [i['pct'] for r in final for i in r['ingredients'] if i['id'] == 'jasmine']
print(f"  n={len(jas_pcts)}, avg={sum(jas_pcts)/len(jas_pcts):.1f}%, range=[{min(jas_pcts):.1f}, {max(jas_pcts):.1f}]")
