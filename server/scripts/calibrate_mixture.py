"""calibrate_mixture.py -- 아코드 매칭률 자동 최적화

CategoryCorrector blend_ratio를 grid search로 최적화하여
12개 아코드 100% 매칭을 목표로 합니다.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from odor_engine import (
    OdorGNN, ConcentrationModulator, PhysicsMixture,
    CategoryCorrector, ODOR_DIMENSIONS, N_ODOR_DIM
)

# Load data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
with open(os.path.join(DATA_DIR, 'accords.json'), 'r', encoding='utf-8') as f:
    ACCORDS = json.load(f)
with open(os.path.join(DATA_DIR, 'ingredients.json'), 'r', encoding='utf-8') as f:
    INGREDIENTS = {ing['id']: ing for ing in json.load(f)}
with open(os.path.join(DATA_DIR, 'molecules.json'), 'r', encoding='utf-8') as f:
    MOLECULES = json.load(f)

# Ingredient -> SMILES mapping
INGREDIENT_TO_SMILES = {}
for mol in MOLECULES:
    for src in mol.get('source_ingredients', []):
        if src not in INGREDIENT_TO_SMILES:
            INGREDIENT_TO_SMILES[src] = mol['smiles']

# Load expanded ingredient_smiles.json (Phase 1: 1,136 ingredients)
_ing_smiles_path = os.path.join(DATA_DIR, 'ingredient_smiles.json')
if os.path.exists(_ing_smiles_path):
    with open(_ing_smiles_path, 'r', encoding='utf-8') as f:
        _ing_smiles = json.load(f)
    for k, v in _ing_smiles.items():
        if v and k not in INGREDIENT_TO_SMILES:
            INGREDIENT_TO_SMILES[k] = v
print(f'SMILES coverage: {len(INGREDIENT_TO_SMILES)} ingredients')

# Expected dimensions per accord
ACCORD_EXPECTED = {
    'classic_chypre':  {'woody', 'earthy', 'floral', 'green'},
    'fougere':         {'herbal', 'floral', 'woody', 'warm'},
    'eau_de_cologne':  {'citrus', 'fresh', 'floral', 'green'},
    'oriental_base':   {'sweet', 'warm', 'amber', 'spicy'},
    'white_floral':    {'floral', 'sweet', 'musk', 'powdery'},
    'woody_amber':     {'woody', 'warm', 'amber', 'musk'},
    'fresh_aquatic':   {'aquatic', 'fresh', 'citrus', 'musk'},
    'gourmand_vanilla':{'sweet', 'warm', 'musk', 'amber'},
    'smoky_leather':   {'smoky', 'leather', 'woody', 'warm'},
    'rose_oud':        {'floral', 'woody', 'smoky', 'musk'},
    'citrus_aromatic': {'citrus', 'fresh', 'herbal', 'woody'},
    'dark_oriental':   {'smoky', 'amber', 'warm', 'sweet'},
}

# Shared instances (created once)
print('Loading shared instances...')
GNN = OdorGNN(device='cpu')
MODULATOR = ConcentrationModulator()
MIXER = PhysicsMixture()

# Pre-encode all molecules + ingredient SMILES
print('Pre-encoding molecules...')
GNN_CACHE = {}
all_smiles_to_encode = set()
for mol in MOLECULES:
    all_smiles_to_encode.add(mol['smiles'])
for s in INGREDIENT_TO_SMILES.values():
    if s:
        all_smiles_to_encode.add(s)

for smiles in all_smiles_to_encode:
    if smiles not in GNN_CACHE:
        try:
            GNN_CACHE[smiles] = GNN.encode(smiles)
        except:
            pass
print(f'Cached {len(GNN_CACHE)} molecule vectors')


def evaluate_accords(blend_ratio, verbose=False):
    """주어진 blend_ratio로 전체 아코드 매칭률 계산"""
    corrector = CategoryCorrector(blend_ratio=blend_ratio)
    
    total_matches = 0
    total_expected = 0
    per_accord = {}
    
    for accord in ACCORDS:
        accord_id = accord['id']
        expected = ACCORD_EXPECTED.get(accord_id)
        if not expected:
            continue
        
        ingredients = accord['ingredients']
        ratios = accord['ratios']
        
        # Build vectors for each ingredient
        vecs = []
        concs = []
        smiles_list = []
        
        for ing_id, ratio in zip(ingredients, ratios):
            smiles = INGREDIENT_TO_SMILES.get(ing_id)
            conc = float(ratio) / 10.0
            
            # GNN encode (from cache)
            if smiles and smiles in GNN_CACHE:
                v = GNN_CACHE[smiles].copy()
            else:
                v = np.zeros(N_ODOR_DIM)
            
            # Category correction
            v = corrector.correct(v, ingredient_id=ing_id)
            
            vecs.append(v)
            concs.append(conc)
            smiles_list.append(smiles)
        
        if len(vecs) < 2:
            continue
        
        vecs = np.array(vecs)
        concs = np.array(concs)
        
        # Modulate
        mod_vecs = MODULATOR.batch_modulate(vecs, concs, smiles_list=smiles_list)
        
        # Mix
        mixture = MIXER.mix(mod_vecs, concs)
        
        # Top 5
        top_indices = np.argsort(mixture)[::-1][:5]
        predicted = set(ODOR_DIMENSIONS[i] for i in top_indices if mixture[i] > 0.005)
        
        matches = predicted & expected
        match_count = len(matches)
        expected_count = len(expected)
        
        total_matches += match_count
        total_expected += expected_count
        
        per_accord[accord_id] = {
            'match_rate': match_count / expected_count if expected_count > 0 else 0,
            'matches': match_count,
            'total': expected_count,
            'predicted': sorted(predicted),
            'expected': sorted(expected),
            'matched': sorted(matches),
            'missing': sorted(expected - predicted),
            'top5': [(ODOR_DIMENSIONS[i], round(float(mixture[i]), 3))
                     for i in np.argsort(mixture)[::-1][:5]],
        }
    
    overall = total_matches / total_expected if total_expected > 0 else 0
    
    if verbose:
        for accord_id, r in per_accord.items():
            icon = '[OK]' if r['match_rate'] >= 1.0 else ('[++]' if r['match_rate'] >= 0.75 else ('[--]' if r['match_rate'] >= 0.5 else '[XX]'))
            print(f"  {icon} [{accord_id}] ({r['matches']}/{r['total']} = {r['match_rate']:.0%})")
            print(f"      expected:  {r['expected']}")
            print(f"      predicted: {r['predicted']}")
            if r['matched']:
                print(f"      matched:   {r['matched']}")
            if r['missing']:
                print(f"      missing:   {r['missing']}")
            print(f"      top5:      {r['top5']}")
        
        ok = sum(1 for r in per_accord.values() if r['match_rate'] >= 1.0)
        mid = sum(1 for r in per_accord.values() if 0.5 <= r['match_rate'] < 1.0)
        low = sum(1 for r in per_accord.values() if r['match_rate'] < 0.5)
        print(f'\n Overall: {total_matches}/{total_expected} ({overall:.0%})')
        print(f' 100%: {ok} | 50-99%: {mid} | <50%: {low}')
    
    return overall, per_accord


def main():
    print('=' * 60)
    print(' Auto-Calibration: CategoryCorrector')
    print('=' * 60)
    
    # Grid search
    print('\n[PHASE 1] Grid search...')
    best_ratio = 0.35
    best_score = 0.0
    
    for ratio in np.arange(0.10, 0.95, 0.05):
        score, _ = evaluate_accords(float(ratio), verbose=False)
        marker = ' <-- BEST' if score > best_score else ''
        print(f'  blend={ratio:.2f} => {score:.0%}{marker}')
        if score > best_score:
            best_score = score
            best_ratio = float(ratio)
    
    # Fine-tune around best
    print(f'\n[PHASE 1.5] Fine-tune around {best_ratio:.2f}...')
    for ratio in np.arange(max(0.05, best_ratio - 0.10), min(0.95, best_ratio + 0.10), 0.01):
        score, _ = evaluate_accords(float(ratio), verbose=False)
        marker = ' <-- BEST' if score > best_score else ''
        if score > best_score:
            print(f'  blend={ratio:.2f} => {score:.0%}{marker}')
            best_score = score
            best_ratio = float(ratio)
    
    print(f'\n  Best: blend_ratio={best_ratio:.2f}, score={best_score:.0%}')
    
    # Detailed evaluation
    print(f'\n[PHASE 2] Detailed evaluation at blend={best_ratio:.2f}')
    print('=' * 60)
    overall, per_accord = evaluate_accords(best_ratio, verbose=True)
    print('=' * 60)
    
    # Report
    all_perfect = all(r['match_rate'] >= 1.0 for r in per_accord.values())
    if all_perfect:
        print('\n*** ALL 12 ACCORDS AT 100%! ***')
    else:
        failing = [(aid, r['match_rate']) for aid, r in per_accord.items() if r['match_rate'] < 1.0]
        print(f'\nFailing: {[(a, f"{r:.0%}") for a, r in failing]}')
    
    return overall, best_ratio


if __name__ == '__main__':
    overall, best_ratio = main()
    sys.exit(0 if overall >= 0.90 else 1)
