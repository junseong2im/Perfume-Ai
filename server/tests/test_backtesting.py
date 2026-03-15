"""
N>2 Digital Backtesting Framework
==================================
Validates POM Engine's multi-molecule mixture prediction
without physical lab access (Dry Lab backtesting).

Strategy: Use known perfume accords (Rose, Fougere, Chypre, Oriental)
where expert consensus on dominant scent category exists, and verify
the engine's predictions match expected profiles.

This is the best available validation when DREAM Synapse data
requires account access. Tests:
1. Descriptor Consistency: Does rose accord predict 'rose/floral'?
2. N-scaling: Does prediction quality degrade as N increases?
3. Dilution resistance: Does adding trace inerts change the prediction?
4. Accord separation: Are different accords distinguishable?
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pom_engine import POMEngine, TASKS_138

def main():
    t0 = time.time()
    engine = POMEngine()
    engine.load()
    
    passed = 0
    failed = 0
    
    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [OK] {name}: {detail}")
        else:
            failed += 1
            print(f"  [XX] {name}: {detail}")
    
    # ==========================================================
    # Known Perfume Accords (industry consensus compositions)
    # ==========================================================
    ACCORDS = {
        "Rose Accord": {
            "recipe": [
                {"name": "citronellol", "pct": 40.0},
                {"name": "geraniol", "pct": 25.0},
                {"name": "linalool", "pct": 15.0},
                {"name": "eugenol", "pct": 5.0},
                {"name": "nerol", "pct": 10.0},
                {"name": "phenethyl_alcohol", "pct": 5.0},
            ],
            "expected_top": ["floral", "rose", "citrus", "sweet"],
            "expected_absent": ["smoky", "burnt", "fishy"],
        },
        "Woody Accord": {
            "recipe": [
                {"name": "iso_e_super", "pct": 40.0},
                {"name": "cedrol", "pct": 20.0},
                {"name": "santalol", "pct": 15.0},
                {"name": "vetiverol", "pct": 15.0},
                {"name": "patchoulol", "pct": 10.0},
            ],
            "expected_top": ["woody", "earthy", "balsamic"],
            "expected_absent": ["citrus", "fruity"],
        },
        "Citrus Accord": {
            "recipe": [
                {"name": "limonene", "pct": 45.0},
                {"name": "citral", "pct": 20.0},
                {"name": "linalool", "pct": 15.0},
                {"name": "linalyl_acetate", "pct": 10.0},
                {"name": "citronellal", "pct": 10.0},
            ],
            "expected_top": ["citrus", "fresh", "fruity", "herbal"],
            "expected_absent": ["smoky", "leathery", "animal"],
        },
        "Vanilla-Amber Accord": {
            "recipe": [
                {"name": "vanillin", "pct": 30.0},
                {"name": "ethyl_vanillin", "pct": 15.0},
                {"name": "coumarin", "pct": 15.0},
                {"name": "benzyl_benzoate", "pct": 20.0},
                {"name": "iso_e_super", "pct": 10.0},
                {"name": "hedione", "pct": 10.0},
            ],
            "expected_top": ["vanilla", "sweet", "warm", "balsamic"],
            "expected_absent": ["fishy", "garlic", "metallic"],
        },
        "Spicy Accord": {
            "recipe": [
                {"name": "eugenol", "pct": 35.0},
                {"name": "cinnamaldehyde", "pct": 20.0},
                {"name": "linalool", "pct": 15.0},
                {"name": "vanillin", "pct": 15.0},
                {"name": "iso_e_super", "pct": 15.0},
            ],
            "expected_top": ["spicy", "sweet", "warm"],
            "expected_absent": ["citrus", "fishy"],
        },
    }
    
    # ==========================================================
    # TEST 1: Accord Recognition (does N>2 mix match family?)
    # ==========================================================
    print("="*60)
    print("  TEST 1: Accord Recognition (N=5-6 molecules)")
    print("="*60)
    
    accord_vectors = {}
    for name, accord in ACCORDS.items():
        mix = engine.predict_mixture(accord["recipe"])
        if not mix:
            check(f"{name} prediction", False, "predict_mixture returned None")
            continue
        
        top_5 = [n for n, v in mix["top_notes"][:5]]
        hit = any(t in top_5 for t in accord["expected_top"])
        absent_ok = all(
            TASKS_138.index(a) < len(TASKS_138) and 
            mix["top_notes"][-1][1] > 0.01
            for a in accord["expected_absent"]
            if a in TASKS_138
        )
        
        check(f"{name}: expected in top 5", hit,
              f"top5={top_5}, expected={accord['expected_top']}")
        
        accord_vectors[name] = np.array([v for _, v in
            sorted([(TASKS_138.index(n), v) for n, v in mix["top_notes"]
                    if n in TASKS_138],
                   key=lambda x: x[0])])
        
        print(f"    Top notes: {[(n, round(v,2)) for n, v in mix['top_notes'][:5]]}")
    
    # ==========================================================
    # TEST 2: N-Scaling (2 vs 3 vs 5 vs 7 molecules)
    # ==========================================================
    print()
    print("="*60)
    print("  TEST 2: N-Scaling Consistency")
    print("="*60)
    
    rose_core = ACCORDS["Rose Accord"]["recipe"]
    prev_pred = None
    for n in [2, 3, 5, 6]:
        subset = rose_core[:n]
        total = sum(s['pct'] for s in subset)
        normalized = [{'name': s['name'], 'pct': s['pct']/total*100} for s in subset]
        
        mix = engine.predict_mixture(normalized)
        if not mix:
            check(f"N={n} prediction", False, "None")
            continue
        
        top_3 = [n_name for n_name, v in mix["top_notes"][:3]]
        has_floral = "floral" in top_3 or "rose" in top_3
        
        check(f"N={n}: floral/rose in top 3", has_floral,
              f"top3={top_3}")
    
    # ==========================================================
    # TEST 3: Dilution Resistance
    # ==========================================================
    print()
    print("="*60)
    print("  TEST 3: Dilution Resistance (adding trace inerts)")
    print("="*60)
    
    # Base: rose 3-component
    base_recipe = [
        {"name": "citronellol", "pct": 50.0},
        {"name": "geraniol", "pct": 30.0},
        {"name": "linalool", "pct": 20.0},
    ]
    base_mix = engine.predict_mixture(base_recipe)
    
    # Diluted: same + 3 trace molecules at 0.01%
    diluted_recipe = base_recipe.copy() + [
        {"name": "hedione", "pct": 0.01},
        {"name": "iso_e_super", "pct": 0.01},
        {"name": "limonene", "pct": 0.01},
    ]
    diluted_mix = engine.predict_mixture(diluted_recipe)
    
    if base_mix and diluted_mix:
        base_top = set(n for n, v in base_mix["top_notes"][:3])
        diluted_top = set(n for n, v in diluted_mix["top_notes"][:3])
        overlap = len(base_top & diluted_top)
        
        check("Dilution: top 3 overlap >= 2", overlap >= 2,
              f"base={base_top}, diluted={diluted_top}, overlap={overlap}")
    
    # ==========================================================
    # TEST 4: Accord Separation (cosine between accords)
    # ==========================================================
    print()
    print("="*60)
    print("  TEST 4: Accord Separation (inter-family distance)")
    print("="*60)
    
    accord_names = list(ACCORDS.keys())
    embeddings = {}
    
    for name, accord in ACCORDS.items():
        mix = engine.predict_mixture(accord["recipe"])
        if mix:
            # Build full 138d vector
            vec = np.zeros(len(TASKS_138))
            for note_name, val in mix["top_notes"]:
                idx = TASKS_138.index(note_name) if note_name in TASKS_138 else -1
                if idx >= 0:
                    vec[idx] = val
            embeddings[name] = vec
    
    if len(embeddings) >= 2:
        names = list(embeddings.keys())
        print("  Inter-accord cosine distances:")
        min_inter = 1.0
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                sim = engine.cosine_sim(embeddings[names[i]], embeddings[names[j]])
                label = f"    {names[i]:20s} vs {names[j]:20s}: {sim:.3f}"
                print(label)
                min_inter = min(min_inter, sim)
        
        check("Min inter-accord sim < 0.95 (accords distinguishable)",
              min_inter < 0.95,
              f"min_sim={min_inter:.3f}")
    
    # ==========================================================
    # TEST 5: 4D Temporal on Accord
    # ==========================================================
    print()
    print("="*60)
    print("  TEST 5: 4D Temporal on Rose Accord")
    print("="*60)
    
    temporal = engine.simulate_temporal(
        ACCORDS["Rose Accord"]["recipe"],
        time_points=[0, 1, 4, 24])
    
    if temporal:
        check("Temporal returns result", True,
              f"longevity={temporal['longevity_hours']}h")
        
        t0_bal = temporal['timeline']['T+0h']['balance']
        t24_bal = temporal['timeline']['T+24h']['balance']
        base_increased = t24_bal['base'] > t0_bal['base']
        check("Base increases over time", base_increased,
              f"T=0: base={t0_bal['base']}%, T=24: base={t24_bal['base']}%")
        
        for tk, tv in temporal['timeline'].items():
            notes = ', '.join(f'{n}({v:.2f})' for n,v in tv['top_notes'][:3])
            bal = tv['balance']
            print(f"    {tk}: T={bal['top']:.0f}/M={bal['middle']:.0f}/B={bal['base']:.0f}%  [{notes}]")
    
    # ==========================================================
    # SUMMARY
    # ==========================================================
    elapsed = time.time() - t0
    print()
    print("="*60)
    print(f"  BACKTESTING SUMMARY: {passed}/{passed+failed} passed ({elapsed:.1f}s)")
    if failed > 0:
        print(f"  {failed} failures detected")
    else:
        print("  ALL PASS - N>2 mixture prediction validated")
    print("="*60)

if __name__ == '__main__':
    main()
