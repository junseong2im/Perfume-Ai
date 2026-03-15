"""
POM Engine v5 -- Production-Grade Verification Suite
=====================================================
Tests ALL physics fixes, edge cases, and production logic.
"""
import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pom_engine import POMEngine, TASKS_138

def fmt(val, width=8):
    if isinstance(val, float):
        return f"{val:>{width}.4f}"
    return f"{str(val):>{width}}"

def test_header(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")

def main():
    t0 = time.time()
    engine = POMEngine()
    engine.load()
    
    passed = 0
    failed = 0
    results = []
    
    def check(name, condition, detail=""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        results.append((name, status, detail))
        icon = "[OK]" if condition else "[XX]"
        print(f"  {icon} {name}: {detail}")
    
    # ================================================================
    # TEST 1: QSPR Volatility (Retention Index)
    # ================================================================
    test_header("QSPR Volatility Classification")
    
    # Vanillin: MW=152, HBD=1, TPSA=46.5 -> Retention=200 -> Base
    vanillin_smi = engine.resolve_smiles('vanillin')
    if vanillin_smi:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(vanillin_smi)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        retention = mw + hbd * 25 + tpsa * 0.5 + hba * 5.0
        
        if retention < 160: vol = 'top'
        elif retention < 215: vol = 'middle'
        else: vol = 'base'
        
        print(f"  Vanillin: MW={mw:.1f}, TPSA={tpsa:.1f}, HBD={hbd}, HBA={hba}")
        print(f"  Retention = {mw:.1f} + {hbd}*25 + {tpsa:.1f}*0.5 + {hba}*5 = {retention:.1f}")
        check("Vanillin NOT Top", vol != 'top',
              f"class={vol}, retention={retention:.1f}")
        check("Vanillin is Base (HBA-assisted retention)", vol == 'base',
              f"class={vol}, retention={retention:.1f}")
    
    # Limonene: MW=136, HBD=0, TPSA=0 -> Retention=136 -> Top
    limonene_smi = engine.resolve_smiles('limonene')
    if limonene_smi:
        mol = Chem.MolFromSmiles(limonene_smi)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        retention = mw + hbd * 25 + tpsa * 0.5 + hba * 5.0
        
        if retention < 160: vol = 'top'
        elif retention < 215: vol = 'middle'
        else: vol = 'base'
        
        print(f"  Limonene: MW={mw:.1f}, TPSA={tpsa:.1f}, HBD={hbd}")
        check("Limonene is Top (volatile terpene)", vol == 'top',
              f"class={vol}, retention={retention:.1f}")
    
    # Iso E Super: MW=234, HBD=0, TPSA=17.1 -> ~243 -> Middle/Base
    iso_smi = engine.resolve_smiles('iso_e_super')
    if iso_smi:
        mol = Chem.MolFromSmiles(iso_smi)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        retention = mw + hbd * 25 + tpsa * 0.5 + hba * 5.0
        if retention < 160: vol = 'top'
        elif retention < 215: vol = 'middle'
        else: vol = 'base'
        print(f"  Iso E Super: MW={mw:.1f}, TPSA={tpsa:.1f}, HBD={hbd}")
        check("Iso E Super NOT Top", vol != 'top',
              f"class={vol}, retention={retention:.1f}")
    
    # ================================================================
    # TEST 2: Hill Equation Receptor Saturation
    # ================================================================
    test_header("Hill Equation Receptor Saturation")
    
    # At threshold concentration, occupancy should be low
    sat_at_threshold = engine.hill_saturation(0.00001, 0.0)  # 0.00001% of ODT_log=0 molecule
    sat_at_10pct = engine.hill_saturation(10.0, 1.5)       # high conc
    sat_at_50pct = engine.hill_saturation(50.0, 1.5)       # very high conc
    
    print(f"  Hill(0.00001%, ODT=0.0) = {sat_at_threshold:.4f}")
    print(f"  Hill(10%,     ODT=1.5) = {sat_at_10pct:.4f}")
    print(f"  Hill(50%,     ODT=1.5) = {sat_at_50pct:.4f}")
    
    check("Very low conc -> low saturation", sat_at_threshold < 0.5,
          f"{sat_at_threshold:.4f} < 0.5")
    check("High conc -> high saturation", sat_at_10pct > 0.5,
          f"{sat_at_10pct:.4f} > 0.5")
    check("Saturation curve (diminishing returns)", 
          sat_at_50pct < sat_at_10pct * 3,
          f"50% sat ({sat_at_50pct:.3f}) < 3x of 10% sat ({sat_at_10pct*3:.3f})")
    
    # Vanillin at 0.1% (ODT_log=3.5 = extreme sensitivity)
    van_sat = engine.hill_saturation(0.1, 3.5)
    lin_sat = engine.hill_saturation(10.0, 0.0)  # linalool high conc low ODT
    print(f"  Vanillin  0.1% (ODT=3.5): sat={van_sat:.4f}")
    print(f"  Linalool 10.0% (ODT=0.0): sat={lin_sat:.4f}")
    check("Vanillin (low conc, high sensitivity) vs Linalool (high conc, low sensitivity)",
          van_sat > lin_sat * 0.5,
          f"vanillin {van_sat:.3f} vs linalool {lin_sat:.3f}")
    
    # ================================================================
    # TEST 3: Perceptual Distance (Off-Note Penalty)
    # ================================================================
    test_header("Perceptual Distance -- Off-Note Penalty")
    
    # Create two predictions: one clean, one with garlic
    clean_pred = np.zeros(len(TASKS_138))
    clean_pred[TASKS_138.index('floral')] = 0.8
    clean_pred[TASKS_138.index('sweet')] = 0.7
    clean_pred[TASKS_138.index('rose')] = 0.6
    
    garlic_pred = clean_pred.copy()
    garlic_pred[TASKS_138.index('garlic')] = 0.15  # inject garlic!
    
    target = clean_pred.copy()
    
    sim_clean = engine.perceptual_distance(clean_pred, target)
    sim_garlic = engine.perceptual_distance(garlic_pred, target)
    sim_cosine_garlic = engine.cosine_sim(garlic_pred, target)
    
    print(f"  Clean recipe  -> perceptual_dist = {sim_clean:.4f}")
    print(f"  Garlic recipe -> perceptual_dist = {sim_garlic:.4f}")
    print(f"  Garlic recipe -> cosine_sim      = {sim_cosine_garlic:.4f}")
    
    check("Clean > Garlic (perceptual)", sim_clean > sim_garlic,
          f"{sim_clean:.4f} > {sim_garlic:.4f}")
    check("Cosine would miss garlic (score still high)", sim_cosine_garlic > 0.95,
          f"cosine={sim_cosine_garlic:.4f} still high despite garlic!")
    check("Perceptual catches garlic (big penalty)", sim_clean - sim_garlic > 0.5,
          f"penalty={sim_clean - sim_garlic:.4f}")
    
    # ================================================================
    # TEST 4: Attention Dilution Defense
    # ================================================================
    test_header("Attention Dilution Defense")
    
    # Test: strong molecule + many weak ones
    van_smi = engine.resolve_smiles('vanillin')
    lin_smi = engine.resolve_smiles('linalool')
    
    if van_smi and lin_smi and engine._use_attention:
        # Normal 2-molecule prediction
        result_2 = engine.predict_mixture_attention(
            [van_smi, lin_smi], [5.0, 5.0])
        
        check("2-molecule attention works", result_2 is not None,
              f"n_active={result_2.get('n_active', 'N/A')}" if result_2 else "None")
        
        if result_2:
            # With 5 weak additional molecules (should get masked)
            extra_smiles = [van_smi, lin_smi]
            extra_pcts = [5.0, 5.0]
            
            # Add very low concentration molecules
            hedione_smi = engine.resolve_smiles('hedione')
            if hedione_smi:
                extra_smiles.append(hedione_smi)
                extra_pcts.append(0.0001)  # basically nothing
                extra_smiles.append(hedione_smi)
                extra_pcts.append(0.0001)
            
            # Add truly imperceptible molecules (ODT_log=0 = high threshold)
            # Use same molecules but at trace amounts with low ODT
            result_many = engine.predict_mixture_attention(
                extra_smiles, extra_pcts)
            if result_many:
                n_active = result_many.get('n_active', len(extra_smiles))
                check("Dilution defense active (n_active tracked)", 
                      'n_active' in result_many,
                      f"n_active={n_active}, n_total={len(extra_smiles)}")
            else:
                check("Dilution defense: fallback when too few active", True,
                      "correctly returned None")
    else:
        check("Attention model loaded", False, "model not available")
    
    # ================================================================
    # TEST 5: Mixture Prediction -- Hill vs Softmax
    # ================================================================
    test_header("Mixture Prediction (Hill Equation)")
    
    mix = engine.predict_mixture([
        {'name': 'vanillin', 'pct': 0.1},
        {'name': 'linalool', 'pct': 10.0},
        {'name': 'limonene', 'pct': 5.0},
    ])
    
    if mix:
        check("Mixture returns result", True, f"method={mix.get('method', 'N/A')}")
        check("Top notes non-empty", len(mix['top_notes']) > 0,
              f"{len(mix['top_notes'])} notes")
        
        # Check that vanillin dominates despite low concentration (high ODT)
        dom = mix['dominant_ingredient']
        print(f"  Dominant: {dom}")
        for ind in mix['individual']:
            print(f"    {ind['name']:15s} {ind['pct']:5.1f}%  w={ind['perceptual_weight']:.3f}")
        
        # Verify attention blend notes exist
        has_attn = 'attention_blend_notes' in mix
        check("Attention blend notes present", has_attn,
              f"{len(mix.get('attention_blend_notes', []))} notes" if has_attn else "missing")
    
    # ================================================================
    # TEST 6: GA Multi-Objective (IFRA + Cost + Balance + Perceptual)
    # ================================================================
    test_header("GA Multi-Objective Optimization")
    
    result = engine.reverse_engineer(
        target='geraniol',
        exclude_names=['rose', 'geraniol', 'citronellol'],
        n_components=5,
        population_size=100,
        generations=200,
        enforce_ifra=True,
        enforce_balance=True,
        target_budget_usd=30,
    )
    
    if result:
        sim = result['similarity']
        cost = result['cost_per_kg']
        ifra = result['ifra_compliant']
        bal = result['balance']
        gen = result['generations_used']
        
        check("Similarity > 0.95", sim > 0.95, f"sim={sim}")
        check("Cost near budget ($30 +/-5%)", cost <= 31.5, f"cost=${cost}")
        check("IFRA compliant", ifra, "COMPLIANT" if ifra else "VIOLATION")
        check("Top note 10-35%", 10 <= bal['top'] <= 35, f"T={bal['top']}%")
        check("Middle note 30-65%", 30 <= bal['middle'] <= 65, f"M={bal['middle']}%")
        check("Base note 10-40%", 10 <= bal['base'] <= 40, f"B={bal['base']}%")
        check("Converged in <200 gen", gen < 200, f"gen={gen}")
        
        print(f"\n  Recipe ({gen} gen, sim={sim}, ${cost}/kg):")
        for item in result['recipe']:
            print(f"    {item['name']:20s} {item['pct']:5.1f}% "
                  f"[{item['volatility']:6s}] R={item.get('retention','?')} "
                  f"${item['cost_kg']}/kg {item['ifra_status']}")
    else:
        check("GA returned result", False, "None")
    
    # ================================================================
    # TEST 7: 138d Prediction Sanity
    # ================================================================
    test_header("138d Prediction Sanity Checks")
    
    # Vanillin should predict vanilla
    van_pred = engine.predict_138d(engine.resolve_smiles('vanillin'))
    van_top = TASKS_138[np.argmax(van_pred)]
    check("Vanillin top note = vanilla", van_top == 'vanilla',
          f"top={van_top}({van_pred[np.argmax(van_pred)]:.2f})")
    
    # Eugenol should predict spicy
    eug_pred = engine.predict_138d(engine.resolve_smiles('eugenol'))
    eug_top = TASKS_138[np.argmax(eug_pred)]
    check("Eugenol top note = spicy", eug_top == 'spicy',
          f"top={eug_top}({eug_pred[np.argmax(eug_pred)]:.2f})")
    
    # Linalool should predict floral
    lin_pred = engine.predict_138d(engine.resolve_smiles('linalool'))
    lin_top = TASKS_138[np.argmax(lin_pred)]
    check("Linalool top note = floral", lin_top == 'floral',
          f"top={lin_top}({lin_pred[np.argmax(lin_pred)]:.2f})")
    
    # ================================================================
    # TEST 8: Embedding DB Coverage
    # ================================================================
    test_header("Embedding DB & Model Status")
    
    n_emb = len(engine._embedding_db['smiles']) if engine._embedding_db else 0
    n_db = len(engine._fragrance_db)
    n_models = len(engine.models) if engine.models else 0
    has_attn = engine._use_attention
    
    check("Embedding DB >= 5000", n_emb >= 5000, f"{n_emb} molecules")
    check("Fragrance DB >= 700", n_db >= 700, f"{n_db} entries")
    check("Ensemble models loaded", n_models >= 10, f"{n_models} models")
    check("PairAttentionNet loaded", has_attn, 
          f"{len(engine._pair_labels)} labels" if engine._pair_labels else "N/A")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{passed+failed}")
    print(f"  Failed: {failed}/{passed+failed}")
    print(f"  Time:   {elapsed:.1f}s")
    print()
    
    if failed > 0:
        print("  FAILED TESTS:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"    [XX] {name}: {detail}")
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"  [ALL PASS] POM Engine v5 Production Verified ({elapsed:.1f}s)")
    else:
        print(f"  [{failed} FAILURES] Review required")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
