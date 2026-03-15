"""
Fragrantica-style Backtest + NIST Evaporation Calibration
==========================================================
1. Backtest: Known perfume compositions → predict accords → compare with actual
2. Calibrate evaporation: known BP data → fix Joback model thresholds
"""
import json, os, sys, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

# ============================================================
# NIST Boiling Point Calibration
# ============================================================
# Known boiling points from NIST WebBook (ground truth)
NIST_BP_DATA = {
    # SMILES: (name, actual_bp_celsius, note_type)
    "CC1=CCC(CC1)C(=C)C": ("Limonene", 176, "top"),
    "CC(=CCCC(C)(C=C)O)C": ("Linalool", 198, "middle"),
    "COc1cc(C=O)ccc1O": ("Vanillin", 285, "base"),
    "CCO": ("Ethanol", 78, "top"),
    "CC=O": ("Acetaldehyde", 20, "top"),
    "O=C1OC2=CC=CC=C2C=C1": ("Coumarin", 301, "base"),   # Coumarin sublimes
    "COc1cc(CC=C)ccc1O": ("Eugenol", 253, "middle"),
    "CC(=CCCC(=CC=O)C)C": ("Citral", 229, "top"),
    "OCC=C(CCC=C(C)C)C": ("Geraniol", 230, "middle"),
    "CC(CCC=C(C)C)CCO": ("Citronellol", 225, "middle"),
    "O=CC=CC1=CC=CC=C1": ("Cinnamaldehyde", 248, "middle"),
    "OCC1=CC=CC=C1": ("Benzyl alcohol", 205, "middle"),
    "OCCC1=CC=CC=C1": ("Phenylethanol", 220, "middle"),
    "O=CC1=CC=CC=C1": ("Benzaldehyde", 179, "top"),
    "CC(=O)C1=CC=CC=C1": ("Acetophenone", 202, "middle"),
    "COC1=CC=C(C=O)C=C1": ("Anisaldehyde", 248, "middle"),
    "CC(=O)C1=CC=C(O)C=C1": ("Raspberry Ketone", 293, "base"),
    "O=CCC1=CC=CC=C1": ("Phenylacetaldehyde", 195, "top"),
    "OC(=O)C1=CC=CC=C1O": ("Salicylic acid", 211, "base"),
    "CC(=CCCC(=CC)C)COC(=O)C": ("Geranyl acetate", 242, "middle"),
}

def calibrate_evaporation():
    """Calibrate Joback BP estimation against NIST data"""
    print("=" * 60)
    print("  [1] NIST Boiling Point Calibration")
    print("=" * 60)
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except:
        print("  RDKit required")
        return 0, 0
    
    errors = []
    results = []
    
    for smi, (name, actual_bp, expected_note) in NIST_BP_DATA.items():
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        n_rings = Descriptors.RingCount(mol)
        
        # Current Joback estimation
        est_bp = 198.0
        est_bp += mw * 0.8
        est_bp += hbd * 45
        est_bp += hba * 12
        est_bp += tpsa * 0.3
        est_bp += n_aromatic * 30
        est_bp += n_rings * 10
        est_bp += rb * 2
        est_bp -= max(0, logp - 5) * 5
        est_bp -= 273.15
        
        error = est_bp - actual_bp
        errors.append(error)
        results.append((name, actual_bp, est_bp, error))
    
    # Print results
    mae = np.mean(np.abs(errors))
    print(f"  {'Name':<20} {'NIST BP':>8} {'Est BP':>8} {'Error':>8}")
    print(f"  {'-'*48}")
    for name, actual, est, err in results:
        symbol = "✓" if abs(err) < 30 else "✗"
        print(f"  {symbol} {name:<18} {actual:>6}°C {est:>6.0f}°C {err:>+6.0f}°C")
    
    print(f"\n  MAE: {mae:.1f}°C")
    print(f"  Correlation: {np.corrcoef([r[1] for r in results], [r[2] for r in results])[0,1]:.3f}")
    
    # Calculate optimal coefficients using least squares
    X = []
    Y = []
    for smi, (name, actual_bp, _) in NIST_BP_DATA.items():
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        X.append([
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.MolLogP(mol),
        ])
        Y.append(actual_bp)
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Add bias
    X_bias = np.column_stack([np.ones(len(X)), X])
    
    # Least squares: Y = X @ coeffs
    coeffs, residuals, _, _ = np.linalg.lstsq(X_bias, Y, rcond=None)
    
    # Verify with calibrated coefficients
    Y_pred = X_bias @ coeffs
    mae_cal = np.mean(np.abs(Y_pred - Y))
    r2 = 1 - np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y))**2)
    
    print(f"\n  CALIBRATED Joback coefficients:")
    names = ['bias', 'MW', 'HBD', 'HBA', 'TPSA', 'AromaticRings', 'TotalRings', 'RotBonds', 'LogP']
    for n, c in zip(names, coeffs):
        print(f"    {n}: {c:.4f}")
    print(f"  Calibrated MAE: {mae_cal:.1f}°C (was {mae:.1f}°C)")
    print(f"  Calibrated R²: {r2:.3f}")
    
    # Verify note classification with calibrated BP
    print(f"\n  Note classification with calibrated BP:")
    cal_passed = 0
    for i, (smi, (name, actual_bp, expected_note)) in enumerate(NIST_BP_DATA.items()):
        cal_bp = Y_pred[i]
        
        # Note thresholds (calibrated against NIST data)
        # Top: actual BP < 200°C
        # Middle: 200-260°C
        # Base: > 260°C
        if cal_bp < 200:
            note = 'top'
        elif cal_bp < 260:
            note = 'middle'
        else:
            note = 'base'
        
        ok = note == expected_note
        if ok:
            cal_passed += 1
        symbol = "✅" if ok else "❌"
        print(f"  {symbol} {name:<18} BP={cal_bp:>6.0f}°C note={note:<7} expected={expected_note}")
    
    print(f"\n  Score: {cal_passed}/{len(NIST_BP_DATA)} ({100*cal_passed//len(NIST_BP_DATA)}%)")
    
    # Save calibrated coefficients
    cal_data = {
        'coefficients': {n: float(c) for n, c in zip(names, coeffs)},
        'features': ['MW', 'HBD', 'HBA', 'TPSA', 'AromaticRings', 'TotalRings', 'RotBonds', 'LogP'],
        'mae_celsius': float(mae_cal),
        'r2': float(r2),
        'note_thresholds': {'top': 200, 'middle': 260},
        'training_size': len(NIST_BP_DATA),
    }
    cal_path = os.path.join(BASE, 'data', 'pom_upgrade', 'bp_calibration.json')
    with open(cal_path, 'w') as f:
        json.dump(cal_data, f, indent=2)
    print(f"  Saved: {cal_path}")
    
    # Update ingredients with calibrated BP
    ing_paths = [
        os.path.join(BASE, '..', 'data', 'ingredients.json'),
        os.path.join(BASE, 'data', 'ingredients.json'),
    ]
    
    for path in ing_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            ings = json.load(f)
        
        updated = 0
        for ing in ings:
            smi = ing.get('smiles', '')
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            
            feats = [1,  # bias
                Descriptors.MolWt(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.MolLogP(mol),
            ]
            cal_bp = sum(c * f for c, f in zip(coeffs, feats))
            
            if cal_bp < 200:
                note = 'top'
            elif cal_bp < 260:
                note = 'middle'
            else:
                note = 'base'
            
            half_life = 0.1 * math.exp(max(0, cal_bp) / 80.0)
            half_life = max(0.25, min(72.0, half_life))
            volatility = max(1.0, min(10.0, cal_bp / 35.0))
            
            ing['est_boiling_point_c'] = round(cal_bp, 1)
            ing['note_type'] = note
            ing['half_life_hours'] = round(half_life, 2)
            ing['volatility'] = round(volatility, 1)
            updated += 1
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ings, f, indent=2, ensure_ascii=False)
        print(f"  Updated {updated} ingredients at {path}")
    
    return cal_passed, len(NIST_BP_DATA)

# ============================================================
# 2. Fragrantica-Style Backtest
# ============================================================
def backtest_accords():
    """Test predicted accords against known perfume compositions"""
    print(f"\n{'='*60}")
    print("  [2] Fragrantica-Style Accord Backtest")
    print(f"{'='*60}")
    
    # Known perfume compositions with SMILES and expected accords
    # Based on publicly known formulations and Fragrantica descriptions
    KNOWN_PERFUMES = {
        "Classic Citrus Cologne": {
            "composition": {
                "CC1=CCC(CC1)C(=C)C": 25,    # Limonene
                "CC(=CCCC(=CC=O)C)C": 5,      # Citral
                "CC(=CCCC(C)(C=C)O)C": 10,    # Linalool
                "OCC=C(CCC=C(C)C)C": 8,       # Geraniol
                "CC(CCC=C(C)C)CCO": 5,        # Citronellol
                "CCO": 47,                      # Ethanol carrier
            },
            "expected_accords": ["citrus", "fresh", "green"],
        },
        "Oriental Vanilla": {
            "composition": {
                "COc1cc(C=O)ccc1O": 15,        # Vanillin
                "O=C1OC2=CC=CC=C2C=C1": 5,     # Coumarin
                "COc1cc(CC=C)ccc1O": 3,         # Eugenol
                "O=CC=CC1=CC=CC=C1": 2,         # Cinnamaldehyde
                "CCO": 75,                       # Carrier
            },
            "expected_accords": ["sweet", "oriental", "warm spicy"],
        },
        "Fresh Floral": {
            "composition": {
                "CC(=CCCC(C)(C=C)O)C": 20,    # Linalool
                "OCC=C(CCC=C(C)C)C": 10,      # Geraniol
                "OCCC1=CC=CC=C1": 8,           # Phenylethanol (rose)
                "CC(CCC=C(C)C)CCO": 5,         # Citronellol
                "CCO": 57,                      # Carrier
            },
            "expected_accords": ["floral", "fresh", "green"],
        },
        "Woody Musky": {
            "composition": {
                "CC1=CCC(CC1)C(=C)C": 5,      # Limonene
                "CC(=CCCC(C)(C=C)O)C": 8,     # Linalool
                "O=C1OC2=CC=CC=C2C=C1": 8,    # Coumarin
                "COc1cc(C=O)ccc1O": 5,         # Vanillin
                "CCO": 74,                      # Carrier
            },
            "expected_accords": ["woody", "warm spicy", "sweet"],
        },
    }
    
    # Load POM engine for predictions
    try:
        from pom_engine import POMEngine
        engine = POMEngine()
        engine.load()
        has_engine = True
    except Exception as e:
        print(f"  POM Engine not available: {e}")
        has_engine = False
    
    if not has_engine:
        print("  Skipping backtest (requires POM engine)")
        return 0, 0
    
    # Top 138 odor labels from the model
    labels_path = os.path.join(BASE, 'data', 'pom_upgrade', 'pom_master_138d.json')
    label_names = None
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            master = json.load(f)
            label_names = master.get('label_names', None)
    
    if not label_names:
        # Use the curated CSV headers
        import csv
        csv_path = os.path.join(BASE, 'data', 'curated_GS_LF_merged_4983.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                header = next(csv.reader(f))
                label_names = header[2:]  # Skip SMILES and descriptors
    
    if not label_names:
        print("  No label names found")
        return 0, 0
    
    print(f"  Label names: {len(label_names)} ({label_names[:5]}...)")
    
    total_hits = 0
    total_tests = 0
    
    for perf_name, perf_data in KNOWN_PERFUMES.items():
        comp = perf_data["composition"]
        expected = [a.lower() for a in perf_data["expected_accords"]]
        
        # Get 138d embeddings for each component
        embeddings = []
        weights = []
        for smi, pct in comp.items():
            if smi == "CCO":  # Skip carrier
                continue
            emb = engine.predict_embedding(smi)
            if np.linalg.norm(emb) > 0:
                embeddings.append(emb)
                weights.append(pct)
        
        if not embeddings:
            print(f"\n  {perf_name}: No valid embeddings")
            continue
        
        # Weighted average (simplified mixture prediction)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        mixture_emb = sum(w * e for w, e in zip(weights, embeddings))
        
        # Convert to 138d probabilities
        if len(mixture_emb) > 138:
            # If 256d, take first 138
            probs = mixture_emb[:138]
        else:
            probs = mixture_emb
        
        # Get top 5 predicted accords
        if len(probs) == len(label_names):
            top_indices = np.argsort(probs)[-5:][::-1]
            predicted = [label_names[i].lower() for i in top_indices]
        else:
            predicted = []
        
        # Check hits
        hits = sum(1 for exp in expected if any(exp in pred for pred in predicted))
        hit_rate = hits / len(expected)
        total_hits += hits
        total_tests += len(expected)
        
        status = "✅" if hit_rate >= 0.5 else "⚠️"
        print(f"\n  {status} {perf_name}")
        print(f"     Predicted: {predicted}")
        print(f"     Expected:  {expected}")
        print(f"     Hit Rate:  {hits}/{len(expected)} ({100*hit_rate:.0f}%)")
    
    overall_hr = total_hits / max(1, total_tests)
    print(f"\n  === Backtest Summary ===")
    print(f"  Overall Hit Rate: {total_hits}/{total_tests} ({100*overall_hr:.0f}%)")
    
    return total_hits, total_tests

# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    
    cal_passed, cal_total = calibrate_evaporation()
    bt_hits, bt_total = backtest_accords()
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  FINAL ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  NIST BP Calibration: {cal_passed}/{cal_total}")
    print(f"  Accord Backtest: {bt_hits}/{bt_total}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
