"""
Fixed Accord Backtest: use predict_138d for proper odor probability vectors
"""
import json, os, sys, csv, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

from pom_engine import POMEngine

def main():
    engine = POMEngine()
    engine.load()
    
    # Load label names
    csv_path = os.path.join(BASE, 'data', 'curated_GS_LF_merged_4983.csv')
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        header = next(csv.reader(f))
        label_names = header[2:]
    print(f"Labels: {len(label_names)}")
    
    # Test single molecule first
    test_smi = "CC1=CCC(CC1)C(=C)C"  # Limonene
    pred_256 = engine.predict_embedding(test_smi)
    print(f"predict_embedding: shape={pred_256.shape if hasattr(pred_256,'shape') else len(pred_256)}")
    
    pred_138 = engine.predict_138d(test_smi)
    print(f"predict_138d: shape={pred_138.shape if hasattr(pred_138,'shape') else len(pred_138)}")
    
    if len(pred_138) >= 138:
        top5 = np.argsort(pred_138[:138])[-5:][::-1]
        print(f"Limonene top 5: {[label_names[i] for i in top5]}")
        print(f"  values: {[f'{pred_138[i]:.3f}' for i in top5]}")
    
    # Test Vanillin
    pred_v = engine.predict_138d("COc1cc(C=O)ccc1O")
    if len(pred_v) >= 138:
        top5 = np.argsort(pred_v[:138])[-5:][::-1]
        print(f"Vanillin top 5: {[label_names[i] for i in top5]}")
    
    # Now backtest perfumes
    PERFUMES = {
        "Classic Citrus": {
            "smiles": ["CC1=CCC(CC1)C(=C)C", "CC(=CCCC(=CC=O)C)C", 
                       "CC(=CCCC(C)(C=C)O)C", "OCC=C(CCC=C(C)C)C"],
            "pcts": [30, 10, 15, 10],
            "expected": ["citrus", "fresh", "floral"],
        },
        "Oriental Vanilla": {
            "smiles": ["COc1cc(C=O)ccc1O", "O=C1OC2=CC=CC=C2C=C1", 
                       "COc1cc(CC=C)ccc1O", "O=CC=CC1=CC=CC=C1"],
            "pcts": [25, 10, 5, 5],
            "expected": ["sweet", "warm spicy", "vanilla"],
        },
        "Fresh Floral": {
            "smiles": ["CC(=CCCC(C)(C=C)O)C", "OCC=C(CCC=C(C)C)C", 
                       "OCCC1=CC=CC=C1", "CC(CCC=C(C)C)CCO"],
            "pcts": [25, 15, 15, 10],
            "expected": ["floral", "fresh", "rose"],
        },
        "Spicy Oriental": {
            "smiles": ["O=CC=CC1=CC=CC=C1", "COc1cc(CC=C)ccc1O",
                       "COc1cc(C=O)ccc1O", "O=C1OC2=CC=CC=C2C=C1"],
            "pcts": [15, 10, 20, 10],
            "expected": ["spicy", "sweet", "balsamic"],
        },
    }
    
    total_hits = 0
    total_tests = 0
    
    for name, data in PERFUMES.items():
        embeddings_138 = []
        valid_pcts = []
        
        for smi, pct in zip(data["smiles"], data["pcts"]):
            pred = engine.predict_138d(smi)
            if pred is not None and len(pred) >= 138:
                embeddings_138.append(pred[:138])
                valid_pcts.append(pct)
        
        if not embeddings_138:
            print(f"\n  {name}: No predictions")
            continue
        
        # OAV-weighted combination
        weights = np.array(valid_pcts, dtype=float)
        odt_db = engine._odt_predicted if hasattr(engine, '_odt_predicted') else {}
        for i, smi in enumerate(data["smiles"][:len(embeddings_138)]):
            odt = odt_db.get(smi, 1.5)
            oav = max(0.1, math.log10(weights[i] * 1e4 + 1) + odt)
            weights[i] = oav
        
        weights = np.exp(weights - weights.max())
        weights /= weights.sum()
        
        mixture_138 = sum(w * np.array(e) for w, e in zip(weights, embeddings_138))
        
        # Top predicted accords
        top_idx = np.argsort(mixture_138)[-8:][::-1]
        predicted = [label_names[i] for i in top_idx if i < len(label_names)]
        
        expected = [e.lower() for e in data["expected"]]
        
        # Fuzzy matching
        hits = 0
        for exp in expected:
            for pred in predicted:
                if exp in pred.lower() or pred.lower() in exp:
                    hits += 1
                    break
        
        total_hits += hits
        total_tests += len(expected)
        
        status = "PASS" if hits >= 2 else ("PARTIAL" if hits >= 1 else "FAIL")
        print(f"\n  [{status}] {name}")
        print(f"    Predicted: {predicted[:5]}")
        print(f"    Expected:  {expected}")
        print(f"    Hit Rate:  {hits}/{len(expected)}")
    
    overall = total_hits / max(1, total_tests)
    print(f"\n  === Overall: {total_hits}/{total_tests} ({100*overall:.0f}%) ===")

if __name__ == '__main__':
    main()
