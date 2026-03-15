"""
DREAM Score Optimization — Step 1: Coverage 100%
=================================================
Bake all 165 DREAM molecules into our embedding DB + ODT cache.
For molecules not in the 5330 DB, use Phase 1 ensemble for zero-shot prediction.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pom_engine import POMEngine, TASKS_138

BASE = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'cid_smiles_cache.json')
DREAM_EMB_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_embeddings.npz')
DREAM_ODT_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')

def main():
    t0 = time.time()
    
    # Load CID→SMILES
    with open(CACHE, 'r') as f:
        cid_smiles = json.load(f)
    non_empty = {k: v for k, v in cid_smiles.items() if v}
    print(f"DREAM molecules: {len(non_empty)}")
    
    # Load engine
    engine = POMEngine()
    engine.load()
    
    # ── Step 1A: Generate 256d embeddings for ALL 165 molecules ──
    print("\n[Step 1A] Generating 256d embeddings for all DREAM molecules...")
    
    embeddings = {}
    in_db = 0
    generated = 0
    failed = 0
    
    for cid, smiles in non_empty.items():
        # Check if in embedding DB
        if smiles in engine.smiles_to_idx:
            idx = engine.smiles_to_idx[smiles]
            emb = engine.embeddings[idx]
            embeddings[cid] = emb
            in_db += 1
        else:
            # Zero-shot: use POM ensemble to generate embedding
            try:
                emb = engine._get_embedding(smiles)
                if emb is not None and len(emb) == engine.emb_dim:
                    embeddings[cid] = emb
                    generated += 1
                else:
                    failed += 1
            except:
                failed += 1
    
    print(f"  In DB: {in_db}, Zero-shot generated: {generated}, Failed: {failed}")
    print(f"  Total coverage: {len(embeddings)}/{len(non_empty)} ({len(embeddings)/len(non_empty)*100:.1f}%)")
    
    # ── Step 1B: Generate ODT (Odor Detection Threshold) for ALL ──
    print("\n[Step 1B] Generating ODT for all DREAM molecules...")
    
    odt_data = {}
    has_db_odt = 0
    xgb_predicted = 0
    no_odt = 0
    
    for cid, smiles in non_empty.items():
        # Try database lookup first
        odt_log = None
        for k, v in engine.frag_db.items():
            if v.get('smiles', '') == smiles and v.get('odt_log') is not None:
                odt_log = v['odt_log']
                has_db_odt += 1
                break
        
        if odt_log is None:
            # XGBoost prediction
            try:
                if hasattr(engine, 'odt_model') and engine.odt_model is not None and cid in embeddings:
                    pred = engine.odt_model.predict(embeddings[cid].reshape(1, -1))
                    odt_log = float(pred[0])
                    xgb_predicted += 1
                else:
                    odt_log = -3.0  # Default: 1 ppm
                    no_odt += 1
            except:
                odt_log = -3.0
                no_odt += 1
        
        odt_data[cid] = odt_log
    
    print(f"  DB ODT: {has_db_odt}, XGBoost: {xgb_predicted}, Default: {no_odt}")
    
    # ── Step 1C: Generate 138d predictions for ALL ──
    print("\n[Step 1C] Generating 138d predictions for all DREAM molecules...")
    
    predictions_138d = {}
    pred_ok = 0
    pred_fail = 0
    
    for cid, smiles in non_empty.items():
        try:
            pred = engine.predict_138d(smiles)
            if pred is not None and np.any(pred > 0):
                predictions_138d[cid] = pred
                pred_ok += 1
            else:
                pred_fail += 1
        except:
            pred_fail += 1
    
    print(f"  138d OK: {pred_ok}, Failed: {pred_fail}")
    
    # ── Save caches ──
    print("\n[Save] Writing DREAM caches...")
    
    # Embeddings
    emb_array = {}
    for cid, emb in embeddings.items():
        emb_array[cid] = emb
    np.savez_compressed(DREAM_EMB_CACHE, **{f"cid_{k}": v for k, v in emb_array.items()})
    
    # 138d predictions
    pred_dict = {}
    for cid, pred in predictions_138d.items():
        pred_dict[cid] = pred.tolist()
    
    # ODT
    with open(DREAM_ODT_CACHE, 'w') as f:
        json.dump({
            'odt': odt_data,
            'pred_138d': pred_dict,
        }, f)
    
    elapsed = time.time() - t0
    print(f"\n[DONE] Step 1 complete in {elapsed:.1f}s")
    print(f"  Embeddings: {len(embeddings)}/{len(non_empty)}")
    print(f"  ODT: {len(odt_data)}/{len(non_empty)}")
    print(f"  138d: {len(predictions_138d)}/{len(non_empty)}")

if __name__ == '__main__':
    main()
