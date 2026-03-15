"""Comprehensive reliability assessment of the odor prediction system"""
import os, sys, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
from rdkit import Chem

BASE = "http://localhost:8001"
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "curated_GS_LF_merged_4983.csv")

# Our 20d dimensions
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
]

# Direct mappable GS labels → our dimensions
DIRECT_MAP = {
    'sweet': 'sweet', 'sour': 'sour', 'woody': 'woody', 'floral': 'floral',
    'citrus': 'citrus', 'spicy': 'spicy', 'musky': 'musk', 'fresh': 'fresh',
    'green': 'green', 'warm': 'warm', 'fruity': 'fruity', 'smoky': 'smoky',
    'powdery': 'powdery', 'herbal': 'herbal', 'amber': 'amber',
    'leather': 'leather', 'earthy': 'earthy', 'ozonic': 'ozonic',
    'metallic': 'metallic',
    # Synonyms
    'vanilla': 'sweet', 'caramellic': 'sweet', 'honey': 'sweet',
    'rose': 'floral', 'jasmine': 'floral', 'violet': 'floral',
    'lemon': 'citrus', 'orange': 'citrus', 'grapefruit': 'citrus',
    'cinnamon': 'spicy', 'clove': 'spicy', 'pepper': 'spicy',
    'pine': 'woody', 'cedar': 'woody', 'sandalwood': 'woody',
    'mint': 'fresh', 'cooling': 'fresh', 'clean': 'fresh',
    'grassy': 'green', 'leafy': 'green',
    'apple': 'fruity', 'banana': 'fruity', 'peach': 'fruity',
    'tropical': 'fruity', 'berry': 'fruity', 'cherry': 'fruity',
    'melon': 'fruity', 'pineapple': 'fruity', 'coconut': 'fruity',
    'roasted': 'smoky', 'burnt': 'smoky', 'coffee': 'smoky',
    'nutty': 'warm', 'balsamic': 'woody',
    'mushroom': 'earthy', 'musty': 'earthy',
    'sulfurous': 'metallic',
    'tobacco': 'smoky',
    'lavender': 'floral',
}

def load_gs_data():
    with open(DATA, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    odor_cols = header[1:]
    return rows, odor_cols

def get_ground_truth_dims(row, odor_cols):
    """Get expected 20d dimensions from GS labels"""
    active_labels = [col.lower() for col, val in zip(odor_cols, row[1:]) if val == "1"]
    expected_dims = set()
    for label in active_labels:
        if label in DIRECT_MAP:
            expected_dims.add(DIRECT_MAP[label])
    return expected_dims, active_labels

def main():
    rows, odor_cols = load_gs_data()
    
    # Random sample (stratified)
    np.random.seed(42)
    indices = np.random.permutation(len(rows))[:500]  # 500 random molecules
    
    print("=" * 60)
    print("  RELIABILITY ASSESSMENT (500 random molecules)")
    print("=" * 60)
    
    # Metrics
    top1_hit = 0      # top1 predicted dim matches any ground truth dim
    top3_hit = 0      # any of top3 predicted dims matches any ground truth dim
    top5_hit = 0      # any of top5 matches
    precision_sum = 0 # how many of top5 predictions are correct
    recall_sum = 0    # how many ground truth dims are captured in top5
    cos_sims = []     # cosine similarity
    tested = 0
    errors = 0
    skipped = 0
    
    t0 = time.time()
    
    for i, idx in enumerate(indices):
        row = rows[idx]
        smiles = row[0]
        expected_dims, active_labels = get_ground_truth_dims(row, odor_cols)
        
        if not expected_dims:
            skipped += 1
            continue
        
        try:
            r = requests.post(f"{BASE}/api/engine/predict-odor", 
                            json={"smiles": smiles}, timeout=10)
            if r.status_code != 200:
                errors += 1
                continue
        except:
            errors += 1
            continue
        
        d = r.json()
        top_dims = d.get("top_dimensions", [])
        pred_dims = [t[0] for t in top_dims[:5]]
        sim = d.get("similarity", 0)
        cos_sims.append(sim)
        
        # Top-1 accuracy
        if pred_dims and pred_dims[0] in expected_dims:
            top1_hit += 1
        
        # Top-3 accuracy
        if any(p in expected_dims for p in pred_dims[:3]):
            top3_hit += 1
        
        # Top-5 accuracy
        if any(p in expected_dims for p in pred_dims[:5]):
            top5_hit += 1
        
        # Precision@5: what fraction of predictions are correct
        if pred_dims:
            correct_in_pred = sum(1 for p in pred_dims if p in expected_dims)
            precision_sum += correct_in_pred / len(pred_dims)
        
        # Recall@5: what fraction of ground truth is captured
        if expected_dims:
            captured = sum(1 for e in expected_dims if e in pred_dims)
            recall_sum += captured / len(expected_dims)
        
        tested += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/500 processed ({elapsed:.0f}s)")
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  RESULTS ({tested} tested, {skipped} skipped, {errors} errors)")
    print(f"{'='*60}")
    
    if tested > 0:
        print(f"\n  === Accuracy ===")
        print(f"  Top-1 Hit Rate:  {top1_hit/tested*100:5.1f}%  ({top1_hit}/{tested})")
        print(f"  Top-3 Hit Rate:  {top3_hit/tested*100:5.1f}%  ({top3_hit}/{tested})")
        print(f"  Top-5 Hit Rate:  {top5_hit/tested*100:5.1f}%  ({top5_hit}/{tested})")
        
        print(f"\n  === Precision & Recall @5 ===")
        print(f"  Avg Precision@5: {precision_sum/tested*100:5.1f}%")
        print(f"  Avg Recall@5:    {recall_sum/tested*100:5.1f}%")
        
        f1 = 2 * (precision_sum/tested) * (recall_sum/tested) / ((precision_sum/tested) + (recall_sum/tested) + 1e-8)
        print(f"  F1 Score:        {f1*100:5.1f}%")
        
        print(f"\n  === Cosine Similarity ===")
        cosarr = np.array(cos_sims)
        print(f"  Mean:    {cosarr.mean():.3f}")
        print(f"  Median:  {np.median(cosarr):.3f}")
        print(f"  Std:     {cosarr.std():.3f}")
        print(f"  Min:     {cosarr.min():.3f}")
        print(f"  Max:     {cosarr.max():.3f}")
        print(f"  >0.8:    {(cosarr > 0.8).sum()}/{tested} ({(cosarr > 0.8).sum()/tested*100:.0f}%)")
        print(f"  >0.7:    {(cosarr > 0.7).sum()}/{tested} ({(cosarr > 0.7).sum()/tested*100:.0f}%)")
        print(f"  >0.5:    {(cosarr > 0.5).sum()}/{tested} ({(cosarr > 0.5).sum()/tested*100:.0f}%)")
        
        # Per-dimension accuracy
        print(f"\n  === Per-Dimension Top-1 Accuracy ===")
        dim_correct = {d: 0 for d in ODOR_DIMENSIONS}
        dim_total = {d: 0 for d in ODOR_DIMENSIONS}
        
        for idx_j in indices[:tested]:
            row = rows[idx_j]
            expected_dims, _ = get_ground_truth_dims(row, odor_cols)
            for ed in expected_dims:
                dim_total[ed] = dim_total.get(ed, 0) + 1
        
        for dim in ODOR_DIMENSIONS:
            total = dim_total.get(dim, 0)
            if total > 5:
                print(f"    {dim:12s} total={total:4d}")
    
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/tested*1000:.0f}ms/molecule)")
    
    # === Reliability Grade ===
    print(f"\n{'='*60}")
    top5_pct = top5_hit/tested*100 if tested > 0 else 0
    f1_pct = f1 * 100 if tested > 0 else 0
    mean_sim = cosarr.mean() if len(cos_sims) > 0 else 0
    
    if top5_pct >= 90 and f1_pct >= 50 and mean_sim >= 0.7:
        grade = "A (Production Ready)"
    elif top5_pct >= 80 and f1_pct >= 40 and mean_sim >= 0.6:
        grade = "B (Good - Minor improvements needed)"
    elif top5_pct >= 60 and f1_pct >= 30:
        grade = "C (Fair - Significant improvements needed)"
    else:
        grade = "D (Prototype - Major improvements needed)"
    
    print(f"  RELIABILITY GRADE: {grade}")
    print(f"  Top-5={top5_pct:.0f}% | F1={f1_pct:.0f}% | CosSim={mean_sim:.3f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
