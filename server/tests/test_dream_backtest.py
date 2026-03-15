"""
DREAM Olfactory Mixtures Backtesting
=====================================
Uses the official DREAM challenge data to validate our POM Engine's
ability to predict perceptual similarity between N>2 molecule mixtures.

Pipeline:
1. Load mixture definitions (CID lists per mixture)
2. Convert CIDs to SMILES via PubChem
3. For each pair of mixtures, predict 138d profile using POM Engine
4. Compute predicted similarity (cosine/perceptual)
5. Compare predicted vs human-rated similarity
6. Calculate correlation (Pearson r) and ranking metrics
"""
import sys, os, csv, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA = r"C:\Users\user\Downloads\GG"
CACHE = r"C:\Users\user\Desktop\Game\server\data\pom_upgrade\cid_smiles_cache.json"

# ============================================================
# Step 1: Parse mixture definitions
# ============================================================
def load_mixtures(path):
    """Parse mixtures: {dataset_label: {dataset, label, cids, n_components}}"""
    mixtures = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0] or not row[0].strip():
                continue
            dataset = row[0].strip()
            label = row[1].strip()
            # Filter: non-empty, non-zero, numeric only
            cids = [c.strip() for c in row[2:] 
                    if c.strip() and c.strip() != '0' and c.strip().isdigit()]
            if not cids:
                continue
            key = f"{dataset}_{label}"
            mixtures[key] = {
                'dataset': dataset,
                'label': label,
                'cids': cids,
                'n_components': len(cids),
            }
    return mixtures

# ============================================================
# Step 2: Parse training data (pairwise similarities)
# ============================================================
def load_training_pairs(path):
    """Parse: [(dataset, mix1, mix2, human_similarity)]"""
    pairs = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 4 or not row[0]:
                continue
            dataset = row[0].strip()
            mix1 = row[1].strip()
            mix2 = row[2].strip()
            sim = float(row[3])
            pairs.append({
                'dataset': dataset,
                'mix1_key': f"{dataset}_{mix1}",
                'mix2_key': f"{dataset}_{mix2}",
                'human_sim': sim,
            })
    return pairs

# ============================================================
# Step 3: CID → SMILES via PubChem (with cache)
# ============================================================
def get_smiles_batch(cids, cache_path):
    """Convert PubChem CIDs to SMILES, with local caching."""
    # Load cache
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    
    missing = [c for c in cids if c not in cache]
    
    if missing:
        print(f"  Fetching {len(missing)} CIDs from PubChem...")
        import urllib.request
        import urllib.error
        
        # Batch API: max 200 per request
        batch_size = 100
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i+batch_size]
            cid_str = ','.join(batch)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IsomericSMILES,CanonicalSMILES/JSON"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    for prop in data.get('PropertyTable', {}).get('Properties', []):
                        cid = str(prop['CID'])
                        # Try multiple SMILES keys
                        smi = (prop.get('IsomericSMILES', '') or 
                               prop.get('CanonicalSMILES', '') or
                               prop.get('ConnectivitySMILES', ''))
                        cache[cid] = smi
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                print(f"    PubChem batch {i//batch_size+1} error: {e}")
                # Don't cache failures as empty
                pass
        
        # Save cache
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        print(f"  Cached {len(cache)} CID→SMILES mappings")
    
    return cache

# ============================================================
# Step 4: Run backtesting
# ============================================================
def main():
    t0 = time.time()
    
    print("="*60)
    print("  DREAM Olfactory Mixtures Backtesting")
    print("="*60)
    
    # Load data
    print("\n[1] Loading DREAM data...")
    mixtures = load_mixtures(f"{DATA}/Mixure_Definitions_Training_set.csv")
    pairs = load_training_pairs(f"{DATA}/TrainingData_mixturedist.csv")
    print(f"  Mixtures: {len(mixtures)} definitions")
    print(f"  Pairs: {len(pairs)} comparisons")
    
    # Collect all CIDs
    all_cids = set()
    for m in mixtures.values():
        all_cids.update(m['cids'])
    print(f"  Unique CIDs: {len(all_cids)}")
    
    # N-component statistics
    n_counts = {}
    for m in mixtures.values():
        n = m['n_components']
        n_counts[n] = n_counts.get(n, 0) + 1
    print(f"  N-component distribution: {dict(sorted(n_counts.items()))}")
    
    # CID → SMILES
    print("\n[2] Converting CIDs to SMILES...")
    smiles_cache = get_smiles_batch(list(all_cids), CACHE)
    resolved = sum(1 for c in all_cids if smiles_cache.get(c, ''))
    print(f"  Resolved: {resolved}/{len(all_cids)} ({resolved/len(all_cids)*100:.1f}%)")
    
    # Load POM Engine
    print("\n[3] Loading POM Engine...")
    from pom_engine import POMEngine, TASKS_138
    engine = POMEngine()
    engine.load()
    
    # Predict 138d for each mixture
    print("\n[4] Predicting 138d profiles for all mixtures...")
    mix_vectors = {}
    
    for key, m in mixtures.items():
        smiles_list = []
        for cid in m['cids']:
            smi = smiles_cache.get(cid, '')
            if smi:
                smiles_list.append(smi)
        
        if len(smiles_list) == 0:
            continue
        
        # Equal parts for each component (DREAM doesn't specify ratios for Snitz set)
        pct = 100.0 / len(smiles_list)
        
        # Predict each molecule individually, then average (simple mixing model)
        pred = np.zeros(len(TASKS_138))
        count = 0
        for smi in smiles_list:
            try:
                p = engine.predict_138d(smi)
                if p is not None and np.any(p):
                    pred += p
                    count += 1
            except:
                pass
        
        if count > 0:
            pred /= count
            mix_vectors[key] = pred
    
    print(f"  Predicted: {len(mix_vectors)}/{len(mixtures)} mixtures ({len(mix_vectors)/len(mixtures)*100:.1f}%)")
    
    # Evaluate pairs
    print("\n[5] Evaluating pairwise similarities...")
    predicted_dists = []
    human_dists = []
    
    for pair in pairs:
        k1 = pair['mix1_key']
        k2 = pair['mix2_key']
        if k1 in mix_vectors and k2 in mix_vectors:
            cos_sim = engine.cosine_sim(mix_vectors[k1], mix_vectors[k2])
            # DREAM ground truth = perceptual DISTANCE, so convert similarity to distance
            pred_dist = 1.0 - cos_sim
            predicted_dists.append(pred_dist)
            human_dists.append(pair['human_sim'])
    
    print(f"  Evaluated: {len(predicted_dists)}/{len(pairs)} pairs")
    
    if len(predicted_dists) < 10:
        print("  [WARN] Too few pairs for reliable statistics")
        return
    
    predicted_dists = np.array(predicted_dists)
    human_dists = np.array(human_dists)
    
    # Pearson correlation
    pearson_r = np.corrcoef(predicted_dists, human_dists)[0, 1]
    
    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(predicted_dists, human_dists)
    
    # RMSE
    rmse = np.sqrt(np.mean((predicted_dists - human_dists)**2))
    
    # Direction accuracy (does our model agree on which pairs are more distant?)
    correct_direction = 0
    total_comparisons = 0
    for i in range(len(predicted_dists)):
        for j in range(i+1, min(i+10, len(predicted_dists))):
            h_order = human_dists[i] > human_dists[j]
            p_order = predicted_dists[i] > predicted_dists[j]
            if h_order == p_order:
                correct_direction += 1
            total_comparisons += 1
    direction_acc = correct_direction / total_comparisons if total_comparisons > 0 else 0
    
    print()
    print("="*60)
    print("  DREAM BACKTESTING RESULTS")
    print("="*60)
    print(f"  Pairs evaluated:     {len(predicted_dists)}")
    print(f"  Pearson r:           {pearson_r:.4f}")
    print(f"  Spearman r:          {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  RMSE:                {rmse:.4f}")
    print(f"  Direction accuracy:  {direction_acc:.1%}")
    print()
    
    # Benchmark against DREAM challenge results
    # Top teams in DREAM Challenge got Pearson r ~ 0.4-0.5
    # Baseline (Dragon descriptors + linear model): r ~ 0.2
    print("  Benchmark comparison:")
    if pearson_r > 0.4:
        grade = "EXCELLENT (top-tier DREAM challenge level)"
    elif pearson_r > 0.3:
        grade = "GOOD (above DREAM challenge median)"
    elif pearson_r > 0.2:
        grade = "FAIR (baseline level)"
    else:
        grade = "WEAK (below baseline)"
    print(f"    Our model:       r={pearson_r:.3f} → {grade}")
    print(f"    DREAM winners:   r~0.49 (Keller et al.)")
    print(f"    DREAM baseline:  r~0.20 (Dragon linear)")
    print()
    
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    print("="*60)

if __name__ == '__main__':
    main()
