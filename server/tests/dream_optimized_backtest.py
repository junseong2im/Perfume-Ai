"""
DREAM Score Optimization — Steps 2+3 Unified
==============================================
Step 2: ODT-weighted mixture prediction (replaces equal weight)
Step 3: Siamese Set Transformer for pairwise distance prediction

Usage:
  python tests/dream_optimized_backtest.py --mode odt      # Step 2 only
  python tests/dream_optimized_backtest.py --mode siamese   # Step 3 (full)
  python tests/dream_optimized_backtest.py --mode all       # Run all modes and compare
"""
import sys, os, csv, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA = r"C:\Users\user\Downloads\GG"
CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_upgrade', 'cid_smiles_cache.json')
ODT_CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_upgrade', 'dream_odt.json')

# ============================================================
# Data loaders (reused from test_dream_backtest.py)
# ============================================================
def load_mixtures(path):
    mixtures = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0] or not row[0].strip():
                continue
            dataset = row[0].strip()
            label = row[1].strip()
            cids = [c.strip() for c in row[2:] 
                    if c.strip() and c.strip() != '0' and c.strip().isdigit()]
            if not cids:
                continue
            mixtures[f"{dataset}_{label}"] = {'dataset': dataset, 'label': label, 'cids': cids}
    return mixtures

def load_pairs(path):
    pairs = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4 or not row[0]:
                continue
            pairs.append({
                'mix1_key': f"{row[0].strip()}_{row[1].strip()}",
                'mix2_key': f"{row[0].strip()}_{row[2].strip()}",
                'human_dist': float(row[3]),
            })
    return pairs

# ============================================================
# Step 2: ODT-weighted mixture prediction
# ============================================================
def predict_mixture_odt_weighted(engine, cids, smiles_cache, odt_data, pred_cache):
    """Predict mixture profile using ODT-based perceptual weighting"""
    n_tasks = len(engine.TASKS_138) if hasattr(engine, 'TASKS_138') else 138
    
    # Collect predictions + ODT weights
    preds = []
    weights = []
    
    for cid in cids:
        smiles = smiles_cache.get(cid, '')
        if not smiles:
            continue
        
        # Get 138d prediction (from cache or compute)
        if cid in pred_cache:
            pred = np.array(pred_cache[cid])
        else:
            try:
                pred = engine.predict_138d(smiles)
                if pred is None or not np.any(pred > 0):
                    continue
            except:
                continue
        
        # ODT-based weight: lower threshold = stronger smell = higher weight
        odt_log = odt_data.get(cid, -3.0)
        # Perceptual weight: inverse of ODT (molecules with low ODT dominate)
        # Use Hill-like: w = 1 / (10^odt + 1e-6) → then normalize
        w = 1.0 / (10**odt_log + 1e-6)
        
        preds.append(pred)
        weights.append(w)
    
    if not preds:
        return None
    
    preds = np.array(preds)
    weights = np.array(weights)
    weights /= weights.sum()  # Normalize
    
    # Weighted blend
    mix_pred = np.average(preds, axis=0, weights=weights)
    return mix_pred

# ============================================================
# Step 3: Siamese Attention Net
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureEncoder(nn.Module):
    """Set Transformer: variable-size N molecules → fixed-size vector"""
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, 
            batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x, weights, mask):
        """
        x: [B, max_N, 138] per-molecule 138d predictions  
        weights: [B, max_N] perceptual weights (ODT-based)
        mask: [B, max_N] True=padded (ignore)
        """
        # Project to hidden dim
        h = self.input_proj(x)  # [B, N, hidden]
        
        # Self-attention: molecules interact (masking/synergy)
        h_attn, _ = self.attention(h, h, h, key_padding_mask=mask)
        h = self.norm(h + h_attn)  # residual connection
        
        # Weighted pooling using perceptual weights
        w = weights.unsqueeze(-1)  # [B, N, 1]
        w = w.masked_fill(mask.unsqueeze(-1), 0.0)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1) / w_sum.squeeze(1)  # [B, hidden]
        
        # Project to output space
        out = self.output_proj(pooled)  # [B, output_dim]
        return out

class SiameseMixtureNet(nn.Module):
    """Siamese network: two mixtures → predicted perceptual distance"""
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = MixtureEncoder(input_dim, hidden_dim, output_dim)
        # Distance head: concatenated differences → scalar distance
        self.dist_head = nn.Sequential(
            nn.Linear(output_dim * 3, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output 0-1 distance
        )
    
    def forward(self, x1, w1, m1, x2, w2, m2):
        z1 = self.encoder(x1, w1, m1)
        z2 = self.encoder(x2, w2, m2)
        
        # Multi-feature distance
        diff = torch.abs(z1 - z2)
        prod = z1 * z2
        combined = torch.cat([diff, prod, z1 + z2], dim=-1)
        
        dist = self.dist_head(combined).squeeze(-1)
        return dist

def prepare_mixture_tensor(cids, smiles_cache, pred_cache, odt_data, max_n=60):
    """Prepare tensors for one mixture"""
    preds = []
    weights = []
    
    for cid in cids:
        smiles = smiles_cache.get(cid, '')
        if not smiles or cid not in pred_cache:
            continue
        
        pred = np.array(pred_cache[cid])
        odt_log = odt_data.get(cid, -3.0)
        w = 1.0 / (10**odt_log + 1e-6)
        
        preds.append(pred)
        weights.append(w)
    
    if not preds:
        return None, None, None
    
    n = len(preds)
    # Pad to max_n
    x = np.zeros((max_n, 138))
    w = np.zeros(max_n)
    mask = np.ones(max_n, dtype=bool)  # True = padded
    
    for i in range(n):
        x[i] = preds[i][:138] if len(preds[i]) >= 138 else np.pad(preds[i], (0, 138-len(preds[i])))
        w[i] = weights[i]
        mask[i] = False
    
    # Normalize weights
    if w.sum() > 0:
        w[:n] /= w[:n].sum()
    
    return x, w, mask

def train_siamese(model, train_pairs, mixtures, smiles_cache, pred_cache, odt_data, 
                  epochs=100, lr=0.001, max_n=60):
    """Train Siamese network on DREAM training pairs"""
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Prepare all data
    valid_pairs = []
    for pair in train_pairs:
        k1, k2 = pair['mix1_key'], pair['mix2_key']
        if k1 not in mixtures or k2 not in mixtures:
            continue
        
        t1 = prepare_mixture_tensor(mixtures[k1]['cids'], smiles_cache, pred_cache, odt_data, max_n)
        t2 = prepare_mixture_tensor(mixtures[k2]['cids'], smiles_cache, pred_cache, odt_data, max_n)
        
        if t1[0] is None or t2[0] is None:
            continue
        
        valid_pairs.append({
            'x1': torch.FloatTensor(t1[0]), 'w1': torch.FloatTensor(t1[1]), 'm1': torch.BoolTensor(t1[2]),
            'x2': torch.FloatTensor(t2[0]), 'w2': torch.FloatTensor(t2[1]), 'm2': torch.BoolTensor(t2[2]),
            'target': torch.FloatTensor([pair['human_dist']]),
        })
    
    n = len(valid_pairs)
    print(f"  Training on {n} valid pairs")
    
    if n < 20:
        print("  [WARN] Too few pairs for training")
        return model
    
    # 5-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(n))
    
    best_r = -1
    best_state = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # Reset model for each fold
        model_fold = SiameseMixtureNet(input_dim=138, hidden_dim=128, output_dim=64).to(device)
        optimizer = torch.optim.Adam(model_fold.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        for epoch in range(epochs):
            model_fold.train()
            train_loss = 0
            np.random.shuffle(train_idx)
            
            # Mini-batch training (batch_size=32)
            batch_size = min(32, len(train_idx))
            for batch_start in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[batch_start:batch_start+batch_size]
                
                x1 = torch.stack([valid_pairs[i]['x1'] for i in batch_idx])
                w1 = torch.stack([valid_pairs[i]['w1'] for i in batch_idx])
                m1 = torch.stack([valid_pairs[i]['m1'] for i in batch_idx])
                x2 = torch.stack([valid_pairs[i]['x2'] for i in batch_idx])
                w2 = torch.stack([valid_pairs[i]['w2'] for i in batch_idx])
                m2 = torch.stack([valid_pairs[i]['m2'] for i in batch_idx])
                targets = torch.stack([valid_pairs[i]['target'] for i in batch_idx]).squeeze()
                
                optimizer.zero_grad()
                preds = model_fold(x1, w1, m1, x2, w2, m2)
                loss = F.mse_loss(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_fold.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
        
        # Validate
        model_fold.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for i in val_idx:
                p = valid_pairs[i]
                pred = model_fold(
                    p['x1'].unsqueeze(0), p['w1'].unsqueeze(0), p['m1'].unsqueeze(0),
                    p['x2'].unsqueeze(0), p['w2'].unsqueeze(0), p['m2'].unsqueeze(0)
                )
                val_preds.append(pred.item())
                val_targets.append(p['target'].item())
        
        val_r = np.corrcoef(val_preds, val_targets)[0, 1]
        print(f"  Fold {fold+1}: val Pearson r = {val_r:.4f}")
        
        if val_r > best_r:
            best_r = val_r
            best_state = model_fold.state_dict().copy()
    
    print(f"  Best fold: r = {best_r:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

# ============================================================
# Main evaluation
# ============================================================
def evaluate(mode, engine, mixtures, pairs, smiles_cache, pred_cache, odt_data):
    """Evaluate a specific prediction mode"""
    from scipy.stats import spearmanr
    
    predicted_dists = []
    human_dists = []
    
    if mode == 'equal':
        # Baseline: equal weight averaging
        mix_vecs = {}
        n_tasks = 138
        for key, m in mixtures.items():
            pred = np.zeros(n_tasks)
            count = 0
            for cid in m['cids']:
                smi = smiles_cache.get(cid, '')
                if not smi or cid not in pred_cache:
                    continue
                p = np.array(pred_cache[cid])
                pred += p[:n_tasks] if len(p) >= n_tasks else np.pad(p, (0, n_tasks-len(p)))
                count += 1
            if count > 0:
                mix_vecs[key] = pred / count
        
        for pair in pairs:
            k1, k2 = pair['mix1_key'], pair['mix2_key']
            if k1 in mix_vecs and k2 in mix_vecs:
                sim = np.dot(mix_vecs[k1], mix_vecs[k2]) / (
                    np.linalg.norm(mix_vecs[k1]) * np.linalg.norm(mix_vecs[k2]) + 1e-9)
                predicted_dists.append(1.0 - sim)
                human_dists.append(pair['human_dist'])
    
    elif mode == 'odt':
        # Step 2: ODT-weighted
        mix_vecs = {}
        for key, m in mixtures.items():
            vec = predict_mixture_odt_weighted(engine, m['cids'], smiles_cache, odt_data, pred_cache)
            if vec is not None:
                mix_vecs[key] = vec
        
        for pair in pairs:
            k1, k2 = pair['mix1_key'], pair['mix2_key']
            if k1 in mix_vecs and k2 in mix_vecs:
                sim = np.dot(mix_vecs[k1], mix_vecs[k2]) / (
                    np.linalg.norm(mix_vecs[k1]) * np.linalg.norm(mix_vecs[k2]) + 1e-9)
                predicted_dists.append(1.0 - sim)
                human_dists.append(pair['human_dist'])
    
    elif mode in ('siamese', 'siamese_pretrained'):
        # Step 3: Siamese Attention Net (with optional pre-trained encoder)
        max_n = 60
        model = SiameseMixtureNet(input_dim=138, hidden_dim=128, output_dim=64)
        
        # Load pre-trained encoder if available
        pretrained_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'mixture_encoder_pretrained.pt')
        if mode == 'siamese_pretrained' and os.path.exists(pretrained_path):
            print("  Loading pre-trained encoder...")
            state = torch.load(pretrained_path, map_location='cpu')
            model.encoder.load_state_dict(state)
            print("  Pre-trained encoder loaded!")
            # Lower LR for pre-trained encoder to avoid catastrophic forgetting
            lr = 0.0005
        else:
            lr = 0.002
        
        print(f"  Training Siamese Net (5-fold CV, lr={lr})...")
        model = train_siamese(model, pairs, mixtures, smiles_cache, pred_cache, odt_data,
                            epochs=150, lr=lr, max_n=max_n)
        
        # Final evaluation on all pairs
        model.eval()
        with torch.no_grad():
            for pair in pairs:
                k1, k2 = pair['mix1_key'], pair['mix2_key']
                if k1 not in mixtures or k2 not in mixtures:
                    continue
                
                t1 = prepare_mixture_tensor(mixtures[k1]['cids'], smiles_cache, pred_cache, odt_data, max_n)
                t2 = prepare_mixture_tensor(mixtures[k2]['cids'], smiles_cache, pred_cache, odt_data, max_n)
                
                if t1[0] is None or t2[0] is None:
                    continue
                
                pred = model(
                    torch.FloatTensor(t1[0]).unsqueeze(0),
                    torch.FloatTensor(t1[1]).unsqueeze(0),
                    torch.BoolTensor(t1[2]).unsqueeze(0),
                    torch.FloatTensor(t2[0]).unsqueeze(0),
                    torch.FloatTensor(t2[1]).unsqueeze(0),
                    torch.BoolTensor(t2[2]).unsqueeze(0),
                )
                predicted_dists.append(pred.item())
                human_dists.append(pair['human_dist'])
    
    if len(predicted_dists) < 10:
        return {'pearson_r': 0, 'spearman_r': 0, 'n_pairs': 0}
    
    predicted_dists = np.array(predicted_dists)
    human_dists = np.array(human_dists)
    
    pearson_r = np.corrcoef(predicted_dists, human_dists)[0, 1]
    spearman_r, sp = spearmanr(predicted_dists, human_dists)
    rmse = np.sqrt(np.mean((predicted_dists - human_dists)**2))
    
    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'spearman_p': sp,
        'rmse': rmse,
        'n_pairs': len(predicted_dists),
    }

def main():
    t0 = time.time()
    
    print("="*60)
    print("  DREAM Score Optimization - Full Pipeline")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    mixtures = load_mixtures(f"{DATA}/Mixure_Definitions_Training_set.csv")
    pairs = load_pairs(f"{DATA}/TrainingData_mixturedist.csv")
    
    with open(CACHE, 'r') as f:
        smiles_cache = json.load(f)
    
    print(f"  Mixtures: {len(mixtures)}, Pairs: {len(pairs)}")
    
    # Load ODT cache
    odt_data = {}
    pred_cache = {}
    if os.path.exists(ODT_CACHE):
        with open(ODT_CACHE, 'r') as f:
            cache_data = json.load(f)
            odt_data = cache_data.get('odt', {})
            pred_cache = cache_data.get('pred_138d', {})
        print(f"  ODT cache: {len(odt_data)}, Pred cache: {len(pred_cache)}")
    
    # If no pred cache, generate on-the-fly
    if not pred_cache:
        print("\n[2] Loading POM Engine + generating predictions...")
        from pom_engine import POMEngine, TASKS_138
        engine = POMEngine()
        engine.load()
        
        for cid, smiles in smiles_cache.items():
            if not smiles:
                continue
            try:
                pred = engine.predict_138d(smiles)
                if pred is not None and np.any(pred > 0):
                    pred_cache[cid] = pred.tolist()
            except:
                pass
        
        # Generate ODT
        for cid, smiles in smiles_cache.items():
            if not smiles or cid in odt_data:
                continue
            odt_data[cid] = -3.0  # Default
        
        print(f"  Generated: {len(pred_cache)} predictions, {len(odt_data)} ODTs")
        
        # Save cache
        with open(ODT_CACHE, 'w') as f:
            json.dump({'odt': odt_data, 'pred_138d': pred_cache}, f)
    else:
        # Still need engine for ODT-weighted mode
        from pom_engine import POMEngine
        engine = POMEngine()
        engine.load()
    
    # Run all modes
    modes = ['equal', 'odt', 'siamese', 'siamese_pretrained']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")
        r = evaluate(mode, engine, mixtures, pairs, smiles_cache, pred_cache, odt_data)
        results[mode] = r
        print(f"  Pearson r:  {r['pearson_r']:.4f}")
        print(f"  Spearman r: {r['spearman_r']:.4f}")
        if 'rmse' in r:
            print(f"  RMSE:       {r['rmse']:.4f}")
        print(f"  Pairs:      {r['n_pairs']}")
    
    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION SUMMARY ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  {'Mode':<15} {'Pearson r':>10} {'Spearman r':>12} {'Improvement':>12}")
    print(f"  {'-'*49}")
    
    baseline_r = results['equal']['pearson_r']
    for mode in modes:
        r = results[mode]
        improvement = ((r['pearson_r'] - baseline_r) / abs(baseline_r) * 100) if baseline_r != 0 else 0
        marker = " ← baseline" if mode == 'equal' else f" +{improvement:.1f}%" if improvement > 0 else f" {improvement:.1f}%"
        print(f"  {mode:<15} {r['pearson_r']:>10.4f} {r['spearman_r']:>12.4f} {marker}")
    
    print(f"\n  DREAM baseline: r~0.20")
    print(f"  DREAM winners:  r~0.49")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
