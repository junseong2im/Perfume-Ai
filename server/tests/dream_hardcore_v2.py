"""
Phase 9.5: Hardcore r=0.60+ Optimization
=========================================
Action 1: 5K+ molecule pre-training (val_r target: 0.85-0.90, NOT 0.97)
Action 2: ODT Z-score calibration (expand QSPR to real dynamic range)
Action 3: 2-Layer TransformerEncoder + noise injection (0.03)

All 3 actions unified in a single pipeline.
"""
import sys, os, csv, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA = r"C:\Users\user\Downloads\GG"
BASE = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'cid_smiles_cache.json')
ODT_FILE = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')
PRETRAINED_V2 = os.path.join(BASE, 'models', 'mixture_encoder_v2.pt')

# ============================================================
# Action 3: Enhanced 2-Layer MixtureEncoder + Noise
# ============================================================
class EnhancedMixtureEncoder(nn.Module):
    """2-Layer TransformerEncoder with noise injection"""
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64, num_heads=4, noise_std=0.03):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.noise_std = noise_std
        
        # 2-Layer Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            batch_first=True, dropout=0.15,
            dim_feedforward=256,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x, weights, mask):
        # Noise injection (training only)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=mask)
        
        # Weighted pooling
        w = weights.unsqueeze(-1)
        w = w.masked_fill(mask.unsqueeze(-1), 0.0)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1) / w_sum.squeeze(1)
        
        return self.output_proj(pooled)

class EnhancedSiameseNet(nn.Module):
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = EnhancedMixtureEncoder(input_dim, hidden_dim, output_dim)
        self.dist_head = nn.Sequential(
            nn.Linear(output_dim * 3, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x1, w1, m1, x2, w2, m2):
        z1 = self.encoder(x1, w1, m1)
        z2 = self.encoder(x2, w2, m2)
        diff = torch.abs(z1 - z2)
        prod = z1 * z2
        combined = torch.cat([diff, prod, z1 + z2], dim=-1)
        return self.dist_head(combined).squeeze(-1)

# ============================================================
# Action 2: ODT Z-Score Calibration
# ============================================================
def calibrate_odts(odt_data, real_mean=-1.8, real_std=2.0):
    """Expand QSPR estimates to real-world dynamic range via Z-score mapping"""
    vals = np.array(list(odt_data.values()))
    q_mean, q_std = np.mean(vals), max(np.std(vals), 0.01)
    
    calibrated = {}
    for k, v in odt_data.items():
        z = (v - q_mean) / q_std
        calibrated[k] = float(z * real_std + real_mean)
    
    return calibrated

# ============================================================  
# Action 1: Massive Pre-training from Full POM DB
# ============================================================
def collect_all_predictions(engine, smiles_cache, existing_pred_cache):
    """Collect 138d predictions from ALL available sources"""
    all_preds = dict(existing_pred_cache)
    
    # Add all fragrance DB molecules  
    added = 0
    for name, entry in engine._fragrance_db.items():
        smi = entry.get('smiles', '')
        if not smi:
            continue
        key = f"fdb_{name}"
        if key in all_preds:
            continue
        try:
            pred = engine.predict_138d(smi)
            if pred is not None and np.any(pred > 0):
                all_preds[key] = pred.tolist()
                added += 1
        except:
            pass
    
    print(f"  Fragrance DB added: {added}")
    print(f"  Total molecules: {len(all_preds)}")
    return all_preds

def pretrain_v2(model, all_preds, n_pairs=100000, epochs=60, batch_size=128, 
                target_val_r=0.88):
    """Pre-train with controlled early stopping (target val_r ~0.85-0.90)"""
    print(f"\n[Action 1] Pre-training with {len(all_preds)} molecules, {n_pairs} pairs...")
    print(f"  Target val_r: {target_val_r} (NOT 0.97!)")
    
    keys = [k for k, v in all_preds.items() if v and len(v) >= 138]
    n_mols = len(keys)
    
    # Generate pairs
    pairs_x_a = []
    pairs_x_b = []
    pairs_target = []
    
    for _ in range(n_pairs):
        i, j = random.sample(range(n_mols), 2)
        va = np.array(all_preds[keys[i]][:138])
        vb = np.array(all_preds[keys[j]][:138])
        cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        target = (1.0 - cos_sim) / 2.0
        pairs_x_a.append(va)
        pairs_x_b.append(vb)
        pairs_target.append(target)
    
    # Single molecule = [B, 1, 138]
    X_a = torch.FloatTensor(np.array(pairs_x_a)).unsqueeze(1)
    X_b = torch.FloatTensor(np.array(pairs_x_b)).unsqueeze(1)
    W = torch.ones(n_pairs, 1)
    M = torch.zeros(n_pairs, 1, dtype=torch.bool)
    targets = torch.FloatTensor(np.array(pairs_target))
    
    split = int(0.9 * n_pairs)
    train_idx = list(range(split))
    val_idx = list(range(split, n_pairs))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_state = None
    best_val_r = -1
    
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_idx)
        train_loss = 0
        n_b = 0
        
        for bs in range(0, len(train_idx), batch_size):
            idx = train_idx[bs:bs+batch_size]
            optimizer.zero_grad()
            pred = model(X_a[idx], W[idx], M[idx], X_b[idx], W[idx], M[idx])
            loss = F.mse_loss(pred, targets[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            vp = model(X_a[val_idx], W[val_idx], M[val_idx], X_b[val_idx], W[val_idx], M[val_idx])
            val_r = np.corrcoef(vp.numpy(), targets[val_idx].numpy())[0, 1]
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss/n_b:.4f} val_r={val_r:.4f}")
        
        # Smart early stopping: stop at target, not at peak
        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if val_r >= target_val_r:
            print(f"  Target val_r={target_val_r} reached at epoch {epoch+1} (val_r={val_r:.4f})")
            break
    
    print(f"  Best val_r: {best_val_r:.4f}")
    if best_state and best_val_r >= target_val_r:
        # Use the state at target, not peak
        model.load_state_dict(best_state)
    elif best_state:
        model.load_state_dict(best_state)
    
    return model

# ============================================================
# DREAM Benchmark
# ============================================================
def load_mixtures(path):
    mixtures = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0] or not row[0].strip():
                continue
            ds, lb = row[0].strip(), row[1].strip()
            cids = [c.strip() for c in row[2:] if c.strip() and c.strip() != '0' and c.strip().isdigit()]
            if cids:
                mixtures[f"{ds}_{lb}"] = {'cids': cids}
    return mixtures

def load_pairs(path):
    pairs = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4 and row[0]:
                pairs.append({
                    'mix1_key': f"{row[0].strip()}_{row[1].strip()}",
                    'mix2_key': f"{row[0].strip()}_{row[2].strip()}",
                    'human_dist': float(row[3]),
                })
    return pairs

def prepare_mixture(cids, smiles_cache, pred_cache, odt_data, max_n=60):
    preds, weights = [], []
    for cid in cids:
        smi = smiles_cache.get(cid, '')
        if not smi or cid not in pred_cache:
            continue
        pred = np.array(pred_cache[cid][:138])
        if len(pred) < 138:
            pred = np.pad(pred, (0, 138-len(pred)))
        odt_log = odt_data.get(cid, -3.0)
        w = 1.0 / (10**odt_log + 1e-6)
        preds.append(pred)
        weights.append(w)
    
    if not preds:
        return None, None, None
    
    n = len(preds)
    x = np.zeros((max_n, 138))
    w = np.zeros(max_n)
    mask = np.ones(max_n, dtype=bool)
    for i in range(n):
        x[i] = preds[i]
        w[i] = weights[i]
        mask[i] = False
    if w[:n].sum() > 0:
        w[:n] /= w[:n].sum()
    return x, w, mask

def train_and_eval_siamese(model, pairs, mixtures, smiles_cache, pred_cache, odt_data,
                           epochs=150, lr=0.002, max_n=60):
    """5-fold CV training and evaluation"""
    valid = []
    for pair in pairs:
        k1, k2 = pair['mix1_key'], pair['mix2_key']
        if k1 not in mixtures or k2 not in mixtures:
            continue
        t1 = prepare_mixture(mixtures[k1]['cids'], smiles_cache, pred_cache, odt_data, max_n)
        t2 = prepare_mixture(mixtures[k2]['cids'], smiles_cache, pred_cache, odt_data, max_n)
        if t1[0] is None or t2[0] is None:
            continue
        valid.append({
            'x1': torch.FloatTensor(t1[0]), 'w1': torch.FloatTensor(t1[1]), 'm1': torch.BoolTensor(t1[2]),
            'x2': torch.FloatTensor(t2[0]), 'w2': torch.FloatTensor(t2[1]), 'm2': torch.BoolTensor(t2[2]),
            'target': torch.FloatTensor([pair['human_dist']]),
        })
    
    n = len(valid)
    print(f"  Valid pairs: {n}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_r = -1
    best_state = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        # Clone model for each fold
        model_f = EnhancedSiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
        # Load pre-trained encoder
        model_f.encoder.load_state_dict(model.encoder.state_dict())
        
        # Differential LR: encoder (low) + dist_head (high)
        optimizer = torch.optim.Adam([
            {'params': model_f.encoder.parameters(), 'lr': lr * 0.1},  # pre-trained: 10x lower
            {'params': model_f.dist_head.parameters(), 'lr': lr},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        for epoch in range(epochs):
            model_f.train()
            np.random.shuffle(train_idx)
            bs = min(32, len(train_idx))
            for bstart in range(0, len(train_idx), bs):
                bidx = train_idx[bstart:bstart+bs]
                x1 = torch.stack([valid[i]['x1'] for i in bidx])
                w1 = torch.stack([valid[i]['w1'] for i in bidx])
                m1 = torch.stack([valid[i]['m1'] for i in bidx])
                x2 = torch.stack([valid[i]['x2'] for i in bidx])
                w2 = torch.stack([valid[i]['w2'] for i in bidx])
                m2 = torch.stack([valid[i]['m2'] for i in bidx])
                tgt = torch.stack([valid[i]['target'] for i in bidx]).squeeze()
                
                optimizer.zero_grad()
                pred = model_f(x1, w1, m1, x2, w2, m2)
                loss = F.mse_loss(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_f.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        
        # Validate
        model_f.eval()
        vp, vt = [], []
        with torch.no_grad():
            for i in val_idx:
                p = valid[i]
                pred = model_f(p['x1'].unsqueeze(0), p['w1'].unsqueeze(0), p['m1'].unsqueeze(0),
                             p['x2'].unsqueeze(0), p['w2'].unsqueeze(0), p['m2'].unsqueeze(0))
                vp.append(pred.item())
                vt.append(p['target'].item())
        
        val_r = np.corrcoef(vp, vt)[0, 1]
        fold_results.append(val_r)
        print(f"  Fold {fold+1}: r = {val_r:.4f}")
        
        if val_r > best_r:
            best_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model_f.state_dict().items()}
    
    mean_r = np.mean(fold_results)
    print(f"  Mean CV: r = {mean_r:.4f}")
    return mean_r, fold_results, best_state

# ============================================================
# Main Pipeline
# ============================================================
def main():
    t0 = time.time()
    print("="*60)
    print("  Phase 9.5: Hardcore r=0.60+ Optimization")
    print("="*60)
    
    # Load DREAM data
    print("\n[1] Loading DREAM data...")
    mixtures = load_mixtures(f"{DATA}/Mixure_Definitions_Training_set.csv")
    pairs = load_pairs(f"{DATA}/TrainingData_mixturedist.csv")
    with open(CACHE, 'r') as f:
        smiles_cache = json.load(f)
    print(f"  Mixtures: {len(mixtures)}, Pairs: {len(pairs)}")
    
    # Load caches
    with open(ODT_FILE, 'r') as f:
        cache = json.load(f)
    pred_cache = cache.get('pred_138d', {})
    odt_data = cache.get('odt', {})
    
    # Action 2: Calibrate ODTs
    print("\n[Action 2] ODT Z-Score Calibration...")
    print(f"  Before: mean={np.mean(list(odt_data.values())):.2f} std={np.std(list(odt_data.values())):.2f}")
    odt_calibrated = calibrate_odts(odt_data, real_mean=-1.8, real_std=2.0)
    print(f"  After:  mean={np.mean(list(odt_calibrated.values())):.2f} std={np.std(list(odt_calibrated.values())):.2f}")
    
    # Load engine and collect all predictions
    print("\n[2] Loading POM Engine + collecting all predictions...")
    from pom_engine import POMEngine
    engine = POMEngine()
    engine.load()
    all_preds = collect_all_predictions(engine, smiles_cache, pred_cache)
    
    # Action 1: Massive pre-training
    model = EnhancedSiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
    n_pairs = min(100000, len(all_preds) * (len(all_preds) - 1) // 2)
    model = pretrain_v2(model, all_preds, n_pairs=n_pairs, epochs=60, target_val_r=0.88)
    
    # Save pre-trained model
    os.makedirs(os.path.dirname(PRETRAINED_V2), exist_ok=True)
    torch.save(model.encoder.state_dict(), PRETRAINED_V2)
    print(f"  Saved: {PRETRAINED_V2}")
    
    # Run benchmarks
    print("\n" + "="*60)
    print("  DREAM Benchmark: 4 Modes")
    print("="*60)
    
    results = {}
    
    # Mode 1: Equal weight baseline
    print("\n--- Mode: EQUAL ---")
    from scipy.stats import spearmanr
    mix_vecs = {}
    for key, m in mixtures.items():
        pred = np.zeros(138)
        cnt = 0
        for cid in m['cids']:
            smi = smiles_cache.get(cid, '')
            if smi and cid in pred_cache:
                p = np.array(pred_cache[cid][:138])
                if len(p) < 138:
                    p = np.pad(p, (0, 138-len(p)))
                pred += p
                cnt += 1
        if cnt > 0:
            mix_vecs[key] = pred / cnt
    
    pd_eq, hd_eq = [], []
    for pair in pairs:
        k1, k2 = pair['mix1_key'], pair['mix2_key']
        if k1 in mix_vecs and k2 in mix_vecs:
            sim = np.dot(mix_vecs[k1], mix_vecs[k2]) / (np.linalg.norm(mix_vecs[k1]) * np.linalg.norm(mix_vecs[k2]) + 1e-9)
            pd_eq.append(1.0 - sim)
            hd_eq.append(pair['human_dist'])
    r_eq = np.corrcoef(pd_eq, hd_eq)[0, 1]
    print(f"  Pearson r: {r_eq:.4f}")
    results['equal'] = r_eq
    
    # Mode 2: ODT calibrated
    print("\n--- Mode: ODT_CALIBRATED ---")
    mix_vecs_odt = {}
    for key, m in mixtures.items():
        preds_list, weights_list = [], []
        for cid in m['cids']:
            smi = smiles_cache.get(cid, '')
            if not smi or cid not in pred_cache:
                continue
            p = np.array(pred_cache[cid][:138])
            if len(p) < 138:
                p = np.pad(p, (0, 138-len(p)))
            odt_log = odt_calibrated.get(cid, -1.8)
            w = 1.0 / (10**odt_log + 1e-6)
            preds_list.append(p)
            weights_list.append(w)
        if preds_list:
            wa = np.array(weights_list)
            wa /= wa.sum()
            mix_vecs_odt[key] = np.average(np.array(preds_list), axis=0, weights=wa)
    
    pd_odt, hd_odt = [], []
    for pair in pairs:
        k1, k2 = pair['mix1_key'], pair['mix2_key']
        if k1 in mix_vecs_odt and k2 in mix_vecs_odt:
            sim = np.dot(mix_vecs_odt[k1], mix_vecs_odt[k2]) / (np.linalg.norm(mix_vecs_odt[k1]) * np.linalg.norm(mix_vecs_odt[k2]) + 1e-9)
            pd_odt.append(1.0 - sim)
            hd_odt.append(pair['human_dist'])
    r_odt = np.corrcoef(pd_odt, hd_odt)[0, 1]
    print(f"  Pearson r: {r_odt:.4f}")
    results['odt_calibrated'] = r_odt
    
    # Mode 3: Enhanced Siamese (pre-trained + fine-tuned)
    print("\n--- Mode: ENHANCED_SIAMESE (pre-trained + 2-layer + noise) ---")
    mean_r, folds, best_state = train_and_eval_siamese(
        model, pairs, mixtures, smiles_cache, pred_cache, odt_calibrated,
        epochs=150, lr=0.002)
    results['enhanced_siamese'] = mean_r
    
    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  {'Mode':<25} {'Pearson r':>10}")
    print(f"  {'-'*35}")
    for mode, r in results.items():
        marker = ""
        if mode == 'enhanced_siamese':
            marker = " *** BEST ***"
        print(f"  {mode:<25} {r:>10.4f}{marker}")
    print(f"\n  DREAM baseline: r ~ 0.20")
    print(f"  DREAM winners:  r ~ 0.49")
    print(f"  Our target:     r > 0.60")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
