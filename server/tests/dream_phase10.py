"""
Phase 10: 5K Contrastive Pre-training + DREAM Fine-tuning
==========================================================
Step 1: Extract 4983 SMILES from curated CSV
Step 2: Batch predict_138d for all (cached to avoid re-computation)
Step 3: Dynamic Contrastive Pre-training (1400万 pair space, no memorization)
Step 4: DREAM 5-Fold Fine-tuning with pre-trained encoder
Step 5: Compare scratch vs pre-trained
"""
import sys, os, csv, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.join(os.path.dirname(__file__), '..')
CURATED_CSV = os.path.join(BASE, 'data', 'curated_GS_LF_merged_4983.csv')
MASTER_138D = os.path.join(BASE, 'data', 'pom_upgrade', 'pom_master_138d.json')
PRETRAINED_V3 = os.path.join(BASE, 'models', 'mixture_encoder_v3.pt')
DREAM_DATA = r"C:\Users\user\Downloads\GG"
DREAM_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'cid_smiles_cache.json')
DREAM_ODT = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')

# ============================================================
# Architecture (FROZEN: 1-Layer Siamese, identical to Phase 9)
# ============================================================
class MixtureEncoder(nn.Module):
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
        h = self.input_proj(x)
        h_attn, _ = self.attention(h, h, h, key_padding_mask=mask)
        h = self.norm(h + h_attn)
        w = weights.unsqueeze(-1)
        w = w.masked_fill(mask.unsqueeze(-1), 0.0)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1) / w_sum.squeeze(1)
        return self.output_proj(pooled)

class SiameseNet(nn.Module):
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = MixtureEncoder(input_dim, hidden_dim, output_dim)
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
# Step 1+2: Extract SMILES → 138d predictions (cached)
# ============================================================
def build_master_138d():
    """Extract 138d predictions for all 4983 curated molecules"""
    if os.path.exists(MASTER_138D):
        with open(MASTER_138D, 'r') as f:
            db = json.load(f)
        if len(db) >= 1000:
            print(f"  [CACHE] Loaded {len(db)} molecules from master 138d DB")
            return db

    print("  [Step 1] Extracting SMILES from curated CSV...")
    smiles_list = []
    with open(CURATED_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row and row[0].strip():
                smiles_list.append(row[0].strip())

    smiles_list = list(set(smiles_list))
    print(f"  Unique SMILES: {len(smiles_list)}")

    print("  [Step 2] Batch generating 138d predictions...")
    from pom_engine import POMEngine
    engine = POMEngine()
    engine.load()

    db = {}
    failed = 0
    for i, smi in enumerate(smiles_list):
        try:
            pred = engine.predict_138d(smi)
            if pred is not None and np.any(pred > 0):
                db[smi] = pred.tolist()
        except:
            failed += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(smiles_list)}: {len(db)} success, {failed} failed")

    # Save cache
    os.makedirs(os.path.dirname(MASTER_138D), exist_ok=True)
    with open(MASTER_138D, 'w') as f:
        json.dump(db, f)

    print(f"  [DONE] Master DB: {len(db)} molecules ({failed} failed)")
    return db

# ============================================================
# Step 3: Dynamic Contrastive Pre-training
# ============================================================
def pretrain_dynamic(model, master_db, epochs=20, batch_size=256):
    """
    Dynamic pair generation from 4983 molecules = ~12.4M unique pairs.
    Each epoch generates FRESH random pairs -> memorization harder.
    
    Key insight: val_r=0.90 at epoch 1 means the linear projection is
    too easy. We need the ATTENTION weights to learn, not just the linear layers.
    Solution: Use multi-molecule mixtures (N=2~5) as pre-training input,
    not single molecules. This forces the attention mechanism to learn
    how to combine multiple vectors.
    """
    keys = list(master_db.keys())
    n_mols = len(keys)
    print(f"\n  [Step 3] Dynamic Contrastive Pre-training (Multi-Mix)")
    print(f"  Molecules: {n_mols}")
    print(f"  Pair space: {n_mols * (n_mols-1) // 2:,} unique pairs")

    # Pre-convert to tensor array for speed
    all_vecs = torch.FloatTensor(np.array([master_db[k][:138] for k in keys]))  # [N, 138]

    max_mix_n = 10  # Max molecules per synthetic mixture
    pairs_per_epoch = 30000
    val_size = 5000

    def make_synthetic_mixture(n_components, max_n=max_mix_n):
        """Create a synthetic mixture of n_components random molecules"""
        mol_ids = random.sample(range(n_mols), n_components)
        x = torch.zeros(max_n, 138)
        w = torch.zeros(max_n)
        mask = torch.ones(max_n, dtype=torch.bool)
        for i, mid in enumerate(mol_ids):
            x[i] = all_vecs[mid]
            w[i] = 1.0 / n_components  # equal weight
            mask[i] = False
        # Mixture centroid (for target distance)
        centroid = all_vecs[mol_ids].mean(dim=0)
        return x, w, mask, centroid

    def make_pair_batch(n_pairs):
        x1_list, w1_list, m1_list = [], [], []
        x2_list, w2_list, m2_list = [], [], []
        targets = []
        for _ in range(n_pairs):
            n1 = random.randint(1, 5)
            n2 = random.randint(1, 5)
            x1, w1, m1, c1 = make_synthetic_mixture(n1)
            x2, w2, m2, c2 = make_synthetic_mixture(n2)
            cos = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()
            dist = (1.0 - cos) / 2.0
            x1_list.append(x1); w1_list.append(w1); m1_list.append(m1)
            x2_list.append(x2); w2_list.append(w2); m2_list.append(m2)
            targets.append(dist)
        return (torch.stack(x1_list), torch.stack(w1_list), torch.stack(m1_list),
                torch.stack(x2_list), torch.stack(w2_list), torch.stack(m2_list),
                torch.FloatTensor(targets))

    # Fixed validation set
    print(f"  Generating {val_size} validation mixture-pairs...")
    vx1, vw1, vm1, vx2, vw2, vm2, vtgt = make_pair_batch(val_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_state = None
    best_val_r = -1
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()

        # Generate FRESH synthetic mixture pairs each epoch
        tx1, tw1, tm1, tx2, tw2, tm2, ttgt = make_pair_batch(pairs_per_epoch)

        train_loss = 0
        n_b = 0
        perm = torch.randperm(pairs_per_epoch)
        for bs in range(0, pairs_per_epoch, batch_size):
            idx = perm[bs:bs+batch_size]
            optimizer.zero_grad()
            pred = model(tx1[idx], tw1[idx], tm1[idx], tx2[idx], tw2[idx], tm2[idx])
            loss = F.mse_loss(pred, ttgt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = []
            vbs = 256
            for b in range(0, val_size, vbs):
                vp = model(vx1[b:b+vbs], vw1[b:b+vbs], vm1[b:b+vbs],
                          vx2[b:b+vbs], vw2[b:b+vbs], vm2[b:b+vbs])
                val_preds.append(vp)
            val_preds = torch.cat(val_preds)
            val_r = np.corrcoef(val_preds.numpy(), vtgt.numpy())[0, 1]

        print(f"  Epoch {epoch+1:3d}: loss={train_loss/n_b:.4f} val_r={val_r:.4f} (30K fresh MIX pairs)")

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    print(f"  Best val_r: {best_val_r:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model

# ============================================================
# Step 4: DREAM Fine-tuning (5-Fold CV)
# ============================================================
def load_dream_data():
    mixtures = {}
    with open(f"{DREAM_DATA}/Mixure_Definitions_Training_set.csv", 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            ds, lb = row[0].strip(), row[1].strip()
            cids = [c.strip() for c in row[2:] if c.strip() and c.strip() != '0' and c.strip().isdigit()]
            if cids:
                mixtures[f"{ds}_{lb}"] = {'cids': cids}

    pairs = []
    with open(f"{DREAM_DATA}/TrainingData_mixturedist.csv", 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4 and row[0]:
                pairs.append({
                    'mix1_key': f"{row[0].strip()}_{row[1].strip()}",
                    'mix2_key': f"{row[0].strip()}_{row[2].strip()}",
                    'human_dist': float(row[3]),
                })

    with open(DREAM_CACHE, 'r') as f:
        smiles_cache = json.load(f)

    with open(DREAM_ODT, 'r') as f:
        cache = json.load(f)

    return mixtures, pairs, smiles_cache, cache.get('pred_138d', {}), cache.get('odt', {})

def prepare_mixture(cids, smiles_cache, pred_cache, odt_data, max_n=60):
    preds, weights = [], []
    for cid in cids:
        smi = smiles_cache.get(cid, '')
        if not smi or cid not in pred_cache:
            continue
        pred = np.array(pred_cache[cid][:138])
        if len(pred) < 138:
            pred = np.pad(pred, (0, 138 - len(pred)))
        odt_log = odt_data.get(cid, -3.0)
        w = 1.0 / (10 ** odt_log + 1e-6)
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

def dream_finetune(model, mixtures, pairs, smiles_cache, pred_cache, odt_data,
                   epochs=150, lr=0.002, pretrained=False):
    """5-Fold CV with optional pre-trained encoder"""
    max_n = 60
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
    pretrained_encoder_state = model.encoder.state_dict()

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        model_f = SiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
        # Load pre-trained encoder
        model_f.encoder.load_state_dict(
            {k: v.clone() for k, v in pretrained_encoder_state.items()}
        )

        if pretrained:
            # Differential LR: encoder 10x lower to preserve pre-trained features
            optimizer = torch.optim.Adam([
                {'params': model_f.encoder.parameters(), 'lr': lr * 0.1},
                {'params': model_f.dist_head.parameters(), 'lr': lr},
            ], weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for epoch in range(epochs):
            model_f.train()
            np.random.shuffle(train_idx)
            bs = min(32, len(train_idx))
            for bstart in range(0, len(train_idx), bs):
                bidx = train_idx[bstart:bstart + bs]
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
        print(f"  Fold {fold + 1}: r = {val_r:.4f}")
        if val_r > best_r:
            best_r = val_r

    mean_r = np.mean(fold_results)
    print(f"  Mean CV: r = {mean_r:.4f}")
    return mean_r, fold_results

# ============================================================
# Main Pipeline
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Phase 10: 5K Contrastive Pre-training Pipeline")
    print("=" * 60)

    # Step 1+2: Build master 138d DB
    print("\n[1-2] Building master 138d DB from 4983 curated SMILES...")
    master_db = build_master_138d()
    print(f"  Master DB size: {len(master_db)} molecules")

    # Step 3: Dynamic Contrastive Pre-training
    model = SiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
    model = pretrain_dynamic(model, master_db, epochs=20)

    # Save
    os.makedirs(os.path.dirname(PRETRAINED_V3), exist_ok=True)
    torch.save(model.encoder.state_dict(), PRETRAINED_V3)
    print(f"  Saved: {PRETRAINED_V3}")

    # Step 4: DREAM Fine-tuning
    print("\n[4] Loading DREAM data...")
    mixtures, pairs, smiles_cache, pred_cache, odt_data = load_dream_data()

    # Mode A: Scratch (baseline)
    print("\n--- Mode: SCRATCH (no pre-training) ---")
    model_scratch = SiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
    r_scratch, folds_scratch = dream_finetune(
        model_scratch, mixtures, pairs, smiles_cache, pred_cache, odt_data,
        pretrained=False)

    # Mode B: Pre-trained
    print("\n--- Mode: PRE-TRAINED (5K contrastive) ---")
    model_pt = SiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
    model_pt.encoder.load_state_dict(torch.load(PRETRAINED_V3, map_location='cpu'))
    r_pt, folds_pt = dream_finetune(
        model_pt, mixtures, pairs, smiles_cache, pred_cache, odt_data,
        pretrained=True)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Phase 10 RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  {'Mode':<25} {'Mean r':>8}  {'Folds':>30}")
    print(f"  {'-' * 65}")
    fs = ', '.join(f'{f:.3f}' for f in folds_scratch)
    fp = ', '.join(f'{f:.3f}' for f in folds_pt)
    print(f"  {'Scratch (464 mol)':<25} {r_scratch:>8.4f}  [{fs}]")
    print(f"  {'Pre-trained (5K mol)':<25} {r_pt:>8.4f}  [{fp}]")
    delta = r_pt - r_scratch
    print(f"\n  Delta: {delta:+.4f} ({'UP!' if delta > 0 else 'DOWN'})")
    print(f"  DREAM Winners: r ~ 0.49")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
