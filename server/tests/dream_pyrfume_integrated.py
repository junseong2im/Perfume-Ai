"""
Phase 10.5: Pyrfume Integration + DREAM Combined Training
==========================================================
Snitz 2013: 360 mixture similarity pairs (CID-based, direct fit)
Bushdid 2014: Individual discrimination trials → aggregate per-stimulus accuracy → distance
DREAM 2017: 500 mixture distance pairs (existing)

Combined: ~860 pairs → 5-Fold CV (fold: ~690 train)
"""
import sys, os, csv, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.join(os.path.dirname(__file__), '..')
DREAM_DATA = r"C:\Users\user\Downloads\GG"
DREAM_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'cid_smiles_cache.json')
DREAM_ODT = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')
SNITZ_DIR = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'snitz_2013')
BUSHDID_DIR = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'bushdid_2014')
MASTER_138D = os.path.join(BASE, 'data', 'pom_upgrade', 'pom_master_138d.json')

# ============================================================
# Architecture (FROZEN: 1-Layer Siamese, same as Phase 9/10)
# ============================================================
class MixtureEncoder(nn.Module):
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(output_dim, output_dim))

    def forward(self, x, weights, mask):
        h = self.input_proj(x)
        h_attn, _ = self.attention(h, h, h, key_padding_mask=mask)
        h = self.norm(h + h_attn)
        w = weights.unsqueeze(-1).masked_fill(mask.unsqueeze(-1), 0.0)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1) / w_sum.squeeze(1)
        return self.output_proj(pooled)

class SiameseNet(nn.Module):
    def __init__(self, input_dim=138, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = MixtureEncoder(input_dim, hidden_dim, output_dim)
        self.dist_head = nn.Sequential(
            nn.Linear(output_dim * 3, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x1, w1, m1, x2, w2, m2):
        z1 = self.encoder(x1, w1, m1)
        z2 = self.encoder(x2, w2, m2)
        diff = torch.abs(z1 - z2)
        combined = torch.cat([diff, z1 * z2, z1 + z2], dim=-1)
        return self.dist_head(combined).squeeze(-1)

# ============================================================
# Data Loading
# ============================================================
def get_138d(cid, smiles_cache, pred_cache, master_db, engine=None):
    """Get 138d prediction for a CID, trying all sources"""
    # Try DREAM cache first
    if cid in pred_cache:
        return np.array(pred_cache[cid][:138])

    # Try via SMILES in master DB
    smi = smiles_cache.get(cid, '')
    if smi and smi in master_db:
        return np.array(master_db[smi][:138])

    # Try engine prediction
    if engine and smi:
        try:
            pred = engine.predict_138d(smi)
            if pred is not None and np.any(pred > 0):
                return pred[:138]
        except:
            pass

    return None

def load_snitz_pairs(smiles_cache, pred_cache, master_db, odt_data, engine=None):
    """Load Snitz 2013: 360 mixture similarity pairs"""
    print("\n  [Snitz 2013] Loading mixture similarity pairs...")

    # Load CID -> SMILES from snitz molecules.csv
    snitz_cid_smi = {}
    with open(os.path.join(SNITZ_DIR, 'molecules.csv'), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '').strip()
            smi = row.get('IsomericSMILES', '').strip()
            if cid and smi:
                snitz_cid_smi[cid] = smi
    print(f"    Snitz molecules: {len(snitz_cid_smi)}")

    # Merge into smiles_cache
    for cid, smi in snitz_cid_smi.items():
        if cid not in smiles_cache:
            smiles_cache[cid] = smi

    # Generate 138d predictions for missing Snitz CIDs
    missing = 0
    for cid, smi in snitz_cid_smi.items():
        if cid not in pred_cache:
            pred = get_138d(cid, smiles_cache, pred_cache, master_db, engine)
            if pred is not None:
                pred_cache[cid] = pred.tolist()
            else:
                missing += 1
    print(f"    Missing predictions: {missing}")

    # Load behavior (pairs)
    pairs = []
    with open(os.path.join(SNITZ_DIR, 'behavior.csv'), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            stim_a = row.get('StimulusA', '').strip()
            stim_b = row.get('StimulusB', '').strip()
            sim_str = row.get('Similarity', '').strip()
            if not stim_a or not stim_b or not sim_str:
                continue
            try:
                similarity = float(sim_str)
            except:
                continue
            cids_a = [c.strip() for c in stim_a.split(',') if c.strip()]
            cids_b = [c.strip() for c in stim_b.split(',') if c.strip()]
            if cids_a and cids_b:
                # Convert similarity to distance (0=same, 1=different)
                # Snitz similarity is typically 0-100 scale
                if similarity > 1:
                    distance = 1.0 - (similarity / 100.0)
                else:
                    distance = 1.0 - similarity
                distance = max(0.0, min(1.0, distance))
                pairs.append({
                    'cids_a': cids_a,
                    'cids_b': cids_b,
                    'distance': distance,
                    'source': 'snitz',
                })
    print(f"    Snitz pairs: {len(pairs)}")
    return pairs

def load_bushdid_pairs(smiles_cache, pred_cache, master_db, odt_data, engine=None):
    """Load Bushdid 2014: discrimination test -> aggregate accuracy -> distance"""
    print("\n  [Bushdid 2014] Loading discrimination trials...")

    # Load molecules
    bush_cid_smi = {}
    bush_stim_cids = {}
    with open(os.path.join(BUSHDID_DIR, 'molecules.csv'), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '').strip()
            smi = row.get('IsomericSMILES', '').strip()
            if cid and smi:
                bush_cid_smi[cid] = smi
                smiles_cache[cid] = smi

    # Generate 138d for bushdid molecules
    for cid, smi in bush_cid_smi.items():
        if cid not in pred_cache:
            pred = get_138d(cid, smiles_cache, pred_cache, master_db, engine)
            if pred is not None:
                pred_cache[cid] = pred.tolist()

    print(f"    Bushdid molecules: {len(bush_cid_smi)}")

    # Load behavior: aggregate per-stimulus pair accuracy
    # Bushdid uses triangle test: 3 stimuli, 1 different, find the odd one out
    # Stimulus = mixture ID, Subject = participant, Correct = True/False
    stim_results = defaultdict(list)
    with open(os.path.join(BUSHDID_DIR, 'behavior.csv'), 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            stim = row.get('Stimulus', '').strip()
            correct = row.get('Correct', '').strip()
            if stim and correct:
                stim_results[stim].append(correct.lower() == 'true')

    # Aggregate: for each stimulus pair, compute discrimination accuracy
    # Higher accuracy = more discriminable = larger distance
    # Bushdid stimuli are mixture IDs (not CID lists) - harder to convert
    # We need the mixture definitions to create pairs
    # For now, check if we can extract mixture CID compositions
    print(f"    Bushdid stimuli: {len(stim_results)}")

    # Bushdid pairs are harder: each "stimulus" is actually a comparison between
    # a reference and a target mixture. Without mixture definitions tied to CIDs,
    # we can't easily create distance pairs. Skip for now.
    print(f"    Bushdid: skipping (no direct mixture CID mapping available)")
    return []

def load_dream_pairs(smiles_cache, pred_cache, odt_data):
    """Load DREAM 2017 training pairs"""
    print("\n  [DREAM 2017] Loading training pairs...")
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
                k1 = f"{row[0].strip()}_{row[1].strip()}"
                k2 = f"{row[0].strip()}_{row[2].strip()}"
                if k1 in mixtures and k2 in mixtures:
                    pairs.append({
                        'cids_a': mixtures[k1]['cids'],
                        'cids_b': mixtures[k2]['cids'],
                        'distance': float(row[3]),
                        'source': 'dream',
                    })
    print(f"    DREAM pairs: {len(pairs)}")
    return pairs

# ============================================================
# Mixture Preparation + Training
# ============================================================
def prepare_mixture(cids, pred_cache, odt_data, max_n=60):
    preds, weights = [], []
    for cid in cids:
        if cid not in pred_cache:
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

def train_cv(all_pairs, pred_cache, odt_data, epochs=150, lr=0.002, tag=""):
    """5-Fold CV training"""
    max_n = 60
    valid = []
    for pair in all_pairs:
        t1 = prepare_mixture(pair['cids_a'], pred_cache, odt_data, max_n)
        t2 = prepare_mixture(pair['cids_b'], pred_cache, odt_data, max_n)
        if t1[0] is None or t2[0] is None:
            continue
        valid.append({
            'x1': torch.FloatTensor(t1[0]), 'w1': torch.FloatTensor(t1[1]), 'm1': torch.BoolTensor(t1[2]),
            'x2': torch.FloatTensor(t2[0]), 'w2': torch.FloatTensor(t2[1]), 'm2': torch.BoolTensor(t2[2]),
            'target': torch.FloatTensor([pair['distance']]),
            'source': pair['source'],
        })

    n = len(valid)
    print(f"  Valid pairs: {n} ({tag})")

    # Count sources
    src_counts = defaultdict(int)
    for v in valid:
        src_counts[v['source']] += 1
    for src, cnt in src_counts.items():
        print(f"    {src}: {cnt}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    dream_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        model = SiameseNet(input_dim=138, hidden_dim=128, output_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_idx)
            bs = 32
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
                pred = model(x1, w1, m1, x2, w2, m2)
                loss = F.mse_loss(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        vp_all, vt_all = [], []
        vp_dream, vt_dream = [], []
        with torch.no_grad():
            for i in val_idx:
                p = valid[i]
                pred = model(p['x1'].unsqueeze(0), p['w1'].unsqueeze(0), p['m1'].unsqueeze(0),
                             p['x2'].unsqueeze(0), p['w2'].unsqueeze(0), p['m2'].unsqueeze(0))
                vp_all.append(pred.item())
                vt_all.append(p['target'].item())
                if p['source'] == 'dream':
                    vp_dream.append(pred.item())
                    vt_dream.append(p['target'].item())

        val_r = np.corrcoef(vp_all, vt_all)[0, 1] if len(vp_all) > 2 else 0
        fold_results.append(val_r)

        # DREAM-only correlation (apples-to-apples comparison)
        dream_r = np.corrcoef(vp_dream, vt_dream)[0, 1] if len(vp_dream) > 2 else 0
        dream_fold_results.append(dream_r)

        print(f"  Fold {fold+1}: r_all={val_r:.4f} r_dream={dream_r:.4f} (dream_val={len(vp_dream)})")

    mean_all = np.mean(fold_results)
    mean_dream = np.mean(dream_fold_results)
    print(f"  Mean CV: r_all={mean_all:.4f} r_dream={mean_dream:.4f}")
    return mean_all, mean_dream, fold_results, dream_fold_results

# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("  Phase 10.5: Pyrfume Integration")
    print("=" * 60)

    # Load caches
    with open(DREAM_CACHE, 'r') as f:
        smiles_cache = json.load(f)
    with open(DREAM_ODT, 'r') as f:
        cache = json.load(f)
    pred_cache = cache.get('pred_138d', {})
    odt_data = cache.get('odt', {})

    # Load master 138d DB
    master_db = {}
    if os.path.exists(MASTER_138D):
        with open(MASTER_138D, 'r') as f:
            master_db = json.load(f)
        print(f"  Master 138d DB: {len(master_db)} molecules")

    # Load engine for missing predictions
    print("\n  Loading POM Engine for missing predictions...")
    from pom_engine import POMEngine
    engine = POMEngine()
    engine.load()

    # Load all pairs
    dream_pairs = load_dream_pairs(smiles_cache, pred_cache, odt_data)
    snitz_pairs = load_snitz_pairs(smiles_cache, pred_cache, master_db, odt_data, engine)
    bushdid_pairs = load_bushdid_pairs(smiles_cache, pred_cache, master_db, odt_data, engine)

    total = len(dream_pairs) + len(snitz_pairs) + len(bushdid_pairs)
    print(f"\n  Total combined pairs: {total}")

    # Run 3 benchmarks
    print(f"\n{'='*60}")
    print("  Benchmark 1: DREAM Only (baseline)")
    print(f"{'='*60}")
    r1_all, r1_dream, f1_all, f1_dream = train_cv(dream_pairs, pred_cache, odt_data, tag="DREAM only")

    print(f"\n{'='*60}")
    print("  Benchmark 2: DREAM + Snitz (integrated)")
    print(f"{'='*60}")
    combined = dream_pairs + snitz_pairs
    random.shuffle(combined)
    r2_all, r2_dream, f2_all, f2_dream = train_cv(combined, pred_cache, odt_data, tag="DREAM+Snitz")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'r_all':>8} {'r_dream':>10}")
    print(f"  {'-'*45}")
    print(f"  {'DREAM only (500)':<25} {r1_all:>8.4f} {r1_dream:>10.4f}")
    print(f"  {'DREAM+Snitz (860)':<25} {r2_all:>8.4f} {r2_dream:>10.4f}")
    delta = r2_dream - r1_dream
    print(f"\n  DREAM-only delta: {delta:+.4f}")
    print(f"  DREAM Winners: r ~ 0.49")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
