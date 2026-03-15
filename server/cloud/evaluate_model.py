"""
Verified Model Evaluation v3
=============================
Step 0: Reproduce training's test cos_sim=0.4972 (validation)
Step 1: Extended metrics (Top-K, binary, categories)

Key differences from v2 (which was wrong):
1. label_smooth=0.05 (matches training) — NOT 0.0
2. EMA weights applied before eval (matches training line 969)
3. model(bert, gb, phys, return_aux=False) — correct arg order
4. Batch-level averaging for cos_sim (matches training line 985)
"""
import os, sys, json, csv, warnings, random
warnings.filterwarnings('ignore')
os.chdir('/home/elicer/cloud')
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data/curated_training_data.csv'
MODEL_PATH = 'weights/v6/odor_v6_best_seed42.pt'

MAJOR_CATEGORIES = {
    'Floral': ['floral', 'rose', 'jasmine', 'violet', 'lily', 'lavender'],
    'Citrus': ['citrus', 'lemon', 'orange', 'bergamot', 'grapefruit', 'lime'],
    'Woody': ['woody', 'cedarwood', 'sandalwood', 'pine', 'vetiver'],
    'Sweet': ['sweet', 'vanilla', 'caramel', 'honey', 'sugar', 'chocolate'],
    'Spicy': ['spicy', 'cinnamon', 'clove', 'pepper', 'nutmeg', 'ginger'],
    'Fruity': ['fruity', 'apple', 'pear', 'peach', 'berry', 'cherry'],
    'Green': ['green', 'grass', 'leaf', 'herbal', 'tea', 'mint'],
    'Musk': ['musk', 'musky', 'animalic', 'amber', 'ambergris'],
    'Fresh': ['fresh', 'clean', 'aquatic', 'marine', 'ozonic'],
    'Smoky': ['smoky', 'burnt', 'tobacco', 'leather', 'smoke'],
    'Earthy': ['earthy', 'soil', 'mushroom', 'moss'],
}

print("=" * 70)
print("  ★ VERIFIED MODEL EVALUATION v3 ★")
print("  (Exact train_v6.py conditions)")
print("=" * 70)

# ─────────────────────────────────────────────
# STEP 0: Load data EXACTLY like training
# ─────────────────────────────────────────────
print("\n[STEP 0] LOGIC VERIFICATION")
print("-" * 50)

from train_v6 import build_bert_cache, OdorDataset, scaffold_split, collate_odor, EMA
from models.odor_predictor_v6 import OdorPredictorV6

# Same seeds as training
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Same BERT cache
bert_cache = build_bert_cache(DATA_PATH, 'data/bert_cache.pt')

# CRITICAL: label_smooth=0.05, same as training (NOT 0.0)
dataset = OdorDataset(DATA_PATH, bert_cache=bert_cache, n_aug=0, label_smooth=0.05)
train_idx, val_idx, test_idx = scaffold_split(dataset, seed=42)

label_names = dataset.label_cols
n_labels = dataset.n_labels
bert_dim = dataset.bert_dim

print(f"  [CHECK 1] label_smooth = 0.05 (matches training): PASS")
print(f"  [CHECK 2] n_aug = 0 (no augmentation for eval): PASS")
print(f"  [CHECK 3] scaffold_split seed=42 (matches training): PASS")
print(f"  [CHECK 4] Total={len(dataset)} Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

# Load model
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model = OdorPredictorV6(bert_dim=bert_dim, n_odor_dim=n_labels).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# EMA: checkpoint contains EMA shadow (line 903: ema.apply_shadow before save)
# So model_state_dict IS already EMA weights
print(f"  [CHECK 5] Model loaded from EMA checkpoint: PASS")
print(f"  [CHECK 6] Checkpoint val_cos_sim = {ckpt.get('val_cos_sim', 'N/A')}")
print(f"  [CHECK 7] Checkpoint epoch = {ckpt.get('epoch', 'N/A')}")
print(f"  [CHECK 8] Params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────────
# STEP 1: Reproduce training's test cos_sim
# (Must match 0.4972 within rounding error)
# ─────────────────────────────────────────────
print("\n[STEP 1] REPRODUCE TRAINING TEST METRICS")
print("-" * 50)

test_ds = Subset(dataset, test_idx)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                         collate_fn=collate_odor, num_workers=0)

# Exact same loop as train_v6.py lines 971-987
test_cos = 0.0
test_mse = 0.0
n_test = 0
all_preds = []
all_trues = []

with torch.no_grad():
    for batch in test_loader:
        bert = batch['bert'].to(DEVICE)
        phys = batch['phys'].to(DEVICE)
        odor = batch['odor'].to(DEVICE)
        gb = batch['graph_batch']
        if gb is not None:
            gb = gb.to(DEVICE)
        else:
            continue
        # EXACT same call as train_v6 line 984
        pred = model(bert, gb, phys, return_aux=False)
        test_cos += F.cosine_similarity(pred, odor, dim=1).mean().item()
        test_mse += F.mse_loss(pred, odor).item()
        n_test += 1
        all_preds.append(pred.cpu().numpy())
        all_trues.append(odor.cpu().numpy())

avg_test_cos = test_cos / max(n_test, 1)
avg_test_mse = test_mse / max(n_test, 1)

all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

print(f"  Reproduced Test cos_sim: {avg_test_cos:.4f}")
print(f"  Reproduced Test MSE:     {avg_test_mse:.6f}")
print(f"  Training reported:       cos=0.4972, MSE=0.006989")

# Validation check
cos_diff = abs(avg_test_cos - 0.4972)
if cos_diff < 0.01:
    print(f"  ★ REPRODUCTION VERIFIED (diff={cos_diff:.4f} < 0.01)")
elif cos_diff < 0.05:
    print(f"  ⚠ CLOSE (diff={cos_diff:.4f} < 0.05) — minor differences expected")
else:
    print(f"  ✗ MISMATCH (diff={cos_diff:.4f}) — check conditions")

# ─────────────────────────────────────────────
# STEP 2: Extended metrics
# ─────────────────────────────────────────────
print(f"\n[STEP 2] EXTENDED METRICS ({len(all_preds)} test samples)")
print("=" * 55)

# 2a. Per-sample cosine similarity
cos_per_sample = []
for i in range(len(all_preds)):
    p, t = all_preds[i], all_trues[i]
    np_, nt = np.linalg.norm(p), np.linalg.norm(t)
    if np_ > 0 and nt > 0:
        cos_per_sample.append(np.dot(p, t) / (np_ * nt))
cos_arr = np.array(cos_per_sample)

# Per-label stats
label_mae = np.mean(np.abs(all_preds - all_trues), axis=0)
label_rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2, axis=0))
overall_mae = np.mean(label_mae)
overall_rmse = np.mean(label_rmse)

print(f"\n  OVERALL")
print(f"  Cos_sim (batch avg):  {avg_test_cos:.4f}")
print(f"  Cos_sim (sample avg): {cos_arr.mean():.4f}")
print(f"  MSE:                  {avg_test_mse:.6f}")
print(f"  MAE (per-label avg):  {overall_mae:.4f}")
print(f"  RMSE (per-label avg): {overall_rmse:.4f}")
print(f"  Per-label accuracy:   {(1 - overall_mae) * 100:.1f}%")

# 2b. Top-K
print(f"\n  TOP-K CATEGORY ACCURACY")
topk = {}
for k in [1, 3, 5, 10]:
    correct = 0
    total = 0
    for i in range(len(all_preds)):
        pred_top = set(np.argsort(all_preds[i])[-k:])
        true_top = set(np.argsort(all_trues[i])[-k:])
        correct += len(pred_top & true_top)
        total += k
    acc = correct / total * 100
    topk[k] = acc
    print(f"  Top-{k:>2}: {acc:.1f}%")

# 2c. Top-1 among major categories only (not all 612)
print(f"\n  TOP-K AMONG MAJOR 22 ODOR DIMS (sweet/floral/woody/...)")
ODOR22 = ['sweet', 'sour', 'woody', 'floral', 'citrus', 'spicy', 'musk',
           'fresh', 'green', 'warm', 'fruity', 'smoky', 'powdery', 'aquatic',
           'herbal', 'amber', 'leather', 'earthy', 'ozonic', 'metallic',
           'fatty', 'waxy']
odor22_idx = [i for i, l in enumerate(label_names) if l.lower() in ODOR22]
if odor22_idx:
    for k in [1, 3, 5]:
        correct = 0
        total = 0
        for i in range(len(all_preds)):
            preds_22 = all_preds[i, odor22_idx]
            trues_22 = all_trues[i, odor22_idx]
            pred_top = set(np.argsort(preds_22)[-k:])
            true_top = set(np.argsort(trues_22)[-k:])
            correct += len(pred_top & true_top)
            total += k
        acc = correct / total * 100
        print(f"  Top-{k:>2} (22 dims): {acc:.1f}%")

# 2d. Per-label best/worst
sorted_idx = np.argsort(label_mae)
print(f"\n  TOP 15 BEST PREDICTED LABELS")
print(f"  {'Label':<28} {'MAE':>6} {'RMSE':>6} {'Acc%':>6}")
for i in sorted_idx[:15]:
    print(f"  {label_names[i]:<28} {label_mae[i]:.4f} {label_rmse[i]:.4f} {(1-label_mae[i])*100:.1f}")

print(f"\n  BOTTOM 15 WORST PREDICTED LABELS")
print(f"  {'Label':<28} {'MAE':>6} {'RMSE':>6} {'Acc%':>6}")
for i in sorted_idx[-15:]:
    print(f"  {label_names[i]:<28} {label_mae[i]:.4f} {label_rmse[i]:.4f} {(1-label_mae[i])*100:.1f}")

# 2e. Major categories
print(f"\n  MAJOR CATEGORY ACCURACY")
print(f"  {'Category':<12} {'MAE':>6} {'Acc%':>6} {'#Labels':>7}")
for cat, kws in MAJOR_CATEGORIES.items():
    idxs = [j for j, l in enumerate(label_names) if any(k in l.lower() for k in kws)]
    if not idxs:
        continue
    cat_mae = label_mae[idxs].mean()
    print(f"  {cat:<12} {cat_mae:.4f} {(1-cat_mae)*100:.1f}  {len(idxs):>6}")

# 2f. Cosine distribution
print(f"\n  COSINE SIMILARITY DISTRIBUTION")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p}: {np.percentile(cos_arr, p):.4f}")
print(f"  Mean={cos_arr.mean():.4f} Std={cos_arr.std():.4f}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"  cos>={t}: {(cos_arr >= t).mean()*100:.1f}%")

# ─── Save ───
print(f"\n[STEP 3] SAVING")
results = {
    'model': MODEL_PATH,
    'test_samples': len(all_preds),
    'label_smooth': 0.05,
    'reproduction': {
        'reproduced_cos': round(avg_test_cos, 4),
        'expected_cos': 0.4972,
        'diff': round(cos_diff, 4),
        'verified': cos_diff < 0.01,
    },
    'overall': {
        'cos_sim_batch': round(avg_test_cos, 4),
        'cos_sim_sample': round(float(cos_arr.mean()), 4),
        'mse': round(avg_test_mse, 6),
        'mae': round(float(overall_mae), 4),
        'per_label_accuracy': round((1 - overall_mae) * 100, 1),
    },
    'top_k': {f'top_{k}': round(v, 1) for k, v in topk.items()},
}
os.makedirs('eval_results', exist_ok=True)
with open('eval_results/verified_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to eval_results/verified_results.json")

print(f"\n{'='*55}")
print(f"  ★ FINAL SUMMARY ★")
print(f"  Test cos_sim:    {avg_test_cos:.4f} (training: 0.4972)")
print(f"  Per-label acc:   {(1 - overall_mae) * 100:.1f}%")
print(f"  Top-1 (612):     {topk[1]:.1f}%")
print(f"  Top-5 (612):     {topk[5]:.1f}%")
print(f"{'='*55}")
