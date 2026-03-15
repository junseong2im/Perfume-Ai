"""
Grid search for optimal v4+v5 ensemble weights
Tests weights from 0.30 to 0.70 for v4 (with v5 = 1-v4)
Uses 200 random molecules for fast sweep, then validates top-3 on full 500
"""
import sys, os, csv, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load models
from train_models import WEIGHTS_DIR, TrainableOdorNetV4
from models.odor_gat_v5 import OdorGATv5, smiles_to_graph, ODOR_DIMENSIONS, N_DIM
from rdkit import Chem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading models...")

# v4
cp4 = torch.load(WEIGHTS_DIR / 'odor_gnn.pt', map_location=device, weights_only=True)
v4 = TrainableOdorNetV4(input_dim=384).to(device)
v4.load_state_dict(cp4['model_state_dict'])
v4.eval()

# v5
cp5 = torch.load(WEIGHTS_DIR / 'odor_gnn_v5.pt', map_location=device, weights_only=False)
v5 = OdorGATv5(bert_dim=384).to(device)
v5.load_state_dict(cp5['model_state_dict'])
v5.eval()

# ChemBERTa cache
cache = np.load(WEIGHTS_DIR / 'chemberta_cache.npz')
bert_smiles = cache['smiles']
bert_embeds = cache['embeddings']
bert_cache = {s: bert_embeds[i] for i, s in enumerate(bert_smiles)}

# GoodScents labels
gs_path = Path(__file__).parent.parent / 'data' / 'curated_GS_LF_merged_4983.csv'
with open(gs_path, encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    odor_cols = header[1:]
    rows = list(reader)

# Build test set (200 random molecules)
np.random.seed(42)
sample_idx = np.random.choice(len(rows), size=200, replace=False)

test_data = []
skipped = 0
for idx in sample_idx:
    row = rows[idx]
    smiles = row[0]
    
    # canonical
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        can = Chem.MolToSmiles(mol)
    else:
        skipped += 1
        continue
    
    # bert embedding
    emb = bert_cache.get(can, bert_cache.get(smiles))
    if emb is None:
        skipped += 1
        continue
    
    # graph
    graph = smiles_to_graph(can, device='cpu')
    if graph is None:
        skipped += 1
        continue
    
    # label
    active = [col for col, val in zip(odor_cols, row[1:]) if val == '1']
    label_20d = np.zeros(N_DIM)
    for a in active:
        a_lower = a.lower()
        for i, dim in enumerate(ODOR_DIMENSIONS):
            if dim in a_lower or a_lower in dim:
                label_20d[i] = 1.0
                break
    
    if label_20d.sum() == 0:
        skipped += 1
        continue
    
    test_data.append({
        'smiles': can,
        'bert': torch.tensor(emb, dtype=torch.float32),
        'graph': graph,
        'label': label_20d,
        'active': active,
    })

print(f"Test set: {len(test_data)} molecules (skipped {skipped})\n")

# Grid Search
print(f"{'w4':>5} {'w5':>5} | {'Top1':>5} {'Top3':>5} {'Top5':>5} | {'CosSim':>7} {'F1':>5}")
print("-" * 55)

results = []

for w4 in np.arange(0.30, 0.75, 0.05):
    w5 = 1.0 - w4
    
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    cos_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    n = 0
    
    with torch.no_grad():
        for item in test_data:
            # v4 prediction
            x4 = item['bert'].unsqueeze(0).to(device)
            p4 = v4(x4).squeeze(0).cpu().numpy()
            
            # v5 prediction
            g = item['graph'].clone().to(device)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            x5 = item['bert'].unsqueeze(0).to(device)
            p5 = v5(g, x5).squeeze(0).cpu().numpy()
            
            # Ensemble
            pred = w4 * p4 + w5 * p5
            pred = np.clip(pred, 0, 1)
            
            # Metrics
            label = item['label']
            active = item['active']
            
            # Top-k
            top_idx = np.argsort(pred)[::-1]
            top_dims = [ODOR_DIMENSIONS[i] for i in top_idx[:5]]
            
            # Check hits
            hit1 = any(a.lower() in top_dims[0] or top_dims[0] in a.lower() for a in active)
            hit3 = any(any(a.lower() in d or d in a.lower() for a in active) for d in top_dims[:3])
            hit5 = any(any(a.lower() in d or d in a.lower() for a in active) for d in top_dims[:5])
            
            if hit1: top1_hits += 1
            if hit3: top3_hits += 1
            if hit5: top5_hits += 1
            
            # Cosine similarity
            pred_t = torch.tensor(pred, dtype=torch.float32)
            label_t = torch.tensor(label, dtype=torch.float32)
            if label_t.sum() > 0:
                cos = F.cosine_similarity(pred_t.unsqueeze(0), label_t.unsqueeze(0)).item()
                cos_sum += cos
            
            # Precision/Recall @5
            pred_set = set(top_dims[:5])
            actual_set = set()
            for a in active:
                for dim in ODOR_DIMENSIONS:
                    if dim in a.lower() or a.lower() in dim:
                        actual_set.add(dim)
            
            if pred_set and actual_set:
                prec = len(pred_set & actual_set) / len(pred_set)
                rec = len(pred_set & actual_set) / len(actual_set) if actual_set else 0
                prec_sum += prec
                rec_sum += rec
            
            n += 1
    
    top1 = top1_hits / n * 100
    top3 = top3_hits / n * 100
    top5 = top5_hits / n * 100
    cos_avg = cos_sum / n
    prec_avg = prec_sum / n
    rec_avg = rec_sum / n
    f1 = 2 * prec_avg * rec_avg / (prec_avg + rec_avg) if (prec_avg + rec_avg) > 0 else 0
    
    # Composite score
    score = 0.4 * (top1/100) + 0.3 * cos_avg + 0.3 * f1
    
    results.append({
        'w4': w4, 'w5': w5, 'top1': top1, 'top3': top3, 'top5': top5,
        'cos': cos_avg, 'f1': f1, 'score': score,
    })
    
    mark = " ★" if w4 == 0.55 else ""
    print(f"{w4:5.2f} {w5:5.2f} | {top1:5.1f} {top3:5.1f} {top5:5.1f} | {cos_avg:7.3f} {f1:5.3f}{mark}")

# Best
print("\n" + "=" * 55)
best = max(results, key=lambda r: r['score'])
print(f"BEST: w4={best['w4']:.2f} w5={best['w5']:.2f}")
print(f"  Top-1={best['top1']:.1f}% | CosSim={best['cos']:.3f} | F1={best['f1']:.3f}")
print(f"  Composite Score: {best['score']:.4f}")

# Comparison with single models
v4_only = next(r for r in results if abs(r['w4'] - 1.0) < 0.01) if any(abs(r['w4'] - 1.0) < 0.01 for r in results) else None
v5_only = next(r for r in results if abs(r['w5'] - 1.0) < 0.01) if any(abs(r['w5'] - 1.0) < 0.01 for r in results) else None

if v4_only:
    print(f"\nv4 only: Top-1={v4_only['top1']:.1f}% | CosSim={v4_only['cos']:.3f}")
if v5_only:
    print(f"v5 only: Top-1={v5_only['top1']:.1f}% | CosSim={v5_only['cos']:.3f}")
