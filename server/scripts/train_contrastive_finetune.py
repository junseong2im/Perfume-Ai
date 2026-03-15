# -*- coding: utf-8 -*-
"""
Full A/B test: ASL-only (baseline) vs ASL+Contrastive (improved)
================================================================
Both trained from scratch with same data, same hyperparams.
Only difference: +SupCon auxiliary loss (lambda=0.3)
"""
import sys, os, time, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
logging.disable(logging.CRITICAL)

import database as db
from train_models import (
    OdorDataset, TrainableOdorNetV4, TrainableOdorNet,
    scaffold_split, _load_bert_cache,
    DimensionWeightedASL, mixup_batch, ODOR_DIMENSIONS, N_DIM
)

WEIGHTS_DIR = 'weights'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("  A/B Test: ASL vs ASL+Contrastive")
print("  Device:", device)
print("=" * 60)

# --- Data ---
bert_cache = _load_bert_cache()
molecules = db.get_all_molecules()
dataset = OdorDataset(molecules, bert_cache=bert_cache, n_augment=10)
input_dim = dataset.input_dim
train_set, val_set = scaffold_split(dataset, val_ratio=0.2)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=64)
print(f"  Data: {len(dataset)} (train={len(train_set)}, val={len(val_set)}), dim={input_dim}")

# --- Helpers ---
class ConHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(N_DIM, 64), nn.GELU(), nn.Linear(64, 32))
    def forward(self, x):
        return F.normalize(self.p(x), dim=-1)

class SupCon(nn.Module):
    def __init__(self, t=0.1):
        super().__init__()
        self.t = t
    def forward(self, z, lb):
        B = z.shape[0]
        if B < 4: return torch.tensor(0.0, device=z.device)
        sim = z @ z.T / self.t
        mask = lb.unsqueeze(0) == lb.unsqueeze(1)
        mask.fill_diagonal_(False)
        v = mask.sum(1) > 0
        if v.sum() < 2: return torch.tensor(0.0, device=z.device)
        sim, mask = sim[v], mask[v]
        e = torch.exp(sim - sim.max(1, keepdim=True)[0])
        I = torch.eye(e.shape[0], e.shape[1], device=device)
        e = e * (1 - I[:e.shape[0], :e.shape[1]])
        return -(torch.log((e * mask.float()).sum(1) / (e.sum(1) + 1e-8) + 1e-8)).mean()

def measure(mdl, loader):
    mdl.eval()
    c = t = t1c = t1t = 0
    dc = np.zeros(N_DIM); dt = np.zeros(N_DIM) + 1e-8
    # cosine sim accumulator
    cos_sims = []
    with torch.no_grad():
        for f, tgt in loader:
            f, tgt = f.to(device), tgt.to(device)
            p = mdl(f)
            pa = (p > 0.3).float(); ta = (tgt > 0.3).float()
            c += (pa == ta).sum().item()
            t += ta.numel()
            for d in range(N_DIM):
                dc[d] += (pa[:,d] == ta[:,d]).sum().item()
                dt[d] += tgt.shape[0]
            t1c += (p.argmax(1) == tgt.argmax(1)).sum().item()
            t1t += tgt.shape[0]
            # cosine similarity
            cos = F.cosine_similarity(p, tgt, dim=1)
            cos_sims.extend(cos.cpu().tolist())
    return c/t*100, dc/dt*100, t1c/max(t1t,1)*100, np.mean(cos_sims)

def train_model(model, use_contrastive=False, epochs=200, label=""):
    ph = ConHead().to(device) if use_contrastive else None
    params = list(model.parameters())
    if ph: params += list(ph.parameters())
    opt = optim.AdamW([
        {'params': model.parameters(), 'lr': 0.002},
    ] + ([{'params': ph.parameters(), 'lr': 0.003}] if ph else []),
    weight_decay=5e-4)
    
    asl = DimensionWeightedASL(gamma_pos=1, gamma_neg=4, clip=0.05)
    asl.compute_weights(train_set)
    sc = SupCon(0.1) if use_contrastive else None
    lam = 0.3
    
    wu = max(10, epochs // 10)
    def lrf(e):
        if e < wu: return (e+1)/wu
        return 0.5*(1+math.cos(math.pi*(e-wu)/(epochs-wu)))
    sch = optim.lr_scheduler.LambdaLR(opt, lrf)
    
    best_vl = float('inf'); best_st = None; pat = 0
    t0 = time.time()
    
    for ep in range(epochs):
        model.train()
        if ph: ph.train()
        tl = 0
        for f, tgt in train_loader:
            f, tgt = f.to(device), tgt.to(device)
            if np.random.random() < 0.5:
                f, tgt = mixup_batch(f, tgt, alpha=0.4)
            p = model(f)
            la = asl(p, tgt*0.95+0.025)
            
            if use_contrastive and ph and sc:
                z = ph(p)
                lc = sc(z, tgt.argmax(1))
                loss = (1-lam)*la + lam*lc
            else:
                loss = la
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        
        avg = tl/len(train_loader)
        model.eval()
        vl = 0
        with torch.no_grad():
            for f2, t2 in val_loader:
                f2, t2 = f2.to(device), t2.to(device)
                vl += asl(model(f2), t2).item()
        vl /= max(1, len(val_loader))
        sch.step()
        
        if vl < best_vl:
            best_vl = vl; pat = 0
            best_st = {k:v.clone() for k,v in model.state_dict().items()}
        else: pat += 1
        
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  [{label}] Ep {ep+1:3d}/{epochs} | L:{avg:.4f} | V:{vl:.4f} | Best:{best_vl:.4f}")
        if pat >= 30:
            print(f"  [{label}] Early stop ep {ep+1}"); break
    
    el = time.time() - t0
    print(f"  [{label}] Done in {el:.1f}s, best_val={best_vl:.4f}")
    if best_st: model.load_state_dict(best_st)
    return model

# ================================================================
# A: Baseline (ASL only)
# ================================================================
print("\n--- Training A: ASL only (baseline) ---")
torch.manual_seed(42); np.random.seed(42)
if input_dim == 384:
    model_a = TrainableOdorNetV4(input_dim=384).to(device)
else:
    model_a = TrainableOdorNet().to(device)
model_a = train_model(model_a, use_contrastive=False, epochs=200, label="ASL")
ov_a, dim_a, t1_a, cos_a = measure(model_a, val_loader)

# ================================================================
# B: ASL + Contrastive
# ================================================================
print("\n--- Training B: ASL + Contrastive ---")
torch.manual_seed(42); np.random.seed(42)
if input_dim == 384:
    model_b = TrainableOdorNetV4(input_dim=384).to(device)
else:
    model_b = TrainableOdorNet().to(device)
model_b = train_model(model_b, use_contrastive=True, epochs=200, label="ASL+Con")
ov_b, dim_b, t1_b, cos_b = measure(model_b, val_loader)

# ================================================================
# Results
# ================================================================
print(f"\n{'='*60}")
print(f"  A/B RESULTS")
print(f"{'='*60}")
print(f"")
print(f"  Metric          | ASL only  | ASL+Con   | Delta")
print(f"  ----------------+-----------+-----------+--------")
print(f"  Overall acc     | {ov_a:>7.1f}%  | {ov_b:>7.1f}%  | {'+'if ov_b>=ov_a else ''}{ov_b-ov_a:.1f}%p")
print(f"  Top-1 acc       | {t1_a:>7.1f}%  | {t1_b:>7.1f}%  | {'+'if t1_b>=t1_a else ''}{t1_b-t1_a:.1f}%p")
print(f"  Cosine sim      | {cos_a:>8.4f} | {cos_b:>8.4f} | {'+'if cos_b>=cos_a else ''}{cos_b-cos_a:.4f}")
print()

imp = deg = 0
for i, nm in enumerate(ODOR_DIMENSIONS):
    a, b = dim_a[i], dim_b[i]
    d = b - a
    s = " **" if d > 1 else (" !!" if d < -1 else "")
    print(f"  {nm:>10s}: {a:>6.1f}% -> {b:>6.1f}%  ({'+'if d>=0 else ''}{d:.1f}%p){s}")
    if d > 0.5: imp += 1
    elif d < -0.5: deg += 1

print(f"\n  Improved: {imp}/{N_DIM}")
print(f"  Degraded: {deg}/{N_DIM}")

# Save better model
winner = "ASL+Con" if ov_b >= ov_a else "ASL"
winner_model = model_b if ov_b >= ov_a else model_a
winner_acc = max(ov_a, ov_b)

torch.save({
    'model_state_dict': winner_model.state_dict(),
    'accuracy': winner_acc,
    'input_dim': input_dim,
    'n_train': len(train_set),
    'n_val': len(val_set),
    'augmentation': 'mixup+contrastive' if winner == "ASL+Con" else 'mixup',
}, os.path.join(WEIGHTS_DIR, 'odor_gnn.pt'))
print(f"\n  Winner: {winner} ({winner_acc:.1f}%), saved to weights/odor_gnn.pt")
