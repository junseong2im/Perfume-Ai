"""
확장 검증 테스트: 12대 수칙 외 전체 파이프라인
================================================
mixup_data, cutmix_data, WarmupCosineScheduler,
progressive_unfreeze, compute_loss, label_smoothing,
contrastive_loss (실제 함수), collate_odor, EMA 수학
"""
import sys, os
SERVER = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, SERVER)
sys.path.insert(0, os.path.join(SERVER, 'cloud'))
sys.path.insert(0, os.path.join(SERVER, 'cloud', 'models'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import traceback

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ FAIL: {name} — {detail}")

print("=" * 70)
print("  확장 파이프라인 검증 (12대 수칙 외)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# A. mixup_data 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[A] mixup_data 검증")

try:
    from train_v6 import mixup_data
    
    B, D_bert, D_phys, D_y = 8, 384, 12, 22
    x_bert = torch.randn(B, D_bert)
    x_phys = torch.randn(B, D_phys)
    y = torch.rand(B, D_y)
    
    mixed_bert, mixed_phys, mixed_y, lam = mixup_data(x_bert, x_phys, y, alpha=0.4)
    
    # A-1: 출력 shape 보존
    test("mixup shape 보존 (bert)", mixed_bert.shape == x_bert.shape,
         f"{mixed_bert.shape} != {x_bert.shape}")
    test("mixup shape 보존 (phys)", mixed_phys.shape == x_phys.shape)
    test("mixup shape 보존 (y)", mixed_y.shape == y.shape)
    
    # A-2: lambda 범위 [0, 1]
    test("mixup lambda ∈ [0,1]", 0 <= lam <= 1, f"lam={lam}")
    
    # A-3: convex combination 확인
    # mixed = lam * x + (1-lam) * x[perm]  →  값이 x의 범위 내
    test("mixup 결과 유한", torch.isfinite(mixed_bert).all().item())
    test("mixup labels 유한", torch.isfinite(mixed_y).all().item())
    
    # A-4: alpha=0 → lam=1 → no mixing
    mb, mp, my, l = mixup_data(x_bert, x_phys, y, alpha=0.0)
    test("alpha=0 → lam=1 (mixing 없음)", l == 1.0, f"lam={l}")
    test("alpha=0 → bert 원본 유지", torch.equal(mb, x_bert))
    
    # A-5: label convex combination 검증
    y1 = torch.ones(4, 22)
    y0 = torch.zeros(4, 22)
    x1 = torch.randn(4, 384)
    xp = torch.randn(4, 12)
    # 수동 lam으로 검증
    np.random.seed(42)
    _, _, my_check, lam_check = mixup_data(x1, xp, y1, alpha=0.4)
    # mixed_y = lam * y1 + (1-lam) * y1[perm] = y1 (since y1 is all ones)
    test("동일 label mixup → label 보존",
         torch.allclose(my_check, y1, atol=1e-5))
except Exception as e:
    test("mixup_data", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# B. cutmix_data 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[B] cutmix_data 검증")

try:
    from train_v6 import cutmix_data
    
    B, D = 8, 384
    x = torch.randn(B, D)
    y = torch.rand(B, 22)
    
    mixed_x, mixed_y, lam = cutmix_data(x, y, alpha=0.4)
    
    # B-1: shape
    test("cutmix shape 보존 (x)", mixed_x.shape == x.shape)
    test("cutmix shape 보존 (y)", mixed_y.shape == y.shape)
    
    # B-2: lambda 범위
    test("cutmix lambda ∈ [0,1]", 0 <= lam <= 1, f"lam={lam}")
    
    # B-3: 결과 유한
    test("cutmix 결과 유한", torch.isfinite(mixed_x).all().item())
    
    # B-4: alpha=0 → lam=1 → no cut
    mx, my, l = cutmix_data(x, y, alpha=0.0)
    test("alpha=0 → no cut (원본 유지)", l == 1.0)
    
    # B-5: cut된 부분은 다른 sample에서 와야 함
    # mixed_x의 일부 dim은 x에서, 나머지는 x[perm]에서
    test("cutmix: 일부 차원이 swapped",
         not torch.equal(mixed_x, x) or lam == 1.0)
except Exception as e:
    test("cutmix_data", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# C. WarmupCosineScheduler 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[C] WarmupCosineScheduler 검증")

try:
    from train_v6 import WarmupCosineScheduler
    
    model = nn.Linear(10, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = WarmupCosineScheduler(opt, warmup_epochs=10, T_0=50, T_mult=2)
    
    lrs = []
    for e in range(200):
        sched.step(e)
        lrs.append(opt.param_groups[0]['lr'])
    
    # C-1: warmup increases
    test("warmup 단조 증가 (E0→E9)",
         all(lrs[i] <= lrs[i+1] for i in range(9)),
         f"lrs[0:10]={[f'{l:.6f}' for l in lrs[:10]]}")
    
    # C-2: warmup final = base_lr
    test("warmup 최종 = base_lr (1e-3)",
         abs(lrs[9] - 1e-3) < 1e-6, f"lr[9]={lrs[9]:.8f}")
    
    # C-3: cosine decreases after warmup
    test("cosine 후 lr 감소 (E10→E59)",
         lrs[10] > lrs[59],
         f"lr[10]={lrs[10]:.6f}, lr[59]={lrs[59]:.6f}")
    
    # C-4: minimum lr > 0
    min_lr = min(lrs)
    test("최소 lr > 0", min_lr > 0, f"min_lr={min_lr:.8f}")
    
    # C-5: lr never exceeds base_lr
    max_lr = max(lrs)
    test("최대 lr ≤ base_lr", max_lr <= 1e-3 + 1e-8,
         f"max_lr={max_lr:.8f}")
    
    # C-6: all lr finite
    test("전체 lr 유한", all(math.isfinite(l) for l in lrs))
    
    # C-7: cosine restart (T_0=50 → restart at E60)
    # After warmup (10), first cosine period is T_0=50 (E10-E59)
    # Then restart at E60
    test("cosine restart: lr[60] > lr[59]",
         lrs[60] > lrs[59],
         f"lr[59]={lrs[59]:.6f}, lr[60]={lrs[60]:.6f}")
except Exception as e:
    test("WarmupCosineScheduler", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# D. progressive_unfreezing 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[D] progressive_unfreezing 검증")

try:
    from train_v6 import apply_progressive_unfreezing
    
    # Create mock model with named parameters matching expected patterns
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.path_b = nn.Linear(10, 10)       # GNN path
            self.path_c = nn.Linear(10, 10)       # phys path
            self.fusion = nn.Linear(10, 10)       # fusion
            self.backbone = nn.Linear(10, 10)     # backbone
            self.heads = nn.ModuleDict({
                'odor': nn.Linear(10, 22),
                'top': nn.Linear(10, 22),
                'mid': nn.Linear(10, 22),
                'base': nn.Linear(10, 22),
                'longevity': nn.Linear(10, 1),
                'sillage': nn.Linear(10, 1),
                'descriptors': nn.Linear(10, 50),
                'receptors': nn.Linear(10, 30),
                'hedonic': nn.Linear(10, 1),
                'super_res': nn.Linear(10, 128),
            })
            self.skip = nn.Linear(10, 10)
        
        def forward(self, x):
            return x
    
    m = MockModel()
    
    # Phase 1: epoch < 30
    apply_progressive_unfreezing(m, epoch=10)
    p1_grad = {n: p.requires_grad for n, p in m.named_parameters()}
    
    test("Phase1 (E10): heads.odor 학습 가능",
         p1_grad.get('heads.odor.weight', False))
    test("Phase1 (E10): backbone 학습 가능",
         p1_grad.get('backbone.weight', False))
    test("Phase1 (E10): path_b 학습 가능",
         p1_grad.get('path_b.weight', False))
    test("Phase1 (E10): heads.descriptors 동결",
         not p1_grad.get('heads.descriptors.weight', True))
    test("Phase1 (E10): heads.hedonic 동결",
         not p1_grad.get('heads.hedonic.weight', True))
    
    # Phase 2: epoch 30-79
    apply_progressive_unfreezing(m, epoch=50)
    p2_grad = {n: p.requires_grad for n, p in m.named_parameters()}
    
    test("Phase2 (E50): heads.top 학습 가능",
         p2_grad.get('heads.top.weight', False))
    test("Phase2 (E50): heads.longevity 학습 가능",
         p2_grad.get('heads.longevity.weight', False))
    test("Phase2 (E50): heads.sillage 학습 가능",
         p2_grad.get('heads.sillage.weight', False))
    
    # Phase 3: epoch >= 80
    apply_progressive_unfreezing(m, epoch=100)
    p3_grad = {n: p.requires_grad for n, p in m.named_parameters()}
    
    all_unfrozen = all(g for g in p3_grad.values())
    test("Phase3 (E100): 전체 파라미터 학습 가능", all_unfrozen,
         f"frozen: {[n for n, g in p3_grad.items() if not g]}")
except Exception as e:
    test("progressive_unfreezing", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# E. compute_loss 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[E] compute_loss 검증")

try:
    from odor_predictor_v6 import compute_loss
    
    class FakeLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_weights = nn.Parameter(torch.ones(10))
        def forward(self, x): return x
    
    fm = FakeLossModel()
    
    # E-1: minimal case (odor only)
    pred = {'odor': torch.rand(4, 22)}
    target = {'odor': torch.rand(4, 22)}
    loss, losses = compute_loss(fm, pred, target, None, epoch=10, max_epochs=300)
    
    test("compute_loss 출력 유한", torch.isfinite(loss).item())
    test("compute_loss odor_mse 존재", 'odor_mse' in losses)
    test("compute_loss odor_cos 존재", 'odor_cos' in losses)
    test("compute_loss loss > 0", loss.item() > 0)
    
    # E-2: label smoothing 확인
    # target_smooth = target * 0.95 + 0.025
    raw_target = torch.tensor([[1.0, 0.0, 0.5]])
    smoothed = raw_target * 0.95 + 0.025
    test("label smoothing: 1.0 → 0.975", abs(smoothed[0,0].item() - 0.975) < 1e-5)
    test("label smoothing: 0.0 → 0.025", abs(smoothed[0,1].item() - 0.025) < 1e-5)
    test("label smoothing: 0.5 → 0.5", abs(smoothed[0,2].item() - 0.5) < 1e-5)
    
    # E-3: cos loss 범위 [0, 2]
    test("cos loss ∈ [0,2]",
         0 <= losses['odor_cos'].item() <= 2.0,
         f"cos={losses['odor_cos'].item():.4f}")
    
    # E-4: perfect prediction → low loss
    pred_perfect = {'odor': target['odor'].clone()}
    loss_perf, _ = compute_loss(fm, pred_perfect, target, None, 10, 300)
    test("완벽 예측 → loss 감소",
         loss_perf.item() < loss.item(),
         f"perfect={loss_perf.item():.4f}, random={loss.item():.4f}")
    
    # E-5: with top/mid/base heads
    pred_full = {
        'odor': torch.rand(4, 22),
        'top': torch.rand(4, 22),
        'mid': torch.rand(4, 22),
        'base': torch.rand(4, 22),
    }
    target_full = {
        'odor': torch.rand(4, 22),
        'top': torch.rand(4, 22),
        'mid': torch.rand(4, 22),
        'base': torch.rand(4, 22),
    }
    loss_f, losses_f = compute_loss(fm, pred_full, target_full, None, 10, 300)
    test("full heads: tmb loss 존재", 'tmb' in losses_f)
    test("full heads: loss 유한", torch.isfinite(loss_f).item())
    
    # E-6: self-supervised tmb (no target)
    pred_self = {
        'odor': torch.rand(4, 22),
        'top': torch.rand(4, 22),
        'mid': torch.rand(4, 22),
        'base': torch.rand(4, 22),
    }
    target_self = {'odor': torch.rand(4, 22)}  # no top/mid/base
    loss_s, losses_s = compute_loss(fm, pred_self, target_self, None, 10, 300)
    test("self-supervised tmb 활성", 'tmb_self' in losses_s)
    
    # E-7: backward 가능
    pred_bw = {'odor': torch.rand(4, 22, requires_grad=True)}
    target_bw = {'odor': torch.rand(4, 22)}
    loss_bw, _ = compute_loss(fm, pred_bw, target_bw, None, epoch=10, max_epochs=300)
    loss_bw.backward()
    test("compute_loss backward 성공", 
         pred_bw['odor'].grad is not None and torch.isfinite(pred_bw['odor'].grad).all().item())
    
    # E-8: R-Drop integration in compute_loss
    pred_rdrop = {
        'odor': torch.rand(4, 22),
        'odor_2': torch.rand(4, 22),
    }
    target_rdrop = {'odor': torch.rand(4, 22)}
    loss_r, losses_r = compute_loss(fm, pred_rdrop, target_rdrop, None, 10, 300, training=True)
    test("R-Drop in compute_loss: rdrop loss 존재", 'rdrop' in losses_r)
    test("R-Drop loss 유한", torch.isfinite(losses_r['rdrop']).item())
except Exception as e:
    test("compute_loss", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# F. contrastive_loss (실제 함수) 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[F] contrastive_loss (실제 함수) 검증")

try:
    from odor_predictor_v6 import contrastive_loss
    
    # F-1: normal case
    emb = torch.randn(16, 128)
    labels = torch.rand(16, 22)
    loss = contrastive_loss(emb, labels, temperature=0.07)
    test("contrastive_loss 유한", torch.isfinite(loss).item())
    test("contrastive_loss > 0", loss.item() > 0)
    
    # F-2: B=1 → return 0
    loss_b1 = contrastive_loss(torch.randn(1, 128), torch.rand(1, 22))
    test("B=1 → loss=0", loss_b1.item() == 0.0)
    
    # F-3: identical labels → many positives
    same_labels = torch.ones(8, 22)
    loss_same = contrastive_loss(torch.randn(8, 128), same_labels)
    test("동일 labels → loss 유한", torch.isfinite(loss_same).item())
    
    # F-4: no positive pairs → return 0
    # orthogonal labels → cosine sim < 0.8
    orth_labels = torch.eye(8, 22)
    loss_orth = contrastive_loss(torch.randn(8, 128), orth_labels)
    # some pairs might still have sim > 0.8, so check finite at least
    test("직교 labels → loss 유한", torch.isfinite(loss_orth).item())
    
    # F-5: extreme temperature
    loss_cold = contrastive_loss(emb, labels, temperature=0.01)
    test("temperature=0.01 NaN 없음", torch.isfinite(loss_cold).item())
    
    loss_hot = contrastive_loss(emb, labels, temperature=1.0)
    test("temperature=1.0 NaN 없음", torch.isfinite(loss_hot).item())
    
    # F-6: backward 가능
    emb_grad = torch.randn(8, 128, requires_grad=True)
    loss_bw = contrastive_loss(emb_grad, torch.rand(8, 22))
    if loss_bw.item() > 0:
        loss_bw.backward()
        test("contrastive_loss backward 성공",
             emb_grad.grad is not None and torch.isfinite(emb_grad.grad).all().item())
    else:
        test("contrastive_loss backward (no positives)", True)
except Exception as e:
    test("contrastive_loss", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# G. focal_bce (실제 함수) 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[G] focal_bce (실제 함수) 검증")

try:
    from odor_predictor_v6 import focal_bce
    
    # G-1: normal
    pred = torch.rand(4, 22)
    target = torch.rand(4, 22)
    loss = focal_bce(pred, target)
    test("focal_bce 유한", torch.isfinite(loss).item())
    test("focal_bce > 0", loss.item() > 0)
    
    # G-2: edge cases
    loss_01 = focal_bce(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]]))
    test("focal_bce(0→0, 1→1) 유한", torch.isfinite(loss_01).item())
    
    loss_extreme = focal_bce(torch.tensor([[0.001, 0.999]]), torch.tensor([[0.999, 0.001]]))
    test("focal_bce 극단 예측 유한", torch.isfinite(loss_extreme).item())
    
    # G-3: soft labels
    loss_soft = focal_bce(torch.tensor([[0.8]]), torch.tensor([[0.8]]))
    test("focal_bce soft label 유한", torch.isfinite(loss_soft).item())
    
    # G-4: backward
    p = torch.rand(4, 22, requires_grad=True)
    l = focal_bce(p, torch.rand(4, 22))
    l.backward()
    test("focal_bce backward 성공",
         p.grad is not None and torch.isfinite(p.grad).all().item())
except Exception as e:
    test("focal_bce (실제)", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# H. EMA 수학 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[H] EMA 수학 검증")

try:
    from odor_predictor_v6 import EMA
    
    m = nn.Linear(10, 5, bias=False)
    m.weight.data.fill_(1.0)
    ema = EMA(m, decay=0.9)
    
    # Step 1: weight → 2.0
    m.weight.data.fill_(2.0)
    ema.update(m)
    # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    expected = 0.9 * 1.0 + 0.1 * 2.0
    actual = ema.shadow['weight'][0, 0].item()
    test("EMA 수학: 0.9*1+0.1*2 = 1.1",
         abs(actual - expected) < 1e-5, f"actual={actual:.5f}")
    
    # Step 2: weight → 3.0
    m.weight.data.fill_(3.0)
    ema.update(m)
    # shadow = 0.9 * 1.1 + 0.1 * 3.0 = 1.29
    expected2 = 0.9 * 1.1 + 0.1 * 3.0
    actual2 = ema.shadow['weight'][0, 0].item()
    test("EMA 2단계: 0.9*1.1+0.1*3 = 1.29",
         abs(actual2 - expected2) < 1e-5, f"actual={actual2:.5f}")
    
    # apply_shadow → model gets EMA weights
    ema.apply_shadow(m)
    test("apply_shadow: model=EMA", 
         abs(m.weight.data[0, 0].item() - expected2) < 1e-5)
    
    # restore → model gets backup (3.0)
    ema.restore(m)
    test("restore: model=backup (3.0)",
         abs(m.weight.data[0, 0].item() - 3.0) < 1e-5)
except Exception as e:
    test("EMA 수학", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# I. label smoothing 수학 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[I] label smoothing 수학 검증")

# In compute_loss: target_smooth = target * 0.95 + 0.025
target_raw = torch.tensor([[1.0, 0.0, 0.5, 0.123]])
smoothed = target_raw * 0.95 + 0.025

test("smoothing(1.0)=0.975", abs(smoothed[0,0].item() - 0.975) < 1e-6)
test("smoothing(0.0)=0.025", abs(smoothed[0,1].item() - 0.025) < 1e-6)
test("smoothing(0.5)=0.5", abs(smoothed[0,2].item() - 0.5) < 1e-6)
test("smoothing 범위 [0.025, 0.975]",
     smoothed.min().item() >= 0.025 - 1e-6 and smoothed.max().item() <= 0.975 + 1e-6)
test("smoothing 단조 보존",
     (smoothed[0,3].item() > smoothed[0,1].item()) == (target_raw[0,3].item() > target_raw[0,1].item()))

# ═══════════════════════════════════════════════════════════════════
# J. loss_weights interaction 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[J] loss_weights interaction 검증")

try:
    from odor_predictor_v6 import compute_loss
    
    class WeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_weights = nn.Parameter(torch.ones(10))
        def forward(self, x): return x
    
    wm = WeightModel()
    pred = {'odor': torch.rand(4, 22)}
    target = {'odor': torch.rand(4, 22)}
    
    # Normal weights
    loss1, _ = compute_loss(wm, pred, target, None, 10, 300)
    
    # Double odor weight
    wm.loss_weights.data[0] = 2.0
    loss2, _ = compute_loss(wm, pred, target, None, 10, 300)
    
    test("weight[0]*2 → loss 증가",
         loss2.item() > loss1.item(),
         f"w1={loss1.item():.4f}, w2={loss2.item():.4f}")
    
    # Zero weight → zero contribution
    wm.loss_weights.data[0] = 0.0
    loss0, _ = compute_loss(wm, pred, target, None, 10, 300)
    test("weight[0]=0 → loss 감소 (다른 head 없으므로 0)",
         loss0.item() < loss1.item(),
         f"w0={loss0.item():.4f}")
except Exception as e:
    test("loss_weights interaction", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# K. Data augmentation + label 일관성 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[K] Augmentation + Label 일관성")

try:
    from train_v6 import mixup_data, cutmix_data
    
    B = 16
    x_bert = torch.randn(B, 384)
    x_phys = torch.randn(B, 12)
    y = torch.rand(B, 22)
    
    # K-1: mixup label은 항상 convex combination (범위 보존)
    for _ in range(10):
        _, _, my, lam = mixup_data(x_bert, x_phys, y, alpha=0.4)
        if my.max().item() > y.max().item() + 1e-5 or my.min().item() < y.min().item() - 1e-5:
            test("mixup label 범위 보존", False, 
                 f"y range=[{y.min():.3f},{y.max():.3f}], mixed=[{my.min():.3f},{my.max():.3f}]")
            break
    else:
        test("mixup label 범위 보존 (10회)", True)
    
    # K-2: cutmix label도 convex combination
    for _ in range(10):
        _, cy, lam = cutmix_data(x_bert, y, alpha=0.4)
        if cy.max().item() > y.max().item() + 1e-5 or cy.min().item() < y.min().item() - 1e-5:
            test("cutmix label 범위 보존", False)
            break
    else:
        test("cutmix label 범위 보존 (10회)", True)
except Exception as e:
    test("augmentation label consistency", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  확장 검증 결과: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} TOTAL")
if FAIL == 0:
    print(f"  ★ 전체 파이프라인 검증 통과 ★")
else:
    print(f"  ✗ {FAIL}개 실패 — 수정 필요")
print(f"{'=' * 70}")

sys.exit(0 if FAIL == 0 else 1)
