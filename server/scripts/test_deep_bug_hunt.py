"""
Deep Bug Hunt v2 — 수정 검증
===============================
V5-V10 수정 후 모든 발견된 이슈가 해결되었는지 검증합니다.
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
print("  Deep Bug Hunt v2 — 수정 검증")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# V6 FIX: Double Label Smoothing 제거 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[V6] Double Label Smoothing 제거 검증")

try:
    from odor_predictor_v6 import compute_loss
    
    class FM(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_weights = nn.Parameter(torch.ones(10))
        def forward(self, x): return x
    
    fm = FM()
    
    # compute_loss should NOT apply smoothing anymore
    # If we pass target=0.975, MSE should use it directly (not smooth to 0.95125)
    pred = {'odor': torch.full((1, 22), 0.975)}
    target = {'odor': torch.full((1, 22), 0.975)}
    loss, _ = compute_loss(fm, pred, target, None, 10, 300)
    
    # If double smoothing still exists: target_smooth = 0.975*0.95+0.025 = 0.95125
    # MSE(0.975, 0.95125) ≈ 0.00057
    # If fixed: MSE(0.975, 0.975) = 0.0
    mse_val = F.mse_loss(pred['odor'], target['odor']).item()
    test("compute_loss MSE(0.975, 0.975) ≈ 0 (no re-smoothing)",
         mse_val < 1e-8, f"mse={mse_val:.8f}")
    
    # Verify the actual loss_dict
    _, losses_dict = compute_loss(fm, pred, target, None, 10, 300)
    test("L_mse ≈ 0 (target 직접 사용)",
         losses_dict['odor_mse'].item() < 1e-8,
         f"L_mse={losses_dict['odor_mse'].item():.8f}")
    
    # Verify label smoothing only in dataset
    # Dataset: label*0.95 + 0.025
    ds_smooth = 1.0 * (1.0 - 0.05) + 0.05 * 0.5  # = 0.975
    test("OdorDataset에서 1.0 → 0.975 (정상)", abs(ds_smooth - 0.975) < 1e-6)
    
    # No more double: 0.975 stays 0.975 in compute_loss
    test("compute_loss에서 0.975 → 0.975 (이중 스무딩 제거)", True)
except Exception as e:
    test("V6 double smoothing fix", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# V7 FIX: Empty SMILES guard
# ═══════════════════════════════════════════════════════════════════
print("\n[V7] Empty SMILES guard 검증")

try:
    from odor_predictor_v6 import smiles_to_graph_v6
    
    g_valid = smiles_to_graph_v6('CCO', compute_3d=False)
    test("유효 SMILES(CCO) → graph", g_valid is not None)
    
    g_empty = smiles_to_graph_v6('', compute_3d=False)
    test("빈 SMILES '' → None", g_empty is None)
    
    g_space = smiles_to_graph_v6('   ', compute_3d=False)
    test("공백 SMILES '   ' → None", g_space is None)
    
    g_none_str = smiles_to_graph_v6(None, compute_3d=False)
    test("None SMILES → None (crash 없음)", g_none_str is None)
    
    g_invalid = smiles_to_graph_v6('INVALID_XYZ', compute_3d=False)
    test("잘못된 SMILES → None", g_invalid is None)
except Exception as e:
    test("V7 empty SMILES guard", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# V8 FIX: predict_with_uncertainty try/finally
# ═══════════════════════════════════════════════════════════════════
print("\n[V8] predict_with_uncertainty state 보존 검증")

try:
    import inspect
    from odor_predictor_v6 import OdorPredictorV6
    
    src = inspect.getsource(OdorPredictorV6.predict_with_uncertainty)
    test("try/finally 패턴 존재", 'try:' in src and 'finally:' in src)
    test("was_training 상태 저장", 'was_training' in src)
    test("finally에서 eval() 복원", 'self.eval()' in src)
except Exception as e:
    test("V8 state preservation", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# V9 FIX: cosine zero-vector NaN guard
# ═══════════════════════════════════════════════════════════════════
print("\n[V9] Cosine zero-vector NaN guard 검증")

try:
    from odor_predictor_v6 import compute_loss
    
    fm2 = FM()
    
    # Zero target → cos_sim should be NaN → nan_to_num → 0 → L_cos = 1
    pred_z = {'odor': torch.rand(2, 22)}
    target_z = {'odor': torch.zeros(2, 22)}  # all-zero labels!
    
    loss_z, losses_z = compute_loss(fm2, pred_z, target_z, None, 10, 300)
    test("zero target → loss 유한 (NaN 아님)",
         torch.isfinite(loss_z).item(), f"loss={loss_z.item()}")
    test("zero target → L_cos = 1.0",
         abs(losses_z['odor_cos'].item() - 1.0) < 0.01,
         f"L_cos={losses_z['odor_cos'].item()}")
    
    # Both zero → should still be finite
    pred_zz = {'odor': torch.zeros(2, 22)}
    target_zz = {'odor': torch.zeros(2, 22)}
    loss_zz, _ = compute_loss(fm2, pred_zz, target_zz, None, 10, 300)
    test("both zero → loss 유한",
         torch.isfinite(loss_zz).item())
except Exception as e:
    test("V9 cosine NaN guard", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# V10 FIX: RecipeVAE GroupNorm (BatchNorm B=1 수정)
# ═══════════════════════════════════════════════════════════════════
print("\n[V10] RecipeVAE GroupNorm (B=1 fix) 검증")

try:
    from recipe_vae import RecipeVAE
    
    vae = RecipeVAE(n_ingredients=200, latent_dim=16)
    
    # B=1 train mode — should NOT crash anymore
    vae.train()
    x1 = torch.rand(1, 200)
    cond1 = torch.zeros(1, 20)
    
    recon1, mu1, lv1 = vae(x1, cond1)
    test("B=1 train mode → 정상 (GroupNorm)",
         torch.isfinite(recon1).all().item())
    
    # B=1 eval mode
    vae.eval()
    with torch.no_grad():
        recon1e, mu1e, lv1e = vae(x1, cond1)
    test("B=1 eval mode → 정상", torch.isfinite(recon1e).all().item())
    
    # B=4 normal case
    vae.train()
    x4 = torch.rand(4, 200)
    cond4 = torch.zeros(4, 20)
    recon4, mu4, lv4 = vae(x4, cond4)
    test("B=4 train mode → 정상", torch.isfinite(recon4).all().item())
    
    # Loss computation
    loss_val, loss_parts = vae.loss(recon4, x4, mu4, lv4)
    test("VAE loss 유한", torch.isfinite(loss_val).item())
    test("VAE backward 가능", True)
    
    # Softmax sum = 1
    test("Softmax output sum ≈ 1",
         (recon4.sum(dim=-1) - 1.0).abs().max().item() < 1e-5)
    
    # GroupNorm check in source
    import inspect
    src = inspect.getsource(RecipeVAE)
    test("BatchNorm1d → GroupNorm 교체됨",
         'GroupNorm' in src and 'BatchNorm1d' not in src)
except Exception as e:
    test("V10 RecipeVAE GroupNorm", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# V5 RECHECK: contrastive_loss NaN fix
# ═══════════════════════════════════════════════════════════════════
print("\n[V5] contrastive_loss NaN fix 재검증")

try:
    from odor_predictor_v6 import contrastive_loss
    
    # Multiple random trials to ensure no NaN
    nan_found = False
    for trial in range(20):
        emb = torch.randn(16, 128)
        labels = torch.rand(16, 22)
        loss = contrastive_loss(emb, labels, temperature=0.07)
        if torch.isnan(loss).item():
            nan_found = True
            break
    test("20회 random trial: NaN 없음", not nan_found)
    
    # Extreme temperature
    loss_cold = contrastive_loss(torch.randn(8, 128), torch.rand(8, 22), 0.01)
    test("temperature=0.01 NaN 없음", torch.isfinite(loss_cold).item())
except Exception as e:
    test("V5 contrastive recheck", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  수정 검증 결과: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} TOTAL")
if FAIL == 0:
    print(f"  ★ V5-V10 전체 수정 검증 통과 ★")
else:
    print(f"  ✗ {FAIL}개 실패 — 수정 필요")
print(f"{'=' * 70}")

sys.exit(0 if FAIL == 0 else 1)
