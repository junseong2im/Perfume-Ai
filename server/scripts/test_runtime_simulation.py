"""
런타임 환경 시뮬레이션 테스트
==============================
GPU 없이 CPU에서 실제 학습 파이프라인을 미니 실행합니다.
- collate_odor 실제 PyG Batch 생성
- OdorPredictorV6 forward + backward
- Gradient accumulation + NaN watchdog
- SWA update
- EMA update
- compute_loss 10 heads
- DataLoader multi-batch
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
from torch.utils.data import DataLoader, Dataset

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
print("  런타임 환경 시뮬레이션 (CPU)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════
# 1. collate_odor + 실제 PyG Graph
# ═══════════════════════════════════════════════════════════════════
print("\n[1] collate_odor + 실제 PyG Batch")

try:
    from odor_predictor_v6 import smiles_to_graph_v6, extract_phys_props
    from train_v6 import collate_odor

    smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCCCC=O']
    batch_items = []
    for smi in smiles_list:
        g = smiles_to_graph_v6(smi, compute_3d=False)
        phys = extract_phys_props(smi)
        batch_items.append({
            'graph': g,
            'bert': torch.randn(384),
            'phys': torch.tensor(phys, dtype=torch.float32),
            'odor': torch.rand(22),
            'weight': 1.0,
            'difficulty': 0.5,
            'smiles': smi,
        })

    batch = collate_odor(batch_items)
    test("collate_odor 성공", batch is not None)
    test("batch['bert'] shape", batch['bert'].shape == (4, 384),
         f"{batch['bert'].shape}")
    test("batch['phys'] shape", batch['phys'].shape == (4, 12),
         f"{batch['phys'].shape}")
    test("batch['odor'] shape", batch['odor'].shape == (4, 22),
         f"{batch['odor'].shape}")
    test("batch['graph_batch'] 유효", hasattr(batch['graph_batch'], 'x'))
    test("graph.batch 존재", hasattr(batch['graph_batch'], 'batch'))
except Exception as e:
    test("collate_odor", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 2. OdorPredictorV6 forward + backward (CPU)
# ═══════════════════════════════════════════════════════════════════
print("\n[2] OdorPredictorV6 forward + backward (CPU)")

try:
    from odor_predictor_v6 import OdorPredictorV6

    model = OdorPredictorV6(bert_dim=384, n_odor_dim=22)
    model.train()

    bert_emb = batch['bert']
    graph_batch = batch['graph_batch']
    phys = batch['phys']

    pred = model(bert_emb, graph_batch, phys, return_aux=True)
    test("forward 성공", isinstance(pred, dict))
    test("pred['odor'] shape", pred['odor'].shape == (4, 22),
         f"{pred['odor'].shape}")
    test("pred['top'] shape", pred['top'].shape == (4, 22))
    test("pred 10 heads 존재",
         all(k in pred for k in ['odor','top','mid','base','longevity','sillage',
                                   'descriptors','receptors','hedonic','super_res']))

    # Backward
    from odor_predictor_v6 import compute_loss
    target = {'odor': batch['odor']}
    loss, losses = compute_loss(model, pred, target, None, epoch=10, max_epochs=300)
    loss.backward()
    test("backward 성공", True)

    # Gradients exist and are finite
    has_grad = any(p.grad is not None for p in model.parameters())
    test("gradients 존재", has_grad)
    all_finite = all(torch.isfinite(p.grad).all().item()
                     for p in model.parameters() if p.grad is not None)
    test("gradients 전부 유한", all_finite)
except Exception as e:
    test("OdorPredictorV6 forward+backward", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 3. Gradient Accumulation + NaN Watchdog 상호작용
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Gradient Accumulation + NaN Watchdog")

try:
    model.zero_grad()
    accumulation_steps = 4
    nan_count = 0

    for step in range(accumulation_steps):
        pred = model(bert_emb, graph_batch, phys, return_aux=True)
        target = {'odor': batch['odor']}
        loss, _ = compute_loss(model, pred, target, None, 10, 300)

        # NaN watchdog
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        (loss / accumulation_steps).backward()

    test(f"4-step 누적: NaN 0개", nan_count == 0)

    # Clip gradients
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    test("clip_grad_norm 유한", math.isfinite(float(grad_norm)),
         f"grad_norm={grad_norm}")

    # Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    test("optimizer.step() 성공", True)

    # Parameters still finite after step
    all_params_finite = all(torch.isfinite(p).all().item()
                            for p in model.parameters())
    test("step 후 파라미터 유한", all_params_finite)
except Exception as e:
    test("gradient accumulation", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 4. EMA update 실제 모델
# ═══════════════════════════════════════════════════════════════════
print("\n[4] EMA update 실제 모델")

try:
    from odor_predictor_v6 import EMA

    ema = EMA(model, decay=0.9995)
    ema.update(model)
    test("EMA update 성공", True)

    ema.apply_shadow(model)
    test("EMA apply_shadow 성공", True)

    all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    test("EMA 적용 후 파라미터 유한", all_finite)

    ema.restore(model)
    test("EMA restore 성공", True)
except Exception as e:
    test("EMA on real model", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 5. SWA update
# ═══════════════════════════════════════════════════════════════════
print("\n[5] SWA update")

try:
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_model.update_parameters(model)
    test("SWA update_parameters 성공", True)

    # SWA forward
    swa_model.eval()
    with torch.no_grad():
        swa_pred = swa_model(bert_emb, graph_batch, phys, return_aux=True)
    test("SWA forward 성공", isinstance(swa_pred, dict))
    test("SWA pred 유한",
         torch.isfinite(swa_pred['odor']).all().item())
except Exception as e:
    test("SWA", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 6. GradNorm 실제 모델
# ═══════════════════════════════════════════════════════════════════
print("\n[6] GradNorm 실제 모델")

try:
    from odor_predictor_v6 import GradNorm

    gn = GradNorm(model, n_tasks=10, alpha=1.5)
    # First update
    gn.update(model, {'odor_mse': torch.tensor(0.5), 'odor_cos': torch.tensor(0.3)}, epoch=1)
    # Second update
    gn.update(model, {'odor_mse': torch.tensor(0.4), 'odor_cos': torch.tensor(0.2)}, epoch=2)
    test("GradNorm 실제 모델 update 성공", True)
    test("loss_weights 유한",
         torch.isfinite(model.loss_weights).all().item())
except Exception as e:
    test("GradNorm real model", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 7. R-Drop 이중 forward
# ═══════════════════════════════════════════════════════════════════
print("\n[7] R-Drop 이중 forward")

try:
    model.train()
    pred1 = model(bert_emb, graph_batch, phys, return_aux=True)
    pred2 = model(bert_emb, graph_batch, phys, return_aux=True)

    # R-Drop: two forward passes should differ (dropout active)
    diff = (pred1['odor'] - pred2['odor']).abs().sum().item()
    test("train mode 2회 forward: 출력 다름 (dropout)", diff > 0.01,
         f"diff={diff:.6f}")

    # Combine for R-Drop loss
    pred_rdrop = {**pred1, 'odor_2': pred2['odor']}
    loss_r, losses_r = compute_loss(model, pred_rdrop,
                                     {'odor': batch['odor']},
                                     None, 20, 300, training=True)
    test("R-Drop loss 유한", torch.isfinite(loss_r).item())
    test("R-Drop loss_dict에 rdrop 존재", 'rdrop' in losses_r)
except Exception as e:
    test("R-Drop dual forward", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 8. Contrastive loss 실제 backbone embeddings
# ═══════════════════════════════════════════════════════════════════
print("\n[8] Contrastive loss 실제 backbone")

try:
    from odor_predictor_v6 import contrastive_loss

    model.eval()
    with torch.no_grad():
        out = model(bert_emb, graph_batch, phys,
                     return_aux=True, return_backbone=True)

    backbone_feat = out['backbone_256']
    test("backbone 256d 출력", backbone_feat.shape == (4, 256),
         f"{backbone_feat.shape}")

    loss_c = contrastive_loss(backbone_feat, batch['odor'], temperature=0.07)
    test("contrastive loss(실제 backbone) 유한",
         torch.isfinite(loss_c).item())
except Exception as e:
    test("contrastive real backbone", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 9. Mini DataLoader 시뮬레이션
# ═══════════════════════════════════════════════════════════════════
print("\n[9] Mini DataLoader")

try:
    class MiniDataset(Dataset):
        def __init__(self, smiles_list):
            self.items = []
            for smi in smiles_list:
                g = smiles_to_graph_v6(smi, compute_3d=False)
                phys = extract_phys_props(smi)
                self.items.append({
                    'graph': g,
                    'bert': torch.randn(384),
                    'phys': torch.tensor(phys, dtype=torch.float32),
                    'odor': torch.rand(22),
                    'weight': 1.0,
                    'difficulty': 0.5,
                    'smiles': smi,
                })
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    mini_ds = MiniDataset(['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCCCC=O',
                           'CC(C)=CC=O', 'OC(=O)c1ccccc1', 'CCCCCCCCC', 'CC=CC'])
    loader = DataLoader(mini_ds, batch_size=4, shuffle=True,
                        collate_fn=collate_odor, num_workers=0)

    n_batches = 0
    for batch_data in loader:
        n_batches += 1
        test(f"batch {n_batches}: bert shape OK",
             batch_data['bert'].shape[1] == 384)

    test(f"DataLoader {n_batches} batches 완료", n_batches == 2)
except Exception as e:
    test("Mini DataLoader", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 10. 미니 학습 루프 (2 epochs × 2 batches)
# ═══════════════════════════════════════════════════════════════════
print("\n[10] 미니 학습 루프 (2 epochs × 2 batches)")

try:
    from train_v6 import WarmupCosineScheduler, apply_progressive_unfreezing

    model2 = OdorPredictorV6(bert_dim=384, n_odor_dim=22)
    model2.train()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = WarmupCosineScheduler(optimizer2, warmup_epochs=2, T_0=5)
    ema2 = EMA(model2, decay=0.999)

    for epoch in range(1, 3):
        apply_progressive_unfreezing(model2, epoch)
        epoch_loss = 0.0
        nan_count = 0

        for batch_data in loader:
            optimizer2.zero_grad()
            pred = model2(batch_data['bert'], batch_data['graph_batch'],
                          batch_data['phys'], return_aux=True)
            loss, _ = compute_loss(model2, pred,
                                    {'odor': batch_data['odor']},
                                    None, epoch, 10)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            optimizer2.step()
            ema2.update(model2)
            epoch_loss += loss.item()

        scheduler2.step(epoch)
        test(f"epoch {epoch}: loss={epoch_loss:.4f}, NaN={nan_count}",
             nan_count == 0 and math.isfinite(epoch_loss))

    # final params finite
    test("학습 후 파라미터 유한",
         all(torch.isfinite(p).all().item() for p in model2.parameters()))
except Exception as e:
    test("mini training loop", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  런타임 시뮬레이션 결과: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} TOTAL")
if FAIL == 0:
    print(f"  ★ 런타임 환경 시뮬레이션 전체 통과 ★")
else:
    print(f"  ✗ {FAIL}개 실패 — 수정 필요")
print(f"{'=' * 70}")

sys.exit(0 if FAIL == 0 else 1)
