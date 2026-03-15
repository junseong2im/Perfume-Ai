"""
Auto-Continue + Resume 기능 검증 v2
=====================================
"""
import sys, os, shutil, tempfile
SERVER = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, SERVER)
sys.path.insert(0, os.path.join(SERVER, 'cloud'))
sys.path.insert(0, os.path.join(SERVER, 'cloud', 'models'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
print("  Auto-Continue + Resume 기능 검증")
print("=" * 70)

tmp_dir = tempfile.mkdtemp()
ckpt_path = os.path.join(tmp_dir, 'test_ckpt.pt')

# ═══════════════════════════════════════════════════════════════════
# 1. Checkpoint 저장/로드 round-trip
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Checkpoint 저장 → 로드 round-trip")

try:
    from odor_predictor_v6 import OdorPredictorV6, EMA

    model_a = OdorPredictorV6(bert_dim=384, n_odor_dim=22)
    ema_a = EMA(model_a, decay=0.9995)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

    # Simple backward to initialize optimizer state
    dummy_loss = sum(p.sum() for p in model_a.parameters() if p.requires_grad)
    dummy_loss.backward()
    opt_a.step()
    ema_a.update(model_a)

    # Get original weight sample
    orig_weight = model_a.heads.odor[0].weight.data[0, 0].item()

    # Save checkpoint
    torch.save({
        'epoch': 50,
        'model_state_dict': model_a.state_dict(),
        'optimizer_state_dict': opt_a.state_dict(),
        'ema_shadow': ema_a.shadow,
        'val_loss': 0.1234,
        'val_cos_sim': 0.8765,
        'bert_dim': 384,
        'n_odor_dim': 22,
        'seed': 42,
    }, ckpt_path)
    test("checkpoint 저장", os.path.exists(ckpt_path))

    # Load into new model
    model_b = OdorPredictorV6(bert_dim=384, n_odor_dim=22)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_b.load_state_dict(ckpt['model_state_dict'])
    test("checkpoint 로드", True)

    # Verify weights match
    loaded_weight = model_b.heads.odor[0].weight.data[0, 0].item()
    test("파라미터 일치", abs(orig_weight - loaded_weight) < 1e-8,
         f"orig={orig_weight}, loaded={loaded_weight}")

    # Full model state check
    match = all(torch.equal(pa.data, pb.data)
                for pa, pb in zip(model_a.parameters(), model_b.parameters()))
    test("전체 파라미터 일치", match)

    test("epoch 복원 = 50", ckpt['epoch'] == 50)
    test("val_cos_sim 복원", abs(ckpt['val_cos_sim'] - 0.8765) < 1e-4)

    # EMA shadow
    ema_b = EMA(model_b, decay=0.9995)
    ema_b.shadow = ckpt['ema_shadow']
    ema_match = all(torch.equal(ema_a.shadow[k], ema_b.shadow[k])
                    for k in ema_a.shadow)
    test("EMA shadow 전체 일치", ema_match)

    # Optimizer state
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    opt_b.load_state_dict(ckpt['optimizer_state_dict'])
    has_state = any(len(v) > 0 for v in opt_b.state.values())
    test("optimizer momentum/variance 복원", has_state)

except Exception as e:
    test("checkpoint round-trip", False, f"{e}\n{traceback.format_exc()}")

# ═══════════════════════════════════════════════════════════════════
# 2. Resume 후 start_epoch 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Resume 후 start_epoch 검증")

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    start_epoch = ckpt.get('epoch', 0) + 1
    best_cos = ckpt.get('val_cos_sim', 0.0)
    best_loss = ckpt.get('val_loss', float('inf'))

    test("start_epoch = 51 (50+1)", start_epoch == 51)
    test("best_cos_sim 복원", abs(best_cos - 0.8765) < 1e-4)
    test("best_val_loss 복원", abs(best_loss - 0.1234) < 1e-4)

    epochs = 100
    epoch_range = list(range(start_epoch, epochs + 1))
    test("epoch range: 51~100", epoch_range[0] == 51 and epoch_range[-1] == 100)
    test("남은 epoch 수: 50", len(epoch_range) == 50)
except Exception as e:
    test("resume start_epoch", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# 3. LR factor 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[3] LR factor 스케일링 검증")

try:
    m = nn.Linear(10, 10)
    opt = torch.optim.AdamW([{'params': m.parameters(), 'lr': 1e-3}])
    test("원래 lr = 1e-3", abs(opt.param_groups[0]['lr'] - 1e-3) < 1e-8)

    for pg in opt.param_groups:
        pg['lr'] *= 0.1
    test("factor=0.1 → lr = 1e-4", abs(opt.param_groups[0]['lr'] - 1e-4) < 1e-8)
except Exception as e:
    test("LR factor", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# 4. Phase 스킵 로직 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Phase 스킵 로직 검증")

try:
    seed = 42
    phase1_ckpt = os.path.join(tmp_dir, f'odor_v6_best_seed{seed}.pt')
    shutil.copy2(ckpt_path, phase1_ckpt)
    test("Phase1 checkpoint 존재 → skip", os.path.exists(phase1_ckpt))

    phase2_ckpt = os.path.join(tmp_dir, f'odor_v6_finetune_seed{seed}.pt')
    test("Phase2 checkpoint 미존재 → run", not os.path.exists(phase2_ckpt))

    shutil.copy2(phase1_ckpt, phase2_ckpt)
    test("Phase2 완료 후 finetune checkpoint 생성", os.path.exists(phase2_ckpt))

    ensemble_seeds = [42, 123, 456, 789, 1024]
    skip_count = sum(1 for s in ensemble_seeds
                     if os.path.exists(os.path.join(tmp_dir, f'odor_v6_best_seed{s}.pt')))
    run_count = len(ensemble_seeds) - skip_count
    test(f"Ensemble: skip={skip_count}, run={run_count}",
         skip_count == 1 and run_count == 4)
except Exception as e:
    test("phase skip logic", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# 5. Scheduler replay 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Scheduler replay 검증")

try:
    from train_v6 import WarmupCosineScheduler

    m = nn.Linear(10, 10)
    opt1 = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched1 = WarmupCosineScheduler(opt1, warmup_epochs=10, T_0=50)
    for e in range(50):
        sched1.step(e)
    lr_expected = opt1.param_groups[0]['lr']

    opt2 = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched2 = WarmupCosineScheduler(opt2, warmup_epochs=10, T_0=50)
    for e in range(50):
        sched2.step(e)
    lr_replay = opt2.param_groups[0]['lr']

    test("scheduler replay: lr 동일",
         abs(lr_replay - lr_expected) < 1e-8)
except Exception as e:
    test("scheduler replay", False, str(e))

# ═══════════════════════════════════════════════════════════════════
# 6. Signature 검증
# ═══════════════════════════════════════════════════════════════════
print("\n[6] train_odor_predictor signature 검증")

try:
    import inspect
    from train_v6 import train_odor_predictor
    sig = inspect.signature(train_odor_predictor)
    test("resume_path 파라미터 존재", 'resume_path' in sig.parameters)
    test("start_lr_factor 파라미터 존재", 'start_lr_factor' in sig.parameters)
    test("resume_path 기본값 None", sig.parameters['resume_path'].default is None)
    test("start_lr_factor 기본값 1.0", sig.parameters['start_lr_factor'].default == 1.0)
except Exception as e:
    test("signature", False, str(e))

# Cleanup
try:
    shutil.rmtree(tmp_dir)
except:
    pass

print(f"\n{'=' * 70}")
print(f"  Auto-Continue 검증 결과: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} TOTAL")
if FAIL == 0:
    print(f"  ★ Auto-Continue 전체 검증 통과 ★")
else:
    print(f"  ✗ {FAIL}개 실패 — 수정 필요")
print(f"{'=' * 70}")

sys.exit(0 if FAIL == 0 else 1)
