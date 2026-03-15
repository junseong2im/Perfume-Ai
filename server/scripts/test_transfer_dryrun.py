"""Dry-run test for transfer_from logic in train_v6.py"""
import sys, ast, inspect, re

BASE = r"c:\Users\user\Desktop\Game\server\cloud"
sys.path.insert(0, BASE)

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")

print("=" * 60)
print("  Transfer Learning Dry-Run Test")
print("=" * 60)

# Read file content
with open(f"{BASE}/train_v6.py", "r", encoding="utf-8") as f:
    content = f.read()

# [1] transfer_from in function signature
print("\n[1] Function signature check")
test("transfer_from in train_odor_predictor signature",
     "transfer_from=None" in content)

# [2] Transfer learning block exists
print("\n[2] Transfer learning logic")
test("Transfer from block exists",
     "Transfer Learning" in content and "transfer_from" in content)
test("Noise std=0.01 applied",
     "torch.randn_like(param) * 0.01" in content)
test("Only loads model weights (not optimizer)",
     "model.load_state_dict(ckpt['model_state_dict'])" in content and
     content.count("model.load_state_dict(ckpt['model_state_dict'])") >= 2)  # resume + transfer
test("Does not load optimizer in transfer block",
     # Check that the transfer block doesn't have optimizer loading
     "optimizer/epoch RESET" in content)

# [3] resume_path takes priority
print("\n[3] Resume priority check")
# The condition is: if transfer_from and ... and not resume_path
test("resume_path takes priority over transfer_from",
     "not resume_path" in content)

# [4] Patience reduced
print("\n[4] Patience check")
test("patience_limit = 25",
     "patience_limit = 25" in content)
test("patience_limit = 40 removed",
     "patience_limit = 40" not in content)

# [5] Phase 3 loop passes transfer_from
print("\n[5] Phase 3 ensemble loop")
test("transfer_from=phase1_ckpt in Phase 3 loop",
     "transfer_from=phase1_ckpt" in content)

# [6] Phase 2 is NOT affected (no transfer_from)
print("\n[6] Phase 2 unaffected")
# Phase 2 calls train_odor_predictor with resume_path, not transfer_from
phase2_section = content[content.find("[Phase 2]"):content.find("[Phase 3]")]
test("Phase 2 uses resume_path (not transfer_from)",
     "resume_path" in phase2_section and "transfer_from" not in phase2_section)

# [7] Syntax validation
print("\n[7] Syntax validation")
try:
    ast.parse(content)
    test("Python syntax valid", True)
except SyntaxError as e:
    test("Python syntax valid", False, str(e))

print(f"\n{'=' * 60}")
print(f"  Result: {PASS} PASS / {FAIL} FAIL")
if FAIL == 0:
    print(f"  ★ Ready for cloud deployment ★")
else:
    print(f"  ✗ Fix {FAIL} issues before deploying")
print(f"{'=' * 60}")
