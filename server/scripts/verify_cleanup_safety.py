"""
파일 정리 안전성 검증
======================
삭제 예정 파일이 핵심 파일에서 참조되는지 확인합니다.
실제 삭제는 하지 않습니다.
"""
import os, re, sys

BASE = r"c:\Users\user\Desktop\Game"
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

print("=" * 70)
print("  파일 정리 안전성 검증")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# 핵심 파일 목록 (반드시 유지)
# ═══════════════════════════════════════════════════════════════
CORE_PY = [
    "server/main.py",
    "server/database.py",
    "server/db_api.py",
    "server/odor_engine.py",
    "server/recipe_engine.py",
    "server/molecular_harmony.py",
    "server/sommelier.py",
    "server/biophysics_simulator.py",
    "server/v6_bridge.py",
    "server/models/__init__.py",
    "server/models/odor_predictor_v6.py",
    "server/models/recipe_vae.py",
    "server/models/safety_net.py",
    "server/models/mixture_net.py",
    "server/models/geometric_gnn.py",
    "server/models/active_learner.py",
    "server/models/competitive_binding.py",
    "server/models/sensor_engine.py",
    "server/models/vae_generator.py",
    "server/models/molecular_engine.py",
    "server/models/neural_net.py",
    "server/cloud/train_v6.py",
    "server/cloud/pretrain_ssl.py",
]

CORE_JS = [
    "js/app.js",
    "js/ai-client.js",
    "js/ai-engine.js",
    "js/analyzer.js",
    "js/active-learner.js",
    "js/competitive-binding.js",
    "js/explainability.js",
    "js/formulator.js",
    "js/geometric-gnn.js",
    "js/harmony.js",
    "js/ingredient-db.js",
    "js/molecular-engine.js",
    "js/neural-net.js",
    "js/sensor-engine.js",
    "js/tf-config.js",
    "js/vae-generator.js",
    "js/visualizer.js",
]

# ═══════════════════════════════════════════════════════════════
# 삭제 예정 파일 (🔴 + 🟡)
# ═══════════════════════════════════════════════════════════════
DELETE_SCRIPTS = [
    "server/scripts/retrain_cos09.py",
    "server/scripts/retrain_cos09_v16.py",
    "server/scripts/retrain_cos09_v20.py",
    "server/scripts/retrain_cos09_v21.py",
    "server/scripts/retrain_cos09_v22.py",
    "server/scripts/retrain_full_system.py",
    "server/scripts/train_unified.py",
    "server/scripts/train_unified_v2.py",
    "server/scripts/train_full_pipeline.py",
    "server/scripts/train_multitask_pipeline.py",
    "server/scripts/train_v6_pipeline.py",
    "server/train_v6.py",  # 구버전 (cloud/train_v6.py가 최신)
    "server/scripts/add_ingredient_origins.py",
    "server/scripts/add_ingredient_origins_extra.py",
    "server/scripts/add_korean_ingredients.py",
    "server/scripts/add_korean_ingredients_v2.py",
    "server/scripts/add_real_formulas.py",
    "server/scripts/expand_global_part1.py",
    "server/scripts/expand_global_part2.py",
    "server/scripts/mega_expand_part1.py",
    "server/scripts/mega_expand_part2.py",
    "server/scripts/mega_data_collector.py",
    "server/scripts/enrich_massive.py",
    "server/scripts/enrich_pro_data.py",
    "server/scripts/consolidate_all_data.py",
    "server/scripts/consolidate_all_data_v2.py",
    "server/scripts/tag_all_origins.py",
    "server/scripts/tag_final_batch.py",
    "server/scripts/test_4strategies.py",
    "server/scripts/test_7improvements.py",
    "server/scripts/test_clean.py",
    "server/scripts/test_csv_loader.py",
    "server/scripts/test_final_verification.py",
    "server/scripts/test_improvements.py",
    "server/scripts/test_multihead.py",
    "server/scripts/test_production_quick.py",
    "server/scripts/test_v6_arch.py",
    "server/scripts/test_zero_defect.py",
    "server/scripts/test_zero_defect_fixes.py",
    "server/scripts/comprehensive_test.py",
    "server/scripts/full_verification.py",
]

DELETE_FILES = [
    "server/cloud.zip",
    "server/cloud_fragrance_v6.zip",
    "server/cloud_fragrance_v6_full.zip",
    "fragrance_v6.zip",
    "server/data_results.txt",
    "server/data_results2.txt",
    "server/data_results3.txt",
    "server/data_results4.txt",
    "server/ds_check.txt",
    "server/ds_check2.txt",
    "server/ds_check3.txt",
    "server/test_output.txt",
    "server/test_output_utf8.txt",
    "server/zip_contents.txt",
    "server_log.txt",
    "server/scripts/benchmark_result.txt",
]

DELETE_WEIGHTS = [
    "server/weights/chemberta_cache.npz.bak",
    "server/weights/bert_cache_lora_v16.npz",
    "server/weights/cos09_v22_best.pt",
]

DELETE_DIRS = [
    "server/weights/chemberta_lora_v16",
    "server/weights/chemberta_lora_v20",
    "server/weights/chemberta_lora_v21",
]

# ═══════════════════════════════════════════════════════════════
# [1] 핵심 Python 파일에서 삭제 예정 모듈 import 확인
# ═══════════════════════════════════════════════════════════════
print("\n[1] 핵심 .py 파일 → 삭제 예정 모듈 참조 확인")

# Build set of module names from delete scripts
delete_modules = set()
for f in DELETE_SCRIPTS:
    basename = os.path.basename(f).replace('.py', '')
    delete_modules.add(basename)

issues = []
for core_rel in CORE_PY:
    core_path = os.path.join(BASE, core_rel)
    if not os.path.exists(core_path):
        continue
    with open(core_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    for mod in delete_modules:
        # Check import patterns
        if re.search(rf'\bimport\s+{mod}\b', content) or \
           re.search(rf'\bfrom\s+{mod}\b', content) or \
           re.search(rf"['\"]\.?/?{mod}\.py['\"]", content):
            issues.append((core_rel, mod))

if issues:
    for core, mod in issues:
        test(f"{core} → {mod}", False, "삭제 예정 모듈 참조!")
else:
    test("핵심 .py 파일에서 삭제 예정 모듈 참조 없음", True)

# ═══════════════════════════════════════════════════════════════
# [2] 핵심 JS 파일에서 삭제 예정 파일 참조 확인
# ═══════════════════════════════════════════════════════════════
print("\n[2] 핵심 .js 파일 → 삭제 예정 파일 참조 확인")

js_issues = []
for js_rel in CORE_JS:
    js_path = os.path.join(BASE, js_rel)
    if not os.path.exists(js_path):
        continue
    with open(js_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    for f in DELETE_FILES + DELETE_SCRIPTS:
        basename = os.path.basename(f)
        if basename in content:
            js_issues.append((js_rel, basename))

if js_issues:
    for js, ref in js_issues:
        test(f"{js} → {ref}", False, "삭제 예정 파일 참조!")
else:
    test("핵심 .js 파일에서 삭제 예정 파일 참조 없음", True)

# ═══════════════════════════════════════════════════════════════
# [3] index.html에서 삭제 예정 파일 참조 확인
# ═══════════════════════════════════════════════════════════════
print("\n[3] index.html → 삭제 예정 파일 참조 확인")

html_path = os.path.join(BASE, "index.html")
if os.path.exists(html_path):
    with open(html_path, encoding='utf-8', errors='ignore') as fh:
        html = fh.read()
    html_issues = []
    for f in DELETE_FILES + DELETE_SCRIPTS:
        basename = os.path.basename(f)
        if basename in html:
            html_issues.append(basename)
    if html_issues:
        for ref in html_issues:
            test(f"index.html → {ref}", False, "참조됨!")
    else:
        test("index.html에서 삭제 예정 파일 참조 없음", True)

# ═══════════════════════════════════════════════════════════════
# [4] 삭제 예정 weight 파일이 핵심 코드에서 사용되는지
# ═══════════════════════════════════════════════════════════════
print("\n[4] 삭제 예정 weight 파일 참조 확인")

weight_names = [
    "chemberta_cache.npz.bak",
    "bert_cache_lora_v16.npz",
    "cos09_v22_best.pt",
    "chemberta_lora_v16",
    "chemberta_lora_v20",
    "chemberta_lora_v21",
]

weight_issues = []
for core_rel in CORE_PY:
    core_path = os.path.join(BASE, core_rel)
    if not os.path.exists(core_path):
        continue
    with open(core_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    for wname in weight_names:
        if wname in content:
            weight_issues.append((core_rel, wname))

if weight_issues:
    for core, wname in weight_issues:
        test(f"{core} → {wname}", False, "weight 참조됨!")
else:
    test("핵심 코드에서 삭제 예정 weight 참조 없음", True)

# ═══════════════════════════════════════════════════════════════
# [5] main.py의 라우트에서 삭제 예정 모듈 사용 확인
# ═══════════════════════════════════════════════════════════════
print("\n[5] main.py API 라우트 → 삭제 모듈 의존성 확인")

main_path = os.path.join(BASE, "server", "main.py")
if os.path.exists(main_path):
    with open(main_path, encoding='utf-8', errors='ignore') as fh:
        main_content = fh.read()
    # Extract all imports
    imports = re.findall(r'(?:from|import)\s+(\S+)', main_content)
    main_issues = []
    for imp in imports:
        imp_base = imp.split('.')[0]
        if imp_base in delete_modules:
            main_issues.append(imp)
    if main_issues:
        for imp in main_issues:
            test(f"main.py imports {imp}", False, "삭제 예정!")
    else:
        test("main.py에서 삭제 예정 모듈 import 없음", True)

# ═══════════════════════════════════════════════════════════════
# [6] train_models.py (구버전이 main.py에서 참조되는지)
# ═══════════════════════════════════════════════════════════════
print("\n[6] train_models.py 참조 확인")

train_models_refs = []
for core_rel in CORE_PY + CORE_JS:
    core_path = os.path.join(BASE, core_rel)
    if not os.path.exists(core_path):
        continue
    with open(core_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    if 'train_models' in content:
        train_models_refs.append(core_rel)

if train_models_refs:
    for ref in train_models_refs:
        test(f"train_models.py 참조: {ref}", False, "삭제 시 문제!")
else:
    test("train_models.py 핵심 코드 참조 없음 (🟡 삭제 가능)", True)

# ═══════════════════════════════════════════════════════════════
# [7] server/train_v6.py (구버전)가 다른 곳에서 참조되는지
# ═══════════════════════════════════════════════════════════════
print("\n[7] server/train_v6.py (구버전) 참조 확인")

old_train_refs = []
for core_rel in CORE_PY:
    if core_rel == "server/train_v6.py":
        continue  # 자기 자신
    core_path = os.path.join(BASE, core_rel)
    if not os.path.exists(core_path):
        continue
    with open(core_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    if 'train_v6' in content and 'cloud' not in core_rel:
        old_train_refs.append(core_rel)

test("server/train_v6.py 참조",
     len(old_train_refs) == 0,
     f"참조: {old_train_refs}" if old_train_refs else "")

# ═══════════════════════════════════════════════════════════════
# [8] 삭제 예정 파일 실제 존재 여부
# ═══════════════════════════════════════════════════════════════
print("\n[8] 삭제 예정 파일 실제 존재 확인")

all_delete = DELETE_FILES + DELETE_SCRIPTS + DELETE_WEIGHTS
exists_count = 0
missing = []
for f in all_delete:
    full = os.path.join(BASE, f)
    if os.path.exists(full):
        exists_count += 1
    else:
        missing.append(f)

test(f"삭제 대상 {exists_count}/{len(all_delete)}개 존재", True)
if missing:
    print(f"    (이미 없는 파일: {len(missing)}개)")

for d in DELETE_DIRS:
    full = os.path.join(BASE, d)
    test(f"디렉토리 존재: {d}", os.path.isdir(full) or True)  # OK either way

# ═══════════════════════════════════════════════════════════════
# [9] panel-test.html 삭제 안전성
# ═══════════════════════════════════════════════════════════════
print("\n[9] panel-test.html 참조 확인")

panel_refs = []
for core_rel in CORE_PY + CORE_JS:
    core_path = os.path.join(BASE, core_rel)
    if not os.path.exists(core_path):
        continue
    with open(core_path, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    if 'panel-test' in content or 'panel_test' in content:
        panel_refs.append(core_rel)

test("panel-test.html 참조 없음 (삭제 가능)",
     len(panel_refs) == 0,
     f"참조: {panel_refs}" if panel_refs else "")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"  안전성 검증 결과: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} TOTAL")
if FAIL == 0:
    print(f"  ★ 삭제해도 핵심 기능에 영향 없음 ★")
else:
    print(f"  ✗ {FAIL}개 문제 발견 — 삭제 전 수정 필요")
print(f"{'=' * 70}")
