"""파일 정리 실행 스크립트 — 검증 완료된 파일만 삭제"""
import os, shutil

BASE = r"c:\Users\user\Desktop\Game"
deleted = 0
freed_bytes = 0

def rm(rel):
    global deleted, freed_bytes
    p = os.path.join(BASE, rel)
    if os.path.isfile(p):
        sz = os.path.getsize(p)
        os.remove(p)
        deleted += 1
        freed_bytes += sz
        print(f"  DEL {rel} ({sz//1024}KB)")
    elif os.path.isdir(p):
        sz = sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fns in os.walk(p) for f in fns)
        shutil.rmtree(p)
        deleted += 1
        freed_bytes += sz
        print(f"  DEL {rel}/ ({sz//1024}KB)")

print("=" * 60)
print("  파일 정리 실행")
print("=" * 60)

# === 🔴 ZIP 중복 ===
print("\n[1] ZIP 중복 삭제")
for f in ["server/cloud.zip", "server/cloud_fragrance_v6.zip",
          "server/cloud_fragrance_v6_full.zip", "fragrance_v6.zip"]:
    rm(f)

# === 🔴 TXT 임시 ===
print("\n[2] 임시 TXT 삭제")
for f in ["server/data_results.txt","server/data_results2.txt",
          "server/data_results3.txt","server/data_results4.txt",
          "server/ds_check.txt","server/ds_check2.txt","server/ds_check3.txt",
          "server/test_output.txt","server/test_output_utf8.txt",
          "server/zip_contents.txt","server_log.txt",
          "server/scripts/benchmark_result.txt"]:
    rm(f)

# === 🔴 캐시 백업 ===
print("\n[3] 캐시 백업 삭제")
rm("server/weights/chemberta_cache.npz.bak")

# === 🔴 panel-test.html ===
print("\n[4] panel-test.html 삭제")
rm("panel-test.html")

# === 🟡 구버전 retrain 스크립트 ===
print("\n[5] 구버전 retrain 스크립트 삭제")
for f in ["server/scripts/retrain_cos09.py","server/scripts/retrain_cos09_v16.py",
          "server/scripts/retrain_cos09_v20.py","server/scripts/retrain_cos09_v21.py",
          "server/scripts/retrain_cos09_v22.py","server/scripts/retrain_full_system.py",
          "server/scripts/train_unified.py","server/scripts/train_unified_v2.py",
          "server/scripts/train_full_pipeline.py","server/scripts/train_multitask_pipeline.py",
          "server/scripts/train_v6_pipeline.py","server/train_v6.py"]:
    rm(f)

# === 🟡 일회성 데이터 수집 스크립트 ===
print("\n[6] 일회성 데이터 수집 스크립트 삭제")
for f in ["server/scripts/add_ingredient_origins.py","server/scripts/add_ingredient_origins_extra.py",
          "server/scripts/add_korean_ingredients.py","server/scripts/add_korean_ingredients_v2.py",
          "server/scripts/add_real_formulas.py","server/scripts/expand_global_part1.py",
          "server/scripts/expand_global_part2.py","server/scripts/mega_expand_part1.py",
          "server/scripts/mega_expand_part2.py","server/scripts/mega_data_collector.py",
          "server/scripts/enrich_massive.py","server/scripts/enrich_pro_data.py",
          "server/scripts/consolidate_all_data.py","server/scripts/consolidate_all_data_v2.py",
          "server/scripts/tag_all_origins.py","server/scripts/tag_final_batch.py"]:
    rm(f)

# === 🟡 이전 테스트 스크립트 ===
print("\n[7] 이전 테스트 스크립트 삭제")
for f in ["server/scripts/test_4strategies.py","server/scripts/test_7improvements.py",
          "server/scripts/test_clean.py","server/scripts/test_csv_loader.py",
          "server/scripts/test_final_verification.py","server/scripts/test_improvements.py",
          "server/scripts/test_multihead.py","server/scripts/test_production_quick.py",
          "server/scripts/test_v6_arch.py","server/scripts/test_zero_defect.py",
          "server/scripts/test_zero_defect_fixes.py","server/scripts/comprehensive_test.py",
          "server/scripts/full_verification.py"]:
    rm(f)

# === 🟡 구버전 weight/LoRA ===
print("\n[8] 구버전 weight 삭제")
rm("server/weights/bert_cache_lora_v16.npz")
rm("server/weights/cos09_v22_best.pt")
rm("server/weights/chemberta_lora_v16")
rm("server/weights/chemberta_lora_v20")
rm("server/weights/chemberta_lora_v21")

# === __pycache__ ===
print("\n[9] __pycache__ 삭제")
for root, dirs, _ in os.walk(BASE):
    for d in dirs:
        if d == "__pycache__":
            p = os.path.join(root, d)
            sz = sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fns in os.walk(p) for f in fns)
            shutil.rmtree(p)
            deleted += 1
            freed_bytes += sz
            rel = os.path.relpath(p, BASE)
            print(f"  DEL {rel}/ ({sz//1024}KB)")

print(f"\n{'=' * 60}")
print(f"  삭제 완료: {deleted}개, {freed_bytes/1e9:.2f}GB 확보")
print(f"{'=' * 60}")
