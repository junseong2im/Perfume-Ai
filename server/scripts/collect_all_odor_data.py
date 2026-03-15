"""
===================================================================
MEGA DATA COLLECTOR — 인터넷에서 수집 가능한 모든 후각 데이터
===================================================================
Pyrfume 20+ 데이터셋 + Zenodo + DREAM 전부 수집
"""
import os
import sys
import json
import time
import csv
import urllib.request
import urllib.error
import urllib.parse

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')
os.makedirs(DATA_DIR, exist_ok=True)

PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"

# ============================================================
# 모든 Pyrfume 데이터셋 목록
# ============================================================
PYRFUME_DATASETS = {
    # ── 인간 후각 실측 데이터 (핵심) ──────────────────
    "arctander_1960":  ["molecules.csv", "behavior.csv"],          # 향 캐릭터 프로파일
    "aromadb":         ["molecules.csv", "behavior.csv"],          # 향기 DB
    "arshamian_2022":  ["molecules.csv", "behavior.csv"],          # 쾌락도 평가
    "bushdid_2014":    ["molecules.csv", "behavior.csv"],          # 혼합물 구분
    "dravnieks_1985":  ["molecules.csv", "behavior.csv"],          # 146 서술어 × 150명
    "flavordb":        ["molecules.csv", "behavior.csv"],          # 음식 향미
    "flavornet":       ["molecules.csv", "behavior.csv"],          # 향미 네트워크
    "foodb":           ["molecules.csv", "behavior.csv"],          # 식품 냄새
    "goodscents":      ["molecules.csv", "behavior.csv"],          # GoodScents
    "ifra_2019":       ["molecules.csv", "behavior.csv"],          # IFRA 규제
    "keller_2012":     ["molecules.csv", "behavior.csv"],          # 강도+쾌락도+캐릭터
    "keller_2016":     ["molecules.csv", "behavior.csv"],          # DREAM challenge
    "leffingwell":     ["molecules.csv", "behavior.csv"],          # 3,522분자
    "snitz_2013":      ["molecules.csv", "behavior.csv"],          # 후각 유사도
    
    # ── 수용체/분자 데이터 ────────────────────────
    "abraham_2012":    ["molecules.csv", "behavior.csv"],          # 역치
    "haddad_2008":     ["molecules.csv", "behavior.csv"],          # 인간+쥐
    
    # ── 추가 향 캐릭터 DB ────────────────────────
    "fragrancedb":     ["molecules.csv", "behavior.csv"],          # 향수 DB
    
    # ── 식품 관련 ────────────────────────────
    "freesolve":       ["molecules.csv", "behavior.csv"],          # 솔벤트
}

# ============================================================
# Zenodo 74 odorants dataset
# ============================================================
ZENODO_FILES = {
    "zenodo_74_odorants": {
        "url": "https://zenodo.org/records/14727277/files",
        "files": []  # Zenodo API로 파일 목록 확인 필요
    }
}


def download_file(url, local_path, retries=2):
    """파일 다운로드 (재시도 포함)"""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 200:
        return True
    
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, local_path)
            if os.path.getsize(local_path) > 100:
                return True
        except Exception as e:
            if attempt == retries:
                return False
            time.sleep(1)
    return False


def count_csv(filepath):
    """CSV 행/열 수 카운트"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = sum(1 for _ in reader)
        return rows, len(header)
    except:
        return 0, 0


def download_all_pyrfume():
    """Pyrfume 전체 데이터셋 다운로드"""
    print("=" * 60)
    print("🧪 Pyrfume 전체 데이터셋 다운로드")
    print(f"   {len(PYRFUME_DATASETS)}개 데이터셋")
    print("=" * 60)
    
    total_files = 0
    total_rows = 0
    total_size = 0
    failed = []
    
    for ds_name, files in PYRFUME_DATASETS.items():
        ds_dir = os.path.join(DATA_DIR, 'pyrfume_all', ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        
        ds_rows = 0
        ds_ok = True
        
        for fname in files:
            url = f"{PYRFUME_BASE}/{ds_name}/{fname}"
            local_path = os.path.join(ds_dir, fname)
            
            success = download_file(url, local_path)
            
            if success:
                total_files += 1
                size = os.path.getsize(local_path)
                total_size += size
                rows, cols = count_csv(local_path)
                ds_rows += rows
                total_rows += rows
            else:
                ds_ok = False
        
        if ds_ok and ds_rows > 0:
            print(f"  ✅ {ds_name:20s}: {ds_rows:>6} rows")
        elif ds_ok:
            print(f"  ✅ {ds_name:20s}: (다운로드됨)")
        else:
            print(f"  ⚠️  {ds_name:20s}: 일부 실패")
            failed.append(ds_name)
        
        time.sleep(0.3)
    
    print(f"\n  총 파일: {total_files}")
    print(f"  총 행수: {total_rows:,}")
    print(f"  총 크기: {total_size/1024/1024:.1f} MB")
    if failed:
        print(f"  실패: {', '.join(failed)}")
    
    return total_files, total_rows


def download_zenodo():
    """Zenodo 74 odorants 데이터셋"""
    print(f"\n{'=' * 60}")
    print("📊 Zenodo — 74 mono-molecular odorants (1,227 participants)")
    print("=" * 60)
    
    zenodo_dir = os.path.join(DATA_DIR, 'zenodo')
    os.makedirs(zenodo_dir, exist_ok=True)
    
    # Zenodo API로 파일 목록 가져오기
    api_url = "https://zenodo.org/api/records/14727277"
    info_path = os.path.join(zenodo_dir, 'record_info.json')
    
    try:
        print(f"  ⬇️  메타데이터 조회...", end="", flush=True)
        urllib.request.urlretrieve(api_url, info_path)
        
        with open(info_path, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        files = record.get('files', [])
        print(f" {len(files)}개 파일 발견")
        
        downloaded = 0
        for file_info in files:
            fname = file_info.get('key', '')
            file_url = file_info.get('links', {}).get('self', '')
            file_size = file_info.get('size', 0)
            
            if not file_url:
                continue
            
            local_path = os.path.join(zenodo_dir, fname)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                print(f"  ✅ {fname} (이미 존재)")
                downloaded += 1
                continue
            
            print(f"  ⬇️  {fname} ({file_size/1024:.1f} KB)...", end="", flush=True)
            success = download_file(file_url, local_path)
            if success:
                print(f" OK")
                downloaded += 1
            else:
                print(f" 실패")
            
            time.sleep(0.5)
        
        print(f"\n  결과: {downloaded}/{len(files)} 파일")
    except Exception as e:
        print(f" 실패: {e}")
        print(f"  📎 수동 다운로드: https://zenodo.org/records/14727277")


def download_dream_trainset():
    """DREAM Olfaction Challenge 학습 데이터 (이미 있으면 스킵)"""
    print(f"\n{'=' * 60}")
    print("🧬 Osmo DREAM TrainSet 확인")
    print("=" * 60)
    
    dream_path = os.path.join(DATA_DIR, 'osmo_pom', 'DREAMTrainSet.txt')
    if os.path.exists(dream_path) and os.path.getsize(dream_path) > 1000:
        size_kb = os.path.getsize(dream_path) / 1024
        print(f"  ✅ DREAMTrainSet.txt: {size_kb:.0f} KB (이미 존재)")
    else:
        url = "https://raw.githubusercontent.com/osmoai/publications/main/lee_et_al_2023/data/DREAMTrainSet.txt"
        os.makedirs(os.path.dirname(dream_path), exist_ok=True)
        download_file(url, dream_path)
        print(f"  ⬇️  DREAMTrainSet.txt 다운로드 완료")


def download_kaggle_check():
    """Kaggle Fragrantica 다운로드 상태 확인"""
    print(f"\n{'=' * 60}")
    print("⭐ Kaggle Fragrantica 향수 데이터 확인")
    print("=" * 60)
    
    frag_dir = os.path.join(DATA_DIR, 'fragrantica')
    os.makedirs(frag_dir, exist_ok=True)
    
    csv_path = os.path.join(frag_dir, 'fra_cleaned.csv')
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        rows, cols = count_csv(csv_path)
        print(f"  ✅ fra_cleaned.csv: {rows}행 × {cols}열")
    else:
        print(f"  ⚠️ Kaggle에서 수동 다운로드 필요:")
        print(f"  📎 https://www.kaggle.com/datasets/miufana1/fragranticacom-fragrance-dataset")
        print(f"  → fra_cleaned.csv를 {frag_dir}/ 에 저장")
        
        # kaggle CLI 시도
        try:
            import subprocess
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', 
                 'miufana1/fragranticacom-fragrance-dataset',
                 '-p', frag_dir, '--unzip'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f"  ✅ Kaggle CLI로 다운로드 성공!")
        except:
            pass


def final_analysis():
    """전체 수집 데이터 분석"""
    print(f"\n{'=' * 60}")
    print("📊 전체 수집 데이터 종합 분석")
    print("=" * 60)
    
    total_files = 0
    total_size = 0
    total_molecules = 0
    total_behavior_rows = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            if sz < 100:
                continue
            total_files += 1
            total_size += sz
            
            if f == 'molecules.csv':
                rows, _ = count_csv(fp)
                total_molecules += rows
            elif f == 'behavior.csv':
                rows, _ = count_csv(fp)
                total_behavior_rows += rows
    
    # DREAM TrainSet 특별 카운트
    dream_path = os.path.join(DATA_DIR, 'osmo_pom', 'DREAMTrainSet.txt')
    if os.path.exists(dream_path):
        with open(dream_path, 'r', encoding='utf-8', errors='replace') as f:
            dream_rows = sum(1 for _ in f) - 1
        total_behavior_rows += dream_rows
    
    print(f"""
  ┌──────────────────────────────────────────┐
  │  총 파일:       {total_files:>8}개                │
  │  총 크기:       {total_size/1024/1024:>8.1f} MB               │
  │  분자 데이터:   {total_molecules:>8}개               │
  │  행동 데이터:   {total_behavior_rows:>8}행               │
  └──────────────────────────────────────────┘

  데이터 유형별:
    🧪 분자 SMILES/구조: {total_molecules:,}개
    📊 실측 향 평가:     {total_behavior_rows:,}행
    📁 총 데이터셋:      {total_files}개 파일
    💾 총 용량:          {total_size/1024/1024:.1f} MB
    """)
    
    # 데이터셋별 상세
    print("  데이터셋별 상세:")
    for ds_name in sorted(os.listdir(os.path.join(DATA_DIR, 'pyrfume_all'))) if os.path.exists(os.path.join(DATA_DIR, 'pyrfume_all')) else []:
        ds_dir = os.path.join(DATA_DIR, 'pyrfume_all', ds_name)
        if not os.path.isdir(ds_dir):
            continue
        
        mol_path = os.path.join(ds_dir, 'molecules.csv')
        beh_path = os.path.join(ds_dir, 'behavior.csv')
        
        mol_rows, mol_cols = count_csv(mol_path) if os.path.exists(mol_path) else (0, 0)
        beh_rows, beh_cols = count_csv(beh_path) if os.path.exists(beh_path) else (0, 0)
        
        if mol_rows > 0 or beh_rows > 0:
            print(f"    {ds_name:20s}: 분자 {mol_rows:>5}개, 평가 {beh_rows:>6}행 × {beh_cols}열")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("🚀 MEGA DATA COLLECTOR — 인터넷 모든 후각 데이터 수집")
    print("=" * 60)
    start = time.time()
    
    # 1. Pyrfume 전체 (18개 데이터셋)
    download_all_pyrfume()
    
    # 2. Zenodo 74 odorants
    download_zenodo()
    
    # 3. DREAM TrainSet
    download_dream_trainset()
    
    # 4. Kaggle Fragrantica
    download_kaggle_check()
    
    # 5. 종합 분석
    final_analysis()
    
    elapsed = time.time() - start
    print(f"\n⏱️ 전체 소요: {elapsed:.1f}초")
    print("✅ 데이터 수집 완료!")
