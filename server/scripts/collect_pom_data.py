"""
===================================================================
Phase 4 — 90%+ 정확도 데이터 수집기
Osmo POM + DREAM + Fragrantica 전체 수집
===================================================================
"""
import os
import sys
import json
import time
import csv
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. Osmo POM 데이터 (Science 2023 — Lee et al.)
# ============================================================
OSMO_BASE = "https://raw.githubusercontent.com/osmoai/publications/main/lee_et_al_2023/data"
OSMO_FILES = [
    "Data%20S1.csv",      # 주요 분자 + SMILES + 138 descriptors
    "Data%20S2.csv",      # 검증 데이터
    "Data%20S3.csv",      # 보충 데이터
    "Data%20S4.csv",
    "Data%20S5.csv",
    "Data%20S6.csv",
    "Data%20S7.csv",
    "DREAMTrainSet.txt",  # DREAM 학습 데이터 (476분자, 19 descriptors)
    "gslf_indices.csv",   # GoodScents+Leffingwell 인덱스
    "calibration_thresholds.csv",  # 보정 역치
    "training_class_counts.csv",   # 학습 클래스 분포
    "triplets.csv",                # 유사도 삼중쌍
]

def download_osmo_pom():
    """Osmo Principal Odor Map 연구 데이터 다운로드"""
    print("=" * 60)
    print("1. Osmo POM 데이터 다운로드 (Science 2023)")
    print("=" * 60)
    
    osmo_dir = os.path.join(DATA_DIR, 'osmo_pom')
    os.makedirs(osmo_dir, exist_ok=True)
    
    downloaded = 0
    for fname in OSMO_FILES:
        url = f"{OSMO_BASE}/{fname}"
        local_name = urllib.parse.unquote(fname)
        local_path = os.path.join(osmo_dir, local_name)
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            print(f"  ✅ {local_name} (이미 존재)")
            downloaded += 1
            continue
        
        try:
            print(f"  ⬇️  {local_name} ...", end="", flush=True)
            urllib.request.urlretrieve(url, local_path)
            size_kb = os.path.getsize(local_path) / 1024
            print(f" {size_kb:.1f} KB")
            downloaded += 1
            time.sleep(0.5)
        except Exception as e:
            print(f" ❌ 실패: {e}")
    
    print(f"\n  결과: {downloaded}/{len(OSMO_FILES)} 파일 다운로드 완료")
    
    # 데이터 요약
    s1_path = os.path.join(osmo_dir, "Data S1.csv")
    if os.path.exists(s1_path):
        try:
            with open(s1_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = sum(1 for _ in reader)
            print(f"  📊 Data S1: {rows}개 분자, {len(header)}개 컬럼")
            # 첫 10개 컬럼명 출력
            print(f"  📋 컬럼 예시: {header[:10]}")
        except Exception as e:
            print(f"  ⚠️ Data S1 파싱 오류: {e}")
    
    return osmo_dir


# ============================================================
# 2. Pyrfume GoodScents+Leffingwell (pip 없이 직접 다운로드)
# ============================================================
PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"
PYRFUME_FILES = {
    "leffingwell": [
        f"{PYRFUME_BASE}/leffingwell/molecules.csv",
        f"{PYRFUME_BASE}/leffingwell/behavior.csv",
    ],
    "goodscents": [
        f"{PYRFUME_BASE}/goodscents/molecules.csv",
        f"{PYRFUME_BASE}/goodscents/behavior.csv",
    ],
    "dream": [
        f"{PYRFUME_BASE}/keller_2016/molecules.csv",
        f"{PYRFUME_BASE}/keller_2016/behavior.csv",
    ],
    "dravnieks": [
        f"{PYRFUME_BASE}/dravnieks_1985/molecules.csv",
        f"{PYRFUME_BASE}/dravnieks_1985/behavior.csv",
    ],
}

def download_pyrfume():
    """Pyrfume 데이터셋 직접 다운로드 (pip 없이)"""
    print("\n" + "=" * 60)
    print("2. Pyrfume 데이터 다운로드 (GS/LF/DREAM/Dravnieks)")
    print("=" * 60)
    
    pyrfume_dir = os.path.join(DATA_DIR, 'pyrfume')
    total = 0
    
    for dataset_name, urls in PYRFUME_FILES.items():
        ds_dir = os.path.join(pyrfume_dir, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)
        
        print(f"\n  [{dataset_name}]")
        for url in urls:
            fname = url.split("/")[-1]
            local_path = os.path.join(ds_dir, fname)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                with open(local_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        rows = sum(1 for _ in reader)
                        print(f"    ✅ {fname}: {rows}행 × {len(header)}열 (이미 존재)")
                    except:
                        print(f"    ✅ {fname} (이미 존재)")
                total += 1
                continue
            
            try:
                print(f"    ⬇️  {fname} ...", end="", flush=True)
                urllib.request.urlretrieve(url, local_path)
                size_kb = os.path.getsize(local_path) / 1024
                
                # 행/열 수 카운트
                with open(local_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        rows = sum(1 for _ in reader)
                        print(f" {rows}행 × {len(header)}열 ({size_kb:.1f} KB)")
                    except:
                        print(f" {size_kb:.1f} KB")
                total += 1
                time.sleep(0.3)
            except Exception as e:
                print(f" ❌ 실패: {e}")
    
    print(f"\n  결과: {total}/{sum(len(v) for v in PYRFUME_FILES.values())} 파일 다운로드 완료")
    return pyrfume_dir


# ============================================================
# 3. Fragrantica 데이터 (Kaggle 대체: 직접 크롤링 대신 구조화)
# ============================================================
def download_fragrantica_sample():
    """Fragrantica 데이터: Kaggle 다운로드 필요 안내 + 페이크-구조 생성"""
    print("\n" + "=" * 60)
    print("3. Fragrantica 향수 데이터")
    print("=" * 60)
    
    frag_dir = os.path.join(DATA_DIR, 'fragrantica')
    os.makedirs(frag_dir, exist_ok=True)
    
    # Kaggle 다운로드 URL 안내
    kaggle_url = "https://www.kaggle.com/datasets/miufana1/fragranticacom-fragrance-dataset"
    
    csv_path = os.path.join(frag_dir, 'fra_cleaned.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                rows = sum(1 for _ in reader)
                print(f"  ✅ fra_cleaned.csv: {rows}행 × {len(header)}열 (이미 존재)")
                return frag_dir
            except:
                pass
    
    print(f"  ⚠️ Kaggle에서 수동 다운로드 필요:")
    print(f"  📎 {kaggle_url}")
    print(f"  → fra_cleaned.csv 를 {frag_dir}/ 에 저장하세요")
    print()
    
    # 구조 설명 생성
    structure_info = {
        "source": "Fragrantica.com via Kaggle",
        "url": kaggle_url,
        "expected_columns": [
            "Name", "Brand", "Year", "Rating", "Votes",
            "Top Notes", "Middle Notes", "Base Notes",
            "Main Accords", "Gender", "Season"
        ],
        "expected_rows": "40,000+",
        "usage": "노트 조합 → 평점 예측 모델 학습",
        "instruction": "fra_cleaned.csv를 이 폴더에 다운로드하세요"
    }
    
    info_path = os.path.join(frag_dir, 'download_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(structure_info, f, indent=2, ensure_ascii=False)
    
    print(f"  📝 다운로드 안내 파일 생성: download_info.json")
    return frag_dir


# ============================================================
# 4. DREAM Olfaction Mixture Challenge (혼합물 데이터)
# ============================================================
DREAM_MIX_FILES = [
    # Snitz 2013 유사도 데이터
    f"{PYRFUME_BASE}/snitz_2013/molecules.csv",
    f"{PYRFUME_BASE}/snitz_2013/behavior.csv",
    # Bushdid 2014 혼합 구분가능성
    f"{PYRFUME_BASE}/bushdid_2014/molecules.csv",
    f"{PYRFUME_BASE}/bushdid_2014/behavior.csv",
    # Ravia 2020 혼합물 데이터  
    f"{PYRFUME_BASE}/ravia_2020/molecules.csv",
    f"{PYRFUME_BASE}/ravia_2020/behavior.csv",
]

def download_dream_mixture():
    """DREAM Olfaction Mixture 데이터 다운로드"""
    print("\n" + "=" * 60)
    print("4. DREAM Mixture Challenge 데이터")
    print("=" * 60)
    
    mix_dir = os.path.join(DATA_DIR, 'dream_mixture')
    total = 0
    
    datasets = {
        "snitz_2013": DREAM_MIX_FILES[0:2],
        "bushdid_2014": DREAM_MIX_FILES[2:4],
        "ravia_2020": DREAM_MIX_FILES[4:6],
    }
    
    for ds_name, urls in datasets.items():
        ds_dir = os.path.join(mix_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        
        print(f"\n  [{ds_name}]")
        for url in urls:
            fname = url.split("/")[-1]
            local_path = os.path.join(ds_dir, fname)
            
            if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                print(f"    ✅ {fname} (이미 존재)")
                total += 1
                continue
            
            try:
                print(f"    ⬇️  {fname} ...", end="", flush=True)
                urllib.request.urlretrieve(url, local_path)
                size_kb = os.path.getsize(local_path) / 1024
                print(f" {size_kb:.1f} KB")
                total += 1
                time.sleep(0.3)
            except Exception as e:
                print(f" ❌ 실패: {e}")
    
    print(f"\n  결과: {total}/{len(DREAM_MIX_FILES)} 파일 다운로드 완료")
    return mix_dir


# ============================================================
# 5. Osmo 향 분류 체계 (GitHub)
# ============================================================
def download_osmo_taxonomy():
    """Osmo 공식 향 분류 체계 다운로드"""
    print("\n" + "=" * 60)
    print("5. Osmo 향 분류 체계 (Taxonomy)")
    print("=" * 60)
    
    tax_dir = os.path.join(DATA_DIR, 'osmo_taxonomy')
    os.makedirs(tax_dir, exist_ok=True)
    
    tax_url = "https://raw.githubusercontent.com/osmoai/scent-taxonomy/main/taxonomy.json"
    local_path = os.path.join(tax_dir, "taxonomy.json")
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
        with open(local_path, 'r', encoding='utf-8') as f:
            tax = json.load(f)
        
        # 구조 분석
        if isinstance(tax, list):
            print(f"  ✅ taxonomy.json: {len(tax)}개 항목 (이미 존재)")
        elif isinstance(tax, dict):
            print(f"  ✅ taxonomy.json: {len(tax)}개 키 (이미 존재)")
        return tax_dir
    
    try:
        print(f"  ⬇️  taxonomy.json ...", end="", flush=True)
        urllib.request.urlretrieve(tax_url, local_path)
        size_kb = os.path.getsize(local_path) / 1024
        print(f" {size_kb:.1f} KB")
        
        with open(local_path, 'r', encoding='utf-8') as f:
            tax = json.load(f)
        if isinstance(tax, list):
            print(f"  📊 {len(tax)}개 향 카테고리")
        elif isinstance(tax, dict):
            print(f"  📊 {len(tax)}개 키")
    except Exception as e:
        print(f" ❌ 실패: {e}")
    
    return tax_dir


# ============================================================
# 6. 통합 데이터 분석 + 요약
# ============================================================
def analyze_collected_data():
    """수집된 데이터 전체 분석"""
    print("\n" + "=" * 60)
    print("📊 수집 데이터 종합 분석")
    print("=" * 60)
    
    total_molecules = 0
    total_files = 0
    total_size_mb = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total_size_mb += size / (1024 * 1024)
            total_files += 1
            
            if f.endswith('.csv') and 'molecule' in f.lower():
                try:
                    with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                        rows = sum(1 for _ in fh) - 1
                    total_molecules += rows
                except:
                    pass
    
    # Osmo Data S1 분자 수 추가
    s1_path = os.path.join(DATA_DIR, 'osmo_pom', 'Data S1.csv')
    if os.path.exists(s1_path):
        try:
            with open(s1_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                s1_rows = sum(1 for _ in reader)
            total_molecules += s1_rows
        except:
            pass
    
    print(f"""
  ┌─────────────────────────────────────┐
  │  총 파일:     {total_files:>6}개              │
  │  총 크기:     {total_size_mb:>6.1f} MB             │
  │  총 분자:    ~{total_molecules:>6}개              │
  └─────────────────────────────────────┘
    """)
    
    # 데이터 소스별 요약
    sources = {
        'osmo_pom': '🔬 Osmo POM (Science 2023)',
        'pyrfume': '🧪 Pyrfume (GS/LF/DREAM/Dravnieks)',
        'dream_mixture': '🌀 DREAM Mixture Challenge',
        'fragrantica': '⭐ Fragrantica (사용자 평점)',
        'osmo_taxonomy': '🏷️ Osmo 향 분류 체계',
    }
    
    for folder, label in sources.items():
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.exists(folder_path):
            n_files = sum(1 for _, _, files in os.walk(folder_path) for f in files)
            size = sum(os.path.getsize(os.path.join(r, f)) 
                      for r, _, files in os.walk(folder_path) 
                      for f in files) / 1024
            print(f"  {label}: {n_files}파일, {size:.1f} KB")
        else:
            print(f"  {label}: ❌ 미수집")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("🚀 Phase 4 — 90%+ 정확도 데이터 수집")
    print("=" * 60)
    start = time.time()
    
    # 1. Osmo POM
    download_osmo_pom()
    
    # 2. Pyrfume
    download_pyrfume()
    
    # 3. Fragrantica
    download_fragrantica_sample()
    
    # 4. DREAM Mixture
    download_dream_mixture()
    
    # 5. Osmo Taxonomy
    download_osmo_taxonomy()
    
    # 6. 분석
    analyze_collected_data()
    
    elapsed = time.time() - start
    print(f"\n⏱️ 전체 소요: {elapsed:.1f}초")
    print("✅ 데이터 수집 완료!")
