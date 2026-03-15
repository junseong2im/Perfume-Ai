"""
향수/향 데이터 대규모 수집기
==============================
인터넷에서 사용 가능한 오픈소스 향 데이터를 수집하여 학습 데이터로 변환

데이터 소스:
  1. Pyrfume/Leffingwell — 3,500+ 분자, SMILES, 113 향 차원 (학술)
  2. Pyrfume/GoodScents — 4,600+ 분자, SMILES, 향 카테고리
  3. Pyrfume/DREAM — 500 분자, 21 향 차원 (퍼셉션 데이터)  
  4. Famous Perfumes DB — 200+ 향수, 노트/어코드 (이미 보유)
"""

import os, json, csv, io, sys
import urllib.request
import urllib.error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'collected')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"


def safe_float(val, default=0.0):
    """NA, '', None 안전 처리"""
    if val is None or val == '' or str(val).strip().upper() in ('NA', 'NAN', 'N/A', 'NONE'):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    """NA, '', None 안전 처리"""
    return int(safe_float(val, default))


def download_csv(url, label=""):
    """URL에서 CSV 다운로드 → 파싱"""
    print(f"  📥 [{label}] {url.split('/')[-1]}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read().decode('utf-8', errors='replace')
        reader = csv.DictReader(io.StringIO(raw))
        rows = list(reader)
        print(f"     → {len(rows)} rows")
        return rows
    except Exception as e:
        print(f"     ⚠ 실패: {e}")
        return []


def download_raw(url, label=""):
    """URL에서 원본 텍스트 다운로드"""
    print(f"  📥 [{label}] {url.split('/')[-1]}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"     ⚠ 실패: {e}")
        return ""


# ================================================================
# 1. Leffingwell — 최대 향 데이터셋 (학술)
# ================================================================
def collect_leffingwell():
    """
    Leffingwell Flavor Base:
    - molecules.csv: CID, SMILES, 이름
    - behavior.csv: CID → 113 향 차원 (binary)
    """
    print("\n" + "=" * 60)
    print("  [1/4] Leffingwell Flavor Base (3,500+ 분자)")
    print("=" * 60)

    mols = download_csv(f"{PYRFUME_BASE}/leffingwell/molecules.csv", "Leffingwell")
    behav = download_csv(f"{PYRFUME_BASE}/leffingwell/behavior.csv", "Leffingwell")

    if not mols or not behav:
        return []

    # molecules 인덱싱
    mol_map = {}
    for m in mols:
        cid = m.get('CID', '')
        mol_map[cid] = {
            'cid': cid,
            'smiles': m.get('IsomericSMILES', ''),
            'name': m.get('name', ''),
            'iupac': m.get('IUPACName', ''),
            'mw': safe_float(m.get('MolecularWeight', 0)),
        }

    # behavior (향 라벨) 합치기
    results = []
    # behavior의 첫 행에서 향 차원 목록 추출
    if behav:
        odor_dims = [k for k in behav[0].keys() if k != 'Stimulus']

    for b in behav:
        cid = b.get('Stimulus', '')
        mol = mol_map.get(cid, {})
        smiles = mol.get('smiles', '')
        if not smiles:
            continue

        # 활성 향 라벨 추출
        active_odors = []
        odor_vector = {}
        for dim in odor_dims:
            val = safe_int(b.get(dim, 0))
            if val > 0:
                active_odors.append(dim)
            odor_vector[dim] = val

        if active_odors:
            results.append({
                'source': 'leffingwell',
                'cid': cid,
                'smiles': smiles,
                'name': mol.get('name', ''),
                'iupac': mol.get('iupac', ''),
                'mw': mol.get('mw', 0),
                'odor_labels': active_odors,
                'odor_vector_113d': odor_vector,
                'n_labels': len(active_odors),
            })

    print(f"  ✅ Leffingwell: {len(results)}개 분자 (SMILES + 113d 향 벡터)")
    return results


# ================================================================
# 2. GoodScents — 추가 향 데이터
# ================================================================
def collect_goodscents():
    """
    GoodScents Company via Pyrfume:
    - molecules.csv: CID, SMILES
    - behavior.csv: 향 라벨
    """
    print("\n" + "=" * 60)
    print("  [2/4] GoodScents (4,600+ 분자)")
    print("=" * 60)

    mols = download_csv(f"{PYRFUME_BASE}/goodscents/molecules.csv", "GoodScents")
    behav = download_csv(f"{PYRFUME_BASE}/goodscents/behavior.csv", "GoodScents")

    if not mols or not behav:
        return []

    mol_map = {}
    for m in mols:
        cid = m.get('CID', '')
        mol_map[cid] = {
            'smiles': m.get('IsomericSMILES', ''),
            'name': m.get('name', ''),
            'mw': safe_float(m.get('MolecularWeight', 0)),
        }

    results = []
    if behav:
        odor_dims = [k for k in behav[0].keys() if k != 'Stimulus']

    for b in behav:
        cid = b.get('Stimulus', '')
        mol = mol_map.get(cid, {})
        smiles = mol.get('smiles', '')
        if not smiles:
            continue

        active_odors = []
        odor_vector = {}
        for dim in odor_dims:
            val = safe_int(b.get(dim, 0))
            if val > 0:
                active_odors.append(dim)
            odor_vector[dim] = val

        if active_odors:
            results.append({
                'source': 'goodscents',
                'cid': cid,
                'smiles': smiles,
                'name': mol.get('name', ''),
                'mw': mol.get('mw', 0),
                'odor_labels': active_odors,
                'odor_vector': odor_vector,
                'n_labels': len(active_odors),
            })

    print(f"  ✅ GoodScents: {len(results)}개 분자")
    return results


# ================================================================
# 3. DREAM Olfaction Challenge — 감각 데이터 (21d)
# ================================================================
def collect_dream():
    """
    DREAM Olfaction Challenge:
    - molecules.csv: CID, SMILES
    - behavior.csv: 21 지각 차원 (연속값, 0~100)
    """
    print("\n" + "=" * 60)
    print("  [3/4] DREAM Olfaction Challenge (500 분자, 21d 연속값)")
    print("=" * 60)

    mols = download_csv(f"{PYRFUME_BASE}/keller_2016/molecules.csv", "DREAM")
    # Try multiple behavior paths
    behav = download_csv(f"{PYRFUME_BASE}/keller_2016/behavior.csv", "DREAM")
    if not behav:
        behav = download_csv(f"{PYRFUME_BASE}/keller_2016/behavior_1/behavior.csv", "DREAM-alt")

    if not behav:
        # Another alt path
        behav = download_csv(f"{PYRFUME_BASE}/keller_2016/behavior_2/behavior.csv", "DREAM-alt2")

    if not mols or not behav:
        return []

    mol_map = {}
    for m in mols:
        cid = m.get('CID', '')
        mol_map[cid] = {
            'smiles': m.get('IsomericSMILES', ''),
            'name': m.get('name', ''),
            'mw': safe_float(m.get('MolecularWeight', 0)),
        }

    results = []
    if behav:
        perception_dims = [k for k in behav[0].keys() 
                          if k not in ('Stimulus', 'Subject', 'Dilution', 'Replicate')]

    for b in behav:
        cid = b.get('Stimulus', '')
        mol = mol_map.get(cid, {})
        smiles = mol.get('smiles', '')
        if not smiles:
            continue

        perception = {}
        for dim in perception_dims:
            try:
                val = safe_float(b.get(dim, 0))
                perception[dim] = val
            except:
                pass

        if perception:
            results.append({
                'source': 'dream_keller2016',
                'cid': cid,
                'smiles': smiles,
                'name': mol.get('name', ''),
                'mw': mol.get('mw', 0),
                'perception_vector': perception,
                'n_dims': len(perception),
            })

    # 동일 분자는 평균
    avg_map = {}
    for r in results:
        cid = r['cid']
        if cid not in avg_map:
            avg_map[cid] = {'count': 0, 'sums': {}, 'meta': r}
        avg_map[cid]['count'] += 1
        for dim, val in r['perception_vector'].items():
            avg_map[cid]['sums'][dim] = avg_map[cid]['sums'].get(dim, 0) + val

    final = []
    for cid, data in avg_map.items():
        entry = data['meta'].copy()
        entry['perception_vector'] = {
            d: round(v / data['count'], 2) 
            for d, v in data['sums'].items()
        }
        final.append(entry)

    print(f"  ✅ DREAM: {len(final)}개 분자 (21d 연속 지각 벡터)")
    return final


# ================================================================
# 4. 추가: arctander/ifra 데이터
# ================================================================
def collect_arctander():
    """Arctander — 향료 백과사전"""
    print("\n" + "=" * 60)
    print("  [4/4] Arctander + IFRA 데이터")
    print("=" * 60)

    mols = download_csv(f"{PYRFUME_BASE}/arctander_1960/molecules.csv", "Arctander")
    behav = download_csv(f"{PYRFUME_BASE}/arctander_1960/behavior.csv", "Arctander")

    if not mols or not behav:
        return []

    mol_map = {}
    for m in mols:
        cid = m.get('CID', '')
        mol_map[cid] = {
            'smiles': m.get('IsomericSMILES', ''),
            'name': m.get('name', ''),
            'mw': safe_float(m.get('MolecularWeight', 0)),
        }

    results = []
    if behav:
        odor_dims = [k for k in behav[0].keys() if k != 'Stimulus']

    for b in behav:
        cid = b.get('Stimulus', '')
        mol = mol_map.get(cid, {})
        smiles = mol.get('smiles', '')
        if not smiles:
            continue

        active_odors = []
        for dim in odor_dims:
            val = b.get(dim, 0)
            ival = safe_int(val)
            if ival > 0:
                active_odors.append(dim)

        if active_odors:
            results.append({
                'source': 'arctander',
                'cid': cid,
                'smiles': smiles,
                'name': mol.get('name', ''),
                'mw': mol.get('mw', 0),
                'odor_labels': active_odors,
                'n_labels': len(active_odors),
            })

    print(f"  ✅ Arctander: {len(results)}개")
    return results


def collect_extra_sources():
    """추가 오픈 데이터셋: Dravnieks, FlavorNet, AromaDB, SuperScent"""
    print("\n" + "=" * 60)
    print("  [EXTRA] 추가 데이터셋")
    print("=" * 60)
    
    extra = []
    
    # Dravnieks 1985 — 144 odors, 146 perceptual dims
    for ds_name in ['dravnieks_1985', 'flavornet', 'aromadb', 'superscent', 'sigma_2014']:
        mols = download_csv(f"{PYRFUME_BASE}/{ds_name}/molecules.csv", ds_name)
        behav = download_csv(f"{PYRFUME_BASE}/{ds_name}/behavior.csv", ds_name)
        
        if not mols or not behav:
            continue
        
        mol_map = {}
        for m in mols:
            cid = m.get('CID', '')
            mol_map[cid] = {
                'smiles': m.get('IsomericSMILES', ''),
                'name': m.get('name', ''),
            }
        
        if not behav:
            continue
        dims = [k for k in behav[0].keys() if k != 'Stimulus']
        
        for b in behav:
            cid = b.get('Stimulus', '')
            mol = mol_map.get(cid, {})
            smiles = mol.get('smiles', '')
            if not smiles:
                continue
            
            active = []
            for dim in dims:
                val = safe_float(b.get(dim, 0))
                if val > 0:
                    active.append(dim)
            
            if active:
                extra.append({
                    'source': ds_name,
                    'cid': cid,
                    'smiles': smiles,
                    'name': mol.get('name', ''),
                    'odor_labels': active,
                    'n_labels': len(active),
                })
    
    print(f"  ✅ Extra: {len(extra)}개")
    return extra


# ================================================================
# 통합 및 저장
# ================================================================
def merge_and_save(leff, gs, dream, arct, extra=None):
    """전체 데이터 병합 + 중복 제거 + SMILES 기반 인덱싱"""
    print("\n" + "=" * 60)
    print("  📊 데이터 병합 및 저장")
    print("=" * 60)

    # SMILES 기반 병합 (중복 제거)
    smiles_index = {}
    
    all_sources = [
        (leff, 'leffingwell'),
        (gs, 'goodscents'),
        (arct, 'arctander'),
    ]
    if extra:
        all_sources.append((extra, None))  # source is already in each entry

    for source_list, source_name in all_sources:
        for entry in source_list:
            smi = entry.get('smiles', '')
            if not smi:
                continue
            if smi not in smiles_index:
                smiles_index[smi] = {
                    'smiles': smi,
                    'name': entry.get('name', ''),
                    'sources': [],
                    'odor_labels': set(),
                    'has_113d': False,
                    'has_21d': False,
                }
            src = source_name or entry.get('source', 'unknown')
            smiles_index[smi]['sources'].append(src)
            smiles_index[smi]['odor_labels'].update(entry.get('odor_labels', []))
            if 'odor_vector_113d' in entry:
                smiles_index[smi]['odor_vector_113d'] = entry['odor_vector_113d']
                smiles_index[smi]['has_113d'] = True

    # DREAM separately (has perception_vector)
    for entry in dream:
        smi = entry.get('smiles', '')
        if not smi:
            continue
        if smi not in smiles_index:
            smiles_index[smi] = {
                'smiles': smi,
                'name': entry.get('name', ''),
                'sources': [],
                'odor_labels': set(),
                'has_113d': False,
                'has_21d': False,
            }
        smiles_index[smi]['sources'].append('dream')
        if 'perception_vector' in entry:
            smiles_index[smi]['perception_21d'] = entry['perception_vector']
            smiles_index[smi]['has_21d'] = True

    # set → list 변환
    merged = []
    for smi, data in smiles_index.items():
        data['odor_labels'] = sorted(list(data['odor_labels']))
        data['n_labels'] = len(data['odor_labels'])
        data['n_sources'] = len(set(data['sources']))
        merged.append(data)

    # 라벨 수 기준 정렬 (많은 정보가 있는 것부터)
    merged.sort(key=lambda x: -x['n_labels'])

    # 통계
    total = len(merged)
    with_113d = sum(1 for m in merged if m.get('has_113d'))
    with_21d = sum(1 for m in merged if m.get('has_21d'))
    multi_source = sum(1 for m in merged if m['n_sources'] > 1)

    # 전체 향 라벨 수집
    all_labels = set()
    for m in merged:
        all_labels.update(m['odor_labels'])

    print(f"  총 고유 분자: {total}")
    print(f"  113d 벡터 보유: {with_113d}")
    print(f"  21d 지각 벡터 보유: {with_21d}")
    print(f"  복수 소스 확인: {multi_source}")
    print(f"  수집 향 차원 수: {len(all_labels)}")

    # JSON 저장
    # sources set 변환
    for m in merged:
        m['sources'] = list(set(m['sources']))

    output_path = os.path.join(OUTPUT_DIR, 'collected_odor_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"\n  💾 저장: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    # 향 차원 인덱스 저장
    label_path = os.path.join(OUTPUT_DIR, 'odor_dimensions.json')
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_labels)), f, indent=2)
    print(f"  💾 향 차원 목록: {label_path} ({len(all_labels)}개)")

    # 통계 저장
    stats = {
        'total_molecules': total,
        'with_113d_vector': with_113d,
        'with_21d_perception': with_21d,
        'multi_source_confirmed': multi_source,
        'total_odor_dimensions': len(all_labels),
        'sources': {
            'leffingwell': len(leff),
            'goodscents': len(gs),
            'dream': len(dream),
            'arctander': len(arct),
            'extra': len(extra) if extra else 0,
        }
    }
    stats_path = os.path.join(OUTPUT_DIR, 'collection_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"  💾 통계: {stats_path}")

    return stats


# ================================================================
# 메인
# ================================================================
if __name__ == '__main__':
    print("🌐 향 데이터 대규모 수집기 시작")
    print("   Pyrfume 오픈 데이터에서 SMILES + 향 라벨 다운로드\n")

    leff = collect_leffingwell()
    gs = collect_goodscents()
    dream = collect_dream()
    arct = collect_arctander()
    extra = collect_extra_sources()

    stats = merge_and_save(leff, gs, dream, arct, extra)

    print("\n" + "=" * 60)
    print("  🎉 데이터 수집 완료!")
    print("=" * 60)
    print(f"  총 고유 분자: {stats['total_molecules']}")
    print(f"  113d 벡터: {stats['with_113d_vector']}")
    print(f"  21d 지각: {stats['with_21d_perception']}")
    print(f"  향 차원: {stats['total_odor_dimensions']}개")
    print(f"\n  소스별:")
    for src, cnt in stats['sources'].items():
        print(f"    {src}: {cnt}")
