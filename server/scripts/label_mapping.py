# -*- coding: utf-8 -*-
"""
label_mapping.py — Co-occurrence 기반 소프트 라벨 매핑
=====================================================
DB에서 138개 라벨 간 동시 출현 통계를 분석하여,
매핑 안 되던 121개 라벨을 22차원 공간에 수학적으로 분배.

Usage:
    # 직접 실행: 매핑 테이블 생성 + 통계 출력
    python scripts/label_mapping.py

    # 임포트: 소프트 벡터 생성
    from scripts.label_mapping import descriptor_to_soft_target, ALL_LABELS
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from collections import defaultdict, Counter

# ================================================================
# 22차원 냄새 공간 (20 + fatty + waxy)
# ================================================================
ODOR_DIMENSIONS_22 = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',  # NEW: 독립 차원
]
N_DIM_22 = len(ODOR_DIMENSIONS_22)

# ================================================================
# 확실한 1:1 하드 매핑 (의미적으로 명확한 것들)
# ================================================================
HARD_MAPPING = {
    # sweet
    'sweet': 'sweet', 'caramellic': 'sweet', 'vanilla': 'sweet',
    'honey': 'sweet', 'chocolate': 'sweet', 'cocoa': 'sweet',
    'butterscotch': 'sweet', 'sugary': 'sweet', 'candy': 'sweet',
    # sour — V22: removed aldehydic (→fresh), cognac/rum/brandy (→warm)
    'sour': 'sour', 'winey': 'sour', 'fermented': 'sour',
    'sharp': 'sour', 'pungent': 'sour', 'acetic': 'sour',
    'cheesy': 'sour', 'vinegar': 'sour', 'acidic': 'sour',
    # woody
    'woody': 'woody', 'pine': 'woody', 'cedar': 'woody',
    'coumarinic': 'woody', 'sandalwood': 'woody', 'birch': 'woody',
    'teak': 'woody', 'oakwood': 'woody',
    # floral
    'floral': 'floral', 'rose': 'floral', 'jasmine': 'floral',
    'violet': 'floral', 'orris': 'floral', 'chamomile': 'floral',
    'muguet': 'floral', 'lily': 'floral', 'lavender': 'floral',
    'geranium': 'floral', 'magnolia': 'floral', 'narcissus': 'floral',
    # citrus
    'citrus': 'citrus', 'lemon': 'citrus', 'orange': 'citrus',
    'grapefruit': 'citrus', 'bergamot': 'citrus', 'lime': 'citrus',
    'mandarin': 'citrus', 'tangerine': 'citrus', 'yuzu': 'citrus',
    # spicy — V22: removed phenolic (→smoky), camphoreous (→fresh)
    'spicy': 'spicy', 'cinnamon': 'spicy', 'anisic': 'spicy',
    'onion': 'spicy', 'garlic': 'spicy', 'alliaceous': 'spicy',
    'horseradish': 'spicy', 'radish': 'spicy', 'pepper': 'spicy',
    'clove': 'spicy', 'cardamom': 'spicy', 'ginger': 'spicy',
    'cumin': 'spicy', 'nutmeg': 'spicy',
    # musk
    'musk': 'musk', 'animal': 'musk', 'catty': 'musk',
    'civet': 'musk', 'ambergris': 'musk', 'castoreum': 'musk',
    # fresh — V22: added aldehydic, camphoreous, ethereal
    'fresh': 'fresh', 'mint': 'fresh', 'ethereal': 'fresh',
    'cucumber': 'fresh', 'medicinal': 'fresh',
    'ketonic': 'fresh', 'cooling': 'fresh', 'mentholic': 'fresh',
    'aldehydic': 'fresh', 'camphoreous': 'fresh', 'clean': 'fresh',
    # green
    'green': 'green', 'leafy': 'green', 'grassy': 'green',
    'hay': 'green', 'tea': 'green', 'vegetable': 'green',
    'cabbage': 'green', 'tomato': 'green', 'galbanum': 'green',
    # warm — V22: added cognac, rum, brandy, alcoholic (warm spirits)
    'warm': 'warm', 'buttery': 'warm', 'creamy': 'warm',
    'milky': 'warm', 'dairy': 'warm', 'nutty': 'warm',
    'hazelnut': 'warm', 'almond': 'warm', 'bread': 'warm',
    'popcorn': 'warm', 'cognac': 'warm', 'rum': 'warm',
    'brandy': 'warm', 'alcoholic': 'warm', 'whiskey': 'warm',
    # fruity
    'fruity': 'fruity', 'apple': 'fruity', 'pear': 'fruity',
    'peach': 'fruity', 'banana': 'fruity', 'melon': 'fruity',
    'cherry': 'fruity', 'berry': 'fruity', 'grape': 'fruity',
    'pineapple': 'fruity', 'tropical': 'fruity', 'coconut': 'fruity',
    'apricot': 'fruity', 'plum': 'fruity', 'strawberry': 'fruity',
    'black currant': 'fruity', 'ripe': 'fruity', 'mango': 'fruity',
    'raspberry': 'fruity', 'blueberry': 'fruity', 'passion fruit': 'fruity',
    'lychee': 'fruity', 'guava': 'fruity',
    # smoky — V22: added phenolic, tarry
    'smoky': 'smoky', 'roasted': 'smoky', 'burnt': 'smoky',
    'coffee': 'smoky', 'tobacco': 'smoky', 'charred': 'smoky',
    'phenolic': 'smoky', 'tarry': 'smoky', 'ashy': 'smoky',
    # powdery
    'powdery': 'powdery', 'dry': 'powdery', 'talc': 'powdery',
    # aquatic
    'aquatic': 'aquatic', 'marine': 'aquatic', 'sea': 'aquatic',
    'oceanic': 'aquatic', 'watery': 'aquatic',
    # herbal — V22: added aromatic (herb family, not amber)
    'herbal': 'herbal', 'thyme': 'herbal', 'basil': 'herbal',
    'oregano': 'herbal', 'sage': 'herbal', 'rosemary': 'herbal',
    'aromatic': 'herbal', 'tarragon': 'herbal', 'dill': 'herbal',
    # amber — V22: removed aromatic (→herbal)
    'amber': 'amber', 'malty': 'amber',
    'balsamic': 'amber', 'resinous': 'amber', 'incense': 'amber',
    'labdanum': 'amber', 'benzoin': 'amber', 'tolu': 'amber',
    # leather
    'leathery': 'leather', 'leather': 'leather', 'musty': 'leather',
    'suede': 'leather', 'birch tar': 'leather',
    # earthy
    'earthy': 'earthy', 'mushroom': 'earthy', 'potato': 'earthy',
    'meaty': 'earthy', 'savory': 'earthy', 'beefy': 'earthy',
    'chicken': 'earthy', 'brothy': 'earthy', 'soil': 'earthy',
    'mossy': 'earthy', 'truffle': 'earthy', 'roots': 'earthy',
    # ozonic
    'ozonic': 'ozonic', 'ozone': 'ozonic',
    # metallic — V22: moved solvent, gasoline, petroleum here
    'metallic': 'metallic', 'sulfurous': 'metallic',
    'fishy': 'metallic', 'solvent': 'metallic',
    'gasoline': 'metallic', 'petroleum': 'metallic',
    'kerosene': 'metallic', 'paint': 'metallic',
    # fatty (독립 차원)
    'fatty': 'fatty', 'oily': 'fatty', 'sebaceous': 'fatty',
    'greasy': 'fatty', 'lard': 'fatty',
    # waxy (독립 차원) — V22: aldehydic also has waxy character but mapped to fresh primarily
    'waxy': 'waxy', 'paraffin': 'waxy', 'candle': 'waxy',
    # misc obvious
    'odorless': None,  # skip
}

# ================================================================
# Co-occurrence 소프트 매핑 계산
# ================================================================
def compute_cooccurrence_mapping(molecules):
    """
    DB 분자 데이터에서 co-occurrence 통계 기반 소프트 매핑 생성.
    
    Returns:
        soft_map: {label: {dim: weight, ...}, ...}
            각 미매핑 라벨이 22차원 중 어디에 얼마나 해당하는지
    """
    dim_set = set(ODOR_DIMENSIONS_22)
    
    # 1) 모든 고유 라벨 수집
    all_labels = set()
    for mol in molecules:
        labels = mol.get('odor_labels', [])
        for l in labels:
            all_labels.add(l.lower())
    
    # 2) 하드 매핑으로 커버 안 되는 라벨 식별
    unmapped_labels = set()
    for label in all_labels:
        if label not in HARD_MAPPING and label not in dim_set:
            unmapped_labels.add(label)
    
    # 3) 각 미매핑 라벨의 co-occurrence 계산
    cooccur = defaultdict(lambda: np.zeros(N_DIM_22, dtype=np.float64))
    cooccur_count = Counter()
    
    for mol in molecules:
        labels = [l.lower() for l in mol.get('odor_labels', [])]
        if not labels:
            continue
        
        # 이 분자가 가진 22차원 라벨들
        dim_hits = np.zeros(N_DIM_22, dtype=np.float64)
        for l in labels:
            mapped = HARD_MAPPING.get(l)
            if mapped and mapped in ODOR_DIMENSIONS_22:
                idx = ODOR_DIMENSIONS_22.index(mapped)
                dim_hits[idx] = 1.0
            elif l in dim_set:
                idx = ODOR_DIMENSIONS_22.index(l)
                dim_hits[idx] = 1.0
        
        # 미매핑 라벨이 이 분자에 있으면, 동시에 나타난 22차원 라벨 기록
        for l in labels:
            if l in unmapped_labels:
                cooccur[l] += dim_hits
                cooccur_count[l] += 1
    
    # 4) 확률로 정규화 → 소프트 매핑
    soft_map = {}
    for label in unmapped_labels:
        total = cooccur_count[label]
        if total == 0:
            continue
        vec = cooccur[label] / total  # 확률 벡터
        # 너무 희소한 건 제거 (5% 미만 co-occurrence)
        vec[vec < 0.05] = 0
        if vec.sum() > 0:
            vec = vec / vec.sum()  # 재정규화
            soft_map[label] = {ODOR_DIMENSIONS_22[i]: float(vec[i]) 
                              for i in range(N_DIM_22) if vec[i] > 0}
    
    return soft_map, unmapped_labels, all_labels


# ================================================================
# 통합 라벨 → 22차원 소프트 벡터 변환
# ================================================================
_cached_soft_map = None

def get_soft_map(molecules=None):
    """소프트 매핑 로드 (캐시됨). molecules가 없으면 DB에서 로드."""
    global _cached_soft_map
    if _cached_soft_map is not None:
        return _cached_soft_map
    
    if molecules is None:
        import database as db
        molecules = db.get_all_molecules()
    
    soft_map, _, _ = compute_cooccurrence_mapping(molecules)
    _cached_soft_map = soft_map
    return soft_map


def descriptor_to_soft_target(odor_labels, soft_map=None):
    """DB 라벨 리스트 → 22d soft target vector.
    
    Hard mapping + co-occurrence soft mapping 결합.
    V22: sum-normalize (intensity 보존) instead of max-normalize.
    """
    vec = np.zeros(N_DIM_22, dtype=np.float32)
    if not odor_labels:
        return vec
    
    if soft_map is None:
        soft_map = get_soft_map()
    
    for label in odor_labels:
        label_lower = label.lower()
        
        # 1) Hard mapping (확실한 1:1)
        if label_lower in HARD_MAPPING:
            mapped = HARD_MAPPING[label_lower]
            if mapped is None:  # 'odorless' → skip
                continue
            if mapped in ODOR_DIMENSIONS_22:
                idx = ODOR_DIMENSIONS_22.index(mapped)
                vec[idx] += 1.0
                continue
        
        # 2) 직접 차원 이름
        if label_lower in ODOR_DIMENSIONS_22:
            idx = ODOR_DIMENSIONS_22.index(label_lower)
            vec[idx] += 1.0
            continue
        
        # 3) Co-occurrence 소프트 매핑
        if label_lower in soft_map:
            for dim_name, weight in soft_map[label_lower].items():
                idx = ODOR_DIMENSIONS_22.index(dim_name)
                vec[idx] += weight
            continue
        
        # 4) 완전 미지 라벨 → 무시 (이제 거의 없음)
    
    # V22: L1 normalize → 강도 정보 보존
    # "fruity, sweet, warm" (3 labels) → [0.33, 0.33, 0.33] 
    # "fruity" (1 label) → [1.0]
    # This preserves relative proportions unlike max-norm
    s = vec.sum()
    if s > 0:
        vec = vec / s
    
    return vec


def descriptor_to_138d_target(odor_labels, all_labels_list):
    """DB 라벨 리스트 → 138d 이진 벡터 (Multi-task aux head용)"""
    vec = np.zeros(len(all_labels_list), dtype=np.float32)
    if not odor_labels:
        return vec
    for label in odor_labels:
        label_lower = label.lower()
        if label_lower in all_labels_list:
            idx = all_labels_list.index(label_lower)
            vec[idx] = 1.0
    return vec


# ================================================================
# ALL_LABELS: 138개 전체 라벨 (aux head용)
# ================================================================
def build_all_labels(molecules):
    """DB에서 고유 라벨 리스트 생성 (빈도 순 정렬)"""
    label_counts = Counter()
    for mol in molecules:
        for l in mol.get('odor_labels', []):
            label_counts[l.lower()] += 1
    # 빈도 순 정렬
    return [label for label, _ in label_counts.most_common()]


# ================================================================
# 직접 실행: 통계 출력
# ================================================================
if __name__ == '__main__':
    import database as db
    
    print("=" * 60)
    print("  Co-occurrence Soft Label Mapping")
    print("=" * 60)
    
    molecules = db.get_all_molecules()
    has_labels = [m for m in molecules if m.get('odor_labels') and m['odor_labels'] != ['odorless']]
    
    soft_map, unmapped, all_labels = compute_cooccurrence_mapping(has_labels)
    all_labels_list = build_all_labels(has_labels)
    
    # 통계
    hard_covered = sum(1 for l in all_labels if l in HARD_MAPPING or l in set(ODOR_DIMENSIONS_22))
    soft_covered = len(soft_map)
    still_unmapped = len(all_labels) - hard_covered - soft_covered
    
    print(f"\n  전체 고유 라벨: {len(all_labels)}")
    print(f"  Hard mapping:   {hard_covered} ({hard_covered/len(all_labels)*100:.0f}%)")
    print(f"  Soft mapping:   {soft_covered} ({soft_covered/len(all_labels)*100:.0f}%)")
    print(f"  미매핑(남음):   {still_unmapped} ({still_unmapped/len(all_labels)*100:.0f}%)")
    
    # 손실 계산
    total_label_instances = sum(len(m.get('odor_labels', [])) for m in has_labels)
    
    mapped_instances = 0
    unmapped_instances = 0
    for mol in has_labels:
        for l in mol.get('odor_labels', []):
            l_lower = l.lower()
            if l_lower in HARD_MAPPING or l_lower in set(ODOR_DIMENSIONS_22) or l_lower in soft_map:
                mapped_instances += 1
            else:
                unmapped_instances += 1
    
    print(f"\n  전체 라벨 인스턴스: {total_label_instances}")
    print(f"  매핑됨:           {mapped_instances} ({mapped_instances/total_label_instances*100:.1f}%)")
    print(f"  버려짐:           {unmapped_instances} ({unmapped_instances/total_label_instances*100:.1f}%)")
    
    # 소프트 매핑 예시
    print(f"\n  소프트 매핑 예시 (상위 15개):")
    sorted_soft = sorted(soft_map.items(), key=lambda x: -max(x[1].values()))
    for label, mapping in sorted_soft[:15]:
        dims = ', '.join(f"{d}:{w:.2f}" for d, w in sorted(mapping.items(), key=lambda x: -x[1])[:4])
        print(f"    {label:>15s} → {dims}")
    
    # 비교: 기존 vs 새 매핑
    print(f"\n  {'='*60}")
    print(f"  데이터 활용률 비교:")
    print(f"    기존 (hard 1:1, 20d):  ~40% 활용")
    print(f"    신규 (soft+22d):       {mapped_instances/total_label_instances*100:.1f}% 활용")
    print(f"  {'='*60}")
