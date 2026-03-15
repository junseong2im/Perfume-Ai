"""
합성 혼합 학습 데이터 대규모 생성기
====================================
477개 원료 × 과학 시뮬레이터 → 수만 건의 혼합 학습 데이터

생성 전략:
  1. 카테고리 커버리지: 모든 향 카테고리 조합 커버
  2. 비율 다양성: 동일 원료도 비율 변경 시 다른 결과
  3. 시간축 포함: t=0h, 1h, 4h, 8h 각각 별도 데이터
  4. 노트 밸런스: 탑+미들+베이스 균형 조합 우선

출력 형식 (각 행):
  입력: [원료1_벡터, 비율1, 원료2_벡터, 비율2, ...]
  출력: [perceived_22d_vector]  (시뮬레이터가 예측한 최종 향)
"""

import json
import os
import sys
import time
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mixture_simulator import MixtureSimulatorV2


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_mixtures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ingredients():
    """원료 DB + SMILES 로드"""
    ingredients = []
    smiles_map = {}
    
    for p in ['data/ingredients.json', '../data/ingredients.json',
              os.path.join(os.path.dirname(__file__), '..', 'data', 'ingredients.json')]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                ingredients = json.load(f)
            break
    
    for p in ['data/ingredient_smiles.json', '../data/ingredient_smiles.json',
              os.path.join(os.path.dirname(__file__), '..', 'data', 'ingredient_smiles.json')]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                smiles_map = json.load(f)
            break
    
    return ingredients, smiles_map


def get_ingredient_vector(ing, sim, smiles_map):
    """원료 → 22d 벡터"""
    return sim.get_odor_vector(ing)


def generate_all_mixtures():
    """대규모 혼합 데이터 생성"""
    print("🧬 합성 혼합 학습 데이터 생성기")
    print("=" * 60)
    
    ingredients, smiles_map = load_ingredients()
    sim = MixtureSimulatorV2()
    
    print(f"  원료: {len(ingredients)}개")
    print(f"  SMILES: {len(smiles_map)}개")
    
    # 1) 모든 원료의 22d 벡터 사전 계산
    print("\n[1/4] 원료 벡터 사전 계산...")
    ing_vectors = {}
    ing_mw = {}
    ing_vol = {}
    
    for ing in ingredients:
        iid = ing.get('id', '')
        vec = get_ingredient_vector(ing, sim, smiles_map)
        if np.sum(vec) > 0.01:  # 유효한 벡터만
            ing_vectors[iid] = vec
            ing_mw[iid] = sim._get_mw_from_data(iid)
            ing_vol[iid] = ing.get('volatility', 5)
    
    valid_ids = list(ing_vectors.keys())
    print(f"  유효 원료: {len(valid_ids)}개 (벡터 > 0)")
    
    # 카테고리별 분류
    cat_groups = {}
    for ing in ingredients:
        iid = ing.get('id', '')
        if iid in ing_vectors:
            cat = ing.get('category', 'unknown')
            cat_groups.setdefault(cat, []).append(iid)
    
    print(f"  카테고리: {len(cat_groups)}개")
    for cat, ids in sorted(cat_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:>12}: {len(ids)}개")
    
    # 노트별 분류
    note_groups = {'top': [], 'middle': [], 'base': []}
    for ing in ingredients:
        iid = ing.get('id', '')
        if iid in ing_vectors:
            note = ing.get('note_type', 'middle')
            note_groups.setdefault(note, []).append(iid)
    
    # 2) 비율 패턴 정의 (하드코딩 아닌 체계적 커버리지)
    # 2~10개 원료 조합, 다양한 비율
    RATIO_TEMPLATES = [
        # 2개 원료
        [70, 30], [50, 50], [80, 20], [60, 40], [90, 10],
        # 3개 원료
        [50, 30, 20], [40, 40, 20], [60, 25, 15], [33, 33, 34],
        # 4개 원료
        [40, 25, 20, 15], [30, 30, 25, 15], [50, 20, 20, 10],
        # 5개 원료  
        [30, 25, 20, 15, 10], [35, 20, 20, 15, 10], [20, 20, 20, 20, 20],
        # 6개 원료
        [25, 20, 18, 15, 12, 10], [30, 20, 15, 15, 12, 8],
        # 8개 원료 (실제 향수 수준)
        [20, 18, 15, 12, 10, 10, 8, 7],
        [25, 15, 13, 12, 10, 10, 8, 7],
    ]
    
    # 3) 혼합 데이터 생성
    print(f"\n[2/4] 혼합 시뮬레이션 시작...")
    
    dataset = []
    categories = list(cat_groups.keys())
    start_time = time.time()
    
    # === 전략 A: 카테고리 간 조합 (체계적) ===
    print("  전략 A: 카테고리 간 조합...")
    cat_pairs = list(itertools.combinations(categories, 2))
    
    for cat_a, cat_b in cat_pairs:
        ids_a = cat_groups[cat_a]
        ids_b = cat_groups[cat_b]
        
        # 각 카테고리에서 대표 원료 선택 (최대 3개)
        sample_a = ids_a[:min(3, len(ids_a))]
        sample_b = ids_b[:min(3, len(ids_b))]
        
        for ia in sample_a:
            for ib in sample_b:
                for ratios in [[70, 30], [50, 50], [30, 70]]:
                    comps = _build_components(
                        [ia, ib], ratios, ing_vectors, ing_mw, ing_vol
                    )
                    for t in [0, 1, 4]:
                        result = sim.simulate_mixture(comps, time_hours=t)
                        dataset.append(_to_training_row(
                            comps, result, t
                        ))
    
    print(f"    → {len(dataset)}개 생성")
    
    # === 전략 B: 노트 피라미드 조합 (탑+미들+베이스) ===
    print("  전략 B: 노트 피라미드 조합...")
    count_before = len(dataset)
    
    tops = note_groups.get('top', [])
    mids = note_groups.get('middle', [])
    bases = note_groups.get('base', [])
    
    # 탑×미들×베이스 조합
    for t_id in tops[:min(15, len(tops))]:
        for m_id in mids[:min(15, len(mids))]:
            for b_id in bases[:min(15, len(bases))]:
                for ratios in [[20, 30, 50], [15, 35, 50], [25, 35, 40], [10, 40, 50]]:
                    comps = _build_components(
                        [t_id, m_id, b_id], ratios, ing_vectors, ing_mw, ing_vol
                    )
                    result = sim.simulate_mixture(comps, time_hours=0)
                    dataset.append(_to_training_row(comps, result, 0))
    
    print(f"    → +{len(dataset) - count_before}개 (총 {len(dataset)})")
    
    # === 전략 C: 실제 향수 스타일 조합 (5~8개 원료) ===
    print("  전략 C: 현실적 향수 조합 (5~8개)...")
    count_before = len(dataset)
    
    # 결정론적 조합: 각 카테고리에서 체계적으로 선택
    # 탑+미들+베이스에서 각 2+2+2 = 6개 조합
    import itertools as it
    
    top_sample = tops[:min(8, len(tops))]
    mid_sample = mids[:min(8, len(mids))]
    base_sample = bases[:min(8, len(bases))]
    
    combo_count = 0
    for t1, t2 in it.combinations(top_sample, 2) if len(top_sample) >= 2 else [(top_sample[0],)] * 1:
        for m1, m2 in it.combinations(mid_sample, 2) if len(mid_sample) >= 2 else [(mid_sample[0],)] * 1:
            for b1, b2 in it.combinations(base_sample, 2) if len(base_sample) >= 2 else [(base_sample[0],)] * 1:
                if combo_count >= 500:  # 최대 500개 조합
                    break
                
                if isinstance(t1, tuple):
                    selected_6 = list(t1) + list(m1) + list(b1)
                else:
                    selected_6 = [t1, t2, m1, m2, b1, b2]
                
                for ratios in [[15, 10, 25, 20, 20, 10], [10, 15, 20, 25, 15, 15]]:
                    comps = _build_components(
                        selected_6[:len(ratios)], ratios, ing_vectors, ing_mw, ing_vol
                    )
                    if comps:
                        for t in [0, 4]:
                            result = sim.simulate_mixture(comps, time_hours=t)
                            dataset.append(_to_training_row(comps, result, t))
                combo_count += 1
            if combo_count >= 500:
                break
        if combo_count >= 500:
            break
    
    print(f"    → +{len(dataset) - count_before}개 (총 {len(dataset)})")
    
    # === 전략 D: 비율 변형 (같은 원료, 다른 비율) ===
    print("  전략 D: 비율 변형...")
    count_before = len(dataset)
    
    # 인기 조합 5개를 비율 변형 (결정론적 linspace)
    popular_combos = [
        ['bergamot', 'lavender', 'sandalwood', 'musk'],
        ['rose', 'jasmine', 'cedarwood', 'vanilla'],
        ['lemon', 'bergamot', 'vetiver', 'amber'],
        ['black_pepper', 'cardamom', 'sandalwood', 'oud'],
        ['neroli', 'iris', 'tonka_bean', 'benzoin'],
    ]
    
    for combo in popular_combos:
        valid_combo = [iid for iid in combo if iid in ing_vectors]
        if len(valid_combo) < 2:
            continue
        
        n = len(valid_combo)
        # 20가지 결정론적 비율: 첫 번째 원료 10~90% 순차적 변화
        for main_pct in np.linspace(10, 90, 20):
            remainder = 100 - main_pct
            # 나머지 원료에 균등 분배
            sub_pct = remainder / max(n - 1, 1)
            ratios = [round(main_pct, 1)] + [round(sub_pct, 1)] * (n - 1)
            
            comps = _build_components(valid_combo, ratios, ing_vectors, ing_mw, ing_vol)
            result = sim.simulate_mixture(comps, time_hours=0)
            dataset.append(_to_training_row(comps, result, 0))
    
    print(f"    → +{len(dataset) - count_before}개 (총 {len(dataset)})")
    
    elapsed = time.time() - start_time
    
    # 4) 저장
    print(f"\n[3/4] 데이터 저장...")
    
    output = {
        'metadata': {
            'total_samples': len(dataset),
            'n_ingredients': len(valid_ids),
            'n_categories': len(cat_groups),
            'generation_time_sec': round(elapsed, 1),
            'simulator': 'MixtureSimulatorV2 (data-driven, Leffingwell 3522)',
            'strategies': ['category_pairs', 'note_pyramid', 'realistic_perfume', 'ratio_variation'],
            'odor_dims': sim.ODOR_DIMS_22,
        },
        'samples': dataset,
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'mixture_training_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  💾 {output_path}")
    print(f"     {file_size:.1f} MB / {len(dataset)}개 샘플")
    
    # 통계 분석
    print(f"\n[4/4] 통계 분석...")
    
    # 복잡도 분포
    complexities = [s['complexity'] for s in dataset]
    intensities = [s['intensity'] for s in dataset]
    n_dims = [s['n_perceived_dims'] for s in dataset]
    n_ings = [s['n_ingredients'] for s in dataset]
    
    print(f"  복잡도: {np.mean(complexities):.2f} ± {np.std(complexities):.2f}")
    print(f"  강도: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
    print(f"  감지 차원: {np.mean(n_dims):.1f} ± {np.std(n_dims):.1f}")
    print(f"  원료 수: {np.mean(n_ings):.1f} ± {np.std(n_ings):.1f}")
    
    # 유니크한 원료 사용 수
    all_ing_ids = set()
    for s in dataset:
        for c in s.get('components', []):
            all_ing_ids.add(c.get('id', ''))
    print(f"  사용된 고유 원료: {len(all_ing_ids)}개 / {len(valid_ids)}개")
    
    # 시너지/마스킹 빈도
    total_syn = sum(len(s.get('synergies', [])) for s in dataset)
    total_mask = sum(len(s.get('masked', [])) for s in dataset)
    print(f"  시너지 발생: {total_syn}회 ({total_syn/len(dataset):.1f}회/샘플)")
    print(f"  마스킹 발생: {total_mask}회 ({total_mask/len(dataset):.1f}회/샘플)")
    
    print(f"\n  ⏱ 소요 시간: {elapsed:.1f}초")
    
    print("\n" + "=" * 60)
    print(f"  🎉 합성 혼합 데이터 생성 완료!")
    print(f"     {len(dataset)}개 샘플 × 22d 벡터")
    print(f"     6가지 과학 원리 기반 시뮬레이션")
    print("=" * 60)
    
    return output


def _build_components(ids, ratios, vectors, mw_map, vol_map):
    """원료 ID + 비율 → 시뮬레이터 입력 형식"""
    comps = []
    for i, iid in enumerate(ids):
        if iid not in vectors:
            continue
        r = ratios[i] if i < len(ratios) else 10
        comps.append({
            'id': iid,
            'ratio': r,
            'odor_vector': vectors[iid].tolist(),
            'mw': mw_map.get(iid, 180),
            'volatility': vol_map.get(iid, 5),
        })
    return comps


def _to_training_row(components, result, time_hours):
    """시뮬레이션 결과 → 학습 데이터 행"""
    return {
        'components': [
            {'id': c['id'], 'ratio': c['ratio'], 'vector': c['odor_vector']}
            for c in components
        ],
        'n_ingredients': len(components),
        'time_hours': time_hours,
        'perceived_vector': result['perceived_vector'].tolist(),
        'intensity': result['intensity'],
        'complexity': result['complexity'],
        'n_perceived_dims': result['n_perceived_dims'],
        'dominant_notes': [n['dimension'] for n in result['dominant_notes'][:5]],
        'synergies': result.get('synergies_applied', []),
        'masked': result.get('masked_notes', []),
    }


if __name__ == '__main__':
    generate_all_mixtures()
