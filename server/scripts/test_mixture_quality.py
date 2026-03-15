#!/usr/bin/env python
"""V22 혼합물 예측 품질 테스트

유명 향수 레시피에 대해:
1. ConcentrationModulator 농도 보정 테스트
2. PhysicsMixture 혼합물 예측 테스트
3. 알려진 향 프로필과의 일치도 검증
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ================================================================
# 테스트 데이터: 유명 향수의 대표 분자 + 농도 + 예상 프로필
# ================================================================
FAMOUS_PERFUMES = [
    {
        'name': 'Chanel No.5 (Aldehydic Floral)',
        'molecules': [
            # SMILES, concentration %, description
            ('CCCCCCCCCCC=O', 0.5, 'undecanal'),        # aldehyde
            ('CC(C)=CCCC(C)=CCO', 3.0, 'geraniol'),     # floral/rose
            ('CC(=CCC/C(=C/CO)/C)C', 2.0, 'linalool'),  # floral/fresh
            ('CC(=O)OC/C=C(\\C)CCC=C(C)C', 2.0, 'linalyl acetate'),  # fresh
            ('O=Cc1ccc(OC)cc1', 1.0, 'anisaldehyde'),   # sweet/anise
        ],
        'expected_top': ['floral', 'fresh', 'sweet', 'powdery'],
        'description': '알데히드+플로럴 — 파우더리하고 우아한 향',
    },
    {
        'name': 'Dior Sauvage (Fresh Spicy)',
        'molecules': [
            ('CC(C)=CCCC(C)=CCO', 3.0, 'linalool'),     # fresh
            ('CC(C)C1CCC(C)CC1O', 2.0, 'menthol-like'),  # fresh/cool
            ('O=CC1=CC=C(C=C1)C', 1.0, 'tolualdehyde'),  # spicy
            ('CC(=O)OC/C=C(\\C)CCC=C(C)C', 2.5, 'linalyl acetate'),  # fresh
            ('CCCCCC(O)CC', 1.0, '2-octanol'),            # fresh
        ],
        'expected_top': ['fresh', 'spicy', 'woody', 'citrus'],
        'description': '프레시 스파이시 — 강렬하고 시원한 향',
    },
    {
        'name': 'Tom Ford Oud Wood (Woody Oriental)',
        'molecules': [
            ('CC1CCC2(C)CCC(O)C2C1C', 4.0, 'santalol-like'),    # woody
            ('CC(C)=CCCC(C)(O)C=C', 3.0, 'linalool variant'),    # woody/fresh  
            ('OC1CCCCC1C', 2.5, 'cyclohexanol deriv'),            # woody
            ('O=Cc1ccccc1', 1.5, 'benzaldehyde'),                 # sweet
            ('CC(C)=CCCC(=CC=O)C', 2.0, 'citral'),               # spicy/warm
        ],
        'expected_top': ['woody', 'warm', 'amber', 'spicy'],
        'description': '우디 오리엔탈 — 깊고 따뜻한 우드',
    },
    {
        'name': 'Jo Malone Lime Basil (Citrus Herbal)',
        'molecules': [
            ('CC(=O)OC/C=C(\\C)CCC=C(C)C', 4.0, 'linalyl acetate'),  # citrus/fresh
            ('C=CC(CC=C(C)C)C', 2.0, 'ocimene'),                       # herbal/green
            ('CC(C)=CCCC(C)=CCO', 3.0, 'geraniol'),                    # citrus
            ('COc1cc(CC=C)ccc1OC', 1.0, 'eugenol'),                    # spicy/herbal
            ('CC(=C)C1CC=C(C)CC1', 1.5, 'limonene-like'),             # citrus
        ],
        'expected_top': ['citrus', 'fresh', 'herbal', 'green'],
        'description': '시트러스 허벌 — 라임+바질의 상쾌함',
    },
    {
        'name': 'Thierry Mugler Angel (Gourmand)',
        'molecules': [
            ('OC(=O)/C=C/c1ccc(O)c(OC)c1', 2.0, 'ferulic acid'),  # sweet
            ('O=Cc1ccc(O)c(OC)c1', 3.0, 'vanillin'),                # sweet/vanilla
            ('CC1=CC(=O)c2ccccc2O1', 1.5, 'coumarin-like'),         # sweet/warm
            ('CCCCCCCCCCCC=O', 0.5, 'dodecanal'),                    # fatty/waxy
            ('OC(=O)CC(O)(CC(=O)O)C(=O)O', 0.3, 'citric acid'),    # fruity/sour
        ],
        'expected_top': ['sweet', 'warm', 'fruity', 'powdery'],
        'description': '구르망 — 초콜릿+카라멜의 달콤함',
    },
]


def test_concentration_modulator():
    """ConcentrationModulator 단위 테스트"""
    from odor_engine import ConcentrationModulator, ODOR_DIMENSIONS
    
    mod = ConcentrationModulator()
    
    print("=" * 70)
    print("TEST 1: ConcentrationModulator 단위 테스트")
    print("=" * 70)
    
    # 테스트 벡터 (floral 향)
    test_vec = np.zeros(22)
    test_vec[ODOR_DIMENSIONS.index('floral')] = 0.8
    test_vec[ODOR_DIMENSIONS.index('sweet')] = 0.4
    test_vec[ODOR_DIMENSIONS.index('fresh')] = 0.3
    
    print(f"\n원본 벡터: floral=0.8, sweet=0.4, fresh=0.3")
    
    # 농도별 보정 테스트
    test_concs = [0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 20.0]
    all_passed = True
    
    for conc in test_concs:
        result, details = mod.modulate(test_vec.copy(), conc, return_details=True)
        fl = result[ODOR_DIMENSIONS.index('floral')]
        sw = result[ODOR_DIMENSIONS.index('sweet')]
        fr = result[ODOR_DIMENSIONS.index('fresh')]
        print(f"  {conc:6.2f}% → floral={fl:.3f} sweet={sw:.3f} fresh={fr:.3f}")
        
        # 기본 검증: 농도 증가에 따라 적절히 변화
        if conc < 0.1:
            # 역치 근처에서는 낮아야 함
            if fl > 0.5:
                print(f"    ⚠ WARNING: 역치 근처에서 floral이 너무 높음")
                all_passed = False
    
    # 고농도 inversion 테스트
    high_conc_vec = np.zeros(22)
    high_conc_vec[ODOR_DIMENSIONS.index('floral')] = 0.7
    high_conc_vec[ODOR_DIMENSIONS.index('earthy')] = 0.1
    
    result = mod.modulate(high_conc_vec.copy(), 8.0)
    earthy_boost = result[ODOR_DIMENSIONS.index('earthy')]
    print(f"\n  고농도 전환 테스트 (floral→earthy, 8%):")
    print(f"    earthy: 0.1 → {earthy_boost:.3f} (전환 확인: {'✅' if earthy_boost > 0.1 else '⚠'})")
    
    if earthy_boost <= 0.1:
        all_passed = False
    
    print(f"\n  ConcentrationModulator: {'✅ PASSED' if all_passed else '⚠ ISSUES FOUND'}")
    return all_passed


def test_physics_mixture():
    """PhysicsMixture 단위 테스트"""
    from odor_engine import PhysicsMixture, ConcentrationModulator, ODOR_DIMENSIONS
    
    mixer = PhysicsMixture()
    mod = ConcentrationModulator()
    
    print("\n" + "=" * 70)
    print("TEST 2: PhysicsMixture 단위 테스트")
    print("=" * 70)
    
    all_passed = True
    
    # 시너지 테스트: floral + musk → powdery 부스트
    vec_a = np.zeros(22)
    vec_a[ODOR_DIMENSIONS.index('floral')] = 0.8
    vec_a[ODOR_DIMENSIONS.index('sweet')] = 0.3
    
    vec_b = np.zeros(22)
    vec_b[ODOR_DIMENSIONS.index('musk')] = 0.8
    vec_b[ODOR_DIMENSIONS.index('warm')] = 0.3
    
    result, analysis = mixer.mix(
        np.array([vec_a, vec_b]), 
        np.array([3.0, 3.0]), 
        return_analysis=True
    )
    
    print(f"\n  시너지 테스트: floral + musk")
    for dim, val in analysis['dominant_dims']:
        print(f"    {dim}: {val:.3f}")
    
    powdery = result[ODOR_DIMENSIONS.index('powdery')]
    print(f"  powdery 부스트: {powdery:.3f} (시너지 확인: {'✅' if powdery > 0.02 else '⚠'})")
    if powdery <= 0.02:
        all_passed = False
    
    # 길항 테스트: fresh + smoky → fresh 감소
    vec_c = np.zeros(22)
    vec_c[ODOR_DIMENSIONS.index('fresh')] = 0.8
    vec_c[ODOR_DIMENSIONS.index('citrus')] = 0.5
    
    vec_d = np.zeros(22)
    vec_d[ODOR_DIMENSIONS.index('smoky')] = 0.8
    vec_d[ODOR_DIMENSIONS.index('woody')] = 0.3
    
    result2, analysis2 = mixer.mix(
        np.array([vec_c, vec_d]),
        np.array([3.0, 3.0]),
        return_analysis=True
    )
    
    fresh_val = result2[ODOR_DIMENSIONS.index('fresh')]
    print(f"\n  길항 테스트: fresh + smoky")
    for dim, val in analysis2['dominant_dims']:
        print(f"    {dim}: {val:.3f}")
    print(f"  fresh 억제: {fresh_val:.3f} (원본 0.8 대비 감소: {'✅' if fresh_val < 0.6 else '⚠'})")
    if fresh_val >= 0.6:
        all_passed = False
    
    # 상호작용 유형 확인
    for ia in analysis2['interactions']:
        print(f"  상호작용: {ia['type']} (synergy={ia['synergy_score']:.3f}, masking_ratio={ia['masking_ratio']:.3f})")
    
    # 마스킹 테스트: 강한 musk가 약한 citrus를 가림
    vec_e = np.zeros(22)
    vec_e[ODOR_DIMENSIONS.index('musk')] = 0.9
    vec_e[ODOR_DIMENSIONS.index('warm')] = 0.5
    
    vec_f = np.zeros(22)
    vec_f[ODOR_DIMENSIONS.index('citrus')] = 0.3
    vec_f[ODOR_DIMENSIONS.index('fresh')] = 0.2
    
    result3, analysis3 = mixer.mix(
        np.array([vec_e, vec_f]),
        np.array([8.0, 1.0]),  # musk 고농도, citrus 저농도
        return_analysis=True
    )
    
    citrus_val = result3[ODOR_DIMENSIONS.index('citrus')]
    print(f"\n  마스킹 테스트: 강한 musk vs 약한 citrus")
    for dim, val in analysis3['dominant_dims']:
        print(f"    {dim}: {val:.3f}")
    print(f"  citrus 마스킹: {citrus_val:.3f} (강하게 억제: {'✅' if citrus_val < 0.15 else '⚠'})")
    if citrus_val >= 0.15:
        all_passed = False
    
    print(f"\n  PhysicsMixture: {'✅ PASSED' if all_passed else '⚠ ISSUES FOUND'}")
    return all_passed


def test_famous_perfumes():
    """유명 향수 레시피로 통합 테스트"""
    from odor_engine import (OdorGNN, ConcentrationModulator, PhysicsMixture, 
                             ODOR_DIMENSIONS, N_ODOR_DIM)
    
    gnn = OdorGNN(device='cpu')
    mod = ConcentrationModulator()
    mixer = PhysicsMixture()
    
    print("\n" + "=" * 70)
    print("TEST 3: 유명 향수 레시피 통합 테스트")
    print("=" * 70)
    
    total_match = 0
    total_expected = 0
    
    for perfume in FAMOUS_PERFUMES:
        print(f"\n  📦 {perfume['name']}")
        print(f"     {perfume['description']}")
        
        # 개별 분자 인코딩 + 농도 보정
        smiles_list = [m[0] for m in perfume['molecules']]
        concentrations = [m[1] for m in perfume['molecules']]
        
        raw_vecs = [gnn.encode(s) for s in smiles_list]
        mod_vecs = mod.batch_modulate(np.array(raw_vecs), np.array(concentrations))
        
        # 혼합
        mixture_vec, analysis = mixer.mix(
            mod_vecs, np.array(concentrations), return_analysis=True
        )
        
        # 상위 5 차원
        top_indices = np.argsort(mixture_vec)[::-1][:5]
        predicted_top = [ODOR_DIMENSIONS[i] for i in top_indices]
        predicted_vals = [float(mixture_vec[i]) for i in top_indices]
        
        print(f"     예상: {perfume['expected_top']}")
        print(f"     예측: {[(d, f'{v:.3f}') for d, v in zip(predicted_top, predicted_vals)]}")
        
        # 매칭 점수: expected_top의 항목이 predicted_top에 포함되는지
        matched = set(predicted_top) & set(perfume['expected_top'])
        match_ratio = len(matched) / len(perfume['expected_top'])
        total_match += len(matched)
        total_expected += len(perfume['expected_top'])
        
        status = '✅' if match_ratio >= 0.5 else '⚠'
        print(f"     매칭: {len(matched)}/{len(perfume['expected_top'])} ({match_ratio:.0%}) {status}")
        
        # 상호작용 요약
        interactions = analysis.get('interactions', [])
        if interactions:
            synergies = [i for i in interactions if i['type'] == 'synergy']
            maskings = [i for i in interactions if i['type'] == 'masking']
            print(f"     상호작용: 시너지 {len(synergies)}, 마스킹 {len(maskings)}, "
                  f"전체 {len(interactions)}")
    
    overall = total_match / max(total_expected, 1)
    print(f"\n  {'='*50}")
    print(f"  전체 매칭률: {total_match}/{total_expected} ({overall:.0%})")
    print(f"  결과: {'✅ GOOD' if overall >= 0.4 else '⚠ NEEDS IMPROVEMENT'}")
    print(f"  (참고: V22 GNN이 미학습 상태이면 규칙 엔진 fallback 사용)")
    return overall >= 0.4


if __name__ == '__main__':
    print("=" * 70)
    print("V22 후처리 품질 테스트")
    print("Concentration Modulator + Physics Mixture")
    print("=" * 70)
    
    results = []
    results.append(('ConcentrationModulator', test_concentration_modulator()))
    results.append(('PhysicsMixture', test_physics_mixture()))
    results.append(('유명 향수 통합', test_famous_perfumes()))
    
    print("\n" + "=" * 70)
    print("최종 결과:")
    for name, passed in results:
        print(f"  {name}: {'✅ PASSED' if passed else '⚠ ISSUES'}")
    print("=" * 70)
    
    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
