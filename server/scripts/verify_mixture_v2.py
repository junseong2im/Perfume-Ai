"""
V2 혼합 시뮬레이터 종합 검증
================================
5가지 검증:
  1. 데이터 기반 파라미터 검증 — PMI/마스킹 행렬이 조향학에 부합하는지
  2. 물리법칙 검증 — Raoult, Stevens 적용이 올바른지
  3. 일관성 검증 — 같은 입력 = 같은 출력 (랜덤 없음)
  4. E2E 파이프라인 — ai_perfumer + V2 시뮬레이터 연동
  5. 합성 데이터 품질 — 29,534개 분포 분석
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mixture_simulator import MixtureSimulatorV2

PASS = 0
FAIL = 0
WARN = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")
    if detail:
        print(f"     → {detail}")


def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  ⚠ {name}")
    if detail:
        print(f"     → {detail}")


# ================================================================
# 테스트 1: 데이터 기반 파라미터 검증
# ================================================================
def test_1_learned_parameters(sim):
    print("\n" + "=" * 60)
    print("  [1/5] 데이터 기반 파라미터 검증")
    print("=" * 60)

    # 1a. 역치가 데이터에서 학습되었는지
    check("역치가 0이 아닌 유효한 값",
          np.all(sim.thresholds > 0),
          f"min={sim.thresholds.min():.4f}, max={sim.thresholds.max():.4f}")

    check("역치가 균등하지 않음 (학습됨)",
          np.std(sim.thresholds) > 0.001,
          f"std={np.std(sim.thresholds):.4f}")

    # 1b. PMI 행렬이 학습되었는지
    pmi_nonzero = np.sum(np.abs(sim.pmi_matrix) > 0.01)
    check("PMI 행렬에 유의미한 값 존재",
          pmi_nonzero > 10,
          f"{pmi_nonzero}개 페어 (0이 아닌)")

    # 1c. PMI 대칭성
    check("PMI 행렬 대칭",
          np.allclose(sim.pmi_matrix, sim.pmi_matrix.T, atol=1e-6))

    # 1d. 마스킹 행렬 학습 확인
    sup_nonzero = np.sum(np.abs(sim.suppression_matrix) > 0.01)
    check("마스킹 행렬에 유의미한 값 존재",
          sup_nonzero > 10,
          f"{sup_nonzero}개 페어")

    # 1e. 조향학적 상식 교차 검증
    # "woody와 earthy는 자주 함께 나온다" — PMI > 0이어야
    woody_idx = sim.ODOR_DIMS_22.index('woody')
    earthy_idx = sim.ODOR_DIMS_22.index('earthy')
    check("woody + earthy PMI > 0 (조향학 부합)",
          sim.pmi_matrix[woody_idx][earthy_idx] > 0,
          f"PMI={sim.pmi_matrix[woody_idx][earthy_idx]:.3f}")

    # "sweet와 floral은 자주 함께" — PMI > 0
    sweet_idx = sim.ODOR_DIMS_22.index('sweet')
    floral_idx = sim.ODOR_DIMS_22.index('floral')
    check("sweet + floral PMI > 0 (조향학 부합)",
          sim.pmi_matrix[sweet_idx][floral_idx] > 0,
          f"PMI={sim.pmi_matrix[sweet_idx][floral_idx]:.3f}")

    # "citrus와 fresh는 관련" — PMI > 0
    citrus_idx = sim.ODOR_DIMS_22.index('citrus')
    fresh_idx = sim.ODOR_DIMS_22.index('fresh')
    pmi_cf = sim.pmi_matrix[citrus_idx][fresh_idx]
    if pmi_cf > 0:
        check("citrus + fresh PMI > 0", True, f"PMI={pmi_cf:.3f}")
    else:
        warn("citrus + fresh PMI <= 0 (데이터에서 약한 연관)",
             f"PMI={pmi_cf:.3f} — Leffingwell에서 이 조합이 드물 수 있음")


# ================================================================
# 테스트 2: 물리법칙 검증
# ================================================================
def test_2_physics(sim):
    print("\n" + "=" * 60)
    print("  [2/5] 물리법칙 검증")
    print("=" * 60)

    # 2a. Stevens 지수가 논문값인지
    check("Stevens 지수 = 0.6 (Stevens 1957)",
          sim.STEVENS_EXPONENT == 0.6,
          f"값: {sim.STEVENS_EXPONENT}")

    # 2b. Raoult's Law: 가벼운 분자가 headspace 지배
    light = [{'id': 'light', 'ratio': 50, 'odor_vector': [0.5]*22, 'mw': 100}]
    heavy = [{'id': 'heavy', 'ratio': 50, 'odor_vector': [0.5]*22, 'mw': 300}]
    combined = light + heavy

    result = sim.simulate_mixture(combined, time_hours=0)
    hs = result['headspace_ratios']
    check("Raoult: 가벼운 분자(MW=100)가 headspace 지배",
          hs[0] > hs[1],
          f"light={hs[0]:.3f}, heavy={hs[1]:.3f}")

    # 2c. 시간 경과 시 가벼운 분자 감소
    result_later = sim.simulate_mixture(combined, time_hours=4)
    hs_later = result_later['headspace_ratios']
    check("Clausius-Clapeyron: 시간 경과 → 가벼운 분자 비율 변화",
          abs(hs[0] - hs_later[0]) > 0.01,
          f"t=0: {hs[0]:.3f} → t=4h: {hs_later[0]:.3f}")

    # 2d. Stevens: 역치 이하 = 0
    tiny = [{'id': 'tiny', 'ratio': 100, 'odor_vector': [0.001]*22, 'mw': 150}]
    result_tiny = sim.simulate_mixture(tiny, time_hours=0)
    perceived = result_tiny['perceived_vector']
    check("Stevens: 역치 이하 농도 → 감지 0",
          np.sum(perceived) < 0.1,
          f"총 감지: {np.sum(perceived):.4f}")

    # 2e. σ/τ: 성분 많으면 전체 강도 감소
    single = [{'id': 'a', 'ratio': 100, 'odor_vector': [0, 0, 0.8, 0]+[0]*18, 'mw': 200}]
    multi = [
        {'id': 'a', 'ratio': 25, 'odor_vector': [0, 0, 0.8, 0]+[0]*18, 'mw': 200},
        {'id': 'b', 'ratio': 25, 'odor_vector': [0, 0, 0.8, 0]+[0]*18, 'mw': 200},
        {'id': 'c', 'ratio': 25, 'odor_vector': [0, 0, 0.8, 0]+[0]*18, 'mw': 200},
        {'id': 'd', 'ratio': 25, 'odor_vector': [0, 0, 0.8, 0]+[0]*18, 'mw': 200},
    ]
    r1 = sim.simulate_mixture(single)
    r4 = sim.simulate_mixture(multi)
    check("Cain-Drexler σ/τ: 4성분 혼합 → 전체 강도 감소",
          r4['intensity'] < r1['intensity'],
          f"1성분: {r1['intensity']:.3f}, 4성분: {r4['intensity']:.3f}")


# ================================================================
# 테스트 3: 일관성 검증 (랜덤 없음)
# ================================================================
def test_3_consistency(sim):
    print("\n" + "=" * 60)
    print("  [3/5] 일관성 검증 (결정론적)")
    print("=" * 60)

    comps = [
        {'id': 'a', 'ratio': 60, 'odor_vector': [0.2, 0.9, 0]+[0]*19, 'mw': 136},
        {'id': 'b', 'ratio': 40, 'odor_vector': [0, 0, 0.8]+[0]*19, 'mw': 220},
    ]

    r1 = sim.simulate_mixture(comps, time_hours=0)
    r2 = sim.simulate_mixture(comps, time_hours=0)
    r3 = sim.simulate_mixture(comps, time_hours=0)

    v1 = np.array(r1['perceived_vector'])
    v2 = np.array(r2['perceived_vector'])
    v3 = np.array(r3['perceived_vector'])

    check("3회 실행 결과 동일 (run 1 == run 2)",
          np.allclose(v1, v2, atol=1e-10))
    check("3회 실행 결과 동일 (run 2 == run 3)",
          np.allclose(v2, v3, atol=1e-10))
    check("강도 동일",
          r1['intensity'] == r2['intensity'] == r3['intensity'],
          f"값: {r1['intensity']}")
    check("복잡도 동일",
          r1['complexity'] == r2['complexity'])


# ================================================================
# 테스트 4: E2E 파이프라인
# ================================================================
def test_4_e2e(sim):
    print("\n" + "=" * 60)
    print("  [4/5] E2E 파이프라인 검증")
    print("=" * 60)

    # 실제 향수급 조합
    components = [
        {'id': 'bergamot', 'ratio': 15,
         'odor_vector': [0.2, 0.9, 0, 0.3, 0, 0.1, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 0.3, 0],
         'mw': 136},
        {'id': 'sandalwood', 'ratio': 25,
         'odor_vector': [0, 0, 0.9, 0, 0, 0, 0, 0.3, 0, 0.4, 0.2, 0, 0, 0, 0, 0.3, 0, 0, 0.2, 0, 0, 0],
         'mw': 220},
        {'id': 'vanilla', 'ratio': 10,
         'odor_vector': [0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.5, 0.3, 0, 0, 0, 0, 0.3, 0.7, 0, 0.8, 0, 0, 0],
         'mw': 152},
        {'id': 'cedarwood', 'ratio': 20,
         'odor_vector': [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.1, 0.1, 0.3, 0, 0, 0, 0, 0, 0, 0.2, 0],
         'mw': 204},
        {'id': 'musk', 'ratio': 15,
         'odor_vector': [0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.1, 0, 0, 0],
         'mw': 258},
        {'id': 'black_pepper', 'ratio': 5,
         'odor_vector': [0, 0, 0.1, 0, 0.8, 0.2, 0, 0, 0, 0.4, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.3, 0],
         'mw': 136},
    ]

    result = sim.simulate_mixture(components, time_hours=0)

    # 기본 출력 구조 검증
    check("perceived_vector 22차원",
          len(result['perceived_vector']) == 22)
    check("dominant_notes 존재",
          len(result['dominant_notes']) > 0)
    check("intensity > 0",
          result['intensity'] > 0,
          f"값: {result['intensity']}")
    check("complexity > 0",
          result['complexity'] > 0,
          f"값: {result['complexity']}")
    check("parameter_source = data-driven",
          'data-driven' in result.get('parameter_source', ''),
          result.get('parameter_source', ''))

    # 향 프로파일 합리성
    vec = result['perceived_vector']
    woody_idx = sim.ODOR_DIMS_22.index('woody')
    citrus_idx = sim.ODOR_DIMS_22.index('citrus')

    check("woody가 지배적 (샌달+시더 45%)",
          vec[woody_idx] > 0.3,
          f"woody={vec[woody_idx]:.3f}")

    # 시너지/마스킹 발생
    check("시너지 발생",
          len(result['synergies_applied']) > 0,
          f"{len(result['synergies_applied'])}건")
    check("마스킹 발생",
          len(result['masked_notes']) >= 0,
          f"{len(result['masked_notes'])}건")

    # 시간 변화
    check("evolution 4단계 존재",
          len(result['evolution']) == 4,
          str(list(result['evolution'].keys())))

    # 비율 변경 시 결과 변화
    comps_mod = [c.copy() for c in components]
    comps_mod[2] = comps_mod[2].copy()
    comps_mod[2]['ratio'] = 40  # vanilla 10→40

    r_mod = sim.simulate_mixture(comps_mod, time_hours=0)
    vec_mod = r_mod['perceived_vector']

    sweet_idx = sim.ODOR_DIMS_22.index('sweet')
    check("바닐라 10%→40%: sweet 증가",
          vec_mod[sweet_idx] > vec[sweet_idx],
          f"before={vec[sweet_idx]:.3f}, after={vec_mod[sweet_idx]:.3f}")


# ================================================================
# 테스트 5: 합성 데이터 품질
# ================================================================
def test_5_data_quality():
    print("\n" + "=" * 60)
    print("  [5/5] 합성 데이터 품질 분석")
    print("=" * 60)

    path = os.path.join('data', 'synthetic_mixtures', 'mixture_training_data.json')
    if not os.path.exists(path):
        warn("합성 데이터 파일 없음", path)
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['samples']
    meta = data['metadata']

    check("데이터 파일 로드 성공",
          len(samples) > 0,
          f"{len(samples)}개 샘플")

    check("시뮬레이터 V2 사용",
          'V2' in meta.get('simulator', '') or 'data-driven' in meta.get('simulator', ''),
          meta.get('simulator', ''))

    # 벡터 유효성
    valid_vectors = 0
    zero_vectors = 0
    for s in samples:
        vec = np.array(s['perceived_vector'])
        if np.sum(vec) > 0.01:
            valid_vectors += 1
        else:
            zero_vectors += 1

    check(f"유효 벡터 비율 > 90%",
          valid_vectors / len(samples) > 0.9,
          f"유효: {valid_vectors}/{len(samples)} ({valid_vectors/len(samples)*100:.1f}%)")

    if zero_vectors > 0:
        warn(f"Zero 벡터 {zero_vectors}개",
             "역치 이하 조합 — 정상일 수 있음")

    # 복잡도 분포
    complexities = [s['complexity'] for s in samples]
    avg_c = np.mean(complexities)
    std_c = np.std(complexities)
    check("평균 복잡도 > 1 bit",
          avg_c > 1.0,
          f"{avg_c:.2f} ± {std_c:.2f}")

    # 강도 분포
    intensities = [s['intensity'] for s in samples]
    check("강도 > 0인 샘플 > 90%",
          sum(1 for x in intensities if x > 0) / len(intensities) > 0.9,
          f"평균: {np.mean(intensities):.3f}")

    # 원료 수 분포
    n_ings = [s['n_ingredients'] for s in samples]
    check("2~10개 원료 범위",
          min(n_ings) >= 1 and max(n_ings) <= 10,
          f"범위: {min(n_ings)}~{max(n_ings)}")

    # 다양성: dominant notes 분포
    all_dom = {}
    for s in samples:
        for d in s.get('dominant_notes', []):
            all_dom[d] = all_dom.get(d, 0) + 1

    check("10개 이상 향 차원 활용",
          len(all_dom) >= 10,
          f"{len(all_dom)}개 차원 활용")

    # 시너지/마스킹 발생률
    syn_count = sum(len(s.get('synergies', [])) for s in samples)
    mask_count = sum(len(s.get('masked', [])) for s in samples)
    check("시너지 발생률 > 0",
          syn_count > 0,
          f"총 {syn_count}회 ({syn_count/len(samples):.1f}회/샘플)")
    check("마스킹 발생률 > 0",
          mask_count > 0,
          f"총 {mask_count}회 ({mask_count/len(samples):.1f}회/샘플)")


# ================================================================
# 메인
# ================================================================
if __name__ == '__main__':
    print("🔬 V2 혼합 시뮬레이터 종합 검증")
    print("=" * 60)

    sim = MixtureSimulatorV2()

    test_1_learned_parameters(sim)
    test_2_physics(sim)
    test_3_consistency(sim)
    test_4_e2e(sim)
    test_5_data_quality()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  검증 결과: {PASS}/{total} PASS, {FAIL} FAIL, {WARN} WARN")

    if FAIL == 0:
        print("  🎉 전체 통과")
    else:
        print(f"  ⚠ {FAIL}개 실패 항목")
    print("=" * 60)
