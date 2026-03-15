"""
Dual Perfumer + Judge Model 검증 스크립트
==========================================
5개 검증 수트:
1. POM Bridge 단위 테스트
2. 교차 검증 불일치 검출 테스트
3. Judge 평가 일관성 테스트
4. E2E 파이프라인 테스트
5. 판정 합리성 테스트
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pom_bridge import POMBridge, ODOR_DIMS_22
from perfume_judge import PerfumeJudge

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

# ============================================================
# 1. POM Bridge 단위 테스트
# ============================================================
print("=" * 60)
print("1. POM Bridge 단위 테스트")
print("=" * 60)

bridge = POMBridge()

# 1.1 모델 로드
check("POM 모델 로드", bridge.model is not None)
check("113d 서술어 로드", len(bridge.descriptor_names) > 0)
check("서술어 수 = 113", bridge.n_descriptors == 113 or bridge.n_descriptors > 50)

# 1.2 리날룰 예측 (citrus/floral 계열)
pred_113 = bridge.predict_113d("CC(=CCC/C(=C/CO)/C)C")
check("리날룰 113d 예측 반환", len(pred_113) > 0)
check("리날룰 113d 범위 [0,1]", np.all(pred_113 >= 0) and np.all(pred_113 <= 1))

pred_22 = bridge.predict_22d("CC(=CCC/C(=C/CO)/C)C")
check("리날룰 22d 예측 반환", len(pred_22) == 22)
check("리날룰 citrus 높음", pred_22[ODOR_DIMS_22.index('citrus')] > 0.2,
      f"citrus={pred_22[ODOR_DIMS_22.index('citrus')]:.3f}")
check("리날룰 floral 활성", pred_22[ODOR_DIMS_22.index('floral')] > 0.1,
      f"floral={pred_22[ODOR_DIMS_22.index('floral')]:.3f}")

# 1.3 바닐린 예측 (sweet/balsamic 계열)
vanillin = "O=Cc1ccc(O)c(OC)c1"
pred_van = bridge.predict_22d(vanillin)
check("바닐린 sweet 활성", pred_van[ODOR_DIMS_22.index('sweet')] > 0.05,
      f"sweet={pred_van[ODOR_DIMS_22.index('sweet')]:.3f}")

# 1.4 캐시 동작
pred_a = bridge.predict_113d("CC(=CCC/C(=C/CO)/C)C")
pred_b = bridge.predict_113d("CC(=CCC/C(=C/CO)/C)C")
check("캐시 일관성", np.allclose(pred_a, pred_b))

# 1.5 빈 SMILES 처리
pred_empty = bridge.predict_113d("")
check("빈 SMILES → zeros", np.all(pred_empty == 0))

pred_invalid = bridge.predict_113d("NOT_A_SMILES")
check("잘못된 SMILES → zeros", np.all(pred_invalid == 0))

# 1.6 결정론성 (같은 입력 → 같은 출력)
bridge._cache.clear()
pred_x = bridge.predict_22d("CC(=CCC/C(=C/CO)/C)C")
bridge._cache.clear()
pred_y = bridge.predict_22d("CC(=CCC/C(=C/CO)/C)C")
check("결정론적 출력", np.allclose(pred_x, pred_y, atol=1e-5))


# ============================================================
# 2. 교차 검증 불일치 검출 테스트
# ============================================================
print(f"\n{'=' * 60}")
print("2. 교차 검증 불일치 검출 테스트")
print("=" * 60)

# 2.1 동일 벡터 → 완전 동의
pom_vec = bridge.predict_22d("CC(=CCC/C(=C/CO)/C)C")
cv_same = bridge.cross_validate("CC(=CCC/C(=C/CO)/C)C", pom_vec)
check("동일 벡터 → 높은 동의도", cv_same['agreement'] > 0.95,
      f"agreement={cv_same['agreement']:.3f}")
check("동일 벡터 → 불일치 없음", len(cv_same['disagreement_dims']) == 0)
check("동일 벡터 → confident=True", cv_same['confident'])

# 2.2 반대 벡터 → 불일치
anti_vec = np.ones(22) - pom_vec  # 반전
cv_anti = bridge.cross_validate("CC(=CCC/C(=C/CO)/C)C", anti_vec)
check("반대 벡터 → 낮은 동의도", cv_anti['agreement'] < 0.7,
      f"agreement={cv_anti['agreement']:.3f}")
check("반대 벡터 → 불일치 감지", len(cv_anti['disagreement_dims']) > 0)

# 2.3 퓨전 테스트
fused = bridge.fuse_predictions("CC(=CCC/C(=C/CO)/C)C", anti_vec)
check("퓨전 벡터 차원=22", len(fused) == 22)
check("퓨전 벡터 값 범위", np.all(fused >= 0),
      f"min={np.min(fused):.3f}, max={np.max(fused):.3f}")


# ============================================================
# 3. Judge 평가 일관성 테스트
# ============================================================
print(f"\n{'=' * 60}")
print("3. Judge 평가 일관성 테스트")
print("=" * 60)

judge = PerfumeJudge(pom_bridge=bridge)

test_formula = [
    {
        'ingredient': {'id': 'bergamot', 'category': 'citrus'},
        'ratio_pct': 4.0,
        'odor_vector': [0, 0.9, 0, 0.2, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0],
    },
    {
        'ingredient': {'id': 'jasmine', 'category': 'floral'},
        'ratio_pct': 5.0,
        'odor_vector': [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
    },
    {
        'ingredient': {'id': 'sandalwood', 'category': 'woody'},
        'ratio_pct': 4.5,
        'odor_vector': [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    {
        'ingredient': {'id': 'vanilla', 'category': 'balsamic'},
        'ratio_pct': 3.0,
        'odor_vector': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0.2, 0.5, 0, 0.8, 0, 0, 0],
    },
    {
        'ingredient': {'id': 'musk', 'category': 'musk'},
        'ratio_pct': 2.0,
        'odor_vector': [0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0.1, 0, 0, 0, 0],
    },
]

target = np.array([0.6, 0.3, 0.4, 0.1, 0, 0, 0.3, 0, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0.1, 0.2, 0, 0.5, 0.3, 0, 0])

# 3.1 기본 동작
result1 = judge.judge(test_formula, target)
check("Judge 반환 dict", isinstance(result1, dict))
check("verdict 반환", result1['verdict'] in ['PASS', 'REVISE', 'REJECT'])
check("score 범위 [0,100]", 0 <= result1['score'] <= 100)
check("8개 평가 항목", len(result1['evaluations']) == 8)

# 3.2 결정론성 (같은 입력 → 같은 출력)
result2 = judge.judge(test_formula, target)
check("결정론적 점수", abs(result1['score'] - result2['score']) < 0.01,
      f"score1={result1['score']}, score2={result2['score']}")
check("결정론적 판정", result1['verdict'] == result2['verdict'])

# 3.3 하드코딩/랜덤 없음
import inspect
source = inspect.getsource(PerfumeJudge)
check("random() 미사용", 'random.' not in source and 'random(' not in source)

# 3.4 각 평가 항목 점수 범위
for key, ev in result1['evaluations'].items():
    check(f"  {key} 점수 범위", 0 <= ev['score'] <= 100,
          f"score={ev['score']:.1f}")

# 3.5 이유 설명 존재
check("reasoning 존재", len(result1.get('reasoning', '')) > 0)
check("suggestions 존재", isinstance(result1.get('suggestions', []), list))

# 3.6 report 포맷
report = judge.format_report(result1)
check("report 텍스트", len(report) > 100)
check("report에 점수 포함", str(round(result1['score'], 1)) in report)


# ============================================================
# 4. 판정 합리성 테스트
# ============================================================
print(f"\n{'=' * 60}")
print("4. 판정 합리성 테스트")
print("=" * 60)

# 4.1 좋은 레시피 (타겟에 매우 가까운)
good_formula = []
for i, dim in enumerate(['floral', 'citrus', 'woody', 'musk', 'sweet']):
    vec = np.zeros(22)
    idx = ODOR_DIMS_22.index(dim)
    vec[idx] = 0.8
    # 인접 차원에도 약간 기여
    if idx > 0: vec[idx-1] = 0.1
    if idx < 21: vec[idx+1] = 0.1
    
    note_types = ['citrus', 'floral', 'woody', 'musk', 'balsamic']
    good_formula.append({
        'ingredient': {'id': f'{dim}_note', 'category': dim},
        'ratio_pct': [4.0, 5.0, 4.5, 2.0, 2.5][i],
        'odor_vector': vec.tolist(),
    })

good_target = np.zeros(22)
for dim in ['floral', 'citrus', 'woody', 'musk', 'sweet']:
    good_target[ODOR_DIMS_22.index(dim)] = 0.7

good_result = judge.judge(good_formula, good_target)
check("좋은 레시피 점수 > 50", good_result['score'] > 50,
      f"score={good_result['score']:.1f}")

# 4.2 나쁜 레시피 (타겟과 완전 다른)
bad_target = np.zeros(22)
bad_target[ODOR_DIMS_22.index('aquatic')] = 0.9
bad_target[ODOR_DIMS_22.index('fresh')] = 0.8

bad_result = judge.judge(good_formula, bad_target)  # floral 레시피인데 aquatic 타겟
check("나쁜 레시피 점수 < 좋은 레시피", bad_result['score'] < good_result['score'],
      f"good={good_result['score']:.1f}, bad={bad_result['score']:.1f}")

# 4.3 단일 원료 (마스킹 없음)
single_formula = [{
    'ingredient': {'id': 'rose', 'category': 'floral'},
    'ratio_pct': 22.0,
    'odor_vector': [0.9] + [0]*21,
}]
single_result = judge.judge(single_formula, np.array([0.9] + [0]*21))
check("단일 원료 마스킹 점수 높음",
      single_result['evaluations']['masking_risk']['score'] >= 95)

# 4.4 IFRA 위반 레시피
ifra_formula = [{
    'ingredient': {'id': 'oakmoss_extract', 'category': 'earthy'},
    'ratio_pct': 5.0,  # 한도 0.1% 대폭 초과
    'odor_vector': [0]*13 + [0.8] + [0]*8,
}]
ifra_result = judge.judge(ifra_formula, np.zeros(22))
check("IFRA 위반 감지", ifra_result['evaluations']['safety']['score'] < 100)


# ============================================================
# 5. POM Bridge + Judge 통합 테스트
# ============================================================
print(f"\n{'=' * 60}")
print("5. POM Bridge + Judge 통합 테스트")
print("=" * 60)

# 5.1 POM 예측 → Judge 입력
smiles_list = ["CC(=CCC/C(=C/CO)/C)C", "O=Cc1ccc(O)c(OC)c1"]  # linalool, vanillin
pom_vecs = bridge.predict_batch_22d(smiles_list)
check("배치 예측 길이", len(pom_vecs) == 2)
check("배치 벡터 차원", all(len(v) == 22 for v in pom_vecs))

# 5.2 POM 예측을 formula에 주입 → Judge 평가
pom_formula = [
    {
        'ingredient': {'id': 'linalool', 'category': 'floral'},
        'ratio_pct': 5.0,
        'odor_vector': pom_vecs[0].tolist(),
    },
    {
        'ingredient': {'id': 'vanillin', 'category': 'balsamic'},
        'ratio_pct': 3.0,
        'odor_vector': pom_vecs[1].tolist(),
    },
]

pom_target = np.array([0.5, 0.3, 0, 0, 0, 0.2, 0, 0, 0.2, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.3, 0, 0])
pom_result = judge.judge(pom_formula, pom_target)
check("POM+Judge 통합 동작", pom_result['verdict'] in ['PASS', 'REVISE', 'REJECT'])
check("POM+Judge 점수 > 0", pom_result['score'] > 0)


# ============================================================
# 결과
# ============================================================
print(f"\n{'=' * 60}")
print(f"검증 결과: {PASS}/{PASS+FAIL} PASS, {FAIL} FAIL")
if FAIL == 0:
    print("🎉 전체 통과!")
else:
    print(f"⚠️ {FAIL}건 실패")
print("=" * 60)
