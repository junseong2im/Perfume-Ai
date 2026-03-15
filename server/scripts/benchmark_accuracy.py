"""Before vs After 정확도 비교 벤치마크"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import json

print('=' * 60)
print('  Before vs After 정확도 비교 벤치마크')
print('=' * 60)

# ================================================================
# 1. 증발 시뮬레이션 정밀도 (GC-MS Unroll 효과)
# ================================================================
print('\n--- 1. 증발 시뮬레이션 정밀도 ---')
from biophysics_simulator import ThermodynamicsEngine
from data.natural_oil_compositions import unroll_ingredient

thermo = ThermodynamicsEngine()

# BEFORE: bergamot = 단일 분자 (geraniol)
single_smiles = ['CC(=CCC=C(C)C)CO']  # geraniol 1개
single_conc = [5.0]

result_before = thermo.simulate_evaporation(single_smiles, single_conc, duration_hours=8)
transitions_before = result_before.get('transitions', [])
longevity_before = result_before.get('longevity_hours', 0)

# AFTER: bergamot = 7개 하위 분자 (GC-MS)
sub = unroll_ingredient('bergamot', 5.0)
multi_smiles = [s for s, _ in sub]
multi_conc = [c for _, c in sub]

result_after = thermo.simulate_evaporation(multi_smiles, multi_conc, duration_hours=8)
transitions_after = result_after.get('transitions', [])
longevity_after = result_after.get('longevity_hours', 0)

print(f'  BEFORE (단일 분자): 전환점 {len(transitions_before)}개, 지속 {longevity_before:.1f}h')
print(f'  AFTER  (7개 분자):  전환점 {len(transitions_after)}개, 지속 {longevity_after:.1f}h')
print(f'  → 증발 정밀도: {len(transitions_after)}/{len(transitions_before)} = {len(transitions_after)/max(len(transitions_before),1):.1f}x')

# Lavender도 테스트
sub_lav = unroll_ingredient('lavender', 10.0)
r_lav_before = thermo.simulate_evaporation(['CC(=CCC=C(C)C)O'], [10.0], duration_hours=8)
r_lav_after = thermo.simulate_evaporation([s for s, _ in sub_lav], [c for _, c in sub_lav], duration_hours=8)
print(f'  라벤더 BEFORE: 전환 {len(r_lav_before.get("transitions",[]))}개')
print(f'  라벤더 AFTER:  전환 {len(r_lav_after.get("transitions",[]))}개')

# ================================================================
# 2. 레시피 평가 정확도 (Proxy Reward vs Harmony Score)
# ================================================================
print('\n--- 2. 레시피 평가 정확도 (Proxy Reward) ---')
from scripts.proxy_reward import ProxyRewardModel, extract_recipe_features, train_proxy
import torch

with open('data/recipe_training_data.json', 'r', encoding='utf-8') as f:
    recipes = json.load(f)

# 100개 레시피로 테스트
model = train_proxy(epochs=80)
errors = []
for i, recipe in enumerate(recipes[:100]):
    true_score = recipe.get('harmony_score', 0.85)
    features = extract_recipe_features(recipe)
    pred_score = model.predict(features)
    errors.append(abs(true_score - pred_score))

mae = np.mean(errors)
within_5pct = sum(1 for e in errors if e < 0.05) / len(errors) * 100
within_10pct = sum(1 for e in errors if e < 0.10) / len(errors) * 100

print(f'  BEFORE (규칙 기반만): 향수 품질 평가 = 물리/화학 수치만')
print(f'  AFTER  (Proxy Reward): MAE={mae:.4f}')
print(f'    ±5% 이내 정확: {within_5pct:.0f}%')
print(f'    ±10% 이내 정확: {within_10pct:.0f}%')

# ================================================================
# 3. 상업적 그럴듯함 판별 정확도 (PMI)
# ================================================================
print('\n--- 3. 상업적 그럴듯함 판별 (PMI) ---')
from scripts.commercial_prior import CommercialPrior
prior = CommercialPrior()

# 성공한 향수 10개 vs 랜덤 10개
success_scores = []
random_scores = []

# 실제 레시피에서 성공 조합 추출
for recipe in recipes[:20]:
    ings = [i['id'] for i in recipe.get('ingredients', [])]
    score = prior.plausibility_score(ings)
    success_scores.append(score)

# 랜덤 조합 (이상한 조합)
import random
random.seed(42)
all_ing_ids = list(prior.ingredient_freq.keys())
for _ in range(20):
    n = random.randint(5, 12)
    rand_ings = random.sample(all_ing_ids, min(n, len(all_ing_ids)))
    score = prior.plausibility_score(rand_ings)
    random_scores.append(score)

print(f'  BEFORE: 조합 적절성 판별 = 없음 (PPO가 맹목적 조합)')
print(f'  AFTER (PMI):')
print(f'    성공 향수 평균: {np.mean(success_scores):.3f} (±{np.std(success_scores):.3f})')
print(f'    랜덤 조합 평균: {np.mean(random_scores):.3f} (±{np.std(random_scores):.3f})')
print(f'    → 분리도: {np.mean(success_scores) - np.mean(random_scores):.3f}')
print(f'    → 판별 AUC: ', end='')

# 간단 AUC 계산
all_scores = [(s, 1) for s in success_scores] + [(s, 0) for s in random_scores]
all_scores.sort(key=lambda x: -x[0])
tp, fp, auc = 0, 0, 0
total_pos = len(success_scores)
total_neg = len(random_scores)
for score, label in all_scores:
    if label == 1:
        tp += 1
    else:
        auc += tp
        fp += 1
auc = auc / (total_pos * total_neg)
print(f'{auc:.3f}')

# ================================================================  
# 4. 활성 수용체 정밀도 (VirtualNose + Unroll)
# ================================================================
print('\n--- 4. 활성 수용체 정밀도 (VirtualNose) ---')
from biophysics_simulator import VirtualNose
nose = VirtualNose()

# Before: 단일 분자
r_before = nose.smell(single_smiles, single_conc)
active_before = r_before['active_receptors']
families_before = sum(1 for v in r_before.get('receptor_families', {}).values() if isinstance(v, (int, float)) and v > 0)

# After: 7개 분자
r_after = nose.smell(multi_smiles, multi_conc)
active_after = r_after['active_receptors']
families_after = sum(1 for v in r_after.get('receptor_families', {}).values() if isinstance(v, (int, float)) and v > 0)

print(f'  BEFORE (단일):   활성 수용체 {active_before}개, 가족 {families_before}개')
print(f'  AFTER  (Unroll): 활성 수용체 {active_after}개, 가족 {families_after}개')
print(f'  → 수용체 커버리지: {active_after/max(active_before,1):.1f}x')

# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 60)
print('  종합 정확도 비교')
print('=' * 60)
print(f'''
  항목              | BEFORE        | AFTER          | 개선
  ─────────────────┼───────────────┼────────────────┼──────
  증발 전환점 수    | {len(transitions_before):>5}개        | {len(transitions_after):>5}개         | {len(transitions_after)/max(len(transitions_before),1):.1f}x
  지속 시간         | {longevity_before:>5.1f}h        | {longevity_after:>5.1f}h         | 정밀화
  품질 예측 MAE     | N/A (없음)     | {mae:>5.4f}          | 신규
  품질 ±5% 정확도   | N/A           | {within_5pct:>5.0f}%          | 신규
  상업성 판별 AUC   | 0.500 (랜덤)   | {auc:>5.3f}          | +{(auc-0.5)*100:.0f}%p
  수용체 커버리지   | {active_before:>5}          | {active_after:>5}           | {active_after/max(active_before,1):.1f}x
  수용체 가족 수    | {families_before:>5}          | {families_after:>5}           | {families_after/max(families_before,1):.1f}x
''')
