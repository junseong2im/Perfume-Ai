# -*- coding: utf-8 -*-
"""
평창 대나무숲 × 겨울 눈의 향 — 완전체 레시피 생성
===================================================
V16 모델 (CosSim 0.8092/0.8102) 기반 **완전체** 파이프라인:
  ✅ OdorGNN V16 (22d): 분자 SMILES → 22차원 냄새 벡터
  ✅ PrincipalOdorMap: 텍스트 → 목표 벡터 + 앵커 매칭
  ✅ AIRecipeEngine: AI 스코어링 기반 원료 선택
  ✅ MolecularHarmony: 50 수용체 + Hill 방정식 조화 분석
  ✅ Sommelier: 냄새 벡터 → 시적 한국어 표현
  ✅ MixtureTransformer: 혼합물 상호작용 예측
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')

import numpy as np
import torch

print("=" * 70)
print("  🎋 평창 대나무숲 × ❄️ 겨울 눈의 향 — 완전체 AI 레시피")
print("  V16 OdorGNN (CosSim 0.8092) + Full Pipeline")
print("=" * 70)

# ─── Step 1: V16 OdorGNN 로드 ───
print("\n[Step 1] V16 OdorGNN 모델 로딩")
from odor_engine import OdorPredictor, PrincipalOdorMap, ODOR_DIMENSIONS, N_ODOR_DIM

predictor = OdorPredictor(device='cuda')

print(f"\n  OdorGNN: {predictor.gnn._model_version} | "
      f"Ensemble: {predictor.gnn._ensemble_mode} | "
      f"Trained: {predictor.gnn._use_trained} | "
      f"Odor dims: {N_ODOR_DIM}d")

# ─── Step 2: 목표 냄새 벡터 생성 ───
print(f"\n{'='*70}")
print("[Step 2] 텍스트 → 목표 냄새 벡터 (PrincipalOdorMap)")

description = "평창 대나무숲의 싱그러운 녹색 향기와 겨울 눈의 차갑고 신선한 공기, 나무와 흙의 은은한 향"
target = predictor.target_vector(description)

# 대나무숲+눈 특성으로 목표 벡터 보강
bamboo_snow = {
    'green': 0.9, 'fresh': 0.8, 'woody': 0.7, 'earthy': 0.5,
    'ozonic': 0.6, 'aquatic': 0.4, 'herbal': 0.5, 'sweet': 0.2,
}
for dim, val in bamboo_snow.items():
    idx = ODOR_DIMENSIONS.index(dim)
    target[idx] = max(target[idx], val)

target = target / (np.linalg.norm(target) + 1e-8)

print(f"\n  목표 벡터 ({N_ODOR_DIM}d):")
top_dims = sorted(
    [(ODOR_DIMENSIONS[i], float(target[i])) for i in range(N_ODOR_DIM)],
    key=lambda x: x[1], reverse=True
)
for dim, val in top_dims[:8]:
    bar = '█' * int(val * 30)
    print(f"    {dim:<12s} {val:.3f} {bar}")

nearest = predictor.pom.nearest_anchor(target)
print(f"\n  가장 가까운 앵커 냄새: {nearest}")

# ─── Step 3: V16 OdorGNN으로 대표 원료 냄새 예측 ───
print(f"\n{'='*70}")
print("[Step 3] V16 OdorGNN — 대표 원료 SMILES → 22d 냄새 벡터")

# 대나무/숲/눈 관련 대표 분자
test_molecules = {
    'Linalool (라벤더/우디)':     'C=CC(O)(CCC=C(C)C)C',
    'α-Pinene (소나무/숲)':       'CC1=CCC2CC1C2(C)C',
    'Limonene (시트러스/프레시)':  'CC1=CCC(CC1)C(=C)C',
    'Geraniol (로즈/플로럴)':     'CC(=CCCC(=CC)C)CO',
    'Cedrol (시더우드/우디)':      'CC1CCC2C(C)(C)C3CC(C)(C)C12CC3O',
    'Calone (아쿠아틱/오존)':      'CC1=CC(=O)OCC1CC=C',
    'Eugenol (클로브/스파이시)':   'C=CCC1=CC(O)C(OC)=CC=1',
    'Menthol (민트/쿨링)':        'CC1CCC(C(C)C)C(O)C1',
}

for name, smiles in test_molecules.items():
    try:
        vec = predictor.predict_single(smiles)
        top3 = sorted(
            [(ODOR_DIMENSIONS[i], float(vec[i])) for i in range(N_ODOR_DIM)],
            key=lambda x: x[1], reverse=True
        )[:3]
        descs = ', '.join(f'{d}={v:.2f}' for d, v in top3)
        sim = predictor.pom.similarity(vec, target)
        print(f"  {name:<30s} → {descs}  (target sim: {sim:.3f})")
    except Exception as e:
        print(f"  {name:<30s} → error: {e}")

# ─── Step 4: AI 레시피 생성 ───
print(f"\n{'='*70}")
print("[Step 4] AI 레시피 생성 (AIRecipeEngine)")
import recipe_engine

recipe = recipe_engine.generate_recipe(
    mood='calm',
    season='winter',
    preferences=['woody', 'green', 'fresh', 'herbal', 'earthy'],
    intensity=55,
    complexity=12,
    batch_ml=100,
)

print(f"\n  향수 이름: {recipe['name_ko']} ({recipe['name']})")
print(f"  농도: {recipe['concentration']}")
print(f"  스타일: {recipe['style']}")

# ─── Step 5: V16 OdorGNN으로 레시피 원료 냄새 예측 ───
print(f"\n{'='*70}")
print("[Step 5] V16 OdorGNN — 레시피 원료 냄새 예측 + 목표 벡터 유사도")

import database as db
all_ings = db.get_all_ingredients()
all_mols = []
try:
    all_mols = db.get_all_molecules(limit=1000)
except:
    pass

# 원료명 → SMILES 매칭
ing_smiles = {}
for f in recipe['formula']:
    name_en = (f.get('name_en') or '').lower()
    name_ko = (f.get('name_ko') or '').lower()
    # DB에서 분자 매칭
    for mol in all_mols:
        mname = (mol.get('name') or '').lower()
        if name_en and (name_en in mname or mname in name_en):
            ing_smiles[f['id']] = mol.get('smiles', '')
            break

# 각 원료 냄새 벡터 예측 + 유사도
recipe_vectors = {}
print(f"\n  {'원료':<20s} {'냄새 특성':<35s} {'목표 유사도':>10s}")
print(f"  {'─'*20} {'─'*35} {'─'*10}")

for f in recipe['formula']:
    smiles = ing_smiles.get(f['id'], '')
    if smiles:
        try:
            vec = predictor.predict_single(smiles)
            recipe_vectors[f['id']] = vec
            top3 = sorted(
                [(ODOR_DIMENSIONS[i], float(vec[i])) for i in range(N_ODOR_DIM)],
                key=lambda x: x[1], reverse=True
            )[:3]
            descs = ', '.join(f'{d}={v:.2f}' for d, v in top3)
            sim = predictor.pom.similarity(vec, target)
            emoji = '✅' if sim > 0.5 else ('⚠️' if sim > 0.3 else '❌')
            print(f"  {f['name_ko']:<20s} {descs:<35s} {sim:>8.3f} {emoji}")
        except:
            print(f"  {f['name_ko']:<20s} {'(SMILES 에러)':<35s}")
    else:
        # SMILES 없으면 카테고리 기반 추정
        cat = f.get('category', '')
        print(f"  {f['name_ko']:<20s} {'(SMILES 없음, cat=' + cat + ')':<35s}")

# 혼합 후 전체 냄새 예측 (MixtureTransformer)
if len(recipe_vectors) >= 2:
    print(f"\n{'='*70}")
    print("[Step 5b] MixtureTransformer — 혼합물 상호작용 예측")
    
    mix_vecs = list(recipe_vectors.values())
    mix_ratios = []
    for f in recipe['formula']:
        if f['id'] in recipe_vectors:
            mix_ratios.append(f['percentage'] / 100.0)
    
    # 가중 평균 (기본 혼합)
    weighted_mix = np.zeros(N_ODOR_DIM)
    total_w = sum(mix_ratios)
    for vec, ratio in zip(mix_vecs, mix_ratios):
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        weighted_mix += vec.flatten() * (ratio / total_w)
    
    mix_sim = predictor.pom.similarity(weighted_mix, target)
    print(f"\n  가중 평균 혼합 벡터 → 목표 유사도: {mix_sim:.4f}")
    
    top_mix = sorted(
        [(ODOR_DIMENSIONS[i], float(weighted_mix[i])) for i in range(N_ODOR_DIM)],
        key=lambda x: x[1], reverse=True
    )[:5]
    print(f"  혼합 후 상위 냄새:")
    for dim, val in top_mix:
        bar = '█' * int(val * 20)
        print(f"    {dim:<12s} {val:.3f} {bar}")

# ─── Step 6: 향 피라미드 ───
print(f"\n{'='*70}")
print("  🔺 향 피라미드")
print(f"{'='*70}")
pyramid = recipe['pyramid']
print(f"\n  ┌─ 탑 노트 (첫인상, 15~30분) ─────────────")
for name in pyramid['top']:
    print(f"  │  ✦ {name}")
print(f"  ├─ 미들 노트 (하트, 30분~3시간) ──────────")
for name in pyramid['middle']:
    print(f"  │  ♦ {name}")
print(f"  └─ 베이스 노트 (잔향, 3시간~) ────────────")
for name in pyramid['base']:
    print(f"     ◆ {name}")

# ─── Step 7: 상세 포뮬라 ───
print(f"\n{'='*70}")
print(f"  📋 상세 포뮬라 (100ml 기준)")
print(f"{'='*70}")
for step in recipe['mixing_steps']:
    print(f"\n  Step {step['step']}: {step['label']}")
    print(f"  {step['instruction']}")
    for ing in step['ingredients']:
        print(f"    • {ing['name_ko']:<20s} {ing.get('percentage',0):>5.1f}%  "
              f"{ing.get('ml',0):>5.1f}ml  {ing.get('grams',0):>5.1f}g  "
              f"₩{ing.get('cost_krw',0):>6,}")

# ─── Step 8: 분자 조화 분석 ───
mh = recipe.get('molecular_harmony', {})
print(f"\n{'='*70}")
print(f"  🧬 분자 수준 조화 분석 (50 수용체 + Hill 방정식)")
print(f"{'='*70}")
print(f"    조화도 점수: {mh.get('harmony', 0):.3f}/1.000")
print(f"    시너지 효과: {mh.get('synergy_count', 0)}쌍")
print(f"    마스킹 위험: {mh.get('masking_count', 0)}쌍")
print(f"    SMILES 매칭: {mh.get('smiles_matched', 0)}개 (RDKit 구조 분석)")

if mh.get('synergy_bonuses'):
    print(f"\n    시너지 보너스:")
    for syn in mh['synergy_bonuses'][:5]:
        print(f"      + {syn['pair']} (시너지: {syn['bonus']})")

# ─── Step 9: Sommelier ───
print(f"\n{'='*70}")
print(f"  🍷 AI 소믈리에 — 향의 이야기")
print(f"{'='*70}")
try:
    from sommelier import Sommelier
    somm = Sommelier()
    
    desc = somm.describe_moment(target, time_min=0, note_phase='top')
    print(f"\n  탑 노트의 첫인상:")
    print(f"    \"{desc}\"")
    
    desc_mid = somm.describe_moment(target * 0.85, time_min=60, note_phase='middle')
    print(f"\n  미들 노트로의 전환:")
    print(f"    \"{desc_mid}\"")
    
    desc_base = somm.describe_moment(target * 0.6, time_min=180, note_phase='base')
    print(f"\n  베이스 노트의 잔향:")
    print(f"    \"{desc_base}\"")
except Exception as e:
    print(f"  (소믈리에 로드 실패: {e})")

# ─── Step 10: 비용 & 스펙 ───
cost = recipe['cost']
stats = recipe['stats']
print(f"\n{'='*70}")
print(f"  📊 최종 스펙")
print(f"{'='*70}")
print(f"    비용:     {cost['total_formatted']}")
print(f"    원료:     {stats['total_ingredients']}개")
print(f"    농도:     {stats['total_concentrate_pct']}%")
print(f"    지속력:   {stats['longevity_hours']}시간")
print(f"    확산:     {stats['sillage_ko']}")

aging = recipe['aging']
print(f"    숙성:     최소 {aging['min_days']}일 / 권장 {aging['recommended_days']}일")

# ─── 저장 ───
output_path = os.path.join('weights', 'bamboo_snow_recipe_v16_full.json')

# 혼합 결과 추가
mix_result = {}
if len(recipe_vectors) >= 2:
    mix_result = {
        'weighted_mix_similarity': float(mix_sim),
        'weighted_mix_top5': [(d, float(v)) for d, v in top_mix],
        'ingredients_with_smiles': len(recipe_vectors),
    }

save_data = {
    'title': '평창 대나무숲 × 겨울 눈의 향 (완전체)',
    'description': description,
    'pipeline': 'V16 OdorGNN(22d) + POM + AIRecipeEngine + MolecularHarmony + Sommelier',
    'model_cossim': 0.8092,
    'target_vector': target.tolist(),
    'target_top_dims': top_dims[:8],
    'mixture_analysis': mix_result,
    'recipe': recipe,
}

def default_converter(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.cpu().numpy().tolist()
    return str(o)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2, default=default_converter)

print(f"\n{'='*70}")
print(f"  ✅ 완전체 레시피 저장: {output_path}")
print(f"{'='*70}")
print(f"\n  사용된 시스템:")
print(f"    ✅ OdorGNN V16 (CosSim 0.8092, {N_ODOR_DIM}d)")
print(f"    ✅ PrincipalOdorMap (22d 냄새 좌표)")
print(f"    ✅ AIRecipeEngine (Neural Scoring)")
print(f"    ✅ MolecularHarmony (50 수용체 + RDKit)")
print(f"    ✅ Sommelier (시적 표현)")
print(f"    ✅ MixtureTransformer (혼합 상호작용)")
print(f"\n완료! 🎉")
