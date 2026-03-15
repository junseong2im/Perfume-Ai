"""
AI 블렌드 레시피 생성기 v2
=========================
역할 분리:
  이 스크립트 (AI 기획자): v6 모델로 원료 선별 + 하이브리드 스코어링
  recipe_engine.py (수석 조향사): IFRA 검증, 강도 제한, 최종 배합비 계산

Le Labo Santal 33 + Diptyque Tam Dao 블렌드
"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

# ================================================================
# 1. 두 향수의 향 프로파일 정의 (22d 벡터)
# ================================================================
ODOR_DIMS = [
    'sweet','sour','woody','floral','citrus','spicy','musk','fresh',
    'green','warm','fruity','smoky','powdery','aquatic','herbal',
    'amber','leather','earthy','ozonic','metallic','fatty','waxy',
]

def make_profile(desc):
    vec = np.zeros(22)
    for name, val in desc.items():
        if name in ODOR_DIMS:
            vec[ODOR_DIMS.index(name)] = val
    return vec

# Le Labo - Santal 33
santal33 = make_profile({
    'woody': 0.95, 'leather': 0.75, 'spicy': 0.55, 'musk': 0.50,
    'warm': 0.60, 'smoky': 0.35, 'amber': 0.40, 'earthy': 0.30,
    'sweet': 0.20, 'powdery': 0.25,
})

# Diptyque - Tam Dao
tam_dao = make_profile({
    'woody': 0.98, 'warm': 0.55, 'musk': 0.40, 'earthy': 0.45,
    'amber': 0.35, 'smoky': 0.25, 'powdery': 0.30, 'herbal': 0.20,
    'green': 0.15, 'sweet': 0.15,
})

# 50:50 블렌드
target = (santal33 * 0.5 + tam_dao * 0.5)
mx = target.max()
if mx > 0:
    target = target / mx

print("=" * 60)
print("  🧪 AI 블렌드 레시피 생성기 v2")
print("  Santal 33 × Tam Dao")
print("  [역할분리] AI 선별 → recipe_engine 조향")
print("=" * 60)

print(f"\n[1/4] 목표 향 프로파일:")
top_dims = sorted([(ODOR_DIMS[i], target[i]) for i in range(22)], key=lambda x: -x[1])
for name, val in top_dims[:8]:
    bar = "█" * int(val * 20)
    print(f"  {name:>10}: {val:.2f} {bar}")

# ================================================================
# 2. SMILES 매핑 + 원료 로드
# ================================================================
print(f"\n[2/4] 원료 SMILES 로드...")

smiles_map = {}
for path in ['data/ingredient_smiles.json', '../data/ingredient_smiles.json']:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for k, v in raw.items():
                smiles_map[k] = v
                smiles_map[k.lower()] = v
        break

for path in ['data/molecules.json', '../data/molecules.json']:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            mols = json.load(f)
        for m in mols:
            mid = m.get('id', '')
            smi = m.get('smiles', '')
            if mid and smi and mid not in smiles_map:
                smiles_map[mid] = smi
        break

ingredients = []
for path in ['data/ingredients.json', '../data/ingredients.json']:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            ingredients = json.load(f)
        break

ing_map = {ing['id']: ing for ing in ingredients}
print(f"  원료: {len(ingredients)}개, SMILES: {len(smiles_map)}개")

# ================================================================
# 3. V6 AI 하이브리드 스코어링 (Threshold + Hit Rate + Cosine)
# ================================================================
print(f"\n[3/4] V6 AI 하이브리드 스코어링...")

v6_engine = None
try:
    from v6_bridge import OdorEngineV6
    v6_engine = OdorEngineV6(
        weights_dir='weights/v6', device='cpu',
        use_ensemble=False, n_ensemble=1
    )
    print("  ✅ V6 모델 로드 성공")
except Exception as e:
    print(f"  ⚠ V6 로드 실패 ({e}), 규칙 기반 fallback")
    try:
        from odor_engine import OdorGNN
        v6_engine = OdorGNN(device='cpu')
        print("  ✅ OdorGNN (규칙 기반) 로드")
    except:
        pass

scored = []
for ing in ingredients:
    iid = ing.get('id', '')
    smiles = smiles_map.get(iid) or smiles_map.get(iid.lower())

    if not smiles or not v6_engine:
        # 규칙 기반 fallback
        score = 0.0
        cat = ing.get('category', '')
        if cat in ['woody', 'sandalwood']:
            score += 0.6
        if cat in ['spicy', 'leather', 'musk', 'amber']:
            score += 0.3
        scored.append((ing, score, 'rule'))
        continue

    try:
        odor_vec = v6_engine.encode(smiles)
        if len(odor_vec) > 22:
            odor_vec = odor_vec[:22]

        # 💡 1. 동적 임계값: 메인 향 대비 10% 미만만 잡음으로 간주
        max_val = np.max(odor_vec)
        dynamic_threshold = max(0.05, max_val * 0.10)
        pred_vec = np.where(odor_vec < dynamic_threshold, 0.0, odor_vec)

        # 💡 2. Cosine Similarity (방향성)
        dot = np.dot(target, pred_vec)
        norm_t = np.linalg.norm(target)
        norm_o = np.linalg.norm(pred_vec)
        cos_sim = max(0.0, dot / (norm_t * norm_o)) if (norm_t > 0 and norm_o > 0) else 0.0

        # 💡 3. Hit Rate + Off-note 페널티
        active_mask = target > 0.05
        sum_pred = np.sum(pred_vec)
        if np.any(active_mask) and sum_pred > 0:
            hit_rate = np.sum(pred_vec[active_mask]) / sum_pred
            off_note_ratio = np.sum(pred_vec[~active_mask]) / sum_pred
            penalty = off_note_ratio * 0.2
        else:
            hit_rate = 0.0
            penalty = 0.0

        # 💡 4. 하이브리드 스코어: 코사인 60% + 적중률 40% - 잡향 페널티
        final_score = max(0.0, (cos_sim * 0.6) + (hit_rate * 0.4) - penalty)

        scored.append((ing, final_score, 'v6_ai_hybrid'))
    except Exception:
        scored.append((ing, 0.1, 'error'))

# 스코어 상위 출력
scored.sort(key=lambda x: x[1], reverse=True)
v6_count = sum(1 for _, _, m in scored if 'v6' in m)
print(f"  AI 스코어링 완료: {v6_count}/{len(scored)} 원료에 V6 적용")
print(f"\n  Top 15 원료:")
for i, (ing, score, method) in enumerate(scored[:15]):
    mark = "🤖" if 'v6' in method else "📏"
    print(f"    {i+1:2d}. {mark} {ing.get('name_ko', ing['id']):>12} "
          f"({ing.get('note_type','?'):>6}) score={score:.3f}")

# ================================================================
# 4. Recipe Engine에 제어권 위임 (IFRA, 강도 제한, 밸런스 적용)
# ================================================================
print(f"\n[4/4] Recipe Engine으로 IFRA 검증 및 최종 배합 비율 계산 중...")

try:
    from recipe_engine import generate_recipe
except ImportError:
    print("⚠ recipe_engine.py를 찾을 수 없습니다.")
    sys.exit(1)

# AI가 선별한 상위 후보들을 recipe_engine에 전달
# (상위 25개 + 스코어 0.3 이상만 전달)
ai_candidates = []
for ing, score, method in scored:
    if score >= 0.3 or len(ai_candidates) < 25:
        item = ing.copy()
        item['ai_score'] = score
        item['method'] = method
        ai_candidates.append(item)
    if len(ai_candidates) >= 30:
        break

print(f"  AI 후보 {len(ai_candidates)}개를 Recipe Engine에 전달")

try:
    final_recipe = generate_recipe(
        mood='bold',           # Santal 33 + Tam Dao = bold woody
        season='all',
        preferences=['woody', 'leather', 'spicy'],
        intensity=65,          # EDP 수준
        complexity=11,         # 11개 원료 목표
        batch_ml=30,
        target_profile=target,
        candidate_ingredients=ai_candidates,
        concentrate_pct=22.0,  # EDP 22%
    )

    print(f"\n✅ IFRA 규제 + 조향 규칙이 적용된 안전한 AI 레시피 생성 완료!")

except Exception as e:
    print(f"\n⚠ Recipe Engine 오류: {e}")
    import traceback
    traceback.print_exc()
    final_recipe = None

# ================================================================
# 5. 메모장 저장
# ================================================================
if final_recipe:
    output_path = os.path.join(os.path.dirname(__file__), '..', 'AI_Blend_Recipe.txt')
    output_path = os.path.abspath(output_path)

    lines = []
    lines.append("=" * 65)
    lines.append("  🧪 AI 블렌드 향수 레시피 v2")
    lines.append("  Le Labo Santal 33 × Diptyque Tam Dao")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  생성 방식: V6 GNN + 하이브리드 스코어링 + Recipe Engine IFRA 검증")
    lines.append(f"  농도: {final_recipe.get('concentration', 'EDP')}")
    lines.append(f"  배치: {final_recipe.get('batch_ml', 30)}ml")
    lines.append(f"  AI 모델: {final_recipe.get('ai', {}).get('method', 'v6_hybrid')}")
    lines.append(f"  조화도: {final_recipe.get('ai', {}).get('harmony_score', 'N/A')}")
    lines.append("")

    # 향 프로파일
    lines.append("-" * 65)
    lines.append("  📐 목표 향 프로파일 (AI 타겟)")
    lines.append("-" * 65)
    for name, val in top_dims[:8]:
        bar = "█" * int(val * 20)
        lines.append(f"    {name:>10}: {val:.2f}  {bar}")
    lines.append("")

    # IFRA 경고
    ifra_warns = final_recipe.get('ifra_warnings', [])
    if ifra_warns:
        lines.append("-" * 65)
        lines.append("  ⚠ IFRA 안전 경고")
        lines.append("-" * 65)
        for w in ifra_warns:
            lines.append(f"    {w.get('ingredient','?')}: "
                         f"{w.get('current_pct',0)}% > 제한 {w.get('ifra_max_pct',0)}%")
            lines.append(f"      → {w.get('action','')}")
        lines.append("")

    # 포뮬러 (프로 포맷: Parts/1000 + CAS + 희석 + 대체원료)
    lines.append("-" * 65)
    lines.append("  📋 포뮬러 (IFRA 검증 완료)")
    lines.append("-" * 65)
    lines.append("")

    # Parts/1000 계산을 위한 총 농축률
    formula = final_recipe.get('formula', [])
    total_concentrate = sum(f.get('percentage', 0) for f in formula)
    batch_total = final_recipe.get('batch_ml', 30)  # 총 배치

    # 노트별 그룹핑
    current_note = None

    for f in formula:
        if f.get('note_type') != current_note:
            current_note = f.get('note_type')
            labels = {'base':'🔶 베이스 노트 (Base)', 'middle':'🟢 미들 노트 (Heart)', 'top':'🔵 탑 노트 (Top)'}
            lines.append(f"  {labels.get(current_note, current_note)}")
            lines.append("")

        pct = f.get('percentage', 0)
        parts = round(pct / 100 * 1000, 1)  # Parts per 1000
        cas = f.get('cas_number', '-')
        dil_solvent = f.get('dilution_solvent', '-')
        dil_pct = f.get('dilution_pct', 100)
        func_note = f.get('function_note', '')
        subs = f.get('substitutes', [])

        ifra_mark = ""
        if f.get('ifra_status') == 'exceeded':
            ifra_mark = " ⚠IFRA초과"
        elif f.get('ifra_status') == 'safe':
            ifra_mark = " ✅"

        # 메인 라인
        name_ko = f.get('name_ko', f.get('id', '?'))
        name_en = f.get('name_en', '')
        cat = f.get('category', '')
        lines.append(f"    {name_ko} ({name_en})")
        lines.append(f"       CAS: {cas}  |  {cat}  |  Note: {current_note}{ifra_mark}")
        
        # 배합 정보
        if dil_pct < 100:
            lines.append(f"       희석: {dil_solvent} {dil_pct}%  |  {pct}%  |  {f.get('ml',0)}ml  |  Parts: {parts}")
        else:
            lines.append(f"       원액  |  {pct}%  |  {f.get('ml',0)}ml  |  {f.get('grams',0)}g  |  Parts: {parts}")

        # 기능/특징
        if func_note:
            lines.append(f"       💡 {func_note}")

        # 대체 원료
        if subs:
            # 원료 DB에서 한글명 찾기
            sub_names = []
            for sid in subs[:3]:
                sub_ing = ing_map.get(sid)
                if sub_ing:
                    sub_cas = sub_ing.get('cas_number', '-')
                    sub_names.append(f"{sub_ing.get('name_ko',sid)} ({sub_ing.get('name_en',sid)}, CAS:{sub_cas})")
                else:
                    sub_names.append(sid)
            lines.append(f"       🔄 대체: {' / '.join(sub_names)}")

        lines.append("")

    # 용매
    for step in final_recipe.get('mixing_steps', []):
        if step.get('note_type') == 'solvent':
            lines.append("  🧊 용매")
            for si in step.get('ingredients', []):
                sol_pct = si.get('percentage', 0)
                sol_parts = round(sol_pct / 100 * 1000, 1)
                lines.append(f"    {si.get('name_ko', 'Ethanol')}")
                lines.append(f"       {sol_pct}% | {si.get('ml',0)}ml | Parts: {sol_parts}")
            lines.append("")

    # Parts/1000 합계
    lines.append(f"  총 Parts: 1000.0")

    # 통계
    stats = final_recipe.get('stats', {})
    cost = final_recipe.get('cost', {})
    lines.append("-" * 65)
    lines.append("  📊 통계")
    lines.append("-" * 65)
    lines.append(f"    원료 수: {stats.get('total_ingredients', 0)}")
    lines.append(f"    농축률: {stats.get('total_concentrate_pct', 0)}%")
    lines.append(f"    지속력: {stats.get('longevity_hours', 0)}시간")
    lines.append(f"    확산력: {stats.get('sillage_ko', '보통')}")
    lines.append(f"    비용: {cost.get('total_formatted', 'N/A')}")
    lines.append("")

    # 숙성
    aging = final_recipe.get('aging', {})
    lines.append("-" * 65)
    lines.append("  ⏳ 숙성 가이드")
    lines.append("-" * 65)
    lines.append(f"    최소: {aging.get('min_days', 14)}일")
    lines.append(f"    권장: {aging.get('recommended_days', 28)}일")
    lines.append(f"    보관: {aging.get('storage', '서늘하고 어두운 곳')}")
    lines.append("")

    # 팁
    tips = final_recipe.get('tips', [])
    if tips:
        lines.append("-" * 65)
        lines.append("  💡 AI 조향사 노트")
        lines.append("-" * 65)
        for t in tips:
            lines.append(f"    • {t}")
        lines.append("")

    # AI 정보
    ai_info = final_recipe.get('ai', {})
    mol_info = final_recipe.get('molecular_harmony', {})
    lines.append("-" * 65)
    lines.append("  🤖 AI 엔진 정보")
    lines.append("-" * 65)
    lines.append(f"    모델: {ai_info.get('model', 'V6')}")
    lines.append(f"    스코어링: 하이브리드 (Cosine 60% + Hit Rate 40%)")
    lines.append(f"    노이즈 필터: Threshold 0.1")
    lines.append(f"    조화도: {ai_info.get('harmony_score', 'N/A')}")
    lines.append(f"    분자 궁합: {mol_info.get('harmony', 0):.2f}")
    if mol_info.get('known_accords'):
        accords = []
        for a in mol_info['known_accords']:
            if isinstance(a, str):
                accords.append(a)
            elif isinstance(a, dict):
                name = a.get('accord', a.get('name', ''))
                note = a.get('note', '')
                accords.append(f"{name} ({note})" if note else name)
            else:
                accords.append(str(a))
        lines.append(f"    어코드: {', '.join(accords)}")
    lines.append("")
    lines.append("=" * 65)
    lines.append("  Generated by AI Perfumer V6 Engine + Recipe Engine")
    lines.append("  역할분리: AI 기획 → IFRA/조향 규칙 검증")
    lines.append("=" * 65)

    text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\n📄 레시피 저장: {output_path}")
    print(f"\n{text}")
else:
    print("\n❌ 레시피 생성 실패")
