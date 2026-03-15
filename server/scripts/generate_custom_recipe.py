"""
Custom Perfume Recipe Generator — 기존 SelfPlayRL 시스템 활용
==============================================================
기존에 구축된 SelfPlayRL (PPO 강화학습) 에이전트를 그대로 사용하여
고객 입력 기반 레시피를 생성합니다.

사용법:
  python generate_custom_recipe.py "따뜻하고 달콤한 겨울 향수"
  python generate_custom_recipe.py "clean fresh summer scent"
"""
import json, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import biophysics_simulator as biophys

# ==============================
# 1) 고객 입력 파싱 (키워드 → 선호 카테고리)
# ==============================
KEYWORD_MAP = {
    # 계절
    '봄': {'cats': ['floral', 'green', 'citrus'], 'mood': 'bright'},
    'spring': {'cats': ['floral', 'green', 'citrus'], 'mood': 'bright'},
    '여름': {'cats': ['citrus', 'aquatic', 'green', 'herbal'], 'mood': 'fresh'},
    'summer': {'cats': ['citrus', 'aquatic', 'green', 'herbal'], 'mood': 'fresh'},
    '가을': {'cats': ['spicy', 'fruity', 'woody', 'amber'], 'mood': 'warm'},
    'autumn': {'cats': ['spicy', 'fruity', 'woody', 'amber'], 'mood': 'warm'},
    '겨울': {'cats': ['spicy', 'amber', 'gourmand', 'woody', 'musk'], 'mood': 'cozy'},
    'winter': {'cats': ['spicy', 'amber', 'gourmand', 'woody', 'musk'], 'mood': 'cozy'},
    # 느낌
    '시원한': {'cats': ['citrus', 'aquatic', 'green'], 'mood': 'fresh'},
    '상쾌한': {'cats': ['citrus', 'green', 'herbal'], 'mood': 'fresh'},
    'fresh': {'cats': ['citrus', 'aquatic', 'green'], 'mood': 'fresh'},
    'clean': {'cats': ['citrus', 'aquatic', 'herbal'], 'mood': 'fresh'},
    '따뜻한': {'cats': ['spicy', 'amber', 'woody', 'gourmand'], 'mood': 'warm'},
    'warm': {'cats': ['spicy', 'amber', 'woody', 'gourmand'], 'mood': 'warm'},
    '달콤한': {'cats': ['fruity', 'gourmand', 'amber', 'floral'], 'mood': 'sweet'},
    'sweet': {'cats': ['fruity', 'gourmand', 'amber', 'floral'], 'mood': 'sweet'},
    '섹시한': {'cats': ['musk', 'amber', 'leather', 'spicy'], 'mood': 'sensual'},
    'sexy': {'cats': ['musk', 'amber', 'leather', 'spicy'], 'mood': 'sensual'},
    '로맨틱': {'cats': ['floral', 'fruity', 'musk'], 'mood': 'romantic'},
    'romantic': {'cats': ['floral', 'fruity', 'musk'], 'mood': 'romantic'},
    '고급스러운': {'cats': ['woody', 'amber', 'leather', 'spicy'], 'mood': 'luxurious'},
    'elegant': {'cats': ['woody', 'amber', 'leather', 'spicy'], 'mood': 'luxurious'},
    '자연스러운': {'cats': ['green', 'herbal', 'woody'], 'mood': 'natural'},
    'natural': {'cats': ['green', 'herbal', 'woody'], 'mood': 'natural'},
    # 향 카테고리
    '플로럴': {'cats': ['floral'], 'mood': 'floral'},
    'floral': {'cats': ['floral'], 'mood': 'floral'},
    '꽃': {'cats': ['floral'], 'mood': 'floral'},
    '우디': {'cats': ['woody'], 'mood': 'woody'},
    'woody': {'cats': ['woody'], 'mood': 'woody'},
    '오리엔탈': {'cats': ['amber', 'spicy', 'musk'], 'mood': 'oriental'},
    'oriental': {'cats': ['amber', 'spicy', 'musk'], 'mood': 'oriental'},
    '시트러스': {'cats': ['citrus'], 'mood': 'citrus'},
    'citrus': {'cats': ['citrus'], 'mood': 'citrus'},
    '머스크': {'cats': ['musk'], 'mood': 'musky'},
    'musk': {'cats': ['musk'], 'mood': 'musky'},
    '바닐라': {'cats': ['gourmand'], 'mood': 'gourmand'},
    'vanilla': {'cats': ['gourmand'], 'mood': 'gourmand'},
    # 성별
    '남성': {'cats': ['woody', 'leather', 'spicy', 'herbal'], 'mood': 'masculine'},
    '남자': {'cats': ['woody', 'leather', 'spicy', 'herbal'], 'mood': 'masculine'},
    '여성': {'cats': ['floral', 'fruity', 'musk'], 'mood': 'feminine'},
    '여자': {'cats': ['floral', 'fruity', 'musk'], 'mood': 'feminine'},
}

def parse_user_input(text):
    """자연어 입력 → 선호 카테고리 목록"""
    text_lower = text.lower().strip()
    preferred = []
    moods = []
    for kw, profile in KEYWORD_MAP.items():
        if kw in text_lower:
            preferred.extend(profile['cats'])
            moods.append(profile['mood'])
    
    if not preferred:
        preferred = ['citrus', 'floral', 'woody', 'musk']
        moods = ['balanced']
    
    # 빈도 기반 정렬
    from collections import Counter
    counts = Counter(preferred)
    preferred_ranked = [item for item, _ in counts.most_common()]
    
    return preferred_ranked, list(set(moods))

# ==============================
# 2) SelfPlayRL 기반 레시피 생성
# ==============================
def generate_for_customer(user_input, n_results=3, n_candidates=20):
    """
    기존 SelfPlayRL 에이전트를 사용하여 레시피 생성.
    
    1) SelfPlayRL.generate_recipe()로 후보 N개 생성
    2) 고객 선호 카테고리에 맞는 것 필터링/정렬
    3) 상위 n_results개 반환
    """
    print("=" * 60)
    print(f"  🧴 AI PERFUMER (SelfPlayRL Engine)")
    print(f"  고객 요청: \"{user_input}\"")
    print("=" * 60)
    
    # 고객 입력 파싱
    preferred_cats, moods = parse_user_input(user_input)
    print(f"\n  📋 분석 결과:")
    print(f"     무드: {', '.join(moods)}")
    print(f"     선호 카테고리: {', '.join(preferred_cats)}")
    
    # SelfPlayRL 에이전트 가져오기
    print(f"\n  🤖 SelfPlayRL 에이전트 로딩...")
    rl = biophys.get_rl()
    print(f"     원료 풀: {rl.n_ingredients}개")
    print(f"     정책 네트워크: PolicyNetwork → {rl.device}")
    
    # 후보 레시피 생성
    print(f"\n  Phase 1: SelfPlayRL로 후보 {n_candidates}개 생성...")
    candidates = []
    for i in range(n_candidates):
        recipe = rl.generate_recipe(n_ingredients=random.randint(7, 12))
        if recipe is None:
            continue
        
        # 선호도 매칭 점수 계산
        ing_cats = [ing.get('category', '').lower() for ing in recipe['ingredients']]
        match_count = sum(1 for cat in ing_cats if cat in preferred_cats)
        match_ratio = match_count / max(len(ing_cats), 1)
        
        recipe['match_ratio'] = match_ratio
        recipe['match_count'] = match_count
        
        eval_data = recipe['evaluation']
        reward = eval_data['reward']
        hedonic = eval_data['hedonic']['hedonic_score']
        longevity = eval_data['thermodynamics']['longevity_hours']
        smoothness = eval_data['thermodynamics']['smoothness']
        receptors = eval_data['nose']['active_receptors']
        
        # 고객 맞춤 종합 점수 = reward * 0.6 + 선호도 매칭 * 0.4
        customer_score = reward * 0.6 + match_ratio * 0.4
        
        recipe['customer_score'] = round(customer_score, 4)
        recipe['scores'] = {
            'reward': reward,
            'hedonic': hedonic,
            'longevity_hours': longevity,
            'smoothness': smoothness,
            'active_receptors': receptors,
        }
        candidates.append(recipe)
    
    print(f"  → 유효 후보: {len(candidates)}개")
    if not candidates:
        print("  ❌ 레시피 생성 실패")
        return []
    
    # 고객 점수 기준 정렬
    candidates.sort(key=lambda x: x['customer_score'], reverse=True)
    
    best = candidates[0]
    print(f"  → 최고 점수: {best['customer_score']:.3f} "
          f"(RL보상={best['scores']['reward']:.3f}, 매칭={best['match_ratio']:.0%})")
    
    # 다양한 Top N 선택 (원료 중복 회피)
    final = []
    for c in candidates:
        if final:
            names = set(i.get('name_ko', '') or i.get('category', '') for i in c['ingredients'])
            too_similar = any(
                len(names & set(j.get('name_ko', '') or j.get('category', '') for j in f['ingredients'])) / 
                max(len(names), 1) > 0.5
                for f in final
            )
            if too_similar:
                continue
        final.append(c)
        if len(final) >= n_results:
            break
    
    # 부족하면 채우기
    while len(final) < n_results and len(candidates) > len(final):
        for c in candidates:
            if c not in final:
                final.append(c)
                break
        if len(final) >= n_results:
            break
    
    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"  🏆 추천 레시피 {len(final)}개")
    print(f"{'=' * 60}")
    
    output = []
    for idx, recipe in enumerate(final, 1):
        s = recipe['scores']
        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  레시피 #{idx}  (SelfPlayRL Gen {recipe['generation']})")
        print(f"  │  종합: {recipe['customer_score']:.3f} | RL보상: {s['reward']:.3f}")
        print(f"  │  쾌락도: {s['hedonic']:.2f} | 지속: {s['longevity_hours']}h")
        print(f"  │  부드러움: {s['smoothness']:.2f} | 수용체: {s['active_receptors']}개")
        print(f"  │  선호 매칭: {recipe['match_ratio']:.0%} ({recipe['match_count']}개 일치)")
        print(f"  └─────────────────────────────────────────────────┘")
        
        for ing in recipe['ingredients']:
            name = ing.get('name_ko', '') or ing.get('category', '')
            cat = ing.get('category', '')
            pct = ing.get('percentage', 0)
            matched = '✓' if cat.lower() in preferred_cats else ' '
            print(f"    {matched} {name:28s} [{cat:8s}] {pct:.1f}%")
        
        output.append({
            'rank': idx,
            'customer_score': recipe['customer_score'],
            'reward': s['reward'],
            'hedonic': s['hedonic'],
            'longevity_hours': s['longevity_hours'],
            'smoothness': s['smoothness'],
            'active_receptors': s['active_receptors'],
            'match_ratio': recipe['match_ratio'],
            'ingredients': recipe['ingredients'],
            'generation': recipe['generation'],
        })
    
    # JSON 저장
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'custom_recipe.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'request': user_input,
            'preferred_categories': preferred_cats,
            'moods': moods,
            'engine': 'SelfPlayRL (PPO)',
            'recipes': output,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 저장: {out_path}")
    
    return output


if __name__ == '__main__':
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        demos = [
            "여름에 어울리는 시원하고 상쾌한 향수",
            "따뜻하고 달콤한 겨울 밤 향수",
            "고급스러운 남성용 우디 향수",
        ]
        print("\n" + "=" * 60)
        print("  🎯 데모 모드: 3가지 고객 요청 테스트")
        print("=" * 60)
        
        for demo in demos:
            generate_for_customer(demo, n_results=1, n_candidates=15)
            print()
        sys.exit(0)
    
    start = time.time()
    generate_for_customer(user_input, n_results=3, n_candidates=20)
    print(f"\n  ⏱ 소요시간: {time.time()-start:.1f}초")
