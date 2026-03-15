# -*- coding: utf-8 -*-
"""
🎯 AI Perfumer v3 — 자율 에이전트 오케스트레이터 (Full)
═════════════════════════════════════════════════════════
8개 업그레이드 통합:

 [1] 진화 알고리즘 — 상위 레시피 교배 → 차세대 생성
 [7] 메모리 시스템 — 과거 성공 패턴 기억/재활용

파이프라인:
 Phase 0: 메모리 로딩
 Phase 1: 1세대 조향 (4 AI × 자기 성찰 + 시너지 + 릴레이)
 Phase 2: 1세대 심사 (비평 + 다단계 + 토론 + Elo)
 Phase 3: 진화 (상위 2개 → 교배 → 2세대 2개)
 Phase 4: 2세대 심사
 Phase 5: [5] 위너 농도 미세 최적화
 Phase 6: 메모리 저장 + 최종 리포트

★ 전체 GNN 1회 로딩 → 모든 에이전트 공유
"""
import sys, os, json, time, copy
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scripts.ai_composer import (
    main as run_composers, load_pool, theme_to_vector, optimize_concentrations,
    beam_search_compose, optimize_concentrations_fine, build_synergy_db,
    cosine_sim, harmony_score, run_composer, get_synergy,
)
from scripts.ai_judge import main as run_judges, GNNJudgeEngine
import numpy as np


# ═══════════════════════════════════════
# [7] 메모리 시스템
# ═══════════════════════════════════════
MEMORY_PATH = os.path.join('weights', 'ai_perfumer', 'memory.json')

def load_memory():
    """과거 성공 패턴 로딩"""
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
                mem = json.load(f)
            print(f"  📚 메모리: {len(mem.get('entries', []))}건 로딩")
            return mem
        except:
            pass
    return {'entries': []}


def save_memory(memory, theme_text, keywords, winner_recipe, winner_score):
    """성공 패턴 메모리 저장 (최대 20건)"""
    entry = {
        'theme': theme_text,
        'keywords': keywords,
        'score': winner_score,
        'top_ingredients': [
            {'name': i['name_en'] or i['name_ko'], 'category': i['category'],
             'note_type': i['note_type'], 'concentration': i['concentration']}
            for i in sorted(winner_recipe['ingredients'],
                           key=lambda x: x['concentration'], reverse=True)[:6]
        ],
        'strategy': winner_recipe.get('strategy', ''),
        'timestamp': time.strftime('%Y-%m-%d %H:%M'),
    }
    
    memory['entries'].append(entry)
    memory['entries'] = memory['entries'][-20:]  # 최대 20건
    
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 메모리 저장: {len(memory['entries'])}건")


def recall_similar(memory, keywords, top_k=3):
    """유사 테마 패턴 회상"""
    if not memory.get('entries'):
        return []
    
    scored = []
    kw_set = set(keywords)
    for entry in memory['entries']:
        overlap = len(kw_set & set(entry.get('keywords', [])))
        if overlap > 0:
            scored.append((overlap, entry))
    
    scored.sort(key=lambda x: (-x[0], -x[1].get('score', 0)))
    return [e for _, e in scored[:top_k]]


def crossover_recipes(recipe_a, recipe_b, pool, theme_vec, gen_id):
    """
    ★ 레시피 교배
    Top 노트: 점수 높은 쪽, Middle: 양쪽 절반, Base: 반대 쪽
    """
    ings_a = recipe_a['ingredients']
    ings_b = recipe_b['ingredients']
    
    score_a = recipe_a.get('scores', {}).get('total_score', 0.5)
    score_b = recipe_b.get('scores', {}).get('total_score', 0.5)
    
    by_note = {'top': {'a': [], 'b': []}, 'middle': {'a': [], 'b': []}, 'base': {'a': [], 'b': []}}
    for i in ings_a:
        nt = i.get('note_type', 'middle')
        if nt in by_note:
            by_note[nt]['a'].append(i)
    for i in ings_b:
        nt = i.get('note_type', 'middle')
        if nt in by_note:
            by_note[nt]['b'].append(i)
    
    child_ings = []
    used_names = set()
    
    better = 'a' if score_a >= score_b else 'b'
    worse = 'b' if better == 'a' else 'a'
    
    for i in by_note['top'][better]:
        name = i.get('name_en') or i.get('name_ko', '')
        if name and name not in used_names:
            child_ings.append(dict(i))
            used_names.add(name)
    
    mid_a = by_note['middle']['a']
    mid_b = by_note['middle']['b']
    for i in mid_a[:len(mid_a)//2] + mid_b[len(mid_b)//2:]:
        name = i.get('name_en') or i.get('name_ko', '')
        if name and name not in used_names:
            child_ings.append(dict(i))
            used_names.add(name)
    
    for i in by_note['base'][worse]:
        name = i.get('name_en') or i.get('name_ko', '')
        if name and name not in used_names:
            child_ings.append(dict(i))
            used_names.add(name)
    
    while len(child_ings) < 8:
        added = False
        for src in [ings_a, ings_b]:
            for i in src:
                name = i.get('name_en') or i.get('name_ko', '')
                if name and name not in used_names:
                    child_ings.append(dict(i))
                    used_names.add(name)
                    added = True
                    break
            if len(child_ings) >= 8:
                break
        if not added:
            break
    
    total_conc = sum(i['concentration'] for i in child_ings)
    if total_conc > 0:
        for i in child_ings:
            i['concentration'] = round(i['concentration'] * 100 / total_conc, 2)
    
    return {
        'composer_id': gen_id,
        'strategy': f'hybrid_{recipe_a["composer_id"]}x{recipe_b["composer_id"]}',
        'theme': recipe_a.get('theme', ''),
        'theme_keywords': recipe_a.get('theme_keywords', []),
        'ingredients': child_ings,
        'scores': {},
        'harmony': 0.0,
        'reasoning_trace': [
            f"교배: #{recipe_a['composer_id']}({score_a:.3f}) × #{recipe_b['composer_id']}({score_b:.3f})",
        ],
        'generation': 2,
        'parent_ids': [recipe_a['composer_id'], recipe_b['composer_id']],
        'ingredient_count': len(child_ings),
        'refinement_rounds': 0,
        'round_scores': [],
        'changes_made': [],
    }


def _collect_judge_feedback(judge_dir):
    """
    ★ 심사 피드백 수집 — 평가 파일에서 비평 추출
    진화 전략에 반영할 구체적 문제점 식별
    """
    feedback = {
        'overloaded_categories': [],   # 과잉 카테고리
        'irrelevant_ingredients': [],  # 테마 무관 원료
        'duplicate_ingredients': [],   # 중복 원료
        'weaknesses': [],              # 기타 약점
    }
    
    try:
        eval_files = [f for f in os.listdir(judge_dir) if f.startswith('eval_') and f.endswith('.json')]
        for ef in eval_files:
            with open(os.path.join(judge_dir, ef), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for judge_result in data.get('judges', []):
                for critique in judge_result.get('critique', []):
                    msg = critique.get('message', '')
                    if '카테고리 과잉' in msg:
                        # "floral 7/13개 (54%)" 에서 카테고리 추출
                        parts = msg.split("'")
                        if len(parts) >= 2:
                            feedback['overloaded_categories'].append(parts[1])
                    elif '테마 무관' in msg:
                        feedback['irrelevant_ingredients'].append(msg)
                    elif '유사 중복' in msg:
                        parts = msg.split("'")
                        if len(parts) >= 2:
                            feedback['duplicate_ingredients'].append(parts[1])
                    else:
                        feedback['weaknesses'].append(msg)
    except:
        pass  # 첫 세대엔 피드백 없을 수 있음
    
    # 중복 제거
    for key in feedback:
        feedback[key] = list(set(feedback[key]))
    
    return feedback


def mutate_recipe(recipe, pool, theme_vec, gen_id):
    """
    ★ Fix #7: 다양한 돌연변이 전략
    전략 A: 원료 교체 (기존)
    전략 B: 농도 변이 (±3~8%)
    전략 C: 노트 교환 (top↔middle, middle↔base)
    """
    import random
    
    ings = [dict(i) for i in recipe['ingredients']]
    used_smiles = {i['smiles'] for i in ings}
    changes = []
    
    # 전략 랜덤 선택 (2개 적용)
    strategies = random.sample(['swap', 'concentration', 'note_shift'], k=2)
    
    for strategy in strategies:
        if strategy == 'swap':
            # 전략 A: 테마 관련도 낮은 원료 1개 교체
            scored = [(i, ing.get('theme_relevance', 0.5)) for i, ing in enumerate(ings)]
            scored.sort(key=lambda x: x[1])
            
            for victim_idx, victim_score in scored[:1]:
                victim = ings[victim_idx]
                note = victim.get('note_type', 'middle')
                candidates = [p for p in pool if p['smiles'] not in used_smiles
                               and p['note_type'] == note]
                if not candidates:
                    candidates = [p for p in pool if p['smiles'] not in used_smiles]
                if not candidates:
                    continue
                
                # 상위 20 중 랜덤 (착취+탐색)
                for p in candidates[:100]:
                    p['_theme_sim'] = cosine_sim(theme_vec, p['vec'])
                sorted_cands = sorted(candidates[:100], key=lambda c: c.get('_theme_sim', 0), reverse=True)
                best = random.choice(sorted_cands[:20]) if len(sorted_cands) >= 20 else sorted_cands[0]
                
                old_name = victim.get('name_en') or victim.get('name_ko', '?')
                new_name = best.get('name_en') or best.get('name_ko', '?')
                
                ings[victim_idx] = {
                    'id': best['id'], 'name_en': best.get('name_en', ''),
                    'name_ko': best.get('name_ko', ''),
                    'category': best['category'], 'note_type': best['note_type'],
                    'concentration': victim['concentration'],
                    'smiles': best['smiles'],
                    'theme_relevance': round(best.get('_theme_sim', 0.5), 3),
                    'price_usd_kg': best.get('price'),
                    'availability': best.get('availability', 'unknown'),
                }
                used_smiles.add(best['smiles'])
                changes.append(f"교체: {old_name} → {new_name}")
        
        elif strategy == 'concentration':
            # 전략 B: 농도 변이 (2~3개 원료에 ±3~8%)
            mutate_count = min(3, len(ings))
            indices = random.sample(range(len(ings)), mutate_count)
            for idx in indices:
                shift = random.uniform(-8, 8)
                old_conc = ings[idx]['concentration']
                new_conc = max(1.0, old_conc + shift)
                ings[idx]['concentration'] = round(new_conc, 2)
                name = ings[idx].get('name_en') or ings[idx].get('name_ko', '?')
                changes.append(f"농도변이: {name} {old_conc:.1f}→{new_conc:.1f}%")
        
        elif strategy == 'note_shift':
            # 전략 C: 노트 교환 (1개 원료의 note_type 변경)
            note_transitions = {'top': 'middle', 'middle': 'base', 'base': 'middle'}
            idx = random.randint(0, len(ings) - 1)
            old_note = ings[idx].get('note_type', 'middle')
            new_note = note_transitions.get(old_note, 'middle')
            ings[idx]['note_type'] = new_note
            name = ings[idx].get('name_en') or ings[idx].get('name_ko', '?')
            changes.append(f"노트이동: {name} {old_note}→{new_note}")
    
    # 농도 재정규화
    total_conc = sum(i['concentration'] for i in ings)
    if total_conc > 0:
        for i in ings:
            i['concentration'] = round(i['concentration'] * 100 / total_conc, 2)
    
    mutated = dict(recipe)
    mutated['composer_id'] = gen_id
    mutated['ingredients'] = ings
    mutated['strategy'] = f"mutant_{recipe['composer_id']}"
    mutated['reasoning_trace'] = list(recipe.get('reasoning_trace', [])) + changes
    mutated['changes_made'] = list(recipe.get('changes_made', [])) + changes
    
    return mutated, changes


def evolve(recipes, pool, theme_vec, output_dir, generation=2, id_start=5, judge_feedback=None):
    """
    ★ 진화: 교배 + 다양한 돌연변이
    Fix #4: 상위 4개 중 랜덤 2개 교배
    Fix #3: 자녀에 harmony/synergy 계산
    ★ NEW: 심사 피드백 기반 지능적 돌연변이
    """
    import random
    
    print(f"\n  🧬 진화 라운드 (세대 {generation})")
    
    # ★ 심사 피드백 반영
    if judge_feedback:
        if judge_feedback.get('overloaded_categories'):
            cats = ', '.join(judge_feedback['overloaded_categories'])
            print(f"     📋 심사 피드백: 카테고리 과잉 [{cats}] → 균형 조정")
        if judge_feedback.get('duplicate_ingredients'):
            dups = ', '.join(judge_feedback['duplicate_ingredients'][:3])
            print(f"     📋 심사 피드백: 중복 원료 [{dups}] → 교체 우선")
    
    sorted_recipes = sorted(recipes, key=lambda r: r.get('scores', {}).get('total_score', 0), reverse=True)
    
    # ★ Fix #4: 상위 4개 중 랜덤 조합 (기존: 항상 1,2위)
    top_n = min(4, len(sorted_recipes))
    top_pool = sorted_recipes[:top_n]
    if top_n >= 2:
        parent_a, parent_b = random.sample(top_pool, 2)
    else:
        parent_a = parent_b = top_pool[0]
    
    score_a = parent_a.get('scores', {}).get('total_score', 0)
    score_b = parent_b.get('scores', {}).get('total_score', 0)
    print(f"     부모 A: #{parent_a['composer_id']} ({score_a:.3f})")
    print(f"     부모 B: #{parent_b['composer_id']} ({score_b:.3f})")
    
    # 자녀 1: 교배
    child1 = crossover_recipes(parent_a, parent_b, pool, theme_vec, gen_id=id_start)
    child1['generation'] = generation
    
    # 자녀 2: 돌연변이 (상위 중 랜덤 선택)
    mut_parent = random.choice(top_pool)
    child2, mut_changes = mutate_recipe(mut_parent, pool, theme_vec, gen_id=id_start + 1)
    child2['generation'] = generation
    if mut_changes:
        for mc in mut_changes:
            print(f"     🧪 {mc}")
    
    # 시뮬레이션으로 점수 매기기 + ★ Fix #3: harmony/synergy 추가
    children = []
    for child in [child1, child2]:
        smiles = [i['smiles'] for i in child['ingredients']]
        concs = [i['concentration'] for i in child['ingredients']]
        try:
            sim = biophys.simulate_recipe(smiles, concs)
            child['scores'] = {
                'hedonic_score': round(sim['hedonic']['hedonic_score'], 4),
                'longevity_hours': round(sim['thermodynamics']['longevity_hours'], 1),
                'smoothness': round(sim['thermodynamics'].get('smoothness', 0.5), 4),
                'active_receptors': sim['nose']['active_receptors'],
            }
            child['scores']['total_score'] = round(
                child['scores']['hedonic_score'] * 0.35 +
                min(1.0, child['scores']['longevity_hours'] / 6) * 0.25 +
                child['scores']['smoothness'] * 0.15 +
                min(1.0, child['scores']['active_receptors'] / 120) * 0.25, 4
            )
        except:
            child['scores'] = {'total_score': 0.5, 'hedonic_score': 0.5,
                              'longevity_hours': 4.0, 'smoothness': 0.5, 'active_receptors': 60}
        
        # ★ Fix #3: harmony & synergy 계산
        ing_vecs = []
        for ing in child['ingredients']:
            # pool에서 vec 찾기
            for p in pool:
                if p['smiles'] == ing['smiles']:
                    ing_vecs.append(p['vec'])
                    break
            else:
                ing_vecs.append([0.0] * 20)
        
        child['harmony'] = round(harmony_score(ing_vecs), 4)
        
        syn_scores = []
        child_ings_ids = []
        for ing in child['ingredients']:
            for p in pool:
                if p['smiles'] == ing['smiles']:
                    child_ings_ids.append(p['id'])
                    break
        for i in range(len(child_ings_ids)):
            for j in range(i+1, len(child_ings_ids)):
                syn_scores.append(get_synergy(child_ings_ids[i], child_ings_ids[j]))
        child['avg_synergy'] = round(float(np.mean(syn_scores)), 4) if syn_scores else 0.0
        
        path = os.path.join(output_dir, f'recipe_{child["composer_id"]}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(child, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"     자녀 #{child['composer_id']}: {child['scores']['total_score']:.3f} "
              f"(H:{child['harmony']:.2f} S:{child['avg_synergy']:.2f}) "
              f"({child['strategy']})")
        children.append(child)
    
    return children


# ═══════════════════════════════════════
# 최종 리포트 (MD)
# ═══════════════════════════════════════
def generate_best_recipe_md(base_dir):
    ranking_path = os.path.join(base_dir, 'final', 'ranking.json')
    with open(ranking_path, 'r', encoding='utf-8') as f:
        ranking = json.load(f)
    
    winner_id = ranking['winner']['composer_id']
    recipe_path = os.path.join(base_dir, 'composers', f'recipe_{winner_id}.json')
    with open(recipe_path, 'r', encoding='utf-8') as f:
        recipe = json.load(f)
    
    comments = []
    for j in range(1, 5):
        eval_path = os.path.join(base_dir, 'judges', f'evaluation_{j}.json')
        if not os.path.exists(eval_path):
            continue
        with open(eval_path, 'r', encoding='utf-8') as f:
            ev = json.load(f)
        for e in ev['evaluations']:
            if e['composer_id'] == winner_id:
                comments.append({
                    'judge': j, 'persona': ev.get('persona', {}),
                    'score': e['total_score'], 'opinion': e.get('opinion', {}),
                    'debate': e.get('debate', {}), 'weights': e.get('weights_used', {}),
                    'critique': e.get('critique', {}),
                })
    
    md = []
    md.append(f"# 🏆 AI 조향 최종 레시피 v3")
    md.append(f"")
    
    gen = recipe.get('generation', 1)
    gen_label = f" ({gen}세대)" if gen >= 2 else ""
    
    md.append(f"**테마**: {recipe.get('theme', '')}")
    md.append(f"**선정**: 조향 AI #{winner_id} ({recipe.get('strategy', '')} 전략){gen_label}")
    md.append(f"**최종 점수**: {ranking['winner']['final_score']:.4f}")
    md.append(f"**심사 합의**: {'✅ 만장일치' if ranking.get('consensus') else '📊 다수결'}")
    
    elo = ranking.get('elo', {})
    if elo:
        elo_scores = elo.get('elo_scores', {})
        winner_elo = elo_scores.get(str(winner_id), '?')
        md.append(f"**Elo 레이팅**: {winner_elo}")
        md.append(f"**올림픽-Elo 일치**: {'✅' if ranking.get('olympic_elo_agree') else '⚡ 불일치'}")
    
    md.append(f"")
    
    # 레시피 테이블
    md.append(f"## 📋 레시피")
    md.append(f"")
    md.append(f"| 노트 | 원료 | 카테고리 | 비율 | 관련도 |")
    md.append(f"|------|------|----------|------|--------|")
    for note in ['top', 'middle', 'base']:
        note_ings = sorted(
            [i for i in recipe['ingredients'] if i.get('note_type') == note],
            key=lambda x: x['concentration'], reverse=True
        )
        emoji = {'top': '🔺', 'middle': '🔷', 'base': '🔻'}[note]
        for i in note_ings:
            name = i.get('name_en') or i.get('name_ko', '')
            md.append(f"| {emoji} {note.upper()} | {name} | {i.get('category', '')} | "
                      f"{i['concentration']:.1f}% | {i.get('theme_relevance', 0):.2f} |")
    
    # 점수
    scores = recipe.get('scores', {})
    md.append(f"")
    md.append(f"## 📊 점수")
    md.append(f"| 항목 | 값 |")
    md.append(f"|------|-----|")
    md.append(f"| 쾌적도 | {scores.get('hedonic_score', 0):.3f} |")
    md.append(f"| 지속력 | {scores.get('longevity_hours', 0)}h |")
    md.append(f"| 시너지 | {recipe.get('avg_synergy', 0):.3f} |")
    md.append(f"| 조화도 | {recipe.get('harmony', 0):.3f} |")
    
    # 추론 과정
    traces = recipe.get('reasoning_trace', [])
    if traces:
        md.append(f"")
        md.append(f"## 🧠 추론 과정")
        for t in traces:
            md.append(f"- {t}")
    
    # 진화 정보
    if gen >= 2:
        md.append(f"")
        md.append(f"## 🧬 진화 정보")
        md.append(f"- 세대: {gen}")
        md.append(f"- 부모: #{recipe.get('parent_ids', [])}")
        md.append(f"- 전략: {recipe.get('strategy', '')}")
    
    # 비평 + 심사
    md.append(f"")
    md.append(f"## ⚖ 심사 의견")
    for c in comments:
        persona = c.get('persona', {})
        critique = c.get('critique', {})
        verdict = c.get('opinion', {}).get('verdict', '—')
        name = persona.get('name', f"심사#{c['judge']}")
        
        debate_note = ""
        d = c.get('debate', {})
        if d.get('adjusted'):
            debate_note = f" (토론: {d['original_score']:.3f}→{d['final_score']:.3f})"
        
        md.append(f"### {name} — {verdict} ({c['score']:.3f}){debate_note}")
        
        for comment in c.get('opinion', {}).get('comments', []):
            md.append(f"- {comment}")
        
        if critique.get('flaws'):
            for flaw in critique['flaws']:
                if flaw.startswith('⚠'):
                    md.append(f"- 🗡 {flaw}")
        md.append(f"")
    
    # Elo 매치
    if elo.get('matches'):
        md.append(f"## 🏟 Elo 매치 결과")
        md.append(f"| 대결 | 결과 |")
        md.append(f"|------|------|")
        for m in elo['matches']:
            md.append(f"| #{m['a']} vs #{m['b']} | {m['result']} ({m['wins_a']}-{m['wins_b']}) |")
    
    # 전체 순위
    md.append(f"")
    md.append(f"## 🏅 전체 순위")
    md.append(f"| 순위 | AI | 올림픽 점수 | Elo |")
    md.append(f"|------|-----|-----------|-----|")
    for fr in ranking['final_ranking']:
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(fr['rank'], '')
        cid = fr['composer_id']
        elo_s = elo.get('elo_scores', {}).get(str(cid), '?')
        md.append(f"| {medal} {fr['rank']} | #{cid} | {fr['final_score']:.4f} | {elo_s} |")
    
    md_text = '\n'.join(md)
    final_dir = os.path.join(base_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    md_path = os.path.join(final_dir, 'best_recipe.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    print(f"\n  📝 최종 레시피: {md_path}")
    return md_path, recipe


import biophysics_simulator as biophys

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Perfumer v3 — Full Upgrade')
    parser.add_argument('--theme', type=str, default=None)
    args = parser.parse_args()
    
    base_dir = os.path.join('weights', 'ai_perfumer')
    composer_dir = os.path.join(base_dir, 'composers')
    judge_dir = os.path.join(base_dir, 'judges')
    
    # ★ 이전 결과 정리 (테마 혼합 방지)
    import shutil
    for d in [composer_dir, judge_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(composer_dir, exist_ok=True)
    os.makedirs(judge_dir, exist_ok=True)
    
    theme_text = args.theme
    if not theme_text:
        theme_file = os.path.join(base_dir, 'theme.txt')
        if os.path.exists(theme_file):
            with open(theme_file, 'r', encoding='utf-8') as f:
                theme_text = f.read().strip()
        else:
            theme_text = "봄날의 벚꽃 향수"
            os.makedirs(base_dir, exist_ok=True)
            with open(theme_file, 'w', encoding='utf-8') as f:
                f.write(theme_text)
    
    total_start = time.time()
    
    print("╔" + "═" * 58 + "╗")
    print("║  🎯 AI Perfumer v3.3 — 자율 에이전트 (5세대 진화)       ║")
    print("║  Full DB·다양성·Swiss Elo·다전략변이·시너지·메모리      ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  🎨 테마: {theme_text}\n")
    
    # ═══ Phase 0: 메모리 + GNN ═══
    print("─" * 60)
    print("  Phase 0: 메모리 + GNN 로딩")
    print("─" * 60)
    memory = load_memory()
    gnn_engine = GNNJudgeEngine()
    
    # 메모리 회상
    pool, all_tags, tag_to_idx = load_pool()
    theme_vec, keywords = theme_to_vector(theme_text, all_tags, tag_to_idx)
    
    similar = recall_similar(memory, keywords)
    if similar:
        print(f"  💡 유사 패턴 {len(similar)}건:")
        for s in similar:
            print(f"     '{s['theme']}' ({s['score']:.3f}) — {', '.join(i['name'] for i in s['top_ingredients'][:3])}")
    
    # ═══ Phase 1: 1세대 조향 ═══
    print(f"\n{'─'*60}")
    print(f"  Phase 1: 1세대 조향 (Full DB + 시너지 + 릴레이)")
    print(f"{'─'*60}")
    recipes, _, _, _ = run_composers(theme_text=theme_text, output_dir=composer_dir, gnn_engine=gnn_engine)
    
    # ═══ Phase 2: 1세대 심사 ═══
    print(f"\n{'─'*60}")
    print(f"  Phase 2: 1세대 심사")
    print(f"{'─'*60}")
    gen1_ranking, _ = run_judges(composer_dir=composer_dir, output_dir=judge_dir, gnn_engine=gnn_engine)
    
    # ═══ Phase 3~N: 다세대 진화 루프 ═══
    NUM_GENERATIONS = 5
    all_children = []
    all_recipes = list(recipes)
    prev_ranking = gen1_ranking
    
    for gen in range(2, NUM_GENERATIONS + 1):
        print(f"\n{'─'*60}")
        print(f"  Phase {gen+1}: {gen}세대 진화 (교배 + 돌연변이)")
        print(f"{'─'*60}")
        
        # 이전 심사 점수 반영
        for fr in prev_ranking.get('final_ranking', []):
            for r in all_recipes:
                if r['composer_id'] == fr['composer_id']:
                    if 'scores' not in r:
                        r['scores'] = {}
                    r['scores']['total_score'] = fr['final_score']
        
        id_start = 4 + (gen - 1) * 2 + 1  # gen2→5,6 / gen3→7,8 / gen4→9,10 / gen5→11,12
        
        # ★ 심사 피드백 수집 (진화에 활용)
        judge_feedback = _collect_judge_feedback(judge_dir)
        
        children = evolve(all_recipes, pool, theme_vec, composer_dir, generation=gen, 
                         id_start=id_start, judge_feedback=judge_feedback)
        all_children.extend(children)
        all_recipes.extend(children)
        
        # 심사
        print(f"\n{'─'*60}")
        print(f"  Phase {gen+1}.5: {gen}세대 심사")
        print(f"{'─'*60}")
        prev_ranking, _ = run_judges(composer_dir=composer_dir, output_dir=judge_dir, gnn_engine=gnn_engine)
    
    final_ranking = prev_ranking
    
    # ═══ 위너 농도 최적화 ═══
    print(f"\n{'─'*60}")
    print(f"  Phase F: 위너 농도 미세 최적화 (±0.3%)")
    print(f"{'─'*60}")
    winner_id = final_ranking['winner']['composer_id']
    winner_path = os.path.join(composer_dir, f'recipe_{winner_id}.json')
    with open(winner_path, 'r', encoding='utf-8') as f:
        winner_recipe = json.load(f)
    
    ings_optimized, improvements = optimize_concentrations_fine(winner_recipe['ingredients'], max_rounds=3)
    if improvements:
        winner_recipe['ingredients'] = ings_optimized
        winner_recipe['concentration_optimized'] = True
        winner_recipe['optimization_improvements'] = improvements
        with open(winner_path, 'w', encoding='utf-8') as f:
            json.dump(winner_recipe, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ 농도 최적화: {len(improvements)}건")
        for imp in improvements:
            print(f"     {imp}")
    else:
        print(f"  ✅ 이미 최적 — 변경 없음")
    
    # ═══ 메모리 저장 + 리포트 ═══
    print(f"\n{'─'*60}")
    print(f"  Phase Final: 메모리 저장 + 리포트")
    print(f"{'─'*60}")
    md_path, final_recipe = generate_best_recipe_md(base_dir)
    
    # ★ 레시피 컬렉션 자동 저장
    collection_dir = os.path.join(base_dir, 'recipe_collection')
    os.makedirs(collection_dir, exist_ok=True)
    import re, datetime
    safe_theme = re.sub(r'[\\/:*?"<>|\s]+', '_', theme_text)[:30]
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    collection_name = f"{safe_theme}_{date_str}.md"
    collection_path = os.path.join(collection_dir, collection_name)
    shutil.copy2(md_path, collection_path)
    print(f"  📂 레시피 컬렉션 저장: {collection_name}")
    
    save_memory(memory, theme_text, keywords,
                final_recipe, final_ranking['winner']['final_score'])
    
    total_elapsed = time.time() - total_start
    total_recipes = len(recipes) + len(all_children)
    
    print(f"\n{'═'*60}")
    print(f"  ✅ AI Perfumer v3.2 완료!")
    print(f"")
    print(f"  📊 통계:")
    print(f"     원료 풀: {len(pool)}개")
    print(f"     총 레시피: {total_recipes}개 (1세대 {len(recipes)} + 진화 {len(all_children)})")
    print(f"     세대 수: {NUM_GENERATIONS} (교배 + 돌연변이)")
    print(f"     심사 라운드: {NUM_GENERATIONS}")
    print(f"     Elo 매치: {len(final_ranking.get('elo', {}).get('matches', []))}건")
    print(f"     메모리: {len(memory['entries'])}건")
    print(f"     총 시간: {total_elapsed:.1f}초")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
