"""
Original Perfume Recipe Generator
==================================
전체 시스템(VirtualNose + HedonicFunction + ThermodynamicsEngine) 사용
진화적 탐색으로 최적 레시피 10개 생성

전략:
 1. DB에서 전체 원료 로딩 (SMILES 매핑)
 2. 스타일별 "씨앗" 레시피 생성 (전문가 지식 기반)
 3. 돌연변이 + 교차로 1000+ 변형 탐색
 4. simulate_recipe()로 전체 평가
 5. Top 10 출력
"""
import json, os, sys, time, random, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import biophysics_simulator as biophys
import database as db

# ==============================
# 1) 원료 풀 로딩
# ==============================
def load_ingredient_pool():
    """DB에서 원료 + SMILES 매핑"""
    ings = db.get_all_ingredients()
    mols = db.get_all_molecules(limit=6000)
    
    mol_smiles = {}
    for mol in mols:
        name = (mol.get('name') or '').lower().strip()
        if name and mol.get('smiles'):
            mol_smiles[name] = mol['smiles']
    
    pool = []
    for ing in ings:
        name_en = (ing.get('name_en') or '').lower().strip()
        name_ko = (ing.get('name_ko') or '').strip()
        cat = (ing.get('category') or '').lower()
        note = (ing.get('note_type') or '').lower()
        
        # SMILES 매칭
        smiles = None
        for mname, msmi in mol_smiles.items():
            if name_en and (name_en == mname or name_en in mname or mname in name_en):
                smiles = msmi
                break
        
        if smiles is None:
            # 카테고리 기반 fallback
            smiles = _cat_smiles(cat)
        
        pool.append({
            'id': ing['id'],
            'name_en': ing.get('name_en', ''),
            'name_ko': name_ko,
            'category': cat,
            'note_type': note,
            'smiles': smiles,
            'has_db_smiles': smiles != _cat_smiles(cat),
        })
    
    return pool

def _cat_smiles(cat):
    defaults = {
        'floral': 'OCC1=CC=CC=C1',
        'citrus': 'CC(=CCC/C(=C/CO)C)C',
        'woody': 'CC1CCC2(C)C(O)CCC12',
        'spicy': 'C=CC1=CC(OC)=C(O)C=C1',
        'fruity': 'CCCCCC(=O)OC',
        'gourmand': 'O=CC1=CC(OC)=C(O)C=C1',
        'musk': 'O=C1CCCCCCCCCCCCC1',
        'amber': 'CC12CCCC(C)(C1)C1CCC(O)CC12',
        'herbal': 'CC(O)CC=CC(C)C',
        'green': 'CC/C=C\\CCO',
        'fresh': 'CC(=O)OCC=C(C)C',
        'aquatic': 'O=CC1CCCCC1',
        'leather': 'CC1=CC=C(O)C=C1',
        'powdery': 'O=C(O)C1=CC=CC=C1',
    }
    return defaults.get(cat, 'CCCCCCO')

# ==============================
# 2) 씨앗 레시피 템플릿 (전문가 지식)
# ==============================
RECIPE_TEMPLATES = [
    {
        'name': 'Fresh Citrus Aromatic',
        'style': 'fresh',
        'structure': {
            'top': {'categories': ['citrus', 'herbal', 'green'], 'count': 3, 'pct': 15},
            'middle': {'categories': ['floral', 'herbal', 'spicy'], 'count': 3, 'pct': 35},
            'base': {'categories': ['woody', 'musk', 'amber'], 'count': 3, 'pct': 50},
        }
    },
    {
        'name': 'Romantic Floral',
        'style': 'floral',
        'structure': {
            'top': {'categories': ['citrus', 'green', 'fruity'], 'count': 2, 'pct': 15},
            'middle': {'categories': ['floral'], 'count': 4, 'pct': 40},
            'base': {'categories': ['musk', 'woody', 'amber'], 'count': 3, 'pct': 45},
        }
    },
    {
        'name': 'Dark Oriental',
        'style': 'oriental',
        'structure': {
            'top': {'categories': ['spicy', 'citrus'], 'count': 2, 'pct': 10},
            'middle': {'categories': ['floral', 'spicy', 'herbal'], 'count': 3, 'pct': 30},
            'base': {'categories': ['amber', 'woody', 'gourmand', 'musk'], 'count': 4, 'pct': 60},
        }
    },
    {
        'name': 'Clean Woody',
        'style': 'woody',
        'structure': {
            'top': {'categories': ['citrus', 'herbal'], 'count': 2, 'pct': 15},
            'middle': {'categories': ['woody', 'herbal', 'floral'], 'count': 3, 'pct': 35},
            'base': {'categories': ['woody', 'amber', 'musk'], 'count': 3, 'pct': 50},
        }
    },
    {
        'name': 'Sensual Gourmand',
        'style': 'oriental',
        'structure': {
            'top': {'categories': ['fruity', 'citrus'], 'count': 2, 'pct': 12},
            'middle': {'categories': ['floral', 'gourmand', 'spicy'], 'count': 3, 'pct': 33},
            'base': {'categories': ['gourmand', 'amber', 'musk', 'woody'], 'count': 4, 'pct': 55},
        }
    },
    {
        'name': 'Mediterranean Fresh',
        'style': 'fresh',
        'structure': {
            'top': {'categories': ['citrus', 'herbal', 'aquatic'], 'count': 3, 'pct': 20},
            'middle': {'categories': ['herbal', 'floral', 'green'], 'count': 3, 'pct': 35},
            'base': {'categories': ['woody', 'musk'], 'count': 2, 'pct': 45},
        }
    },
    {
        'name': 'Elegant Leather',
        'style': 'leather',
        'structure': {
            'top': {'categories': ['citrus', 'spicy'], 'count': 2, 'pct': 12},
            'middle': {'categories': ['leather', 'floral', 'spicy'], 'count': 3, 'pct': 33},
            'base': {'categories': ['leather', 'woody', 'amber', 'musk'], 'count': 4, 'pct': 55},
        }
    },
    {
        'name': 'Fruity Floral',
        'style': 'floral',
        'structure': {
            'top': {'categories': ['fruity', 'citrus', 'green'], 'count': 3, 'pct': 18},
            'middle': {'categories': ['floral', 'fruity'], 'count': 3, 'pct': 37},
            'base': {'categories': ['musk', 'woody', 'amber'], 'count': 3, 'pct': 45},
        }
    },
    {
        'name': 'Spicy Night',
        'style': 'oriental',
        'structure': {
            'top': {'categories': ['spicy', 'citrus', 'herbal'], 'count': 3, 'pct': 15},
            'middle': {'categories': ['spicy', 'floral'], 'count': 3, 'pct': 30},
            'base': {'categories': ['amber', 'woody', 'musk', 'gourmand'], 'count': 4, 'pct': 55},
        }
    },
    {
        'name': 'Green Tea Garden',
        'style': 'fresh',
        'structure': {
            'top': {'categories': ['green', 'citrus'], 'count': 2, 'pct': 18},
            'middle': {'categories': ['green', 'herbal', 'floral'], 'count': 3, 'pct': 37},
            'base': {'categories': ['woody', 'musk'], 'count': 2, 'pct': 45},
        }
    },
]

# ==============================
# 3) 레시피 생성 + 진화적 탐색
# ==============================
def generate_recipe(template, pool, mutation_rate=0.3):
    """템플릿 기반 레시피 생성 (랜덤 원료 선택)"""
    # 카테고리별 원료 인덱스
    cat_pool = {}
    for ing in pool:
        cat_pool.setdefault(ing['category'], []).append(ing)
    
    smiles_list = []
    concentrations = []
    ingredients = []
    
    for note_type in ['top', 'middle', 'base']:
        spec = template['structure'][note_type]
        cats = spec['categories']
        n = spec['count']
        pct_total = spec['pct']
        
        selected = []
        for _ in range(n):
            # 랜덤 카테고리 선택
            cat = random.choice(cats)
            pool_for_cat = cat_pool.get(cat, cat_pool.get('floral', []))
            if not pool_for_cat:
                continue
            
            # 가능하면 DB SMILES가 있는 원료 우선
            db_ings = [i for i in pool_for_cat if i['has_db_smiles']]
            if db_ings and random.random() > 0.3:
                ing = random.choice(db_ings)
            else:
                ing = random.choice(pool_for_cat)
            
            selected.append(ing)
        
        if not selected:
            continue
        
        # 비율 분배 (약간의 랜덤성)
        raw_ratios = [random.uniform(0.5, 1.5) for _ in selected]
        total = sum(raw_ratios)
        for ing, ratio in zip(selected, raw_ratios):
            conc = (ratio / total) * pct_total
            smiles_list.append(ing['smiles'])
            concentrations.append(round(conc, 2))
            ingredients.append({
                'name': ing['name_en'] or ing['name_ko'],
                'category': ing['category'],
                'note_type': note_type,
                'concentration': round(conc, 2),
            })
    
    return smiles_list, concentrations, ingredients

def mutate_recipe(smiles_list, concentrations, ingredients, pool, mutation_rate=0.3):
    """레시피 돌연변이 — 원료 교체 또는 비율 변경"""
    new_smi = list(smiles_list)
    new_conc = list(concentrations)
    new_ings = [dict(i) for i in ingredients]
    
    cat_pool = {}
    for ing in pool:
        cat_pool.setdefault(ing['category'], []).append(ing)
    
    for idx in range(len(new_smi)):
        if random.random() < mutation_rate:
            # 같은 카테고리에서 다른 원료로 교체
            cat = new_ings[idx]['category']
            candidates = cat_pool.get(cat, [])
            db_candidates = [c for c in candidates if c['has_db_smiles']]
            if db_candidates:
                replacement = random.choice(db_candidates)
            elif candidates:
                replacement = random.choice(candidates)
            else:
                continue
            new_smi[idx] = replacement['smiles']
            new_ings[idx]['name'] = replacement['name_en'] or replacement['name_ko']
        
        # 비율 변경
        if random.random() < mutation_rate:
            new_conc[idx] *= random.uniform(0.7, 1.4)
    
    # 비율 정규화 (합 = 100)
    total = sum(new_conc)
    if total > 0:
        new_conc = [c / total * 100 for c in new_conc]
        for i, c in enumerate(new_conc):
            new_ings[i]['concentration'] = round(c, 2)
    
    return new_smi, new_conc, new_ings

def evaluate_recipe(smiles_list, concentrations):
    """전체 시스템으로 레시피 평가"""
    try:
        result = biophys.simulate_recipe(smiles_list, concentrations)
        
        hedonic = result['hedonic']['hedonic_score']
        longevity_h = result['thermodynamics']['longevity_hours']
        longevity_score = min(1.0, longevity_h / 6)
        smoothness = result['thermodynamics'].get('smoothness', 0.5)
        receptors = result['nose']['active_receptors']
        receptor_score = min(1.0, receptors / 120)
        
        # 종합 점수 (가중합)
        total = (hedonic * 0.35 +
                 longevity_score * 0.25 +
                 smoothness * 0.15 +
                 receptor_score * 0.25)
        
        return {
            'total_score': round(total, 4),
            'hedonic_score': round(hedonic, 4),
            'longevity_hours': round(longevity_h, 1),
            'longevity_score': round(longevity_score, 4),
            'smoothness': round(smoothness, 4),
            'active_receptors': receptors,
            'receptor_score': round(receptor_score, 4),
            'raw_result': result,
        }
    except Exception as e:
        return None

# ==============================
# 4) 메인: 진화적 탐색
# ==============================
def main():
    print("=" * 70)
    print("  🧪 ORIGINAL PERFUME RECIPE GENERATOR")
    print("  Using: VirtualNose + HedonicFunction + ThermodynamicsEngine")
    print("=" * 70)
    
    # Load ingredient pool
    pool = load_ingredient_pool()
    db_matched = sum(1 for i in pool if i['has_db_smiles'])
    print(f"\n  Ingredients: {len(pool)} (DB SMILES: {db_matched})")
    
    # Phase 1: Generate seed recipes (10 templates × 10 variants = 100)
    print(f"\n  Phase 1: Generating seed recipes ({len(RECIPE_TEMPLATES)} templates)...")
    candidates = []
    
    for template in RECIPE_TEMPLATES:
        for v in range(10):
            smi, conc, ings = generate_recipe(template, pool)
            if not smi:
                continue
            score = evaluate_recipe(smi, conc)
            if score is None:
                continue
            candidates.append({
                'template': template['name'],
                'style': template['style'],
                'smiles': smi,
                'concentrations': conc,
                'ingredients': ings,
                'score': score,
            })
    
    print(f"  Seeds generated: {len(candidates)}")
    print(f"  Best seed score: {max(c['score']['total_score'] for c in candidates):.4f}")
    
    # Phase 2: Evolutionary optimization (top 20 → mutate × 20 → select)
    GENERATIONS = 5
    POP_SIZE = 20
    MUTATIONS_PER = 15
    
    print(f"\n  Phase 2: Evolution ({GENERATIONS} generations × {POP_SIZE} × {MUTATIONS_PER} mutations)...")
    
    for gen in range(GENERATIONS):
        # Sort and keep top POP_SIZE
        candidates.sort(key=lambda x: x['score']['total_score'], reverse=True)
        elite = candidates[:POP_SIZE]
        
        # Mutate each elite
        new_gen = list(elite)
        for parent in elite:
            for _ in range(MUTATIONS_PER):
                m_smi, m_conc, m_ings = mutate_recipe(
                    parent['smiles'], parent['concentrations'],
                    parent['ingredients'], pool, mutation_rate=0.25
                )
                score = evaluate_recipe(m_smi, m_conc)
                if score is None:
                    continue
                new_gen.append({
                    'template': parent['template'],
                    'style': parent['style'],
                    'smiles': m_smi,
                    'concentrations': m_conc,
                    'ingredients': m_ings,
                    'score': score,
                })
        
        candidates = new_gen
        candidates.sort(key=lambda x: x['score']['total_score'], reverse=True)
        best = candidates[0]['score']
        print(f"    Gen {gen+1}: Best={best['total_score']:.4f} "
              f"(H={best['hedonic_score']:.2f} L={best['longevity_hours']}h "
              f"R={best['active_receptors']})")
    
    # Phase 3: Select top 10 (diverse)
    print(f"\n  Phase 3: Selecting top 10 diverse recipes...")
    candidates.sort(key=lambda x: x['score']['total_score'], reverse=True)
    
    # Deduplicate — max 1 per template, max 2 per style, low overlap
    final = []
    used_templates = {}
    used_styles = {}
    for c in candidates:
        tmpl = c['template']
        style = c['style']
        t_count = used_templates.get(tmpl, 0)
        s_count = used_styles.get(style, 0)
        if t_count >= 1:  # max 1 per template
            continue
        if s_count >= 2:  # max 2 per style
            continue
        # Check ingredient diversity against already selected
        if final:
            names = set(i['name'] for i in c['ingredients'])
            too_similar = False
            for f in final:
                f_names = set(i['name'] for i in f['ingredients'])
                overlap = len(names & f_names) / max(len(names | f_names), 1)
                if overlap > 0.35:
                    too_similar = True
                    break
            if too_similar:
                continue
        
        final.append(c)
        used_templates[tmpl] = t_count + 1
        used_styles[style] = s_count + 1
        if len(final) >= 10:
            break

    
    # Fill up if needed — relax constraints progressively
    if len(final) < 10:
        # Pass 2: allow max 2 per template
        for c in candidates:
            if c in final:
                continue
            tmpl = c['template']
            t_count = sum(1 for f in final if f['template'] == tmpl)
            if t_count >= 2:
                continue
            names = set(i['name'] for i in c['ingredients'])
            too_similar = False
            for f in final:
                f_names = set(i['name'] for i in f['ingredients'])
                overlap = len(names & f_names) / max(len(names | f_names), 1)
                if overlap > 0.5:
                    too_similar = True
                    break
            if too_similar:
                continue
            final.append(c)
            if len(final) >= 10:
                break
    
    # Pass 3: just fill remaining
    if len(final) < 10:
        for c in candidates:
            if c not in final:
                final.append(c)
                if len(final) >= 10:
                    break

    
    # ==============================
    # 5) 결과 출력
    # ==============================
    print(f"\n{'=' * 70}")
    print(f"  🏆 TOP 10 ORIGINAL PERFUME RECIPES")
    print(f"{'=' * 70}\n")
    
    output_recipes = []
    
    for rank, recipe in enumerate(final, 1):
        s = recipe['score']
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  #{rank} | {recipe['template']} ({recipe['style']})")
        print(f"  Score: {s['total_score']:.3f} | Hedonic: {s['hedonic_score']:.2f} | "
              f"Longevity: {s['longevity_hours']}h | Receptors: {s['active_receptors']}")
        print(f"  Smoothness: {s['smoothness']:.2f}")
        print(f"  ─────────────────────────────────────────────")
        
        # Group by note
        for note in ['top', 'middle', 'base']:
            note_ings = [i for i in recipe['ingredients'] if i['note_type'] == note]
            if note_ings:
                note_total = sum(i['concentration'] for i in note_ings)
                print(f"  {note.upper():7s} ({note_total:.1f}%):")
                for i in note_ings:
                    print(f"    • {i['name']:25s} {i['category']:12s} {i['concentration']:.1f}%")
        print()
        
        output_recipes.append({
            'rank': rank,
            'name': recipe['template'],
            'style': recipe['style'],
            'total_score': s['total_score'],
            'hedonic_score': s['hedonic_score'],
            'longevity_hours': s['longevity_hours'],
            'smoothness': s['smoothness'],
            'active_receptors': s['active_receptors'],
            'ingredients': recipe['ingredients'],
            'smiles': recipe['smiles'],
            'concentrations': recipe['concentrations'],
        })
    
    # Save to file
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_recipes.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_recipes, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    return output_recipes

if __name__ == '__main__':
    start = time.time()
    recipes = main()
    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")
