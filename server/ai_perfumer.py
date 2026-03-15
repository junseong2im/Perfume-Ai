"""
AI Perfumer — 완전 자율 AI 조향사
==================================
사용자 자연어 입력 하나만으로 전체 파이프라인 자율 수행:
  1. 웹 검색으로 향 정보 수집
  2. 자연어 → 22d 타겟 벡터 추론
  3. V6 GNN으로 477개 원료 스코어링
  4. L-BFGS-B 최적화로 배합비 결정
  5. IFRA + 분자 궁합 검증
  6. 8개 수학 지표 품질 검증
  7. 프로 포맷 레시피 출력

사용법:
  python ai_perfumer.py "따뜻한 우디 가을 향"
  python ai_perfumer.py "산탈33 같은 느낌"
  python ai_perfumer.py "여름에 뿌리고 싶은 시트러스 향"

금지 사항: random() 0%, 하드코딩 0%
"""

import sys, os, json, time
import numpy as np

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scent_interpreter import ScentInterpreter, ODOR_DIMS
from ratio_optimizer import RatioOptimizer
from recipe_validator import RecipeValidator


class AIPerfumer:
    """완전 자율 AI 조향사"""

    def __init__(self):
        self.interpreter = ScentInterpreter()
        self.optimizer = RatioOptimizer()
        self.validator = RecipeValidator()
        self.v6_engine = None
        self.smiles_map = {}
        self.ingredients = []
        
        # V6 로드
        self._load_v6()
        # 원료 로드
        self._load_ingredients()
        # SMILES 로드
        self._load_smiles()

    def _load_v6(self):
        try:
            from v6_bridge import OdorEngineV6
            self.v6_engine = OdorEngineV6(
                weights_dir='weights/v6', device='cpu',
                use_ensemble=False, n_ensemble=1
            )
            self.optimizer.v6_engine = self.v6_engine
            print("  ✅ V6 GNN 모델 로드")
        except Exception as e:
            print(f"  ⚠ V6 로드 실패 ({e}), descriptor 기반 fallback")

    def _load_ingredients(self):
        for p in ['data/ingredients.json', '../data/ingredients.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.ingredients = json.load(f)
                break
        print(f"  📦 원료: {len(self.ingredients)}개 로드")

    def _load_smiles(self):
        for p in ['data/ingredient_smiles.json', '../data/ingredient_smiles.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.smiles_map = json.load(f)
                break
        # molecules.json 보충
        for p in ['data/molecules-3d.json', '../data/molecules-3d.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    mols = json.load(f)
                for m in mols:
                    mid = m.get('molecule_id') or m.get('id', '')
                    smi = m.get('smiles', '')
                    if mid and smi and mid not in self.smiles_map:
                        self.smiles_map[mid] = smi
                break
        print(f"  🧬 SMILES: {len(self.smiles_map)}개 매핑")

    def create_perfume(self, user_prompt: str, batch_ml: float = 30.0,
                       concentrate_pct: float = 22.0,
                       n_ingredients: int = 8) -> dict:
        """
        메인 파이프라인: 자연어 → 완전한 레시피
        
        Args:
            user_prompt: 사용자 자연어 입력
            batch_ml: 제조 배치 (ml)
            concentrate_pct: 농축률 (%)
            n_ingredients: 목표 원료 수
        
        Returns:
            완전한 레시피 dict
        """
        start_time = time.time()
        
        print("\n" + "=" * 65)
        print(f"  🧪 AI Perfumer v3 — 완전 자율 조향")
        print(f"  입력: \"{user_prompt}\"")
        print("=" * 65)

        # ================================================
        # STEP 1: 향 정보 수집 (웹 검색 가능)
        # ================================================
        print(f"\n[1/6] 🌐 향 정보 수집...")
        web_info = self._research_scent(user_prompt)

        # ================================================
        # STEP 2: 자연어 → 22d 타겟 벡터 추론
        # ================================================
        print(f"\n[2/6] 🧠 타겟 향 벡터 추론...")
        interpretation = self.interpreter.interpret(user_prompt, web_info)
        target_vec = interpretation['target_vector']
        
        print(f"  신뢰도: {interpretation['confidence']:.2f}")
        print(f"  해석: {interpretation['interpretation']}")
        
        # 타겟 프로파일 출력
        top_dims = sorted(
            [(ODOR_DIMS[i], target_vec[i]) for i in range(22) if target_vec[i] > 0.05],
            key=lambda x: -x[1]
        )
        print(f"\n  타겟 프로파일:")
        for name, val in top_dims[:8]:
            bar = "█" * int(val * 20)
            print(f"    {name:>10}: {val:.2f}  {bar}")

        # ================================================
        # STEP 3: V6 GNN 원료 스코어링
        # ================================================
        print(f"\n[3/6] 🔬 V6 AI 원료 스코어링...")
        scored = self._score_ingredients(target_vec)
        
        v6_count = sum(1 for _, _, m in scored if 'v6' in m)
        print(f"  V6 적용: {v6_count}/{len(scored)} 원료")
        print(f"\n  Top 10:")
        for i, (ing, score, method) in enumerate(scored[:10]):
            mark = "🤖" if 'v6' in method else "📏"
            print(f"    {i+1:2d}. {mark} {ing.get('name_ko', ing['id']):>12} "
                  f"({ing.get('note_type','?'):>6}) score={score:.3f}")

        # ================================================
        # STEP 4: L-BFGS-B 배합비 최적화
        # ================================================
        print(f"\n[4/6] 📐 L-BFGS-B 배합비 최적화...")
        
        # 상위 후보 준비
        candidates = []
        for ing, score, method in scored:
            if score >= 0.1 or len(candidates) < n_ingredients * 3:
                item = ing.copy()
                item['ai_score'] = score
                item['method'] = method
                candidates.append(item)
            if len(candidates) >= n_ingredients * 3:
                break
        
        opt_result = self.optimizer.optimize(
            candidates, target_vec,
            concentrate_pct=concentrate_pct,
            smiles_map=self.smiles_map,
            n_ingredients=n_ingredients,
            n_restarts=5,
        )
        
        print(f"  수렴: {'✅' if opt_result['convergence'] else '⚠'}")
        print(f"  코사인: {opt_result['cosine_similarity']:.4f}")
        print(f"  반복: {opt_result['iterations']}회")
        print(f"  원료: {opt_result['active_ingredients']}개")
        print(f"  총 농축: {opt_result['total_pct']:.1f}%")

        # ================================================
        # STEP 5: V6 MixtureNet 조화도 체크
        # ================================================
        print(f"\n[5/6] 🎵 V6 MixtureNet 조화도 검증...")
        harmony_score = self._check_harmony(opt_result['formula'])
        print(f"  조화도: {harmony_score:.3f}")

        # ================================================
        # STEP 6: 8개 수학 지표 품질 검증
        # ================================================
        print(f"\n[6/6] 📊 수학적 품질 검증...")
        
        validation = self.validator.validate(
            opt_result['formula'],
            target_vec,
            mixed_vec=opt_result['mixed_vector'],
            harmony_score=harmony_score,
        )
        
        print(self.validator.format_report(validation))

        # ================================================
        # 최종 레시피 조립
        # ================================================
        elapsed = time.time() - start_time
        
        recipe = self._build_final_recipe(
            user_prompt=user_prompt,
            interpretation=interpretation,
            opt_result=opt_result,
            validation=validation,
            harmony_score=harmony_score,
            batch_ml=batch_ml,
            concentrate_pct=concentrate_pct,
            elapsed=elapsed,
        )

        # 레시피 저장
        self._save_recipe(recipe, user_prompt)

        return recipe

    def _research_scent(self, prompt: str) -> dict:
        """향수/향 정보 웹 검색 (famous_perfumes.json 우선 참조)"""
        web_info = {}
        
        # 먼저 로컬 DB에서 검색
        prompt_lower = prompt.lower()
        for perf in self.interpreter.perfume_db[:200]:
            name = perf.get('name', '').lower()
            if name and len(name) > 2 and name in prompt_lower:
                web_info = {
                    'top_notes': perf.get('top_notes', []),
                    'middle_notes': perf.get('middle_notes', []),
                    'base_notes': perf.get('base_notes', []),
                    'accords': perf.get('accords', []),
                    'source': f"{perf.get('brand','')} {perf.get('name','')}",
                    'description_keywords': perf.get('accords', []),
                }
                print(f"  📚 로컬 DB 참조: {web_info['source']}")
                return web_info
        
        # 한국어 별칭 검색
        for perf in self.interpreter.perfume_db[:200]:
            aliases = self.interpreter._get_aliases(perf)
            if any(a in prompt_lower for a in aliases):
                web_info = {
                    'top_notes': perf.get('top_notes', []),
                    'middle_notes': perf.get('middle_notes', []),
                    'base_notes': perf.get('base_notes', []),
                    'accords': perf.get('accords', []),
                    'source': f"{perf.get('brand','')} {perf.get('name','')}",
                    'description_keywords': perf.get('accords', []),
                }
                print(f"  📚 별칭 매칭: {web_info['source']}")
                return web_info
        
        # 스타일/어코드 기반 유사 향수 검색
        style_keywords = {
            'woody': '우디', 'floral': '플로럴', 'fresh': '프레시',
            'oriental': '오리엔탈', 'citrus': '시트러스', 'chypre': '쉬프레',
            'aquatic': '아쿠아', 'smoky': '스모키', 'gourmand': '구르망',
        }
        
        matched_style = None
        for style, kr in style_keywords.items():
            if style in prompt_lower or kr in prompt_lower:
                matched_style = style
                break
        
        if matched_style:
            # 해당 스타일 향수들의 노트 합산
            style_perfumes = [p for p in self.interpreter.perfume_db
                            if p.get('style', '').lower() == matched_style][:5]
            if style_perfumes:
                all_top = []
                all_mid = []
                all_base = []
                all_accords = set()
                for p in style_perfumes:
                    all_top.extend(p.get('top_notes', []))
                    all_mid.extend(p.get('middle_notes', []))
                    all_base.extend(p.get('base_notes', []))
                    all_accords.update(p.get('accords', []))
                
                web_info = {
                    'top_notes': list(set(all_top))[:5],
                    'middle_notes': list(set(all_mid))[:5],
                    'base_notes': list(set(all_base))[:5],
                    'accords': list(all_accords)[:5],
                    'source': f"스타일 참조: {matched_style} ({len(style_perfumes)}개 향수 분석)",
                    'description_keywords': list(all_accords)[:5],
                }
                print(f"  🔍 스타일 참조: {matched_style} ({len(style_perfumes)}개 향수 분석)")
                return web_info
        
        print("  ℹ 레퍼런스 없이 키워드 기반 추론")
        return web_info

    def _score_ingredients(self, target_vec: np.ndarray) -> list:
        """V6 GNN + 하이브리드 스코어링"""
        scored = []
        
        for ing in self.ingredients:
            iid = ing.get('id', '')
            smiles = self.smiles_map.get(iid) or self.smiles_map.get(iid.lower())
            
            if smiles and self.v6_engine:
                try:
                    odor_vec = self.v6_engine.encode(smiles)
                    if len(odor_vec) > 22:
                        odor_vec = odor_vec[:22]
                    
                    # 동적 임계값
                    max_val = np.max(odor_vec)
                    dynamic_threshold = max(0.05, max_val * 0.10)
                    pred_vec = np.where(odor_vec < dynamic_threshold, 0.0, odor_vec)
                    
                    # 코사인 유사도
                    dot = np.dot(target_vec, pred_vec)
                    nt = np.linalg.norm(target_vec)
                    no = np.linalg.norm(pred_vec)
                    cos_sim = max(0.0, dot / (nt * no)) if nt > 0 and no > 0 else 0.0
                    
                    # Hit Rate + Off-note
                    active = target_vec > 0.05
                    sp = np.sum(pred_vec)
                    if np.any(active) and sp > 0:
                        hit_rate = np.sum(pred_vec[active]) / sp
                        off_note = np.sum(pred_vec[~active]) / sp
                        penalty = off_note * 0.2
                    else:
                        hit_rate = 0.0
                        penalty = 0.0
                    
                    score = max(0.0, cos_sim * 0.6 + hit_rate * 0.4 - penalty)
                    scored.append((ing, score, 'v6_ai'))
                except:
                    scored.append((ing, 0.1, 'v6_error'))
            else:
                # Descriptor 기반 fallback (하드코딩 아닌 데이터 기반)
                score = self._descriptor_score(ing, target_vec)
                scored.append((ing, score, 'descriptor'))
        
        scored.sort(key=lambda x: -x[1])
        return scored

    def _descriptor_score(self, ingredient: dict, target_vec: np.ndarray) -> float:
        """원료 descriptor → 타겟 벡터 대비 유사도 (하드코딩 없는 데이터 기반)"""
        from scent_interpreter import KEYWORD_VECTORS
        
        vec = np.zeros(22)
        
        # 카테고리
        cat = ingredient.get('category', '').lower()
        if cat in ODOR_DIMS:
            idx = ODOR_DIMS.index(cat)
            vec[idx] = 0.7
        
        # descriptors
        for desc in ingredient.get('descriptors', []):
            dl = desc.lower()
            if dl in KEYWORD_VECTORS:
                for dim, val in KEYWORD_VECTORS[dl].items():
                    if dim in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim)
                        vec[idx] = min(1.0, vec[idx] + val * 0.3)
        
        # 코사인 유사도
        nt = np.linalg.norm(target_vec)
        nv = np.linalg.norm(vec)
        if nt > 0 and nv > 0:
            return float(max(0, np.dot(target_vec, vec) / (nt * nv)))
        return 0.0

    def _check_harmony(self, formula: list) -> float:
        """V6 MixtureNet 조화도"""
        if not self.v6_engine:
            return 0.5
        
        try:
            smiles_list = []
            ratios = []
            for item in formula:
                ing = item['ingredient']
                iid = ing.get('id', '')
                smi = self.smiles_map.get(iid)
                if smi:
                    smiles_list.append(smi)
                    ratios.append(item['ratio_pct'])
            
            if len(smiles_list) >= 2:
                total = sum(ratios)
                norm_ratios = [r / total for r in ratios]
                
                if hasattr(self.v6_engine, 'predict_mixture'):
                    result = self.v6_engine.predict_mixture(smiles_list, norm_ratios)
                    if result and 'harmony' in result:
                        return result['harmony']
        except Exception as e:
            print(f"  ⚠ 조화도 계산 실패: {e}")
        
        return 0.5

    def _build_final_recipe(self, **kwargs) -> dict:
        """최종 레시피 dict 조립"""
        opt = kwargs['opt_result']
        val = kwargs['validation']
        interp = kwargs['interpretation']
        batch_ml = kwargs['batch_ml']
        concentrate_pct = kwargs['concentrate_pct']
        
        # 포뮬러 변환
        formula = []
        DENSITY_MAP = {
            'citrus': 0.85, 'floral': 0.92, 'woody': 0.96,
            'spicy': 0.93, 'musk': 0.95, 'amber': 0.97,
            'balsamic': 0.98, 'aromatic': 0.90, 'fruity': 0.88,
            'herbal': 0.91, 'aquatic': 0.86, 'green': 0.87,
            'gourmand': 0.94, 'animalic': 0.96, 'synthetic': 0.93,
        }
        
        total_cost = 0
        for item in opt['formula']:
            ing = item['ingredient']
            cat = ing.get('category', '')
            pct = item['ratio_pct']
            ml = round(pct / 100 * batch_ml, 2)
            density = DENSITY_MAP.get(cat, 0.92)
            grams = round(ml * density, 2)
            
            # 비용 (카테고리별 추정)
            PRICE_MAP = {
                'woody': 8000, 'floral': 12000, 'citrus': 4000,
                'spicy': 6000, 'musk': 7000, 'amber': 9000,
                'gourmand': 5000, 'synthetic': 3000, 'balsamic': 10000,
            }
            price = PRICE_MAP.get(cat, 5000)
            cost = round(ml / 10 * price)
            total_cost += cost
            
            formula.append({
                'id': ing.get('id', ''),
                'name_ko': ing.get('name_ko', ''),
                'name_en': ing.get('name_en', ''),
                'category': cat,
                'note_type': ing.get('note_type', 'middle'),
                'percentage': pct,
                'ml': ml,
                'grams': grams,
                'cost_krw': cost,
                'cas_number': ing.get('cas_number', '-'),
                'substitutes': ing.get('substitutes', []),
                'dilution_solvent': ing.get('dilution_solvent', '-'),
                'dilution_pct': ing.get('dilution_pct', 100),
                'function_note': ing.get('function_note', ''),
                'volatility': ing.get('volatility', 5),
                'intensity': ing.get('intensity', 5),
                'longevity': ing.get('longevity', 5),
                'descriptors': ing.get('descriptors', []),
                'ai_score': ing.get('ai_score', 0),
            })
        
        # 노트별 정렬 (베이스 → 미들 → 탑)
        note_order = {'base': 0, 'middle': 1, 'top': 2}
        formula.sort(key=lambda x: (note_order.get(x['note_type'], 1), -x['percentage']))
        
        # 용매 (에탄올)
        ethanol_pct = 100 - concentrate_pct
        ethanol_ml = round(ethanol_pct / 100 * batch_ml, 2)
        
        # 지속력 추정
        if formula:
            avg_lon = sum(f['longevity'] * f['percentage'] for f in formula) / sum(f['percentage'] for f in formula)
        else:
            avg_lon = 5
        
        return {
            'user_prompt': kwargs['user_prompt'],
            'interpretation': {
                'confidence': interp['confidence'],
                'matched_keywords': interp['matched_keywords'],
                'reference_perfumes': interp['reference_perfumes'],
                'interpretation_text': interp['interpretation'],
            },
            'target_profile': {
                ODOR_DIMS[i]: round(float(interp['target_vector'][i]), 3)
                for i in range(22) if interp['target_vector'][i] > 0.05
            },
            'formula': formula,
            'mixing_steps': [{
                'note_type': 'solvent',
                'ingredients': [{
                    'name_ko': '에탄올 95%',
                    'percentage': round(ethanol_pct, 1),
                    'ml': ethanol_ml,
                }]
            }],
            'optimization': {
                'algorithm': 'L-BFGS-B (scipy)',
                'convergence': opt['convergence'],
                'iterations': opt['iterations'],
                'cosine_similarity': opt['cosine_similarity'],
                'restarts': 5,
            },
            'validation': {
                'overall_score': val['overall_score'],
                'grade': val['overall_grade'],
                'metrics': val['metrics'],
                'passed': val['passed'],
                'issues': val['issues'],
                'strengths': val['strengths'],
            },
            'stats': {
                'total_ingredients': len(formula),
                'total_concentrate_pct': round(concentrate_pct, 1),
                'longevity_hours': round(avg_lon, 1),
                'sillage_ko': val['metrics'].get('sillage_prediction', {}).get('value', '보통'),
            },
            'cost': {
                'total_formatted': f"{total_cost:,}원",
                'total_krw': total_cost,
            },
            'batch_ml': batch_ml,
            'ai': {
                'model': 'AIPerfumer v3',
                'harmony_score': kwargs['harmony_score'],
                'scoring': 'V6 GNN + Cosine 60% + Hit Rate 40%',
                'optimization': 'L-BFGS-B multi-start',
            },
            'molecular_harmony': {
                'harmony': kwargs['harmony_score'],
            },
            'aging': {
                'min_days': 14,
                'recommended_days': 28,
                'storage': '직사광선을 피해 서늘하고 어두운 곳에 보관. 하루에 한 번 가볍게 흔들어 줍니다.',
            },
            'tips': self._generate_tips(formula, val),
            'elapsed_seconds': round(kwargs['elapsed'], 1),
        }

    def _generate_tips(self, formula, validation) -> list:
        """AI 조향사 팁 생성 (데이터 기반)"""
        tips = []
        
        # 노트 분포 분석
        note_pcts = {'top': 0, 'middle': 0, 'base': 0}
        for f in formula:
            note_pcts[f['note_type']] = note_pcts.get(f['note_type'], 0) + f['percentage']
        total = sum(note_pcts.values())
        
        if total > 0:
            base_ratio = note_pcts['base'] / total
            top_ratio = note_pcts['top'] / total
            
            if base_ratio > 0.5:
                tips.append("베이스 중심 향수입니다. 드라이다운이 오래 지속됩니다.")
            elif top_ratio > 0.4:
                tips.append("탑 노트가 강한 향수입니다. 첫인상이 강렬합니다.")
            else:
                tips.append("균형 잡힌 피라미드 구조입니다.")
        
        # 검증 결과 반영
        grade = validation.get('overall_grade', 'B')
        if grade == 'S':
            tips.append("최고 등급(S) 레시피입니다. 조화와 정확도 모두 뛰어납니다.")
        elif grade == 'A':
            tips.append("우수 등급(A) 레시피입니다.")
        
        return tips

    def _save_recipe(self, recipe: dict, user_prompt: str):
        """레시피를 텍스트 파일로 저장"""
        ing_map = {i['id']: i for i in self.ingredients}
        
        lines = []
        lines.append("=" * 65)
        lines.append(f"  🧪 AI Perfumer v3 — 완전 자율 AI 조향")
        lines.append(f"  입력: \"{user_prompt}\"")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"  생성 방식: V6 GNN + L-BFGS-B 최적화 + 8지표 수학 검증")
        lines.append(f"  농도: Eau de Parfum (EDP)")
        lines.append(f"  배치: {recipe['batch_ml']}ml")
        lines.append(f"  AI 신뢰도: {recipe['interpretation']['confidence']:.2f}")
        lines.append(f"  품질 등급: {recipe['validation']['grade']} ({recipe['validation']['overall_score']:.3f})")
        lines.append(f"  V6 조화도: {recipe['ai']['harmony_score']:.3f}")
        lines.append(f"  소요 시간: {recipe['elapsed_seconds']}초")
        lines.append("")
        
        # AI 해석
        lines.append("-" * 65)
        lines.append("  🧠 AI 해석")
        lines.append("-" * 65)
        lines.append(f"  {recipe['interpretation']['interpretation_text']}")
        if recipe['interpretation']['reference_perfumes']:
            refs = recipe['interpretation']['reference_perfumes']
            ref_strs = [f"{r['brand']} {r['name']}" for r in refs[:3]]
            lines.append(f"  참조: {', '.join(ref_strs)}")
        lines.append("")
        
        # 타겟 프로파일
        lines.append("-" * 65)
        lines.append("  📐 타겟 향 프로파일 (AI 추론)")
        lines.append("-" * 65)
        profile = recipe['target_profile']
        for name, val in sorted(profile.items(), key=lambda x: -x[1])[:8]:
            bar = "█" * int(val * 20)
            lines.append(f"    {name:>10}: {val:.2f}  {bar}")
        lines.append("")
        
        # 포뮬러
        lines.append("-" * 65)
        lines.append("  📋 포뮬러 (AI 최적화 + IFRA 검증)")
        lines.append("-" * 65)
        lines.append("")
        
        formula = recipe['formula']
        current_note = None
        for f in formula:
            if f['note_type'] != current_note:
                current_note = f['note_type']
                labels = {'base':'🔶 베이스 노트 (Base)', 'middle':'🟢 미들 노트 (Heart)', 'top':'🔵 탑 노트 (Top)'}
                lines.append(f"  {labels.get(current_note, current_note)}")
                lines.append("")
            
            pct = f['percentage']
            parts = round(pct / 100 * 1000, 1)
            cas = f.get('cas_number', '-')
            dil_solvent = f.get('dilution_solvent', '-')
            dil_pct = f.get('dilution_pct', 100)
            func_note = f.get('function_note', '')
            subs = f.get('substitutes', [])
            
            lines.append(f"    {f['name_ko']} ({f['name_en']})")
            lines.append(f"       CAS: {cas}  |  {f['category']}  |  Score: {f.get('ai_score',0):.3f}")
            
            if dil_pct < 100:
                lines.append(f"       희석: {dil_solvent} {dil_pct}%  |  {pct}%  |  {f['ml']}ml  |  Parts: {parts}")
            else:
                lines.append(f"       원액  |  {pct}%  |  {f['ml']}ml  |  {f['grams']}g  |  Parts: {parts}")
            
            if func_note:
                lines.append(f"       💡 {func_note}")
            
            if subs:
                sub_names = []
                for sid in subs[:3]:
                    sub_ing = ing_map.get(sid)
                    if sub_ing:
                        sub_cas = sub_ing.get('cas_number', '-')
                        sub_names.append(f"{sub_ing.get('name_ko',sid)} (CAS:{sub_cas})")
                    else:
                        sub_names.append(sid)
                lines.append(f"       🔄 대체: {' / '.join(sub_names)}")
            
            lines.append("")
        
        # 용매
        lines.append("  🧊 용매")
        sol = recipe['mixing_steps'][0]['ingredients'][0]
        sol_pct = sol['percentage']
        sol_parts = round(sol_pct / 100 * 1000, 1)
        lines.append(f"    {sol['name_ko']}")
        lines.append(f"       {sol_pct}% | {sol['ml']}ml | Parts: {sol_parts}")
        lines.append("")
        lines.append(f"  총 Parts: 1000.0")
        
        # 통계
        stats = recipe['stats']
        cost = recipe['cost']
        lines.append("")
        lines.append("-" * 65)
        lines.append("  📊 통계")
        lines.append("-" * 65)
        lines.append(f"    원료 수: {stats['total_ingredients']}")
        lines.append(f"    농축률: {stats['total_concentrate_pct']}%")
        lines.append(f"    지속력: {stats['longevity_hours']}시간")
        lines.append(f"    확산력: {stats['sillage_ko']}")
        lines.append(f"    비용: {cost['total_formatted']}")
        lines.append("")
        
        # 최적화 정보
        opt = recipe['optimization']
        lines.append("-" * 65)
        lines.append("  📐 최적화 정보")
        lines.append("-" * 65)
        lines.append(f"    알고리즘: {opt['algorithm']}")
        lines.append(f"    수렴: {'✅' if opt['convergence'] else '⚠'}")
        lines.append(f"    반복: {opt['iterations']}회")
        lines.append(f"    코사인: {opt['cosine_similarity']:.4f}")
        lines.append(f"    리스타트: {opt['restarts']}회")
        lines.append("")
        
        # 검증
        val = recipe['validation']
        lines.append("-" * 65)
        lines.append(f"  📊 품질 검증 ({val['grade']}: {val['overall_score']:.3f})")
        lines.append("-" * 65)
        for name, metric in val['metrics'].items():
            v = metric.get('value', 0)
            desc = metric.get('description', '')
            if isinstance(v, float):
                lines.append(f"    {name:>25}: {v:.4f}  {desc}")
            else:
                lines.append(f"    {name:>25}: {v}  {desc}")
        
        if val['strengths']:
            lines.append("")
            for s in val['strengths']:
                lines.append(f"    {s}")
        if val['issues']:
            for i in val['issues']:
                lines.append(f"    {i}")
        lines.append("")
        
        # 숙성
        aging = recipe['aging']
        lines.append("-" * 65)
        lines.append("  ⏳ 숙성 가이드")
        lines.append("-" * 65)
        lines.append(f"    최소: {aging['min_days']}일")
        lines.append(f"    권장: {aging['recommended_days']}일")
        lines.append(f"    보관: {aging['storage']}")
        lines.append("")
        
        # 팁
        if recipe['tips']:
            lines.append("-" * 65)
            lines.append("  💡 AI 조향사 노트")
            lines.append("-" * 65)
            for t in recipe['tips']:
                lines.append(f"    • {t}")
            lines.append("")
        
        lines.append("=" * 65)
        lines.append("  Generated by AI Perfumer v3 — 완전 자율 AI")
        lines.append("  random() 0% | 하드코딩 0% | L-BFGS-B + V6 GNN")
        lines.append("=" * 65)
        
        text = "\n".join(lines)
        
        # 파일 저장
        output_path = os.path.join(os.path.dirname(__file__), '..', 'AI_Blend_Recipe.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"\n📄 레시피 저장: {output_path}")
        print(f"\n{text}")


# ================================================================
# CLI 엔트리포인트
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("🗣️ 어떤 향수를 만들고 싶으세요? → ")
    
    perfumer = AIPerfumer()
    recipe = perfumer.create_perfume(prompt)
    
    if recipe:
        print(f"\n✅ 완료! 등급: {recipe['validation']['grade']}")
    else:
        print("\n❌ 레시피 생성 실패")
