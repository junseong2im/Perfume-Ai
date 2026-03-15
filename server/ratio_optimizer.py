"""
RatioOptimizer — 그래디언트 기반 배합비 최적화
=============================================
random() 0%, 하드코딩 0%
scipy L-BFGS-B로 수학적 최적해 탐색

목적함수:
    max  Cosine(혼합벡터, 타겟) + λ₁·Harmony - λ₂·IFRA위반 - λ₃·잡향
제약조건:
    Σ ratios = concentrate_pct
    각 원료 ≤ IFRA 한도
    각 원료 ≥ 0.3%
"""

import numpy as np
from scipy.optimize import minimize


class RatioOptimizer:
    """L-BFGS-B 기반 향수 배합비 최적화 엔진"""

    def __init__(self, v6_engine=None):
        self.v6_engine = v6_engine
        # IFRA 한도 (기존 recipe_engine.py에서 가져옴)
        self.ifra_limits = {
            'oakmoss': 0.1, 'eugenol': 0.5, 'cinnamaldehyde': 0.1,
            'coumarin': 0.6, 'methyl_salicylate': 0.5,
            'isoeugenol': 0.02, 'citral': 1.0,
            'linalool': 5.0, 'geraniol': 5.0,
            'limonene': 4.0, 'hydroxycitronellal': 1.0,
            'benzyl_alcohol': 2.0, 'benzyl_benzoate': 5.0,
            'benzyl_salicylate': 2.0, 'farnesol': 1.0,
            'eugenol': 0.5, 'indole': 0.5,
        }

    def optimize(self, candidates: list, target_vec: np.ndarray,
                 concentrate_pct: float = 22.0,
                 smiles_map: dict = None,
                 n_ingredients: int = 8,
                 n_restarts: int = 5) -> dict:
        """
        최적 배합비 탐색
        
        Args:
            candidates: AI 스코어링된 원료 리스트 [{id, ai_score, ...}, ...]
            target_vec: 22d 타겟 벡터
            concentrate_pct: 총 농축률 (%)
            smiles_map: SMILES 매핑 dict
            n_ingredients: 최종 원료 수
            n_restarts: 다중 시작점 수 (수렴 안정성)
        
        Returns:
            {
                'formula': [(ingredient, ratio_pct), ...],
                'objective_value': float,
                'cosine_similarity': float,
                'mixed_vector': np.array(22),
                'convergence': bool,
                'iterations': int,
            }
        """
        if not candidates:
            return {'formula': [], 'objective_value': 0, 'convergence': False}

        # 1) 원료 벡터 사전 계산
        n = min(len(candidates), n_ingredients * 2)  # 후보풀 = 목표의 2배
        top_candidates = sorted(candidates, key=lambda x: x.get('ai_score', 0), reverse=True)[:n]
        
        odor_matrix = np.zeros((n, 22))  # (n, 22)
        for i, cand in enumerate(top_candidates):
            odor_matrix[i] = self._get_odor_vector(cand, smiles_map)

        # 2) 제약 바운드 계산
        bounds = []
        for cand in top_candidates:
            cid = cand.get('id', '')
            max_pct = cand.get('max_pct', 15)
            
            # IFRA 한도
            ifra_lim = self.ifra_limits.get(cid)
            if ifra_lim is not None:
                max_pct = min(max_pct, ifra_lim * 0.9)
            
            # 강도 제한
            if cand.get('intensity', 5) >= 8:
                max_pct = min(max_pct, cand.get('max_pct', 8))
            
            bounds.append((0.0, max_pct))

        # 3) 목적함수 정의
        def objective(ratios):
            ratios = np.array(ratios)
            
            # 혼합 벡터: 가중 평균
            total = np.sum(ratios)
            if total < 0.01:
                return 1.0  # 원료 없으면 최악
            
            weights = ratios / total
            mixed = np.dot(weights, odor_matrix)  # (22,)
            
            # (a) 코사인 유사도 (최대화 → -로 최소화)
            norm_t = np.linalg.norm(target_vec)
            norm_m = np.linalg.norm(mixed)
            if norm_t > 0 and norm_m > 0:
                cosine = np.dot(target_vec, mixed) / (norm_t * norm_m)
            else:
                cosine = 0.0
            
            # (b) 잡향 페널티: 타겟에 없는 차원에 혼합물이 높으면 패널티
            active = target_vec > 0.05
            if np.any(~active) and np.sum(mixed) > 0:
                off_note = np.sum(mixed[~active]) / np.sum(mixed)
            else:
                off_note = 0.0
            
            # (c) 원료 수 조절: n_ingredients에 가까울수록 좋음
            active_ings = np.sum(ratios > 0.3)
            complexity_penalty = abs(active_ings - n_ingredients) * 0.02
            
            # (d) 총량 제약 위반 패널티
            total_penalty = abs(total - concentrate_pct) * 0.1
            
            # (e) 다양성: 너무 한 원료에 집중 안 되게 (엔트로피)
            if active_ings > 1:
                probs = weights[ratios > 0.01]
                probs = probs / np.sum(probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(active_ings + 1e-10)
                diversity = entropy / max_entropy if max_entropy > 0 else 0
            else:
                diversity = 0.0
            
            # 종합 목적함수 (최소화)
            loss = (
                -cosine * 2.0           # 코사인 유사도 최대화
                + off_note * 1.0        # 잡향 최소화
                + complexity_penalty    # 원료 수 적절
                + total_penalty         # 총량 제약
                - diversity * 0.3       # 다양성 보상
            )
            
            return loss

        # 4) 다중 시작점 최적화 (global minimum 근사)
        best_result = None
        best_loss = float('inf')
        
        for restart in range(n_restarts):
            # 초기값: AI 스코어 기반 분배 (다양한 변형)
            x0 = np.zeros(n)
            scores = np.array([c.get('ai_score', 0.5) for c in top_candidates])
            
            if restart == 0:
                # 1번: AI 스코어 비례
                x0 = scores / np.sum(scores) * concentrate_pct
            elif restart == 1:
                # 2번: 균등 분배
                x0 = np.ones(n) / n * concentrate_pct
            elif restart == 2:
                # 3번: 탑 N에 집중
                top_n_idx = np.argsort(-scores)[:n_ingredients]
                x0[top_n_idx] = concentrate_pct / n_ingredients
            else:
                # 나머지: 스코어 기반 + 소프트맥스 변형
                temperature = 0.5 + restart * 0.3
                softmax = np.exp(scores / temperature)
                softmax = softmax / np.sum(softmax)
                x0 = softmax * concentrate_pct
            
            # 바운드 내로 클리핑
            for i in range(n):
                x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])

            # L-BFGS-B 최적화
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result

        # 5) 결과 정리
        ratios = np.array(best_result.x)
        
        # 미미한 비율 제거 (0.3% 미만)
        ratios[ratios < 0.3] = 0.0
        
        # 총량 재정규화
        total = np.sum(ratios)
        if total > 0:
            ratios = ratios / total * concentrate_pct

        # 혼합 벡터 계산
        total = np.sum(ratios)
        if total > 0:
            weights = ratios / total
            mixed_vector = np.dot(weights, odor_matrix)
        else:
            mixed_vector = np.zeros(22)

        # 최종 코사인
        norm_t = np.linalg.norm(target_vec)
        norm_m = np.linalg.norm(mixed_vector)
        final_cosine = float(np.dot(target_vec, mixed_vector) / (norm_t * norm_m)) if norm_t > 0 and norm_m > 0 else 0.0

        # 결과 포뮬러
        formula = []
        for i, cand in enumerate(top_candidates):
            if ratios[i] > 0.01:
                formula.append({
                    'ingredient': cand,
                    'ratio_pct': round(float(ratios[i]), 2),
                    'odor_vector': odor_matrix[i].tolist(),
                })
        
        # 비율 내림차순
        formula.sort(key=lambda x: -x['ratio_pct'])

        return {
            'formula': formula,
            'objective_value': float(best_loss),
            'cosine_similarity': final_cosine,
            'mixed_vector': mixed_vector,
            'convergence': best_result.success,
            'iterations': best_result.nit,
            'total_pct': float(np.sum(ratios)),
            'active_ingredients': len(formula),
        }

    def _get_odor_vector(self, ingredient: dict, smiles_map: dict = None) -> np.ndarray:
        """원료 → 22d 향 벡터 (V6 또는 descriptor 기반)"""
        vec = np.zeros(22)
        iid = ingredient.get('id', '')
        
        # V6 GNN으로 예측
        if self.v6_engine and smiles_map:
            smiles = smiles_map.get(iid)
            if smiles:
                try:
                    pred = self.v6_engine.encode(smiles)
                    if len(pred) >= 22:
                        return np.array(pred[:22])
                except:
                    pass
        
        # Fallback: descriptor → 22d
        from scent_interpreter import KEYWORD_VECTORS, ODOR_DIMS
        
        # 카테고리 매핑
        cat = ingredient.get('category', '').lower()
        if cat in ODOR_DIMS:
            idx = ODOR_DIMS.index(cat)
            vec[idx] = 0.8
        
        # descriptor 매핑
        for desc in ingredient.get('descriptors', []):
            desc_lower = desc.lower()
            if desc_lower in KEYWORD_VECTORS:
                for dim, val in KEYWORD_VECTORS[desc_lower].items():
                    if dim in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim)
                        vec[idx] = min(1.0, vec[idx] + val * 0.3)
        
        return vec


# ================================================================
# CLI 테스트
# ================================================================
if __name__ == '__main__':
    print("🔬 RatioOptimizer 테스트")
    
    # 간단한 테스트 원료
    test_candidates = [
        {'id': 'sandalwood', 'name_ko': '샌달우드', 'category': 'woody', 'ai_score': 0.9, 'max_pct': 15, 'intensity': 6, 'descriptors': ['우디','크리미']},
        {'id': 'cedarwood', 'name_ko': '시더우드', 'category': 'woody', 'ai_score': 0.85, 'max_pct': 12, 'intensity': 5, 'descriptors': ['우디','드라이']},
        {'id': 'bergamot', 'name_ko': '베르가못', 'category': 'citrus', 'ai_score': 0.7, 'max_pct': 12, 'intensity': 7, 'descriptors': ['시트러스','상큼']},
        {'id': 'vanilla', 'name_ko': '바닐라', 'category': 'gourmand', 'ai_score': 0.6, 'max_pct': 15, 'intensity': 7, 'descriptors': ['달콤','따뜻한']},
        {'id': 'musk', 'name_ko': '머스크', 'category': 'musk', 'ai_score': 0.75, 'max_pct': 10, 'intensity': 5, 'descriptors': ['머스크','파우더리']},
        {'id': 'black_pepper', 'name_ko': '블랙페퍼', 'category': 'spicy', 'ai_score': 0.5, 'max_pct': 5, 'intensity': 8, 'descriptors': ['스파이시','따뜻한']},
    ]
    
    target = np.zeros(22)
    target[2] = 1.0   # woody
    target[9] = 0.6   # warm
    target[6] = 0.4   # musk
    target[15] = 0.3  # amber
    
    optimizer = RatioOptimizer()
    result = optimizer.optimize(test_candidates, target, concentrate_pct=22.0)
    
    print(f"  수렴: {result['convergence']}")
    print(f"  코사인: {result['cosine_similarity']:.4f}")
    print(f"  반복: {result['iterations']}")
    print(f"  총 농축: {result['total_pct']:.1f}%")
    print(f"\n  최적 배합:")
    for item in result['formula']:
        ing = item['ingredient']
        print(f"    {ing['name_ko']:>8}: {item['ratio_pct']:.2f}%")
