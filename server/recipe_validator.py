"""
RecipeValidator — 고급 수학 기반 레시피 품질 검증
=================================================
8개 독립 수학 지표로 레시피를 다각 검증

지표:
1. Cosine Similarity (방향 정확도)
2. KL Divergence (분포 유사도)
3. Shannon Entropy (복잡도 적절성)
4. Note Pyramid Balance (탑:미들:베이스 밸런스)
5. V6 MixtureNet Harmony (분자 수준 조화)
6. IFRA Compliance Rate (안전성)
7. Simpson Diversity Index (원료 다양성)
8. Sillage Prediction (확산력 예측)
"""

import numpy as np


class RecipeValidator:
    """수학적 다차원 레시피 검증 엔진"""

    # 검증 기준 임계값
    THRESHOLDS = {
        'cosine_similarity': {'min': 0.4, 'good': 0.6, 'excellent': 0.8},
        'kl_divergence': {'max': 2.0, 'good': 1.0, 'excellent': 0.5},
        'shannon_entropy': {'min': 1.0, 'max': 4.0, 'good_min': 1.5, 'good_max': 3.5},
        'pyramid_balance': {'min': 0.5, 'good': 0.7, 'excellent': 0.85},
        'harmony_score': {'min': 0.3, 'good': 0.5, 'excellent': 0.7},
        'ifra_compliance': {'min': 0.9, 'good': 0.95, 'excellent': 1.0},
        'diversity_index': {'min': 0.3, 'good': 0.5, 'excellent': 0.7},
        'sillage_score': {'min': 0.3, 'good': 0.5, 'excellent': 0.7},
    }

    # 이상적 탑:미들:베이스 비율 범위
    IDEAL_PYRAMID = {
        'top': (0.10, 0.30),     # 10~30%
        'middle': (0.20, 0.50),  # 20~50%
        'base': (0.30, 0.55),    # 30~55%
    }

    def __init__(self, v6_engine=None):
        self.v6_engine = v6_engine

    def validate(self, formula: list, target_vec: np.ndarray,
                 mixed_vec: np.ndarray = None, 
                 harmony_score: float = None) -> dict:
        """
        레시피 종합 검증
        
        Args:
            formula: [{'ingredient': {...}, 'ratio_pct': float, 'odor_vector': [...]}, ...]
            target_vec: 22d 타겟 벡터
            mixed_vec: 혼합 벡터 (없으면 계산)
            harmony_score: V6 MixtureNet 조화도 (외부 제공)
        
        Returns:
            {
                'overall_score': float (0~1),
                'overall_grade': str (S/A/B/C/F),
                'metrics': {각 지표별 세부},
                'passed': bool,
                'issues': [str],
                'strengths': [str],
            }
        """
        issues = []
        strengths = []
        
        # 혼합 벡터 계산 (필요시)
        if mixed_vec is None:
            mixed_vec = self._calc_mixed_vector(formula)
        
        # === 1. Cosine Similarity ===
        cosine = self._cosine_similarity(target_vec, mixed_vec)
        
        # === 2. KL Divergence ===
        kl_div = self._kl_divergence(target_vec, mixed_vec)
        
        # === 3. Shannon Entropy ===
        entropy = self._shannon_entropy(formula)
        
        # === 4. Note Pyramid Balance ===
        pyramid = self._pyramid_balance(formula)
        
        # === 5. V6 Harmony ===
        h_score = harmony_score if harmony_score is not None else 0.5
        
        # === 6. IFRA Compliance ===
        ifra_rate = self._ifra_compliance(formula)
        
        # === 7. Simpson Diversity Index ===
        diversity = self._simpson_diversity(formula)
        
        # === 8. Sillage Prediction ===
        sillage = self._predict_sillage(formula)
        
        # === 종합 점수 (가중 평균) ===
        # 정규화된 점수들
        scores = {
            'cosine_similarity': cosine,
            'kl_divergence': max(0, 1.0 - kl_div / 3.0),  # 반전: 낮을수록 좋으므로
            'shannon_entropy': self._entropy_score(entropy),
            'pyramid_balance': pyramid['score'],
            'harmony_score': h_score,
            'ifra_compliance': ifra_rate,
            'diversity_index': diversity,
            'sillage_score': sillage['score'],
        }
        
        # 가중치 (중요도 반영)
        weights = {
            'cosine_similarity': 0.25,  # 타겟 매칭이 가장 중요
            'kl_divergence': 0.10,
            'shannon_entropy': 0.08,
            'pyramid_balance': 0.12,
            'harmony_score': 0.20,      # V6 조화도 중요
            'ifra_compliance': 0.15,    # 안전성
            'diversity_index': 0.05,
            'sillage_score': 0.05,
        }
        
        overall = sum(scores[k] * weights[k] for k in scores)
        
        # 등급
        if overall >= 0.85:
            grade = 'S'
        elif overall >= 0.70:
            grade = 'A'
        elif overall >= 0.55:
            grade = 'B'
        elif overall >= 0.40:
            grade = 'C'
        else:
            grade = 'F'
        
        # 이슈 & 강점 분석
        T = self.THRESHOLDS
        if cosine < T['cosine_similarity']['min']:
            issues.append(f"⚠ 코사인 유사도 {cosine:.3f} — 타겟과 방향이 다름")
        elif cosine >= T['cosine_similarity']['excellent']:
            strengths.append(f"✨ 코사인 유사도 {cosine:.3f} — 타겟 정밀 매칭")
        
        if kl_div > T['kl_divergence']['max']:
            issues.append(f"⚠ KL 발산 {kl_div:.3f} — 분포 차이 큼")
        elif kl_div <= T['kl_divergence']['excellent']:
            strengths.append(f"✨ KL 발산 {kl_div:.3f} — 분포 매우 유사")
        
        ent_min, ent_max = T['shannon_entropy']['good_min'], T['shannon_entropy']['good_max']
        if entropy < T['shannon_entropy']['min'] or entropy > T['shannon_entropy']['max']:
            issues.append(f"⚠ 엔트로피 {entropy:.3f} — 복잡도 부적절")
        elif ent_min <= entropy <= ent_max:
            strengths.append(f"✨ 엔트로피 {entropy:.3f} — 적절한 복잡도")
        
        if pyramid['score'] < T['pyramid_balance']['min']:
            issues.append(f"⚠ 노트 피라미드 불균형: {pyramid['detail']}")
        elif pyramid['score'] >= T['pyramid_balance']['excellent']:
            strengths.append(f"✨ 노트 피라미드 밸런스 우수: {pyramid['detail']}")
        
        if ifra_rate < T['ifra_compliance']['min']:
            issues.append(f"⚠ IFRA 준수 {ifra_rate:.0%} — 안전 검토 필요")
        elif ifra_rate >= T['ifra_compliance']['excellent']:
            strengths.append(f"✨ IFRA 100% 준수")
        
        if diversity < T['diversity_index']['min']:
            issues.append(f"⚠ 다양성 지수 {diversity:.3f} — 원료 편중")
        
        passed = overall >= 0.40 and ifra_rate >= 0.9

        metrics = {
            'cosine_similarity': {'value': cosine, 'unit': '', 'description': '타겟 벡터와 방향 일치도'},
            'kl_divergence': {'value': kl_div, 'unit': 'nats', 'description': '타겟 대비 분포 발산도'},
            'shannon_entropy': {'value': entropy, 'unit': 'bits', 'description': '배합 복잡도'},
            'pyramid_balance': {'value': pyramid['score'], 'detail': pyramid['detail'], 'description': '탑:미들:베이스 비율 적절성'},
            'harmony_score': {'value': h_score, 'description': 'V6 MixtureNet 분자 조화도'},
            'ifra_compliance': {'value': ifra_rate, 'description': 'IFRA 안전 규제 준수율'},
            'diversity_index': {'value': diversity, 'description': 'Simpson 다양성 지수'},
            'sillage_prediction': {'value': sillage['level'], 'score': sillage['score'], 'description': '예측 확산력'},
        }

        return {
            'overall_score': round(overall, 4),
            'overall_grade': grade,
            'metrics': metrics,
            'passed': passed,
            'issues': issues,
            'strengths': strengths,
            'scores': scores,
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            return float(max(0, np.dot(a, b) / (na * nb)))
        return 0.0

    def _kl_divergence(self, target: np.ndarray, mixed: np.ndarray) -> float:
        """KL(target || mixed) — 타겟 분포 대비 혼합물 분포 발산"""
        # 분포로 변환 (음수 제거, 정규화)
        p = np.maximum(target, 0) + 1e-10
        q = np.maximum(mixed, 0) + 1e-10
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))

    def _shannon_entropy(self, formula: list) -> float:
        """배합비의 Shannon Entropy — 복잡도 지표"""
        ratios = np.array([f['ratio_pct'] for f in formula if f['ratio_pct'] > 0])
        if len(ratios) < 1:
            return 0.0
        total = np.sum(ratios)
        if total <= 0:
            return 0.0
        probs = ratios / total
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def _entropy_score(self, entropy: float) -> float:
        """엔트로피를 0~1 점수로 변환"""
        # 2.0~3.0 사이가 최적
        if 2.0 <= entropy <= 3.0:
            return 1.0
        elif entropy < 1.0:
            return max(0, entropy / 1.0)
        elif entropy > 4.0:
            return max(0, 1.0 - (entropy - 4.0) / 2.0)
        elif entropy < 2.0:
            return 0.5 + 0.5 * (entropy - 1.0) / 1.0
        else:  # 3.0 < entropy <= 4.0
            return 0.5 + 0.5 * (4.0 - entropy) / 1.0

    def _pyramid_balance(self, formula: list) -> dict:
        """노트 피라미드 밸런스 점수"""
        note_pcts = {'top': 0, 'middle': 0, 'base': 0}
        for f in formula:
            note = f.get('ingredient', {}).get('note_type', 'middle')
            note_pcts[note] = note_pcts.get(note, 0) + f['ratio_pct']
        
        total = sum(note_pcts.values())
        if total <= 0:
            return {'score': 0, 'detail': 'N/A'}
        
        ratios = {k: v / total for k, v in note_pcts.items()}
        
        # 각 노트가 이상 범위 안에 있는지
        score = 0
        for note_type, (lo, hi) in self.IDEAL_PYRAMID.items():
            r = ratios.get(note_type, 0)
            if lo <= r <= hi:
                score += 1.0
            elif r < lo:
                score += max(0, r / lo)
            else:
                score += max(0, 1.0 - (r - hi) / 0.3)
        
        score /= 3.0
        
        detail = f"T:{ratios.get('top',0):.0%} M:{ratios.get('middle',0):.0%} B:{ratios.get('base',0):.0%}"
        return {'score': min(1.0, score), 'detail': detail}

    def _ifra_compliance(self, formula: list) -> float:
        """IFRA 준수율 (위반 원료 / 전체 원료)"""
        from ratio_optimizer import RatioOptimizer
        ifra = RatioOptimizer().ifra_limits
        
        total = len(formula)
        if total == 0:
            return 1.0
        
        violations = 0
        for f in formula:
            iid = f.get('ingredient', {}).get('id', '')
            limit = ifra.get(iid)
            if limit is not None and f['ratio_pct'] > limit:
                violations += 1
        
        return 1.0 - (violations / total)

    def _simpson_diversity(self, formula: list) -> float:
        """Simpson Diversity Index — 1 - Σ(pᵢ²)"""
        ratios = np.array([f['ratio_pct'] for f in formula if f['ratio_pct'] > 0])
        if len(ratios) < 2:
            return 0.0
        total = np.sum(ratios)
        if total <= 0:
            return 0.0
        probs = ratios / total
        return float(1.0 - np.sum(probs ** 2))

    def _predict_sillage(self, formula: list) -> dict:
        """확산력 예측: 원료의 volatility × ratio 기반"""
        if not formula:
            return {'level': 'N/A', 'score': 0}
        
        weighted_vol = 0
        total_pct = 0
        for f in formula:
            vol = f.get('ingredient', {}).get('volatility', 5)
            pct = f['ratio_pct']
            weighted_vol += vol * pct
            total_pct += pct
        
        if total_pct > 0:
            avg_vol = weighted_vol / total_pct
        else:
            avg_vol = 5
        
        # 높은 volatility + 높은 농축률 = 높은 확산력
        sillage_raw = (avg_vol / 10) * 0.4 + (total_pct / 30) * 0.6
        sillage_raw = min(1.0, sillage_raw)
        
        if sillage_raw >= 0.7:
            level = '강함'
        elif sillage_raw >= 0.5:
            level = '보통'
        elif sillage_raw >= 0.3:
            level = '은은함'
        else:
            level = '약함'
        
        return {'level': level, 'score': sillage_raw}

    def _calc_mixed_vector(self, formula: list) -> np.ndarray:
        """포뮬러에서 혼합 벡터 계산"""
        vec = np.zeros(22)
        total = sum(f['ratio_pct'] for f in formula)
        if total <= 0:
            return vec
        for f in formula:
            ov = np.array(f.get('odor_vector', [0]*22))
            if len(ov) < 22:
                ov = np.pad(ov, (0, 22 - len(ov)))
            vec += ov * (f['ratio_pct'] / total)
        return vec

    def format_report(self, result: dict) -> str:
        """검증 결과를 텍스트 리포트로"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  📊 AI 레시피 품질 검증 리포트")
        lines.append(f"  종합: {result['overall_score']:.3f} / 1.000  등급: {result['overall_grade']}")
        lines.append("=" * 60)
        
        for name, metric in result['metrics'].items():
            val = metric.get('value', 0)
            desc = metric.get('description', '')
            if isinstance(val, float):
                lines.append(f"  {name:>25}: {val:.4f}  — {desc}")
            else:
                lines.append(f"  {name:>25}: {val}  — {desc}")
        
        lines.append("")
        
        if result['strengths']:
            lines.append("  💎 강점:")
            for s in result['strengths']:
                lines.append(f"    {s}")
        
        if result['issues']:
            lines.append("  ⚠ 개선 필요:")
            for i in result['issues']:
                lines.append(f"    {i}")
        
        lines.append("")
        lines.append(f"  {'✅ PASS' if result['passed'] else '❌ FAIL'}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
