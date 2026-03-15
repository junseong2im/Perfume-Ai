// formulator.js - 포뮬레이션 엔진 (배합비 최적화)
class Formulator {
    constructor(db, harmony) {
        this.db = db;
        this.harmony = harmony;
    }

    // 메인 포뮬레이션 함수
    formulate(params) {
        const { mood, season, preferences, intensity: userIntensity } = params;

        // 1. 타겟 프로필 생성
        const profile = this._buildTargetProfile(mood, season, preferences, userIntensity);

        // 2. 후보 향료 선별
        const candidates = this._selectCandidates(profile);

        // 3. 최적 조합 생성
        const formula = this._optimize(candidates, profile);

        // 4. 배합비 정규화
        return this._normalizeFormula(formula);
    }

    _buildTargetProfile(mood, season, preferences, userIntensity) {
        const moodMap = {
            romantic: { categories: ['floral', 'fruity', 'gourmand'], intensityMod: 0, warmth: 0.6, sweetness: 0.7 },
            fresh: { categories: ['citrus', 'aquatic', 'green', 'aromatic'], intensityMod: -1, warmth: 0.2, sweetness: 0.3 },
            elegant: { categories: ['floral', 'woody', 'chypre'], intensityMod: 0, warmth: 0.5, sweetness: 0.4 },
            sexy: { categories: ['amber', 'animalic', 'musk', 'gourmand'], intensityMod: 1, warmth: 0.8, sweetness: 0.6 },
            mysterious: { categories: ['balsamic', 'woody', 'amber', 'spicy'], intensityMod: 1, warmth: 0.7, sweetness: 0.3 },
            cheerful: { categories: ['citrus', 'fruity', 'floral', 'green'], intensityMod: -1, warmth: 0.3, sweetness: 0.6 },
            calm: { categories: ['aromatic', 'woody', 'green', 'aquatic'], intensityMod: -1, warmth: 0.4, sweetness: 0.3 },
            luxurious: { categories: ['woody', 'amber', 'floral', 'animalic'], intensityMod: 1, warmth: 0.7, sweetness: 0.5 },
            natural: { categories: ['green', 'aromatic', 'woody', 'citrus'], intensityMod: -1, warmth: 0.3, sweetness: 0.2 },
            cozy: { categories: ['gourmand', 'amber', 'woody', 'spicy'], intensityMod: 0, warmth: 0.9, sweetness: 0.8 }
        };

        const seasonMap = {
            spring: { volatilityBias: 0.6, lightness: 0.7, bonusCategories: ['floral', 'green', 'citrus'] },
            summer: { volatilityBias: 0.8, lightness: 0.8, bonusCategories: ['citrus', 'aquatic', 'fruity'] },
            fall: { volatilityBias: 0.3, lightness: 0.3, bonusCategories: ['woody', 'spicy', 'amber'] },
            winter: { volatilityBias: 0.2, lightness: 0.2, bonusCategories: ['gourmand', 'amber', 'balsamic'] }
        };

        const moodProfile = moodMap[mood] || moodMap.romantic;
        const seasonProfile = seasonMap[season] || seasonMap.spring;

        const targetCategories = [...new Set([
            ...moodProfile.categories,
            ...seasonProfile.bonusCategories,
            ...(preferences || [])
        ])];

        return {
            categories: targetCategories,
            warmth: moodProfile.warmth,
            sweetness: moodProfile.sweetness,
            intensityTarget: 5 + (moodProfile.intensityMod || 0) + (userIntensity || 0),
            volatilityBias: seasonProfile.volatilityBias,
            lightness: seasonProfile.lightness,
            noteRatios: { top: 0.25, middle: 0.40, base: 0.35 }
        };
    }

    _selectCandidates(profile) {
        const all = this.db.getAll();
        const scored = all.map(ing => {
            let score = 0;
            // 카테고리 매칭
            if (profile.categories.includes(ing.category)) score += 5;
            // 강도 적합성
            score += (1 - Math.abs(ing.intensity - profile.intensityTarget) / 10) * 3;
            // 따뜻함-차가움 축
            const warmIngredients = ['amber', 'balsamic', 'gourmand', 'spicy', 'animalic'];
            const coolIngredients = ['citrus', 'aquatic', 'green', 'aromatic'];
            if (warmIngredients.includes(ing.category)) score += profile.warmth * 3;
            if (coolIngredients.includes(ing.category)) score += (1 - profile.warmth) * 3;
            // 랜덤성 추가 (창의적 조합을 위해)
            score += Math.random() * 2;
            return { ...ing, candidateScore: score };
        });

        scored.sort((a, b) => b.candidateScore - a.candidateScore);

        // 노트 타입별 최소 수 보장
        const tops = scored.filter(i => i.note_type === 'top').slice(0, 8);
        const mids = scored.filter(i => i.note_type === 'middle').slice(0, 10);
        const bases = scored.filter(i => i.note_type === 'base').slice(0, 8);

        return [...tops, ...mids, ...bases];
    }

    _optimize(candidates, profile) {
        const formula = [];
        const topCandidates = candidates.filter(c => c.note_type === 'top');
        const midCandidates = candidates.filter(c => c.note_type === 'middle');
        const baseCandidates = candidates.filter(c => c.note_type === 'base');

        // 탑 노트 3-4개 선택
        this._greedySelect(topCandidates, formula, 3 + (Math.random() > 0.5 ? 1 : 0));
        // 미들 노트 3-5개 선택
        this._greedySelect(midCandidates, formula, 3 + Math.floor(Math.random() * 3));
        // 베이스 노트 2-4개 선택
        this._greedySelect(baseCandidates, formula, 2 + Math.floor(Math.random() * 3));

        // 배합비 할당
        const totalItems = formula.length;
        for (const item of formula) {
            const ing = this.db.getById(item.id);
            let basePct = ing.typical_pct;
            // 노트 타입별 가중치
            if (ing.note_type === 'top') basePct *= profile.noteRatios.top / 0.25;
            if (ing.note_type === 'middle') basePct *= profile.noteRatios.middle / 0.40;
            if (ing.note_type === 'base') basePct *= profile.noteRatios.base / 0.35;
            // 약간의 랜덤 변동
            basePct *= (0.8 + Math.random() * 0.4);
            item.percentage = Math.max(1, Math.min(ing.max_pct, basePct));
        }

        return formula;
    }

    _greedySelect(candidates, formula, count) {
        const selectedIds = formula.map(f => f.id);
        const usedCategories = new Set(formula.map(f => this.db.getById(f.id)?.category));

        for (let i = 0; i < count && candidates.length > 0; i++) {
            let bestIdx = -1, bestScore = -Infinity;

            for (let j = 0; j < candidates.length; j++) {
                const c = candidates[j];
                if (selectedIds.includes(c.id)) continue;

                let score = c.candidateScore;
                // 기존 향료와의 조화도
                score += this.harmony.fitScore(c.id, selectedIds) * 5;
                // 카테고리 다양성 보너스
                if (!usedCategories.has(c.category)) score += 2;

                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = j;
                }
            }

            if (bestIdx >= 0) {
                const selected = candidates[bestIdx];
                formula.push({ id: selected.id, percentage: 0 });
                selectedIds.push(selected.id);
                usedCategories.add(selected.category);
                candidates.splice(bestIdx, 1);
            }
        }
    }

    _normalizeFormula(formula) {
        const total = formula.reduce((sum, f) => sum + f.percentage, 0);
        if (total === 0) return formula;
        for (const item of formula) {
            item.percentage = Math.round((item.percentage / total) * 1000) / 10;
        }
        // 반올림 보정
        const newTotal = formula.reduce((sum, f) => sum + f.percentage, 0);
        if (newTotal !== 100 && formula.length > 0) {
            formula[0].percentage = Math.round((formula[0].percentage + (100 - newTotal)) * 10) / 10;
        }
        return formula;
    }
}

export default Formulator;
