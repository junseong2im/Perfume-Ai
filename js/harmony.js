// harmony.js - 노트 조화 알고리즘
class HarmonyEngine {
    constructor(db) {
        this.db = db;
    }

    // 두 향료 간 조화도 점수 (-1 ~ 1)
    pairScore(idA, idB) {
        return this.db.getHarmonyScore(idA, idB);
    }

    // 전체 포뮬러 조화도 (0-100)
    formulaHarmony(ingredientIds) {
        if (ingredientIds.length < 2) return 100;
        let totalScore = 0, pairs = 0;
        for (let i = 0; i < ingredientIds.length; i++) {
            for (let j = i + 1; j < ingredientIds.length; j++) {
                totalScore += this.pairScore(ingredientIds[i], ingredientIds[j]);
                pairs++;
            }
        }
        const avg = totalScore / pairs;
        return Math.round(Math.max(0, Math.min(100, (avg + 1) * 50)));
    }

    // 노트 피라미드 밸런스 점수 (0-100)
    pyramidBalance(formula) {
        const noteWeights = { top: 0, middle: 0, base: 0 };
        for (const item of formula) {
            const ing = this.db.getById(item.id);
            if (ing) noteWeights[ing.note_type] += item.percentage;
        }
        const total = noteWeights.top + noteWeights.middle + noteWeights.base;
        if (total === 0) return 0;
        const topR = noteWeights.top / total;
        const midR = noteWeights.middle / total;
        const baseR = noteWeights.base / total;

        // 이상적비율: 탑 15-30%, 미들 30-50%, 베이스 20-35%
        const idealTop = [0.15, 0.30], idealMid = [0.30, 0.50], idealBase = [0.20, 0.35];
        let score = 100;
        score -= this._deviationPenalty(topR, idealTop) * 30;
        score -= this._deviationPenalty(midR, idealMid) * 40;
        score -= this._deviationPenalty(baseR, idealBase) * 30;
        return Math.round(Math.max(0, Math.min(100, score)));
    }

    _deviationPenalty(value, [min, max]) {
        if (value >= min && value <= max) return 0;
        if (value < min) return (min - value) / min;
        return (value - max) / (1 - max);
    }

    // 복합성 점수 (0-100) - 향의 다양성
    complexityScore(formula) {
        const categories = new Set();
        const descriptors = new Set();
        for (const item of formula) {
            const ing = this.db.getById(item.id);
            if (ing) {
                categories.add(ing.category);
                ing.descriptors.forEach(d => descriptors.add(d));
            }
        }
        const catScore = Math.min(100, categories.size * 15);
        const descScore = Math.min(100, descriptors.size * 5);
        const countScore = Math.min(100, formula.length * 8);
        return Math.round((catScore + descScore + countScore) / 3);
    }

    // 특정 향료가 기존 포뮬러에 어울리는지 점수
    fitScore(ingredientId, existingIds) {
        if (existingIds.length === 0) return 0.5;
        let total = 0;
        for (const id of existingIds) {
            total += this.pairScore(ingredientId, id);
        }
        return total / existingIds.length;
    }

    // 향료 추천 - 현재 포뮬러에 가장 잘 어울리는 향료
    recommend(existingIds, noteType = null, limit = 10) {
        let candidates = this.db.getAll().filter(i => !existingIds.includes(i.id));
        if (noteType) candidates = candidates.filter(i => i.note_type === noteType);

        return candidates
            .map(i => ({ ...i, fit: this.fitScore(i.id, existingIds) }))
            .sort((a, b) => b.fit - a.fit)
            .slice(0, limit);
    }
}

export default HarmonyEngine;
