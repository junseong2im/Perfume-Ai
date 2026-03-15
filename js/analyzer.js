// analyzer.js - 향 프로필 분석기
class Analyzer {
    constructor(db, harmony) {
        this.db = db;
        this.harmony = harmony;
    }

    analyze(formula) {
        const ingredients = formula.map(f => ({
            ...f,
            data: this.db.getById(f.id)
        })).filter(f => f.data);

        return {
            harmony: this.harmony.formulaHarmony(formula.map(f => f.id)),
            balance: this.harmony.pyramidBalance(formula),
            complexity: this.harmony.complexityScore(formula),
            longevity: this._predictLongevity(ingredients),
            sillage: this._predictSillage(ingredients),
            character: this._extractCharacter(ingredients),
            notePyramid: this._buildPyramid(ingredients),
            categoryBreakdown: this._categoryBreakdown(ingredients),
            timeline: this._buildTimeline(ingredients),
            overallScore: 0 // computed below
        };
    }

    _predictLongevity(ingredients) {
        let weightedLongevity = 0, totalPct = 0;
        for (const ing of ingredients) {
            weightedLongevity += ing.data.longevity * ing.percentage;
            totalPct += ing.percentage;
        }
        const hours = totalPct > 0 ? weightedLongevity / totalPct : 0;
        return {
            hours: Math.round(hours * 10) / 10,
            label: hours >= 8 ? '매우 오래 지속' : hours >= 6 ? '오래 지속' : hours >= 4 ? '보통' : hours >= 2 ? '가벼운' : '매우 가벼운',
            score: Math.round(Math.min(100, hours * 12))
        };
    }

    _predictSillage(ingredients) {
        let weightedIntensity = 0, totalPct = 0;
        for (const ing of ingredients) {
            weightedIntensity += ing.data.intensity * ing.percentage * (1 + (10 - ing.data.volatility) / 20);
            totalPct += ing.percentage;
        }
        const sillage = totalPct > 0 ? weightedIntensity / totalPct : 0;
        return {
            value: Math.round(sillage * 10) / 10,
            label: sillage >= 8 ? '강렬한 확산' : sillage >= 6 ? '좋은 확산' : sillage >= 4 ? '보통 확산' : '은은한',
            score: Math.round(Math.min(100, sillage * 13))
        };
    }

    _extractCharacter(ingredients) {
        const descriptorCount = {};
        for (const ing of ingredients) {
            for (const desc of ing.data.descriptors) {
                descriptorCount[desc] = (descriptorCount[desc] || 0) + ing.percentage;
            }
        }
        return Object.entries(descriptorCount)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 6)
            .map(([keyword, weight]) => ({ keyword, weight: Math.round(weight) }));
    }

    _buildPyramid(ingredients) {
        const pyramid = { top: [], middle: [], base: [] };
        for (const ing of ingredients) {
            pyramid[ing.data.note_type].push({
                id: ing.id,
                name_ko: ing.data.name_ko,
                name_en: ing.data.name_en,
                percentage: ing.percentage,
                category: ing.data.category,
                descriptors: ing.data.descriptors
            });
        }
        for (const key of ['top', 'middle', 'base']) {
            pyramid[key].sort((a, b) => b.percentage - a.percentage);
        }
        return pyramid;
    }

    _categoryBreakdown(ingredients) {
        const cats = {};
        for (const ing of ingredients) {
            const cat = ing.data.category;
            cats[cat] = (cats[cat] || 0) + ing.percentage;
        }
        return Object.entries(cats)
            .map(([name, pct]) => ({ name, percentage: Math.round(pct * 10) / 10 }))
            .sort((a, b) => b.percentage - a.percentage);
    }

    _buildTimeline(ingredients) {
        const phases = [
            { name: '탑 노트 (0-30분)', time: '0-30min', notes: [] },
            { name: '미들 노트 (30분-3시간)', time: '30min-3h', notes: [] },
            { name: '베이스 노트 (3시간+)', time: '3h+', notes: [] }
        ];
        for (const ing of ingredients) {
            const idx = ing.data.note_type === 'top' ? 0 : ing.data.note_type === 'middle' ? 1 : 2;
            phases[idx].notes.push(ing.data.name_ko);
        }
        return phases;
    }

    getOverallScore(analysis) {
        return Math.round(
            analysis.harmony * 0.30 +
            analysis.balance * 0.25 +
            analysis.complexity * 0.20 +
            analysis.longevity.score * 0.15 +
            analysis.sillage.score * 0.10
        );
    }
}

export default Analyzer;
