// ingredient-db.js - 향료 데이터베이스 관리 모듈
class IngredientDB {
    constructor() {
        this.ingredients = [];
        this.accords = [];
        this.compatibility = { synergy: [], clash: [] };
        this.loaded = false;
    }

    async load() {
        try {
            const [ingRes, accRes, compRes] = await Promise.all([
                fetch('./data/ingredients.json'),
                fetch('./data/accords.json'),
                fetch('./data/compatibility.json')
            ]);
            this.ingredients = await ingRes.json();
            this.accords = await accRes.json();
            this.compatibility = await compRes.json();
            this.loaded = true;
            console.log(`[IngredientDB] Loaded ${this.ingredients.length} ingredients, ${this.accords.length} accords`);
        } catch (e) {
            console.error('[IngredientDB] Load failed:', e);
        }
    }

    getAll() { return this.ingredients; }
    getById(id) { return this.ingredients.find(i => i.id === id); }
    getByCategory(cat) { return this.ingredients.filter(i => i.category === cat); }
    getByNoteType(type) { return this.ingredients.filter(i => i.note_type === type); }

    getByCategories(cats) {
        return this.ingredients.filter(i => cats.includes(i.category));
    }

    search(query) {
        const q = query.toLowerCase();
        return this.ingredients.filter(i =>
            i.name_ko.includes(q) || i.name_en.toLowerCase().includes(q) ||
            i.descriptors.some(d => d.includes(q)) || i.category.includes(q)
        );
    }

    getSimilar(id, limit = 5) {
        const target = this.getById(id);
        if (!target) return [];
        return this.ingredients
            .filter(i => i.id !== id)
            .map(i => ({
                ...i,
                similarity: this._calcSimilarity(target, i)
            }))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, limit);
    }

    _calcSimilarity(a, b) {
        let score = 0;
        if (a.category === b.category) score += 3;
        if (a.note_type === b.note_type) score += 2;
        score += (1 - Math.abs(a.intensity - b.intensity) / 10);
        score += (1 - Math.abs(a.volatility - b.volatility) / 10);
        const shared = a.descriptors.filter(d => b.descriptors.includes(d)).length;
        score += shared * 1.5;
        return score;
    }

    getHarmonyScore(idA, idB) {
        for (const [a, b, score] of this.compatibility.synergy) {
            if ((a === idA && b === idB) || (a === idB && b === idA)) return score;
        }
        for (const [a, b, score] of this.compatibility.clash) {
            if ((a === idA && b === idB) || (a === idB && b === idA)) return score;
        }
        // Default: neutral with slight bonus for same category
        const ingA = this.getById(idA), ingB = this.getById(idB);
        if (ingA && ingB && ingA.category === ingB.category) return 0.3;
        return 0.1;
    }

    getAccord(id) { return this.accords.find(a => a.id === id); }
    getAllAccords() { return this.accords; }

    getCategories() {
        return [...new Set(this.ingredients.map(i => i.category))];
    }

    getCategoryMap() {
        const map = {};
        for (const i of this.ingredients) {
            if (!map[i.category]) map[i.category] = [];
            map[i.category].push(i);
        }
        return map;
    }
}

export default IngredientDB;
