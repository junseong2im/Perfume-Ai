// ai-client.js — Python GPU 백엔드 API 클라이언트
// ==================================================
// TF.js 제거, Python FastAPI 서버로 모든 AI 연산 위임
// 기존 ai-engine.js와 동일 인터페이스 → app.js 최소 변경
// ==================================================

const API_BASE = 'http://localhost:8001/api';

class AIClient {
    constructor() {
        this.level = 1;
        this.ingredients = [];
        this.molecules = [];
        this.molecules3d = [];
        this.primaryOdors = null;
        this.initialized = false;
    }

    // =========================================
    // 초기화 + 데이터 로딩
    // =========================================
    async init() {
        try {
            const health = await this._get('/health');
            console.log(`[AIClient] Connected: GPU=${health.gpu}, Level=${health.level}`);
            this.level = health.level;

            const [ingredients, molecules, mol3d, odors] = await Promise.all([
                this._get('/data/ingredients'),
                this._get('/data/molecules'),
                this._get('/data/molecules3d'),
                this._get('/data/primary-odors')
            ]);
            this.ingredients = ingredients;
            this.molecules = molecules;
            this.molecules3d = mol3d || [];
            this.primaryOdors = odors;
            this.initialized = true;
            console.log(`[AIClient] Data: ${ingredients.length} ingredients, ${molecules.length} molecules`);
        } catch (e) {
            console.error('[AIClient] Server unavailable:', e.message);
            throw new Error('Python GPU 서버에 연결할 수 없습니다. python server/main.py 를 실행하세요.');
        }
    }

    // =========================================
    // 학습 (SSE 스트리밍 진행률)
    // =========================================
    async trainAll(onProgress) {
        return new Promise((resolve, reject) => {
            const eventSource = new EventSource(`${API_BASE}/train`);

            // SSE는 GET만 지원 → POST를 대신 fetch로
            // EventSource 대신 fetch + ReadableStream 사용
            fetch(`${API_BASE}/train`, { method: 'POST' })
                .then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    const processStream = async () => {
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            const text = decoder.decode(value);
                            const lines = text.split('\n').filter(l => l.startsWith('data: '));
                            for (const line of lines) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (onProgress) onProgress(data.stage, data.message);
                                    if (data.stage === 'complete') {
                                        this.level = 5;
                                    }
                                } catch (e) { }
                            }
                        }
                        resolve({ level: this.level });
                    };
                    processStream().catch(reject);
                })
                .catch(reject);
        });
    }

    // =========================================
    // 향수 생성 (Level 1~5)
    // =========================================
    async createPerfume(mood, season, preferences, intensity = 50) {
        return this._post('/perfume/create', { mood, season, preferences, intensity });
    }

    // VAE 생성 (Level 3+)
    async generateNovelFormula() {
        return this._post('/perfume/generate', {});
    }

    async generateFromLatent(x, y) {
        return this._post('/perfume/latent', { x, y });
    }

    // =========================================
    // 분자 탐색 (Level 4+)
    // =========================================
    async exploreMolecule(id) {
        return this._post('/molecule/explore', { id });
    }

    async predictSmell(smiles) {
        return this._post('/molecule/predict', { smiles });
    }

    async getVariants(id) {
        return this._post('/molecule/variants', { id });
    }

    // =========================================
    // 혼합 시뮬레이션 (Level 5 - Pillar 2)
    // =========================================
    async simulateBinding(moleculeIds) {
        return this._post('/binding/simulate', { moleculeIds });
    }

    // =========================================
    // 센서 시뮬레이션 (Level 5 - Pillar 3)
    // =========================================
    async simulateSensor(id) {
        return this._post('/sensor/simulate', { id });
    }

    // =========================================
    // 텍스트→냄새 (Level 5 - Pillar 4)
    // =========================================
    async predictFromText(text) {
        return this._post('/text/predict', { text });
    }

    // =========================================
    // Chirality (Level 5 - Pillar 1)
    // =========================================
    async getChiralPairs() {
        return this._get('/chirality/pairs');
    }

    // =========================================
    // 데이터 접근
    // =========================================
    getIngredients() { return this.ingredients; }
    getMolecules() { return this.molecules; }
    getMolecules3D() { return this.molecules3d; }
    getPrimaryOdors() { return this.primaryOdors; }
    getLevel() { return this.level; }

    getMoleculeById(id) {
        return this.molecules.find(m => m.id === id);
    }

    getIngredientById(id) {
        return this.ingredients.find(i => i.id === id);
    }

    // =========================================
    // 서버 상태
    // =========================================
    async getHealth() {
        return this._get('/health');
    }

    // =========================================
    // HTTP 유틸리티
    // =========================================
    async _get(path) {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
        return res.json();
    }

    async _post(path, body) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(`API ${path}: ${err.detail || res.status}`);
        }
        return res.json();
    }
}

export default AIClient;
