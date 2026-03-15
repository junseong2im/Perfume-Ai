// active-learner.js — Pillar 4: 진짜 Embedding + GRU + 임베딩 Zero-shot
// =======================================================================
// 핵심 수정:
// 1. Dense 우회 → tf.layers.embedding 직접 사용
// 2. SMILES 토큰 → 학습 가능한 임베딩 벡터 → GRU 시퀀스 인코딩
// 3. MC Dropout + 코사인 유사도 기반 Zero-shot
// =======================================================================

class ActiveLearner {
    constructor() {
        // SMILES 토큰 사전
        this.tokenMap = {};
        this.vocabSize = 0;
        this.maxSeqLen = 64;
        this.embedDim = 32;
        this.gruHiddenDim = 64;

        // 모델
        this.chemLangModel = null;      // Embedding + GRU SMILES → 냄새
        this.uncertaintyModel = null;    // MC Dropout 모델
        this.trained = false;

        // 냄새 임베딩 (학습됨)
        this.odorEmbeddings = null;
        this.odorLabels = [
            'floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh',
            'green', 'warm', 'musk', 'fruity', 'rose', 'jasmine',
            'cedar', 'vanilla', 'amber', 'clean', 'smoky', 'powdery',
            'aquatic', 'herbal'
        ];

        // 텍스트-냄새 시드 매핑
        this.textOdorSeeds = {
            '장미': ['floral', 'rose'], '재스민': ['floral', 'jasmine'],
            '레몬': ['citrus', 'fresh'], '오렌지': ['citrus', 'fruity'],
            '백단향': ['woody', 'warm'], '시더': ['woody', 'cedar'],
            '바닐라': ['sweet', 'vanilla'], '꿀': ['sweet', 'warm'],
            '라벤더': ['herbal', 'fresh'], '민트': ['fresh', 'green'],
            '계피': ['spicy', 'warm'], '후추': ['spicy'],
            '바다': ['aquatic', 'fresh'], '이끼': ['green', 'woody'],
            '머스크': ['musk', 'powdery'], '앰버': ['amber', 'warm'],
            '연기': ['smoky', 'woody'], '비누': ['clean', 'fresh'],
            '복숭아': ['fruity', 'sweet'], '사과': ['fruity', 'green'],
            '숲': ['green', 'woody', 'fresh'], '꽃': ['floral'],
            '향신료': ['spicy'], '과일': ['fruity'], '나무': ['woody'],
            '달콤': ['sweet'], '신선': ['fresh'], '깨끗': ['clean'],
            '따뜻': ['warm'], '부드러운': ['powdery', 'musk']
        };
    }

    // ==========================================================
    // SMILES 토크나이저
    // ==========================================================
    buildVocab(molecules) {
        const tokenSet = new Set(['<PAD>', '<START>', '<END>', '<UNK>']);

        for (const mol of molecules) {
            const smiles = mol.smiles || '';
            const tokens = this._tokenizeSmiles(smiles);
            tokens.forEach(t => tokenSet.add(t));
        }

        this.tokenMap = {};
        let idx = 0;
        for (const token of tokenSet) {
            this.tokenMap[token] = idx++;
        }
        this.vocabSize = idx;
        console.log(`[ActiveLearner] Vocab built: ${this.vocabSize} tokens`);
    }

    // SMILES → 토큰 배열 (multi-char 원소 처리)
    _tokenizeSmiles(smiles) {
        const tokens = [];
        let i = 0;
        while (i < smiles.length) {
            if (smiles[i] === '[') {
                const close = smiles.indexOf(']', i);
                if (close > i) {
                    tokens.push(smiles.substring(i, close + 1));
                    i = close + 1;
                    continue;
                }
            }
            if (i + 1 < smiles.length) {
                const two = smiles.substring(i, i + 2);
                if (['Cl', 'Br', 'Si', 'Na', 'Li', 'Se', 'Te'].includes(two)) {
                    tokens.push(two);
                    i += 2;
                    continue;
                }
            }
            tokens.push(smiles[i]);
            i++;
        }
        return tokens;
    }

    // SMILES → 정수 시퀀스 (패딩)
    encodeSmiles(smiles) {
        const tokens = this._tokenizeSmiles(smiles);
        const seq = [this.tokenMap['<START>'] || 1];
        for (const token of tokens) {
            seq.push(this.tokenMap[token] || this.tokenMap['<UNK>'] || 3);
        }
        seq.push(this.tokenMap['<END>'] || 2);
        while (seq.length < this.maxSeqLen) seq.push(0);
        return seq.slice(0, this.maxSeqLen);
    }

    // ==========================================================
    // 진짜 Embedding + GRU Chemical Language Model
    // ==========================================================
    buildChemLangModel() {
        const input = tf.input({ shape: [this.maxSeqLen] });

        // ★ 진짜 tf.layers.embedding: 정수 토큰 → 학습 가능 벡터
        // inputDim = 어휘 크기, outputDim = 임베딩 차원
        // 파라미터: vocabSize × embedDim 개의 학습 가능 가중치
        const embedded = tf.layers.embedding({
            inputDim: Math.max(this.vocabSize, 100), // 최소 100
            outputDim: this.embedDim,                 // 32차원 임베딩
            inputLength: this.maxSeqLen,
            embeddingsInitializer: 'glorotUniform',
            maskZero: true,  // <PAD>=0 마스킹
            name: 'smiles_embedding'
        }).apply(input);   // [batch, maxSeqLen, embedDim]

        // GRU 인코더 — 임베딩 시퀀스를 순차 처리
        const gruOut = tf.layers.gru({
            units: this.gruHiddenDim,
            returnSequences: false,     // 마지막 hidden state만
            goBackwards: false,
            recurrentDropout: 0,        // WebGL에서 recurrentDropout 지원 제한
            dropout: 0.1,
            name: 'gru_encoder'
        }).apply(embedded);   // [batch, gruHiddenDim]

        // GRU 출력 → 냄새 임베딩 공간
        const odorEmbed = tf.layers.dense({
            units: this.embedDim,
            activation: 'tanh',
            name: 'odor_embedding'
        }).apply(gruOut);     // [batch, embedDim]

        // 냄새 분류
        const odorOutput = tf.layers.dense({
            units: this.odorLabels.length,
            activation: 'sigmoid',
            name: 'odor_classifier'
        }).apply(odorEmbed);  // [batch, 20]

        this.chemLangModel = tf.model({ inputs: input, outputs: odorOutput });
        this.chemLangModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        // 임베딩 레이어 정보 출력
        const embLayer = this.chemLangModel.getLayer('smiles_embedding');
        const embParams = embLayer.countParams();
        console.log(`[ActiveLearner] ★ Real Embedding Layer: ${this.vocabSize} tokens × ${this.embedDim}d = ${embParams} learnable params`);
        console.log('[ActiveLearner] Embedding + GRU Chemical Language Model built');
    }

    // ==========================================================
    // MC Dropout 불확실성 모델 (Embedding + Dense)
    // ==========================================================
    buildUncertaintyModel() {
        const input = tf.input({ shape: [this.maxSeqLen] });

        // 동일한 Embedding 사용
        const embedded = tf.layers.embedding({
            inputDim: Math.max(this.vocabSize, 100),
            outputDim: 16,
            inputLength: this.maxSeqLen,
            maskZero: true,
            name: 'unc_embedding'
        }).apply(input);    // [batch, maxSeqLen, 16]

        // GlobalAveragePooling으로 시퀀스 축소
        const pooled = tf.layers.globalAveragePooling1d().apply(embedded); // [batch, 16]

        // MC Dropout 레이어들
        let x = tf.layers.dense({ units: 48, activation: 'relu' }).apply(pooled);
        x = tf.layers.dropout({ rate: 0.3 }).apply(x);          // MC Dropout
        x = tf.layers.dense({ units: 32, activation: 'relu' }).apply(x);
        x = tf.layers.dropout({ rate: 0.3 }).apply(x);          // MC Dropout
        const output = tf.layers.dense({
            units: this.odorLabels.length,
            activation: 'sigmoid'
        }).apply(x);

        this.uncertaintyModel = tf.model({ inputs: input, outputs: output });
        this.uncertaintyModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy'
        });

        console.log('[ActiveLearner] MC Dropout Uncertainty Model built (with Embedding)');
    }

    // ==========================================================
    // 학습
    // ==========================================================
    async train(moleculeDB, epochs = 50, onProgress = null) {
        const molecules = moleculeDB.getAll ? moleculeDB.getAll() : moleculeDB;
        if (!molecules || molecules.length === 0) return;

        // 어휘 구축
        this.buildVocab(molecules);

        if (!this.chemLangModel) this.buildChemLangModel();
        if (!this.uncertaintyModel) this.buildUncertaintyModel();

        // 학습 데이터 준비
        const inputs = [];
        const targets = [];

        for (const mol of molecules) {
            const seq = this.encodeSmiles(mol.smiles || '');
            const odorVec = this.odorLabels.map(l => (mol.odor_labels || []).includes(l) ? 1 : 0);
            inputs.push(seq);
            targets.push(odorVec);
        }

        // 데이터 증강: 토큰 수준 노이즈
        const augInputs = [...inputs];
        const augTargets = [...targets];
        for (let a = 0; a < 5; a++) {
            for (let i = 0; i < inputs.length; i++) {
                const noisy = inputs[i].map(v => Math.random() > 0.9 ? 0 : v);
                augInputs.push(noisy);
                augTargets.push(targets[i]);
            }
        }

        const xs = tf.tensor2d(augInputs, undefined, 'int32'); // ★ int32 — Embedding 입력
        const ys = tf.tensor2d(augTargets);

        // GRU 모델 학습
        console.log('[ActiveLearner] Training Embedding + GRU CLM...');
        await this.chemLangModel.fit(xs, ys, {
            epochs: Math.floor(epochs * 0.6),
            batchSize: 8,
            validationSplit: 0.15,
            callbacks: {
                onEpochEnd: (e, logs) => {
                    if (onProgress) onProgress(e + 1, `CLM: loss=${logs.loss.toFixed(4)}`);
                }
            }
        });

        // 불확실성 모델 학습
        console.log('[ActiveLearner] Training Uncertainty Model...');
        await this.uncertaintyModel.fit(xs, ys, {
            epochs: Math.floor(epochs * 0.4),
            batchSize: 8,
            callbacks: {
                onEpochEnd: (e, logs) => {
                    if (onProgress) onProgress(e + 1, `Uncertainty: loss=${logs.loss.toFixed(4)}`);
                }
            }
        });

        xs.dispose();
        ys.dispose();

        // 냄새 임베딩 추출
        this._extractOdorEmbeddings(molecules);

        this.trained = true;
        console.log('[ActiveLearner] Training complete (real Embedding + GRU)');
    }

    // 냄새 임베딩 추출 (GRU의 odor_embedding 레이어에서)
    _extractOdorEmbeddings(molecules) {
        const embeddingModel = tf.model({
            inputs: this.chemLangModel.input,
            outputs: this.chemLangModel.getLayer('odor_embedding').output
        });

        this.odorEmbeddings = {};

        for (const label of this.odorLabels) {
            const mols = molecules.filter(m => (m.odor_labels || []).includes(label));
            if (mols.length === 0) {
                this.odorEmbeddings[label] = new Array(this.embedDim).fill(0);
                continue;
            }

            const seqs = mols.map(m => this.encodeSmiles(m.smiles || ''));
            const input = tf.tensor2d(seqs, undefined, 'int32'); // ★ int32
            const embeddings = embeddingModel.predict(input);
            const mean = embeddings.mean(0).dataSync();
            this.odorEmbeddings[label] = Array.from(mean);

            input.dispose();
            embeddings.dispose();
        }

        embeddingModel.dispose();
        console.log('[ActiveLearner] Odor embeddings extracted from Embedding+GRU');
    }

    // ==========================================================
    // MC Dropout 불확실성 추정
    // ==========================================================
    estimateUncertainty(smiles, numSamples = 20) {
        if (!this.uncertaintyModel) return null;

        const seq = this.encodeSmiles(smiles);
        const input = tf.tensor2d([seq], undefined, 'int32'); // ★ int32

        const predictions = [];
        for (let i = 0; i < numSamples; i++) {
            const pred = this.uncertaintyModel.predict(input, { training: true });
            predictions.push(Array.from(pred.dataSync()));
            pred.dispose();
        }
        input.dispose();

        const mean = this.odorLabels.map((_, idx) => {
            const vals = predictions.map(p => p[idx]);
            return vals.reduce((a, b) => a + b, 0) / numSamples;
        });

        const variance = this.odorLabels.map((_, idx) => {
            const vals = predictions.map(p => p[idx]);
            const m = mean[idx];
            return vals.reduce((a, b) => a + (b - m) ** 2, 0) / numSamples;
        });

        const totalUncertainty = variance.reduce((a, b) => a + b, 0) / this.odorLabels.length;

        return {
            predictions: this.odorLabels.map((label, i) => ({
                label,
                mean: mean[i],
                variance: variance[i],
                confidence: 1 - Math.sqrt(variance[i])
            })).filter(p => p.mean > 0.2).sort((a, b) => b.mean - a.mean),
            totalUncertainty,
            isHighUncertainty: totalUncertainty > 0.1,
            numSamples
        };
    }

    // ==========================================================
    // Zero-shot 텍스트→냄새 (임베딩 코사인 유사도)
    // ==========================================================
    predictFromText(text) {
        if (!this.odorEmbeddings) return this._fallbackTextPrediction(text);

        const seedLabels = new Set();
        for (const [keyword, odors] of Object.entries(this.textOdorSeeds)) {
            if (text.includes(keyword)) {
                odors.forEach(o => seedLabels.add(o));
            }
        }

        if (seedLabels.size === 0) return this._fallbackTextPrediction(text);

        const queryVec = new Array(this.embedDim).fill(0);
        for (const label of seedLabels) {
            const emb = this.odorEmbeddings[label];
            if (emb) {
                for (let i = 0; i < this.embedDim; i++) queryVec[i] += emb[i];
            }
        }
        const norm = Math.sqrt(queryVec.reduce((s, v) => s + v * v, 0)) || 1;
        for (let i = 0; i < this.embedDim; i++) queryVec[i] /= norm;

        const results = [];
        for (const label of this.odorLabels) {
            const emb = this.odorEmbeddings[label];
            if (!emb) continue;

            const embNorm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0)) || 1;
            const dotProduct = queryVec.reduce((s, v, i) => s + v * emb[i], 0);
            const similarity = dotProduct / embNorm;
            const seedBonus = seedLabels.has(label) ? 0.2 : 0;

            results.push({
                label,
                score: Math.max(0, Math.min(1, (similarity + 1) / 2 + seedBonus)),
                cosineSimilarity: similarity,
                isDirectMatch: seedLabels.has(label)
            });
        }

        results.sort((a, b) => b.score - a.score);

        return {
            query: text,
            seedKeywords: Array.from(seedLabels),
            predictions: results.slice(0, 10),
            method: 'embedding_cosine_similarity',
            confidence: results[0]?.score || 0
        };
    }

    _fallbackTextPrediction(text) {
        const matched = [];
        for (const [keyword, odors] of Object.entries(this.textOdorSeeds)) {
            if (text.includes(keyword)) {
                odors.forEach(o => matched.push(o));
            }
        }
        const counts = {};
        matched.forEach(o => { counts[o] = (counts[o] || 0) + 1; });

        const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
        const predictions = Object.entries(counts)
            .map(([label, count]) => ({ label, score: count / total, cosineSimilarity: 0, isDirectMatch: true }))
            .sort((a, b) => b.score - a.score);

        return {
            query: text,
            seedKeywords: Object.keys(counts),
            predictions,
            method: 'keyword_fallback',
            confidence: predictions[0]?.score || 0
        };
    }

    // ==========================================================
    // 능동 학습: 불확실한 분자 추천
    // ==========================================================
    suggestNextSamples(candidates, topK = 5) {
        if (!this.trained) return candidates.slice(0, topK);

        const scored = candidates.map(mol => {
            const uncertainty = this.estimateUncertainty(mol.smiles || '');
            return {
                ...mol,
                uncertainty: uncertainty?.totalUncertainty || 0,
                isHighUncertainty: uncertainty?.isHighUncertainty || false
            };
        });

        scored.sort((a, b) => b.uncertainty - a.uncertainty);
        return scored.slice(0, topK);
    }

    // ==========================================================
    // 기존 API 호환
    // ==========================================================
    predictOdorFromSmiles(smiles) {
        if (!this.chemLangModel || !this.trained) return null;

        const seq = this.encodeSmiles(smiles);
        const input = tf.tensor2d([seq], undefined, 'int32'); // ★ int32
        const output = this.chemLangModel.predict(input);
        const scores = Array.from(output.dataSync());
        input.dispose();
        output.dispose();

        return this.odorLabels
            .map((label, i) => ({ label, score: scores[i] }))
            .filter(p => p.score > 0.2)
            .sort((a, b) => b.score - a.score);
    }
}

export default ActiveLearner;
