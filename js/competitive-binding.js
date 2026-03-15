// competitive-binding.js — Pillar 2: 실제 Scaled Dot-Product Multi-Head Attention
// =================================================================================
// 핵심 변경: Dense 레이어 → 진짜 Attention 메커니즘
//   Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
//   Multi-Head: 4 헤드 → concat → linear projection
// =================================================================================

class CompetitiveBinding {
    constructor() {
        // 아키텍처 파라미터
        this.moleculeDim = 32;     // 분자 임베딩 차원
        this.numHeads = 4;
        this.headDim = 8;          // d_k = moleculeDim / numHeads
        this.numReceptors = 8;

        // 학습 가능 가중치
        this.attentionWeights = null;
        this.receptorWeights = null;
        this.projectionModel = null;
        this.trained = false;

        // 가상 수용체 정의 (특화 영역)
        this.receptorProfiles = [
            { name: 'OR1: 플로럴', sensitivity: ['floral', 'rose', 'jasmine', 'violet'] },
            { name: 'OR2: 시트러스', sensitivity: ['citrus', 'lemon', 'orange', 'bergamot'] },
            { name: 'OR3: 우디', sensitivity: ['woody', 'cedar', 'sandalwood', 'vetiver'] },
            { name: 'OR4: 스파이시', sensitivity: ['spicy', 'clove', 'cinnamon', 'pepper'] },
            { name: 'OR5: 스위트', sensitivity: ['sweet', 'vanilla', 'caramel', 'honey'] },
            { name: 'OR6: 프레시', sensitivity: ['fresh', 'clean', 'aquatic', 'ozonic'] },
            { name: 'OR7: 그린', sensitivity: ['green', 'leafy', 'herbal', 'grassy'] },
            { name: 'OR8: 머스크', sensitivity: ['musk', 'amber', 'warm', 'powdery'] }
        ];
    }

    // ==========================================================
    // Attention 가중치 초기화
    // ==========================================================
    initAttentionWeights() {
        const d = this.moleculeDim;
        const h = this.numHeads;
        const dk = this.headDim;

        this.attentionWeights = {};

        // 각 헤드의 Q, K, V 프로젝션
        for (let i = 0; i < h; i++) {
            this.attentionWeights[`W_Q_${i}`] = tf.variable(
                tf.randomNormal([d, dk]).mul(tf.scalar(Math.sqrt(2 / d)))
            );
            this.attentionWeights[`W_K_${i}`] = tf.variable(
                tf.randomNormal([d, dk]).mul(tf.scalar(Math.sqrt(2 / d)))
            );
            this.attentionWeights[`W_V_${i}`] = tf.variable(
                tf.randomNormal([d, dk]).mul(tf.scalar(Math.sqrt(2 / d)))
            );
        }

        // 멀티헤드 출력 프로젝션: [numHeads * dk] → [d]
        this.attentionWeights['W_O'] = tf.variable(
            tf.randomNormal([h * dk, d]).mul(tf.scalar(Math.sqrt(2 / (h * dk))))
        );
        this.attentionWeights['b_O'] = tf.variable(tf.zeros([d]));

        // 수용체별 쿼리 벡터 (학습 가능)
        this.receptorWeights = [];
        for (let r = 0; r < this.numReceptors; r++) {
            this.receptorWeights.push(
                tf.variable(tf.randomNormal([1, d]).mul(tf.scalar(0.1)))
            );
        }

        console.log(`[CompetitiveBinding] Attention weights initialized (${h} heads, d_k=${dk})`);
    }

    // ==========================================================
    // Scaled Dot-Product Attention
    // ==========================================================
    scaledDotProductAttention(Q, K, V) {
        return tf.tidy(() => {
            const dk = Q.shape[Q.shape.length - 1];
            const scale = tf.scalar(Math.sqrt(dk));

            // QK^T / √d_k
            const scores = tf.matMul(Q, K, false, true).div(scale); // [N, N] or [1, N]

            // Softmax
            const weights = tf.softmax(scores, -1);

            // Weighted sum of V
            const output = tf.matMul(weights, V); // [N, dk] or [1, dk]

            return { output, weights };
        });
    }

    // ==========================================================
    // Multi-Head Attention
    // ==========================================================
    multiHeadAttention(X, queryOverride = null) {
        return tf.tidy(() => {
            const headOutputs = [];
            const allWeights = [];

            for (let i = 0; i < this.numHeads; i++) {
                const W_Q = this.attentionWeights[`W_Q_${i}`];
                const W_K = this.attentionWeights[`W_K_${i}`];
                const W_V = this.attentionWeights[`W_V_${i}`];

                // Q, K, V 프로젝션
                const Q = queryOverride
                    ? tf.matMul(queryOverride, W_Q) // [1, dk]
                    : tf.matMul(X, W_Q); // [N, dk]
                const K = tf.matMul(X, W_K); // [N, dk]
                const V = tf.matMul(X, W_V); // [N, dk]

                const { output, weights } = this.scaledDotProductAttention(Q, K, V);
                headOutputs.push(output);
                allWeights.push(weights);
            }

            // 헤드 Concat + 출력 프로젝션
            const concatenated = tf.concat(headOutputs, -1); // [N, numHeads*dk] or [1, ...]
            const W_O = this.attentionWeights['W_O'];
            const b_O = this.attentionWeights['b_O'];
            const projected = tf.matMul(concatenated, W_O).add(b_O); // [N, d]

            return { output: projected, attentionWeights: allWeights };
        });
    }

    // ==========================================================
    // 분자 임베딩 (입력 변환)
    // ==========================================================
    buildModel() {
        this.initAttentionWeights();

        // 분자 벡터 → 임베딩 공간 프로젝션
        this.projectionModel = tf.sequential();
        this.projectionModel.add(tf.layers.dense({
            inputShape: [29],  // molecular-engine fullMoleculeVector
            units: this.moleculeDim,
            activation: 'relu'
        }));
        this.projectionModel.add(tf.layers.dense({
            units: this.moleculeDim,
            activation: 'tanh'  // [-1, 1] 정규화
        }));

        // 결합 예측 모델 (attention 결과 → 냄새 강도)
        this.bindingModel = tf.sequential();
        this.bindingModel.add(tf.layers.dense({
            inputShape: [this.moleculeDim],
            units: 32,
            activation: 'relu'
        }));
        this.bindingModel.add(tf.layers.dense({
            units: this.numReceptors,
            activation: 'sigmoid'
        }));
        this.bindingModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        console.log('[CompetitiveBinding] Models built');
    }

    async train(moleculeDB, epochs = 30, onProgress = null) {
        if (!this.attentionWeights) this.buildModel();

        const molecules = moleculeDB.getAll ? moleculeDB.getAll() : moleculeDB;
        if (molecules.length === 0) return;

        // 학습 데이터: 개별 분자 → 수용체 활성화 패턴
        const inputs = [];
        const targets = [];

        for (const mol of molecules) {
            const vec = moleculeDB.fullMoleculeVector
                ? moleculeDB.fullMoleculeVector(mol)
                : new Array(29).fill(0);
            inputs.push(vec);

            // 수용체 활성화 할당 (odor_labels와 receptor sensitivity 매칭)
            const activation = this.receptorProfiles.map(r => {
                const labels = mol.odor_labels || [];
                const match = r.sensitivity.filter(s => labels.includes(s)).length;
                return Math.min(1.0, match / r.sensitivity.length + Math.random() * 0.05);
            });
            targets.push(activation);
        }

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(targets);

        await this.bindingModel.fit(xs, ys, {
            epochs,
            batchSize: 8,
            callbacks: {
                onEpochEnd: (e, logs) => {
                    if (onProgress) onProgress(e + 1, epochs, logs.loss);
                }
            }
        });

        xs.dispose();
        ys.dispose();
        this.trained = true;
        console.log('[CompetitiveBinding] Training complete');
    }

    // ==========================================================
    // 혼합물 시뮬레이션 (진짜 어텐션 기반)
    // ==========================================================
    simulateBinding(molecules, moleculeDB) {
        if (!this.attentionWeights) this.buildModel();

        return tf.tidy(() => {
            const N = molecules.length;
            if (N === 0) return null;

            // 1. 분자 벡터 → 임베딩
            const molVectors = molecules.map(mol =>
                moleculeDB?.fullMoleculeVector
                    ? moleculeDB.fullMoleculeVector(mol)
                    : new Array(29).fill(Math.random() * 0.5)
            );
            const X_raw = tf.tensor2d(molVectors); // [N, 29]
            const X = this.projectionModel.predict(X_raw); // [N, d]

            // 2. Self-Attention: 분자 간 상호작용 학습
            const { output: attended, attentionWeights: selfAttnWeights } = this.multiHeadAttention(X);

            // 3. 수용체별 Cross-Attention: 각 수용체가 분자 혼합물에 주의
            const receptorResults = [];
            for (let r = 0; r < this.numReceptors; r++) {
                const receptorQuery = this.receptorWeights[r]; // [1, d]

                const { output: rOutput, attentionWeights: rWeights } =
                    this.multiHeadAttention(attended, receptorQuery);

                // 수용체 활성화 강도
                const activation = rOutput.abs().mean().dataSync()[0];

                // 각 분자에 대한 어텐션 점수 (첫 번째 헤드)
                const moleculeBindings = rWeights[0].dataSync();

                receptorResults.push({
                    receptorName: this.receptorProfiles[r].name,
                    totalActivation: Math.min(1.0, activation),
                    moleculeBindings: Array.from(moleculeBindings).slice(0, N),
                    dominantMolecule: Array.from(moleculeBindings).slice(0, N)
                        .indexOf(Math.max(...Array.from(moleculeBindings).slice(0, N)))
                });
            }

            // 4. 상호작용 분석 (self-attention 가중치에서)
            const interactions = this._analyzeInteractions(selfAttnWeights, molecules, N);

            // 5. 결과 요약
            const summary = this._generateSummary(receptorResults, interactions, molecules);

            return {
                receptorResults,
                interactions,
                summary,
                numMolecules: N,
                selfAttentionWeights: selfAttnWeights[0].dataSync()
            };
        });
    }

    // 어텐션 가중치 기반 상호작용 분석
    _analyzeInteractions(attentionWeights, molecules, N) {
        const interactions = [];
        const headWeights = attentionWeights[0].dataSync();

        for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
                // 양방향 어텐션 평균
                const attnIJ = headWeights[i * N + j] || 0;
                const attnJI = headWeights[j * N + i] || 0;
                const mutualAttention = (attnIJ + attnJI) / 2;

                // 상호작용 유형 판단
                let type = 'neutral';
                if (mutualAttention > 0.3) {
                    type = 'synergy'; // 서로 강하게 주의 → 시너지
                } else if (mutualAttention < 0.1 && N > 2) {
                    type = 'masking'; // 매우 낮은 관심 → 마스킹
                }

                interactions.push({
                    moleculeA: molecules[i]?.name || `mol_${i}`,
                    moleculeB: molecules[j]?.name || `mol_${j}`,
                    mutualAttention,
                    type,
                    attnAtoB: attnIJ,
                    attnBtoA: attnJI
                });
            }
        }

        return interactions;
    }

    // 결과 요약 생성
    _generateSummary(receptorResults, interactions, molecules) {
        const summary = [];

        // 가장 활성화된 수용체
        const maxReceptor = receptorResults.reduce((max, r) =>
            r.totalActivation > max.totalActivation ? r : max, receptorResults[0]);
        summary.push(`가장 강한 수용체 반응: ${maxReceptor.receptorName} (${(maxReceptor.totalActivation * 100).toFixed(0)}%)`);

        // 지배적 분자
        if (maxReceptor.dominantMolecule < molecules.length) {
            const dom = molecules[maxReceptor.dominantMolecule];
            summary.push(`지배적 분자: ${dom?.name || 'unknown'}`);
        }

        // 시너지 상호작용
        const synergies = interactions.filter(i => i.type === 'synergy');
        if (synergies.length > 0) {
            const best = synergies[0];
            summary.push(`✨ 시너지: ${best.moleculeA} ↔ ${best.moleculeB} (상호 어텐션 ${(best.mutualAttention * 100).toFixed(0)}%)`);
        }

        // 마스킹 상호작용
        const maskings = interactions.filter(i => i.type === 'masking');
        if (maskings.length > 0) {
            summary.push(`🚫 마스킹 효과: ${maskings.length}건 감지`);
        }

        return summary;
    }

    // ==========================================================
    // 혼합물 냄새 벡터 계산 (진짜 어텐션 가중치 기반)
    // ==========================================================
    computeMixtureOdor(molecules, moleculeDB) {
        if (!this.trained || !this.bindingModel) return null;

        return tf.tidy(() => {
            const N = molecules.length;
            const molVectors = molecules.map(mol =>
                moleculeDB?.fullMoleculeVector
                    ? moleculeDB.fullMoleculeVector(mol)
                    : new Array(29).fill(0)
            );
            const X_raw = tf.tensor2d(molVectors);
            const X = this.projectionModel.predict(X_raw);

            // Self-attention → 가중된 분자 표현
            const { output: attended } = this.multiHeadAttention(X);

            // 평균 풀링 → 혼합물 벡터
            const mixtureVec = attended.mean(0).reshape([1, this.moleculeDim]);

            // 수용체 활성화 예측
            const activation = this.bindingModel.predict(mixtureVec);
            return Array.from(activation.dataSync());
        });
    }
}

export default CompetitiveBinding;
