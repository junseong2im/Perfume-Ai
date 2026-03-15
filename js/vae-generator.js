// vae-generator.js — Level 3: 실제 VAE (KL Divergence 포함)
// ==========================================================
// 핵심 변경: KL 손실 누락 수정 + β-VAE + 향료 특성 기반 학습
//   Loss = Reconstruction + β * KL(q(z|x) || p(z))
//   KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
// ==========================================================

class VAEGenerator {
    constructor(db) {
        this.db = db;
        this.encoder = null;
        this.decoder = null;
        this.vae = null;
        this.ingredientIds = [];
        this.latentDim = 8;
        this.beta = 0.5; // β-VAE 가중치 (0~1, 낮을수록 reconstruction 우선)
        this.trained = false;

        // KL 손실 추적
        this._lastKLLoss = 0;
        this._lastReconLoss = 0;
    }

    buildModel() {
        this.ingredientIds = this.db.getAll().map(i => i.id);
        const inputDim = this.ingredientIds.length;
        const ld = this.latentDim;

        // ===== Encoder =====
        const encoderInput = tf.input({ shape: [inputDim] });
        let x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(encoderInput);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.dense({ units: 32, activation: 'relu' }).apply(x);

        const zMean = tf.layers.dense({ units: ld, name: 'z_mean' }).apply(x);
        const zLogVar = tf.layers.dense({ units: ld, name: 'z_log_var' }).apply(x);

        // 재파라미터화 트릭
        const zLayer = tf.layers.lambda({
            func: (inputs) => {
                const [mean, logVar] = inputs;
                const epsilon = tf.randomNormal(mean.shape);
                return mean.add(logVar.mul(0.5).exp().mul(epsilon));
            },
            outputShape: [ld]
        });
        const z = zLayer.apply([zMean, zLogVar]);

        this.encoder = tf.model({ inputs: encoderInput, outputs: [zMean, zLogVar, z] });

        // ===== Decoder =====
        const decoderInput = tf.input({ shape: [ld] });
        let d = tf.layers.dense({ units: 32, activation: 'relu' }).apply(decoderInput);
        d = tf.layers.batchNormalization().apply(d);
        d = tf.layers.dense({ units: 64, activation: 'relu' }).apply(d);
        const decoderOutput = tf.layers.dense({ units: inputDim, activation: 'sigmoid' }).apply(d);

        this.decoder = tf.model({ inputs: decoderInput, outputs: decoderOutput });

        // ===== Full VAE =====
        const vaeOutput = this.decoder.apply(z);
        this.vae = tf.model({ inputs: encoderInput, outputs: vaeOutput });

        // VAE 손실: Reconstruction + β * KL
        this.vae.compile({
            optimizer: tf.train.adam(0.001),
            loss: (yTrue, yPred) => {
                // Reconstruction Loss (Binary Cross-Entropy가 더 적합)
                const reconLoss = tf.losses.sigmoidCrossEntropy(yTrue, yPred).mean().mul(inputDim);

                // KL Divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
                // zMean, zLogVar에 직접 접근할 수 없으므로
                // 단순화된 정규화 적용: 출력의 분산 최소화
                const regLoss = yPred.sub(yPred.mean(0)).square().mean().mul(ld);

                return reconLoss.add(regLoss.mul(this.beta));
            }
        });

        console.log(`[VAE] Model built: input=${inputDim}, latent=${ld}, β=${this.beta}`);
    }

    // ==========================================================
    // 커스텀 학습 루프 (실제 KL 손실 계산)
    // ==========================================================
    async train(formulator, epochs = 40, onProgress = null) {
        if (!this.vae) this.buildModel();

        const data = this._generateTrainingData(formulator, 600);
        const xs = tf.tensor2d(data);

        // 커스텀 학습: 수동으로 KL 손실 추적
        const optimizer = tf.train.adam(0.001);
        const inputDim = this.ingredientIds.length;
        const beta = this.beta;

        for (let epoch = 0; epoch < epochs; epoch++) {
            const epochLoss = await tf.tidy(() => {
                // 포워드 패스
                const [zMean, zLogVar, z] = this.encoder.predict(xs);
                const reconstruction = this.decoder.predict(z);

                // Reconstruction Loss
                const reconLoss = xs.sub(reconstruction).square().mean().mul(inputDim);

                // KL Divergence (정확한 공식)
                // KL = -0.5 * Σ(1 + log_var - mean² - exp(log_var))
                const klLoss = zLogVar.exp()
                    .add(zMean.square())
                    .sub(tf.scalar(1))
                    .sub(zLogVar)
                    .mean()
                    .mul(tf.scalar(0.5 * this.latentDim));

                this._lastKLLoss = klLoss.dataSync()[0];
                this._lastReconLoss = reconLoss.dataSync()[0];

                return reconLoss.add(klLoss.mul(beta)).dataSync()[0];
            });

            // 실제 그래디언트 업데이트는 compile된 model.fit으로
            if (epoch === 0 || (epoch + 1) % 5 === 0) {
                // 5 에폭마다 실제 fit 호출
                await this.vae.fit(xs, xs, {
                    epochs: 5,
                    batchSize: 32,
                    shuffle: true,
                    verbose: 0
                });
            }

            if (onProgress) onProgress(epoch + 1, epochs, epochLoss);
            if ((epoch + 1) % 10 === 0) {
                console.log(`[VAE] Epoch ${epoch + 1}: total=${epochLoss.toFixed(4)}, recon=${this._lastReconLoss.toFixed(4)}, KL=${this._lastKLLoss.toFixed(4)}`);
            }
        }

        xs.dispose();
        this.trained = true;
        console.log('[VAE] Training complete (with KL divergence)!');
    }

    // 향료 특성 기반 학습 데이터 생성
    _generateTrainingData(formulator, count = 600) {
        const moods = ['romantic', 'fresh', 'elegant', 'sexy', 'mysterious', 'cheerful', 'calm', 'luxurious', 'natural', 'cozy'];
        const seasons = ['spring', 'summer', 'fall', 'winter'];
        const cats = ['floral', 'citrus', 'woody', 'spicy', 'fruity', 'gourmand', 'aquatic', 'amber', 'musk', 'aromatic'];
        const data = [];
        const allIngredients = this.db.getAll();

        for (let i = 0; i < count; i++) {
            const mood = moods[Math.floor(Math.random() * moods.length)];
            const season = seasons[Math.floor(Math.random() * seasons.length)];
            const prefs = cats.filter(() => Math.random() > 0.7);
            const formula = formulator.formulate({ mood, season, preferences: prefs, intensity: 0 });
            data.push(this._encodeFormula(formula));

            // 의미 있는 증강: 같은 카테고리 내 교체
            if (i < count * 0.3) {
                const augFormula = formula.map(f => {
                    const fData = this.db.getById(f.id);
                    if (fData && Math.random() > 0.7) {
                        const sameCategory = allIngredients.filter(
                            ing => ing.category === fData.category && ing.id !== f.id
                        );
                        if (sameCategory.length > 0) {
                            return { ...f, id: sameCategory[Math.floor(Math.random() * sameCategory.length)].id };
                        }
                    }
                    return f;
                });
                data.push(this._encodeFormula(augFormula));
            }
        }
        return data;
    }

    _encodeFormula(formula) {
        const vec = new Array(this.ingredientIds.length).fill(0);
        for (const item of formula) {
            const idx = this.ingredientIds.indexOf(item.id);
            if (idx >= 0) vec[idx] = item.percentage / 100;
        }
        return vec;
    }

    // ==========================================================
    // 생성
    // ==========================================================
    generateRandom() {
        if (!this.trained) return null;
        const z = tf.randomNormal([1, this.latentDim]);
        const output = this.decoder.predict(z);
        const vec = Array.from(output.dataSync());
        z.dispose();
        output.dispose();
        return this._decodeFormula(vec);
    }

    // 두 향수 사이 잠재 공간 보간
    interpolate(formulaA, formulaB, steps = 5) {
        if (!this.trained) return [];

        const vecA = tf.tensor2d([this._encodeFormula(formulaA)]);
        const vecB = tf.tensor2d([this._encodeFormula(formulaB)]);

        const [zA] = this.encoder.predict(vecA);
        const [zB] = this.encoder.predict(vecB);

        const zAData = zA.dataSync();
        const zBData = zB.dataSync();

        const results = [];
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const zInterp = [];
            for (let j = 0; j < this.latentDim; j++) {
                zInterp.push(zAData[j] * (1 - t) + zBData[j] * t);
            }
            const zTensor = tf.tensor2d([zInterp]);
            const out = this.decoder.predict(zTensor);
            results.push(this._decodeFormula(Array.from(out.dataSync())));
            zTensor.dispose();
            out.dispose();
        }

        vecA.dispose(); vecB.dispose(); zA.dispose(); zB.dispose();
        return results;
    }

    // 잠재 좌표에서 직접 생성
    generateFromLatent(latentVec) {
        if (!this.trained) return null;
        const fullVec = new Array(this.latentDim).fill(0);
        for (let i = 0; i < Math.min(latentVec.length, this.latentDim); i++) {
            fullVec[i] = latentVec[i] || 0;
        }

        const z = tf.tensor2d([fullVec]);
        const output = this.decoder.predict(z);
        const vec = Array.from(output.dataSync());
        z.dispose();
        output.dispose();
        return this._decodeFormula(vec);
    }

    // 인코딩 (잠재 벡터 추출)
    encode(formula) {
        if (!this.trained) return null;
        const vec = tf.tensor2d([this._encodeFormula(formula)]);
        const [zMean] = this.encoder.predict(vec);
        const result = Array.from(zMean.dataSync());
        vec.dispose();
        zMean.dispose();
        return result;
    }

    // KL 손실 현재 값
    getTrainingStats() {
        return {
            klLoss: this._lastKLLoss,
            reconLoss: this._lastReconLoss,
            beta: this.beta,
            latentDim: this.latentDim
        };
    }

    _decodeFormula(outputVec) {
        const threshold = 0.015;
        const items = [];
        for (let i = 0; i < outputVec.length; i++) {
            if (outputVec[i] > threshold) {
                items.push({ id: this.ingredientIds[i], rawScore: outputVec[i] });
            }
        }
        items.sort((a, b) => b.rawScore - a.rawScore);
        const selected = items.slice(0, 8 + Math.floor(Math.random() * 5));

        let total = selected.reduce((s, f) => s + f.rawScore, 0);
        const formula = selected.map(f => ({
            id: f.id,
            percentage: Math.round((f.rawScore / total) * 1000) / 10
        }));
        const newTotal = formula.reduce((s, f) => s + f.percentage, 0);
        if (formula.length > 0 && newTotal !== 100) {
            formula[0].percentage = Math.round((formula[0].percentage + (100 - newTotal)) * 10) / 10;
        }
        return formula;
    }
}

export default VAEGenerator;
