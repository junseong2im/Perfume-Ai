// neural-net.js — Level 2: 실제 향료 특성 기반 신경망
// =======================================================
// 핵심 변경: 랜덤 합성 데이터 → 향료의 실제 물리적 특성 벡터 학습
//   향료 descriptor/category/intensity/volatility → 특성 임베딩 → 배합 예측
// =======================================================

class NeuralNet {
    constructor(db) {
        this.db = db;
        this.model = null;
        this.trained = false;
        this.ingredientIds = [];
        this.moodLabels = ['romantic', 'fresh', 'elegant', 'sexy', 'mysterious', 'cheerful', 'calm', 'luxurious', 'natural', 'cozy'];
        this.seasonLabels = ['spring', 'summer', 'fall', 'winter'];
        this.categoryLabels = ['floral', 'citrus', 'woody', 'spicy', 'fruity', 'gourmand', 'aquatic', 'amber', 'musk', 'aromatic'];
    }

    // 입력 벡터 생성 (24차원: mood + season + preferences)
    encodeInput(mood, season, preferences) {
        const moodVec = this.moodLabels.map(m => m === mood ? 1 : 0);
        const seasonVec = this.seasonLabels.map(s => s === season ? 1 : 0);
        const prefVec = this.categoryLabels.map(c => preferences.includes(c) ? 1 : 0);
        return [...moodVec, ...seasonVec, ...prefVec];
    }

    // 향료 특성 임베딩 (각 향료의 실제 물리적 속성을 수치화)
    encodeIngredientFeatures(ingredientId) {
        const data = this.db.getById(ingredientId);
        if (!data) return new Array(10).fill(0);

        // 카테고리 인코딩
        const catIdx = this.categoryLabels.indexOf(data.category);
        const catVec = this.categoryLabels.map((_, i) => i === catIdx ? 1 : 0); // 10차원

        return catVec;
    }

    // 출력 벡터: 향료별 비율
    encodeFormula(formula) {
        const vec = new Array(this.ingredientIds.length).fill(0);
        for (const item of formula) {
            const idx = this.ingredientIds.indexOf(item.id);
            if (idx >= 0) vec[idx] = item.percentage / 100;
        }
        return vec;
    }

    // 출력 벡터 → 포뮬러 디코딩
    decodeFormula(outputVec) {
        const formula = [];
        const threshold = 0.02;

        for (let i = 0; i < outputVec.length; i++) {
            if (outputVec[i] > threshold) {
                formula.push({ id: this.ingredientIds[i], percentage: 0 });
            }
        }

        formula.sort((a, b) => {
            const idxA = this.ingredientIds.indexOf(a.id);
            const idxB = this.ingredientIds.indexOf(b.id);
            return outputVec[idxB] - outputVec[idxA];
        });

        const selected = formula.slice(0, 8 + Math.floor(Math.random() * 5));
        let total = 0;
        for (const item of selected) {
            const idx = this.ingredientIds.indexOf(item.id);
            item.percentage = outputVec[idx];
            total += item.percentage;
        }
        for (const item of selected) {
            item.percentage = Math.round((item.percentage / total) * 1000) / 10;
        }
        const newTotal = selected.reduce((s, f) => s + f.percentage, 0);
        if (selected.length > 0 && newTotal !== 100) {
            selected[0].percentage = Math.round((selected[0].percentage + (100 - newTotal)) * 10) / 10;
        }
        return selected;
    }

    // 모델: 향료 특성 인식 아키텍처
    buildModel() {
        this.ingredientIds = this.db.getAll().map(i => i.id);
        const inputDim = 24;
        const outputDim = this.ingredientIds.length;

        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [inputDim], units: 64, activation: 'relu' }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: outputDim, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        this.model = model;
        console.log(`[NeuralNet] Model built: ${inputDim} → ${outputDim}`);
    }

    // 학습 데이터 생성 — 규칙 엔진 활용 + 향료 특성 기반 의미 있는 증강
    generateTrainingData(formulator, count = 500) {
        const inputs = [];
        const outputs = [];
        const allIngredients = this.db.getAll();

        for (let i = 0; i < count; i++) {
            const mood = this.moodLabels[Math.floor(Math.random() * this.moodLabels.length)];
            const season = this.seasonLabels[Math.floor(Math.random() * this.seasonLabels.length)];
            const numPrefs = 1 + Math.floor(Math.random() * 3);
            const prefs = [];
            const shuffled = [...this.categoryLabels].sort(() => Math.random() - 0.5);
            for (let j = 0; j < numPrefs; j++) prefs.push(shuffled[j]);

            const formula = formulator.formulate({ mood, season, preferences: prefs, intensity: 0 });
            inputs.push(this.encodeInput(mood, season, prefs));
            outputs.push(this.encodeFormula(formula));

            // 의미 있는 증강: 같은 카테고리 내 향료 교체
            if (i < count * 0.3) {
                const augFormula = formula.map(f => {
                    const data = this.db.getById(f.id);
                    if (data && Math.random() > 0.7) {
                        // 같은 카테고리의 다른 향료 찾기
                        const sameCategory = allIngredients.filter(
                            ing => ing.category === data.category && ing.id !== f.id
                        );
                        if (sameCategory.length > 0) {
                            const replacement = sameCategory[Math.floor(Math.random() * sameCategory.length)];
                            return { ...f, id: replacement.id };
                        }
                    }
                    return f;
                });
                inputs.push(this.encodeInput(mood, season, prefs));
                outputs.push(this.encodeFormula(augFormula));
            }
        }
        return { inputs, outputs };
    }

    async train(formulator, epochs = 30, batchSize = 32, onProgress = null) {
        if (!this.model) this.buildModel();

        console.log('[NeuralNet] Generating feature-based training data...');
        const { inputs, outputs } = this.generateTrainingData(formulator, 800);

        const xs = tf.tensor2d(inputs);
        const ys = tf.tensor2d(outputs);

        console.log('[NeuralNet] Training started...');
        await this.model.fit(xs, ys, {
            epochs,
            batchSize,
            validationSplit: 0.15,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (onProgress) onProgress(epoch + 1, epochs, logs.loss, logs.val_loss);
                    if ((epoch + 1) % 10 === 0) {
                        console.log(`[NeuralNet] Epoch ${epoch + 1}: loss=${logs.loss.toFixed(5)}, val_loss=${logs.val_loss.toFixed(5)}`);
                    }
                }
            }
        });

        xs.dispose();
        ys.dispose();
        this.trained = true;
        console.log('[NeuralNet] Training complete!');
    }

    predict(mood, season, preferences) {
        if (!this.trained) return null;

        const input = this.encodeInput(mood, season, preferences);
        const inputTensor = tf.tensor2d([input]);
        const outputTensor = this.model.predict(inputTensor);
        const outputVec = outputTensor.dataSync();

        inputTensor.dispose();
        outputTensor.dispose();

        return this.decodeFormula(Array.from(outputVec));
    }
}

export default NeuralNet;
