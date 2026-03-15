// sensor-engine.js — Pillar 3: Conv1D 초해상도 + 바이오센서 시뮬레이션
// =====================================================================
// 핵심 변경: Dense 오토인코더 → Conv1D 기반 Super-Resolution
//   Low-res [20] → Conv1D → UpSampling → Conv1D → High-res [200]
// =====================================================================

class SensorEngine {
    constructor() {
        this.numChannels = 4;
        this.lowRes = 20;
        this.highRes = 200;

        // 초해상도 모델 (Conv1D)
        this.superResModel = null;
        this.trained = false;

        // 가상 센서 프로파일
        this.sensorProfiles = [
            { name: 'MOX-1', type: '금속산화물', sensitivity: ['citrus', 'fresh', 'green'], peakTime: 5, decayRate: 0.3 },
            { name: 'SAW-2', type: 'SAW', sensitivity: ['floral', 'rose', 'jasmine'], peakTime: 8, decayRate: 0.15 },
            { name: 'QCM-3', type: 'QCM 크리스탈', sensitivity: ['woody', 'amber', 'musk'], peakTime: 12, decayRate: 0.08 },
            { name: 'BIO-4', type: '바이오센서', sensitivity: ['sweet', 'vanilla', 'fruity'], peakTime: 6, decayRate: 0.2 }
        ];
    }

    // ==========================================================
    // Conv1D 기반 Super-Resolution 모델
    // ==========================================================
    buildSuperResModel() {
        // 입력: [N, lowRes, 1] → 출력: [N, highRes, 1]
        const input = tf.input({ shape: [this.lowRes, 1] });

        // Conv1D 인코더
        let x = tf.layers.conv1d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(input);
        x = tf.layers.batchNormalization().apply(x);
        x = tf.layers.conv1d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }).apply(x);

        // 업샘플링: 20 → 200 (10배)
        // TF.js: upSampling1d로 단계적 업샘플링
        x = tf.layers.upSampling1d({ size: 2 }).apply(x);  // 20 → 40
        x = tf.layers.conv1d({ filters: 32, kernelSize: 5, padding: 'same', activation: 'relu' }).apply(x);
        x = tf.layers.batchNormalization().apply(x);

        x = tf.layers.upSampling1d({ size: 5 }).apply(x);  // 40 → 200
        x = tf.layers.conv1d({ filters: 16, kernelSize: 5, padding: 'same', activation: 'relu' }).apply(x);

        // 잔차 연결을 위한 Skip Connection (선형 업샘플링)
        let skip = tf.layers.upSampling1d({ size: 10 }).apply(input); // 20 → 200
        skip = tf.layers.conv1d({ filters: 16, kernelSize: 1, padding: 'same' }).apply(skip);

        // 잔차 합산
        x = tf.layers.add().apply([x, skip]);
        x = tf.layers.activation({ activation: 'relu' }).apply(x);

        // 최종 출력
        const output = tf.layers.conv1d({ filters: 1, kernelSize: 3, padding: 'same', activation: 'sigmoid' }).apply(x);

        this.superResModel = tf.model({ inputs: input, outputs: output });
        this.superResModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        console.log('[SensorEngine] Conv1D Super-Resolution model built');
    }

    // ==========================================================
    // 센서 응답 곡선 시뮬레이션 (물리 기반)
    // ==========================================================
    simulateSensorResponse(molecule, channelIdx = 0) {
        const profile = this.sensorProfiles[channelIdx % this.numChannels];
        const odorLabels = molecule.odor_labels || [];

        // 기본 감도 계산 (센서별 민감도와 분자 냄새의 교집합)
        const matchCount = profile.sensitivity.filter(s => odorLabels.includes(s)).length;
        const baseSensitivity = matchCount / Math.max(profile.sensitivity.length, 1);
        const amplitude = 0.3 + baseSensitivity * 0.7; // 0.3 ~ 1.0

        // 물리 기반 응답 곡선: 1차 지수 상승/하강 모델
        // R(t) = A * (1 - exp(-t/τ_rise)) * exp(-t/τ_decay) + noise
        const highRes = [];
        const tRise = profile.peakTime;
        const tDecay = profile.decayRate;
        const noiseLevel = 0.02;

        for (let i = 0; i < this.highRes; i++) {
            const t = i / this.highRes * 30; // 0~30초
            const rise = 1 - Math.exp(-t / tRise);
            const decay = Math.exp(-tDecay * Math.max(0, t - tRise * 2));
            const signal = amplitude * rise * decay;
            const noise = (Math.random() - 0.5) * noiseLevel;
            highRes.push(Math.max(0, Math.min(1, signal + noise)));
        }

        // 저해상도 버전 (다운샘플링)
        const lowRes = [];
        const stride = this.highRes / this.lowRes;
        for (let i = 0; i < this.lowRes; i++) {
            const start = Math.floor(i * stride);
            const end = Math.floor((i + 1) * stride);
            let sum = 0;
            for (let j = start; j < end; j++) sum += highRes[j];
            lowRes.push(sum / (end - start));
        }

        return { highRes, lowRes, sensor: profile, amplitude, sensitivity: baseSensitivity };
    }

    // ==========================================================
    // 학습: 저해상도 → 고해상도 복원
    // ==========================================================
    async train(moleculeDB, epochs = 40, onProgress = null) {
        if (!this.superResModel) this.buildSuperResModel();

        const molecules = moleculeDB.getAll ? moleculeDB.getAll() : moleculeDB;
        if (molecules.length === 0) return;

        // 학습 데이터: 각 분자 × 각 채널의 응답 곡선
        const lowResData = [];
        const highResData = [];

        for (const mol of molecules) {
            for (let ch = 0; ch < this.numChannels; ch++) {
                const response = this.simulateSensorResponse(mol, ch);
                lowResData.push(response.lowRes.map(v => [v]));   // [20, 1]
                highResData.push(response.highRes.map(v => [v])); // [200, 1]

                // 노이즈 증강
                for (let a = 0; a < 2; a++) {
                    const noisyLow = response.lowRes.map(v => {
                        const noise = (Math.random() - 0.5) * 0.08;
                        return [Math.max(0, Math.min(1, v + noise))];
                    });
                    lowResData.push(noisyLow);
                    highResData.push(response.highRes.map(v => [v]));
                }
            }
        }

        const xs = tf.tensor3d(lowResData);   // [N, 20, 1]
        const ys = tf.tensor3d(highResData);   // [N, 200, 1]

        await this.superResModel.fit(xs, ys, {
            epochs,
            batchSize: 8,
            validationSplit: 0.15,
            callbacks: {
                onEpochEnd: (e, logs) => {
                    if (onProgress) onProgress(e + 1, epochs, logs.loss);
                    if ((e + 1) % 10 === 0) {
                        console.log(`[SensorEngine] Epoch ${e + 1}: loss=${logs.loss.toFixed(6)}`);
                    }
                }
            }
        });

        xs.dispose();
        ys.dispose();
        this.trained = true;
        console.log('[SensorEngine] Conv1D SR training complete');
    }

    // ==========================================================
    // 초해상도 복원
    // ==========================================================
    superResolve(lowResSignal) {
        if (!this.trained || !this.superResModel) return null;

        return tf.tidy(() => {
            const input = tf.tensor3d([lowResSignal.map(v => [v])]); // [1, 20, 1]
            const output = this.superResModel.predict(input); // [1, 200, 1]
            const result = Array.from(output.dataSync());
            return result;
        });
    }

    // 분자의 전체 센서 프로파일 (4채널 + 초해상도)
    getFullSensorProfile(molecule) {
        const channels = [];

        for (let ch = 0; ch < this.numChannels; ch++) {
            const response = this.simulateSensorResponse(molecule, ch);
            const srSignal = this.trained ? this.superResolve(response.lowRes) : null;

            channels.push({
                sensor: response.sensor,
                lowRes: response.lowRes,
                highRes: response.highRes,
                superResolved: srSignal,
                psnr: srSignal ? this._computePSNR(response.highRes, srSignal) : null,
                amplitude: response.amplitude,
                sensitivity: response.sensitivity
            });
        }

        return channels;
    }

    // PSNR 계산 (초해상도 품질 측정)
    _computePSNR(original, reconstructed) {
        const n = Math.min(original.length, reconstructed.length);
        let mse = 0;
        for (let i = 0; i < n; i++) {
            mse += (original[i] - reconstructed[i]) ** 2;
        }
        mse /= n;
        if (mse === 0) return Infinity;
        return 10 * Math.log10(1.0 / mse);
    }
}

export default SensorEngine;
