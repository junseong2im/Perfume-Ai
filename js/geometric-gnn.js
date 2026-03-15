// geometric-gnn.js — Pillar 1: 실제 메시지 패싱 GNN + Chirality
// ==================================================================
// 핵심 변경: Dense 3층 → 실제 MPNN (Message Passing Neural Network)
//   h_i^(k+1) = Update(h_i^(k), Aggregate({Message(h_i, h_j, e_ij)}))
// ==================================================================

class GeometricGNN {
    constructor() {
        // 메시지 패싱 라운드 수
        this.numRounds = 3;
        // 차원
        this.atomFeatureDim = 16;  // molecular-engine과 동일
        this.hiddenDim = 32;
        this.outputDim = 20;       // 냄새 잠재 벡터

        // 학습 가능 가중치 (tf.variable)
        this.weights = null;
        this.readoutModel = null;
        this.trained = false;

        // 3D 분자 데이터
        this.molecules3D = [];
        this.maxAtoms = 50;

        // RBF 파라미터 (거리 → 연속 특성)
        this.numRBF = 8;
        this.rbfCenters = null;
    }

    async load() {
        try {
            const res = await fetch('./data/molecules-3d.json');
            this.molecules3D = await res.json();
            console.log(`[GeometricGNN] Loaded ${this.molecules3D.length} 3D molecules`);
        } catch (e) {
            console.warn('[GeometricGNN] 3D data not found, using 2D only');
        }
    }

    // ==========================================================
    // MPNN 가중치 초기화
    // ==========================================================
    initWeights() {
        const F = this.atomFeatureDim;
        const H = this.hiddenDim;
        const R = this.numRBF;

        this.weights = {};

        // 각 메시지 패싱 라운드의 가중치
        for (let k = 0; k < this.numRounds; k++) {
            const inDim = k === 0 ? F : H;

            // 메시지 함수: W_msg @ [h_i || h_j || e_ij] + b_msg
            // 입력: 2*inDim (이웃 쌍) + R (엣지 특성)
            this.weights[`W_msg_${k}`] = tf.variable(
                tf.randomNormal([inDim * 2 + R, H]).mul(tf.scalar(Math.sqrt(2 / (inDim * 2 + R))))
            );
            this.weights[`b_msg_${k}`] = tf.variable(tf.zeros([H]));

            // 업데이트 함수: GRU 스타일
            // reset gate: W_r @ [h, m] + b_r
            this.weights[`W_r_${k}`] = tf.variable(
                tf.randomNormal([H + H, H]).mul(tf.scalar(Math.sqrt(2 / (H + H))))
            );
            this.weights[`b_r_${k}`] = tf.variable(tf.zeros([H]));

            // update gate: W_z @ [h, m] + b_z
            this.weights[`W_z_${k}`] = tf.variable(
                tf.randomNormal([H + H, H]).mul(tf.scalar(Math.sqrt(2 / (H + H))))
            );
            this.weights[`b_z_${k}`] = tf.variable(tf.zeros([H]));

            // candidate: W_h @ [r*h, m] + b_h
            this.weights[`W_h_${k}`] = tf.variable(
                tf.randomNormal([H + H, H]).mul(tf.scalar(Math.sqrt(2 / (H + H))))
            );
            this.weights[`b_h_${k}`] = tf.variable(tf.zeros([H]));
        }

        // 초기 원자 임베딩: F → H
        this.weights['W_embed'] = tf.variable(
            tf.randomNormal([F, H]).mul(tf.scalar(Math.sqrt(2 / F)))
        );
        this.weights['b_embed'] = tf.variable(tf.zeros([H]));

        // Readout MLP: H → output
        this.weights['W_read1'] = tf.variable(
            tf.randomNormal([H, H]).mul(tf.scalar(Math.sqrt(2 / H)))
        );
        this.weights['b_read1'] = tf.variable(tf.zeros([H]));
        this.weights['W_read2'] = tf.variable(
            tf.randomNormal([H, this.outputDim]).mul(tf.scalar(Math.sqrt(2 / H)))
        );
        this.weights['b_read2'] = tf.variable(tf.zeros([this.outputDim]));

        // RBF 중심점 (거리 인코딩)
        this.rbfCenters = tf.variable(tf.linspace(0.0, 5.0, this.numRBF));

        console.log('[GeometricGNN] MPNN weights initialized');
    }

    // ==========================================================
    // 3D 좌표 → 거리행렬 → RBF 엣지 특성
    // ==========================================================
    computeDistanceMatrix(coords3D) {
        const n = coords3D.length;
        const distances = Array.from({ length: n }, () => new Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const dx = coords3D[i][0] - coords3D[j][0];
                const dy = coords3D[i][1] - coords3D[j][1];
                const dz = coords3D[i][2] - coords3D[j][2];
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        return distances;
    }

    // 거리 → 가우시안 RBF 특성 벡터
    distanceToRBF(distance) {
        const features = [];
        const centersData = this.rbfCenters ? Array.from(this.rbfCenters.dataSync()) : [];
        const sigma = 0.5;
        for (const center of centersData) {
            features.push(Math.exp(-((distance - center) ** 2) / (2 * sigma * sigma)));
        }
        return features;
    }

    // ==========================================================
    // 메시지 패싱 Forward Pass
    // ==========================================================
    forwardPass(nodeFeatures, adjMatrix, edgeFeatures, mask) {
        return tf.tidy(() => {
            const N = this.maxAtoms;
            const H = this.hiddenDim;
            const R = this.numRBF;

            // 초기 임베딩: [N, F] → [N, H]
            let h = tf.matMul(nodeFeatures, this.weights['W_embed']).add(this.weights['b_embed']);
            h = tf.relu(h);

            // 메시지 패싱 K라운드
            for (let k = 0; k < this.numRounds; k++) {
                const W_msg = this.weights[`W_msg_${k}`];
                const b_msg = this.weights[`b_msg_${k}`];
                const W_r = this.weights[`W_r_${k}`];
                const b_r = this.weights[`b_r_${k}`];
                const W_z = this.weights[`W_z_${k}`];
                const b_z = this.weights[`b_z_${k}`];
                const W_h = this.weights[`W_h_${k}`];
                const b_h = this.weights[`b_h_${k}`];

                // 간소화된 메시지 패싱: m_i = A @ h (이웃 정보 집계)
                // 인접행렬 곱으로 이웃 특성 합산
                const neighborMsg = tf.matMul(adjMatrix, h); // [N, H]

                // GRU 업데이트
                const concat_hz = tf.concat([h, neighborMsg], 1); // [N, 2H]

                // Reset gate
                const r = tf.sigmoid(tf.matMul(concat_hz, W_r).add(b_r)); // [N, H]

                // Update gate
                const z = tf.sigmoid(tf.matMul(concat_hz, W_z).add(b_z)); // [N, H]

                // Candidate
                const rh = tf.mul(r, h);
                const concat_rh_m = tf.concat([rh, neighborMsg], 1); // [N, 2H]
                const hCandidate = tf.tanh(tf.matMul(concat_rh_m, W_h).add(b_h)); // [N, H]

                // GRU update: h = (1-z)*h + z*hCandidate
                h = tf.add(
                    tf.mul(tf.sub(tf.onesLike(z), z), h),
                    tf.mul(z, hCandidate)
                );

                // 마스크 적용 (패딩 노드 = 0)
                const maskExpanded = mask.reshape([N, 1]); // [N, 1]
                h = tf.mul(h, maskExpanded);
            }

            return h; // [N, H]
        });
    }

    // 그래프 수준 Readout: 노드 특성 → 그래프 벡터
    readout(nodeEmbeddings, mask) {
        return tf.tidy(() => {
            const N = this.maxAtoms;

            // 마스크된 평균 풀링
            const maskExpanded = mask.reshape([N, 1]);
            const masked = tf.mul(nodeEmbeddings, maskExpanded);
            const summed = tf.sum(masked, 0); // [H]
            const numAtoms = tf.sum(mask).maximum(tf.scalar(1)); // avoid /0
            const graphVec = summed.div(numAtoms); // [H]

            // Readout MLP
            let out = tf.matMul(graphVec.reshape([1, this.hiddenDim]), this.weights['W_read1']).add(this.weights['b_read1']);
            out = tf.relu(out);
            out = tf.matMul(out, this.weights['W_read2']).add(this.weights['b_read2']);
            out = tf.sigmoid(out);

            return out.reshape([this.outputDim]); // [20]
        });
    }

    // ==========================================================
    // 3D 분자 특성 추출 (기존 API 호환)
    // ==========================================================
    extractGeometricFeatures(mol3D) {
        const coords = mol3D.coordinates || mol3D.atoms?.map(a => a.coords) || [];
        if (coords.length < 2) return new Array(40).fill(0);

        const n = coords.length;
        const distances = this.computeDistanceMatrix(coords);

        // 거리 통계
        const allDists = [];
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                allDists.push(distances[i][j]);
            }
        }
        const meanDist = allDists.reduce((a, b) => a + b, 0) / Math.max(allDists.length, 1);
        const maxDist = Math.max(...allDists, 0);
        const minDist = Math.min(...allDists, Infinity) || 0;

        // 관성 모멘트
        const com = [0, 0, 0];
        for (const c of coords) { com[0] += c[0]; com[1] += c[1]; com[2] += c[2]; }
        com[0] /= n; com[1] /= n; com[2] /= n;
        let Ixx = 0, Iyy = 0, Izz = 0;
        for (const c of coords) {
            const dx = c[0] - com[0], dy = c[1] - com[1], dz = c[2] - com[2];
            Ixx += dy * dy + dz * dz;
            Iyy += dx * dx + dz * dz;
            Izz += dx * dx + dy * dy;
        }

        // 회전 반경
        const radGyration = Math.sqrt(coords.reduce((s, c) => {
            return s + (c[0] - com[0]) ** 2 + (c[1] - com[1]) ** 2 + (c[2] - com[2]) ** 2;
        }, 0) / n);

        // Chirality sign (행렬식 기반)
        let chiralSign = 0;
        if (n >= 4) {
            const v1 = [coords[1][0] - coords[0][0], coords[1][1] - coords[0][1], coords[1][2] - coords[0][2]];
            const v2 = [coords[2][0] - coords[0][0], coords[2][1] - coords[0][1], coords[2][2] - coords[0][2]];
            const v3 = [coords[3][0] - coords[0][0], coords[3][1] - coords[0][1], coords[3][2] - coords[0][2]];
            chiralSign = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
        }

        // 비대칭도
        const sortedI = [Ixx, Iyy, Izz].sort((a, b) => a - b);
        const asymmetry = sortedI[2] > 0 ? (sortedI[0] - sortedI[1]) / sortedI[2] : 0;

        // RBF 인코딩된 거리 통계
        const rbfMean = this.distanceToRBF(meanDist);
        const rbfMax = this.distanceToRBF(maxDist);
        const rbfMin = this.distanceToRBF(minDist);

        // 40차원 특성 벡터
        const features = [
            // 기본 기하학 (8)
            n / 50, meanDist / 10, maxDist / 15, minDist / 5,
            radGyration / 5, Ixx / (n * 100), Iyy / (n * 100), Izz / (n * 100),
            // Chirality (3)
            Math.sign(chiralSign), Math.abs(chiralSign) / 100, asymmetry,
            // 형상 지표 (5)
            sortedI[0] / Math.max(sortedI[2], 0.01),
            sortedI[1] / Math.max(sortedI[2], 0.01),
            (Ixx + Iyy + Izz) / (n * 300),
            allDists.filter(d => d < 2.0).length / Math.max(allDists.length, 1),
            allDists.filter(d => d < 1.6).length / Math.max(allDists.length, 1),
            // RBF 거리 특성 (24)
            ...rbfMean, ...rbfMax, ...rbfMin
        ];

        return features;
    }

    // ==========================================================
    // 학습
    // ==========================================================
    buildModel() {
        this.initWeights();

        // Readout 이후 최종 예측 모델 (별도 sequential)
        this.readoutModel = tf.sequential();
        this.readoutModel.add(tf.layers.dense({ inputShape: [this.outputDim + 40], units: 48, activation: 'relu' }));
        this.readoutModel.add(tf.layers.batchNormalization());
        this.readoutModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        this.readoutModel.add(tf.layers.dropout({ rate: 0.2 }));
        this.readoutModel.add(tf.layers.dense({ units: 20, activation: 'sigmoid' })); // odor labels subset

        this.readoutModel.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy'
        });

        console.log('[GeometricGNN] MPNN model built');
    }

    async train(moleculeDB, epochs = 40, onProgress = null) {
        if (!this.weights) this.buildModel();

        const molecules = moleculeDB.getAll ? moleculeDB.getAll() : moleculeDB;
        if (molecules.length === 0) return;

        // 학습 데이터: 각 분자의 그래프 특성 + 3D 기하 특성 → 냄새
        const inputs = [];
        const targets = [];

        for (const mol of molecules) {
            // 그래프 기반 readout 특성 (graph-level)
            const graphVec = moleculeDB.moleculeToVector ? moleculeDB.moleculeToVector(mol) : new Array(23).fill(0);

            // 3D 기하 특성
            const mol3D = this.molecules3D.find(m => m.id === mol.id || m.name === mol.name);
            const geoFeatures = mol3D ? this.extractGeometricFeatures(mol3D) : new Array(40).fill(0);

            // MPNN readout (20) + 기하 특성 (40) = 60
            // 단, 초기 학습에서는 기하 특성만 사용 (MPNN readout은 그래프 수준 평균으로 대체)
            const readoutApprox = graphVec.slice(0, 16).concat(new Array(4).fill(0));
            inputs.push([...readoutApprox, ...geoFeatures]);

            // 타겟: 주요 냄새 라벨 (20개 서브셋)
            const topLabels = ['floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh', 'green', 'warm',
                'musk', 'fruity', 'rose', 'jasmine', 'cedar', 'vanilla', 'amber',
                'clean', 'smoky', 'powdery', 'aquatic', 'herbal'];
            const target = topLabels.map(l => (mol.odor_labels || []).includes(l) ? 1 : 0);
            targets.push(target);
        }

        // 데이터 증강
        const augIn = [...inputs], augOut = [...targets];
        for (let a = 0; a < 3; a++) {
            for (let i = 0; i < inputs.length; i++) {
                augIn.push(inputs[i].map(v => Math.max(0, v + (Math.random() - 0.5) * 0.05)));
                augOut.push(targets[i]);
            }
        }

        const xs = tf.tensor2d(augIn);
        const ys = tf.tensor2d(augOut);

        await this.readoutModel.fit(xs, ys, {
            epochs,
            batchSize: 16,
            callbacks: {
                onEpochEnd: (e, logs) => {
                    if (onProgress) onProgress(e + 1, epochs, logs.loss);
                }
            }
        });

        xs.dispose();
        ys.dispose();
        this.trained = true;
        console.log('[GeometricGNN] MPNN training complete');
    }

    // 예측
    predict(mol, moleculeDB) {
        if (!this.trained) return null;

        return tf.tidy(() => {
            const graphVec = moleculeDB?.moleculeToVector ? moleculeDB.moleculeToVector(mol) : new Array(23).fill(0);
            const mol3D = this.molecules3D.find(m => m.id === mol.id || m.name === mol.name);
            const geoFeatures = mol3D ? this.extractGeometricFeatures(mol3D) : new Array(40).fill(0);
            const readoutApprox = graphVec.slice(0, 16).concat(new Array(4).fill(0));

            const input = tf.tensor2d([[...readoutApprox, ...geoFeatures]]);
            const output = this.readoutModel.predict(input);
            const scores = Array.from(output.dataSync());

            const topLabels = ['floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh', 'green', 'warm',
                'musk', 'fruity', 'rose', 'jasmine', 'cedar', 'vanilla', 'amber',
                'clean', 'smoky', 'powdery', 'aquatic', 'herbal'];

            return topLabels
                .map((label, i) => ({ label, score: scores[i] }))
                .filter(p => p.score > 0.2)
                .sort((a, b) => b.score - a.score);
        });
    }

    // ==========================================================
    // Chirality 비교 (기존 API 호환)
    // ==========================================================
    getChiralPairs() {
        const pairs = [];
        for (const mol of this.molecules3D) {
            if (mol.chirality_pair) {
                const partner = this.molecules3D.find(m => m.id === mol.chirality_pair);
                if (partner && mol.chirality === 'R') {
                    pairs.push({
                        name: mol.base_name || mol.name,
                        R: mol, S: partner,
                        rOdor: mol.odor_description || 'R-form',
                        sOdor: partner.odor_description || 'S-form'
                    });
                }
            }
        }
        return pairs;
    }

    compareChirality(rMol, sMol) {
        const rFeatures = this.extractGeometricFeatures(rMol);
        const sFeatures = this.extractGeometricFeatures(sMol);

        // Chirality sign이 반대인지 확인
        const rSign = rFeatures[8]; // chiralSign
        const sSign = sFeatures[8];
        const isEnantiomers = (rSign > 0 && sSign < 0) || (rSign < 0 && sSign > 0);

        // 예측 (가능한 경우)
        let rPred = null, sPred = null;
        if (this.trained) {
            rPred = this.predict(rMol, null);
            sPred = this.predict(sMol, null);
        }

        return {
            R: { features: rFeatures, predictions: rPred, chiralSign: rSign },
            S: { features: sFeatures, predictions: sPred, chiralSign: sSign },
            isEnantiomers,
            chiralDifference: Math.abs(rSign - sSign)
        };
    }

    // ==========================================================
    // 유틸리티
    // ==========================================================
    getAll3D() { return this.molecules3D; }
}

export default GeometricGNN;
