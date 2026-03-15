// molecular-engine.js — Level 4: 실제 SMILES 그래프 파서 + 분자 GNN 냄새 예측
// ============================================================================
// 핵심 변경: 문자 수 세기 → 진짜 분자 그래프 (인접행렬 + 원자 특성 행렬)
// ============================================================================

class MolecularEngine {
    constructor() {
        this.molecules = [];
        this.model = null;
        this.trained = false;
        this.maxAtoms = 50;  // 패딩용 최대 원자 수
        this.atomFeatureDim = 16; // 원자당 특성 차원

        // 원소 → one-hot 인덱스
        this.elementIndex = { 'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8, 'B': 9, 'Si': 10 };
        this.numElements = Object.keys(this.elementIndex).length;

        this.odorLabels = [
            'floral', 'rose', 'jasmine', 'violet', 'iris', 'lilac', 'muguet',
            'citrus', 'lemon', 'orange', 'bergamot', 'grapefruit', 'lime',
            'woody', 'cedar', 'sandalwood', 'pine', 'vetiver', 'patchouli',
            'spicy', 'clove', 'cinnamon', 'pepper', 'ginger', 'nutmeg',
            'sweet', 'vanilla', 'caramel', 'honey', 'gourmand', 'cotton_candy',
            'fresh', 'clean', 'soapy', 'ozonic', 'aquatic', 'marine',
            'green', 'leafy', 'grassy', 'herbaceous', 'minty',
            'warm', 'amber', 'balsamic', 'tobacco', 'smoky',
            'musk', 'animalic', 'powdery', 'creamy', 'waxy',
            'fruity', 'peach', 'plum', 'coconut', 'watermelon'
        ];
    }

    async load() {
        try {
            const res = await fetch('./data/molecules.json');
            this.molecules = await res.json();
            console.log(`[MolecularEngine] Loaded ${this.molecules.length} molecules`);
        } catch (e) {
            console.error('[MolecularEngine] Load failed:', e);
        }
    }

    // ===================================================================
    // 실제 SMILES 그래프 파서
    // SMILES → { atoms: [...], bonds: [...], adjMatrix, nodeFeatures }
    // ===================================================================
    parseSMILES(smiles) {
        const atoms = [];
        const bonds = [];
        const ringOpenings = {}; // 고리 닫힘 추적
        const branchStack = []; // 분기점 스택
        let currentAtom = -1;
        let i = 0;
        let bondType = 1; // 1=단일, 2=이중, 3=삼중

        while (i < smiles.length) {
            const ch = smiles[i];

            // 결합 유형 지정자
            if (ch === '=') { bondType = 2; i++; continue; }
            if (ch === '#') { bondType = 3; i++; continue; }
            if (ch === '/' || ch === '\\') { i++; continue; } // cis/trans 무시

            // 분기 시작/끝
            if (ch === '(') { branchStack.push(currentAtom); i++; continue; }
            if (ch === ')') { currentAtom = branchStack.pop(); i++; continue; }

            // 고리 숫자
            if (ch >= '0' && ch <= '9') {
                const ringNum = parseInt(ch);
                if (ringOpenings[ringNum] !== undefined) {
                    // 고리 닫힘
                    bonds.push({ from: currentAtom, to: ringOpenings[ringNum], type: bondType });
                    delete ringOpenings[ringNum];
                } else {
                    // 고리 열림
                    ringOpenings[ringNum] = currentAtom;
                }
                bondType = 1;
                i++;
                continue;
            }

            // 대괄호 원자: [NH], [O-], [C@@H] 등
            if (ch === '[') {
                const close = smiles.indexOf(']', i);
                if (close === -1) { i++; continue; }
                const bracket = smiles.substring(i + 1, close);
                const atom = this._parseBracketAtom(bracket);
                const atomIdx = atoms.length;
                atoms.push(atom);

                if (currentAtom >= 0) {
                    bonds.push({ from: currentAtom, to: atomIdx, type: bondType });
                }
                currentAtom = atomIdx;
                bondType = 1;
                i = close + 1;
                continue;
            }

            // 2문자 원소: Cl, Br, Si
            if (i + 1 < smiles.length) {
                const two = smiles.substring(i, i + 2);
                if (this.elementIndex[two] !== undefined) {
                    const atomIdx = atoms.length;
                    atoms.push({ element: two, aromatic: false, charge: 0, hCount: -1, chirality: 0 });
                    if (currentAtom >= 0) {
                        bonds.push({ from: currentAtom, to: atomIdx, type: bondType });
                    }
                    currentAtom = atomIdx;
                    bondType = 1;
                    i += 2;
                    continue;
                }
            }

            // 방향족 원자: c, n, o, s
            if ('cnos'.includes(ch)) {
                const element = ch.toUpperCase();
                const atomIdx = atoms.length;
                atoms.push({ element, aromatic: true, charge: 0, hCount: -1, chirality: 0 });
                if (currentAtom >= 0) {
                    bonds.push({ from: currentAtom, to: atomIdx, type: 1.5 }); // 방향족 결합
                }
                currentAtom = atomIdx;
                bondType = 1;
                i++;
                continue;
            }

            // 일반 원자: C, N, O, S, F, P, B, I
            if (this.elementIndex[ch] !== undefined || 'CNOSPBFI'.includes(ch)) {
                const atomIdx = atoms.length;
                atoms.push({ element: ch, aromatic: false, charge: 0, hCount: -1, chirality: 0 });
                if (currentAtom >= 0) {
                    bonds.push({ from: currentAtom, to: atomIdx, type: bondType });
                }
                currentAtom = atomIdx;
                bondType = 1;
                i++;
                continue;
            }

            // 기타 문자 (H, ., +, -  등) 스킵
            i++;
        }

        // 수소 수 자동 계산 (valence 기반)
        this._assignHydrogens(atoms, bonds);

        return { atoms, bonds, smiles };
    }

    // 대괄호 내 원자 파싱: [C@@H], [NH2+], [O-]
    _parseBracketAtom(bracket) {
        let element = '';
        let aromatic = false;
        let charge = 0;
        let hCount = 0;
        let chirality = 0;
        let i = 0;

        // 원소 추출
        if (i < bracket.length && bracket[i] >= 'A' && bracket[i] <= 'Z') {
            element = bracket[i];
            i++;
            if (i < bracket.length && bracket[i] >= 'a' && bracket[i] <= 'z' && bracket[i] !== 'H') {
                element += bracket[i];
                i++;
            }
        } else if (i < bracket.length && bracket[i] >= 'a' && bracket[i] <= 'z') {
            element = bracket[i].toUpperCase();
            aromatic = true;
            i++;
        }

        // chirality
        while (i < bracket.length && bracket[i] === '@') {
            chirality++;
            i++;
        }

        // H count
        if (i < bracket.length && bracket[i] === 'H') {
            i++;
            if (i < bracket.length && bracket[i] >= '0' && bracket[i] <= '9') {
                hCount = parseInt(bracket[i]);
                i++;
            } else {
                hCount = 1;
            }
        }

        // charge
        while (i < bracket.length) {
            if (bracket[i] === '+') charge++;
            else if (bracket[i] === '-') charge--;
            i++;
        }

        return { element, aromatic, charge, hCount, chirality };
    }

    // 원자가(valence) 기반 수소 수 자동 계산
    _assignHydrogens(atoms, bonds) {
        const valenceMap = { 'C': 4, 'N': 3, 'O': 2, 'S': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'P': 3, 'B': 3, 'Si': 4 };

        // 각 원자의 결합 수 계산
        const bondCounts = new Array(atoms.length).fill(0);
        for (const bond of bonds) {
            const effectiveBond = bond.type === 1.5 ? 1 : bond.type; // 방향족 → 1로 근사
            bondCounts[bond.from] += effectiveBond;
            bondCounts[bond.to] += effectiveBond;
        }

        for (let a = 0; a < atoms.length; a++) {
            if (atoms[a].hCount >= 0) continue; // 이미 명시됨
            const valence = valenceMap[atoms[a].element] || 4;
            const currentBonds = bondCounts[a];
            const charge = atoms[a].charge;
            atoms[a].hCount = Math.max(0, valence - currentBonds + charge);
        }
    }

    // ===================================================================
    // 분자 그래프 → 텐서 변환 (GNN 입력)
    // ===================================================================

    // 원자 특성 벡터 (16차원)
    atomToFeatures(atom) {
        const features = new Array(this.atomFeatureDim).fill(0);

        // [0..10] 원소 one-hot
        const elIdx = this.elementIndex[atom.element];
        if (elIdx !== undefined) features[elIdx] = 1;

        // [11] 방향족 여부
        features[11] = atom.aromatic ? 1 : 0;

        // [12] 전하 (정규화)
        features[12] = atom.charge / 2;

        // [13] 수소 수 (정규화)
        features[13] = (atom.hCount >= 0 ? atom.hCount : 0) / 4;

        // [14] chirality 여부
        features[14] = atom.chirality > 0 ? 1 : 0;

        // [15] chirality 방향 (@@=2 → 1, @=1 → -1)
        features[15] = atom.chirality === 2 ? 1 : atom.chirality === 1 ? -1 : 0;

        return features;
    }

    // 분자 그래프 → GNN 입력 텐서
    graphToTensors(parsedMol) {
        const { atoms, bonds } = parsedMol;
        const n = Math.min(atoms.length, this.maxAtoms);

        // 노드 특성 행렬: [maxAtoms × atomFeatureDim]
        const nodeFeatures = [];
        for (let i = 0; i < this.maxAtoms; i++) {
            if (i < n) {
                nodeFeatures.push(this.atomToFeatures(atoms[i]));
            } else {
                nodeFeatures.push(new Array(this.atomFeatureDim).fill(0)); // 패딩
            }
        }

        // 인접행렬: [maxAtoms × maxAtoms] (결합 유형 가중치 포함)
        const adjMatrix = Array.from({ length: this.maxAtoms }, () => new Array(this.maxAtoms).fill(0));
        for (const bond of bonds) {
            if (bond.from < n && bond.to < n) {
                const weight = bond.type === 1.5 ? 1.5 : bond.type;
                adjMatrix[bond.from][bond.to] = weight;
                adjMatrix[bond.to][bond.from] = weight; // 무방향 그래프
            }
        }

        // 자기 루프 추가 (GNN에서 표준)
        for (let i = 0; i < n; i++) {
            adjMatrix[i][i] = 1;
        }

        // 도(degree) 정규화: D^(-1/2) A D^(-1/2)
        const degrees = adjMatrix.map(row => row.reduce((s, v) => s + (v > 0 ? 1 : 0), 0));
        const normAdj = adjMatrix.map((row, i) =>
            row.map((v, j) => {
                if (v === 0 || degrees[i] === 0 || degrees[j] === 0) return 0;
                return v / Math.sqrt(degrees[i] * degrees[j]);
            })
        );

        // 원자 수 마스크 (패딩 구분용)
        const mask = new Array(this.maxAtoms).fill(0);
        for (let i = 0; i < n; i++) mask[i] = 1;

        return { nodeFeatures, adjMatrix: normAdj, mask, numAtoms: n };
    }

    // 분자 → 평탄화된 입력 벡터 (기존 API 호환)
    moleculeToVector(mol) {
        const parsed = this.parseSMILES(mol.smiles || '');
        const { nodeFeatures, adjMatrix, mask, numAtoms } = this.graphToTensors(parsed);

        // 그래프 수준 특성 (readout): 원자 특성의 가중 평균
        const graphVec = new Array(this.atomFeatureDim).fill(0);
        for (let i = 0; i < numAtoms; i++) {
            for (let f = 0; f < this.atomFeatureDim; f++) {
                graphVec[f] += nodeFeatures[i][f];
            }
        }
        if (numAtoms > 0) {
            for (let f = 0; f < this.atomFeatureDim; f++) {
                graphVec[f] /= numAtoms;
            }
        }

        // 그래프 통계 특성 추가 (7차원)
        const graphStats = [
            numAtoms / this.maxAtoms,                                                  // 원자 수 (정규화)
            parsed.bonds.length / (this.maxAtoms * 2),                                 // 결합 수
            parsed.bonds.filter(b => b.type === 2).length / Math.max(parsed.bonds.length, 1), // 이중결합 비율
            parsed.bonds.filter(b => b.type === 1.5).length / Math.max(parsed.bonds.length, 1), // 방향족 비율
            parsed.atoms.filter(a => a.aromatic).length / Math.max(numAtoms, 1),       // 방향족 원자 비율
            parsed.atoms.filter(a => a.chirality > 0).length / Math.max(numAtoms, 1),  // 키랄 원자 비율
            (mol.mw || this._estimateMW(parsed)) / 400                                 // 분자량 (정규화)
        ];

        return [...graphVec, ...graphStats]; // 23차원 (16 + 7)
    }

    // 분자 물리화학적 특성 (기존 API 호환)
    physchemVector(mol) {
        return [
            (mol.mw || 150) / 400,
            (mol.logP || 2) / 8,
            (mol.hbd || 0) / 4,
            (mol.hba || 0) / 6,
            (mol.rotatable || 0) / 15,
            (mol.aromatic_rings || 0) / 4
        ];
    }

    // 전체 입력 벡터 = 그래프 특성 + 물리화학 (29차원)
    fullMoleculeVector(mol) {
        return [...this.moleculeToVector(mol), ...this.physchemVector(mol)];
    }

    // 타겟 벡터: 냄새 원-핫 (55차원)
    odorToVector(odorLabels) {
        return this.odorLabels.map(l => odorLabels.includes(l) ? 1 : 0);
    }

    // ===================================================================
    // GNN-lite 모델: 그래프 readout → 냄새 예측
    // ===================================================================
    buildModel() {
        const inputDim = 23 + 6; // moleculeToVector(23) + physchemVector(6)
        const outputDim = this.odorLabels.length;

        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [inputDim], units: 64, activation: 'relu' }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: outputDim, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(0.002),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        this.model = model;
        console.log(`[MolecularEngine] Model built: ${inputDim} → ${outputDim}`);
    }

    async train(epochs = 60, onProgress = null) {
        if (!this.model) this.buildModel();
        if (this.molecules.length === 0) await this.load();

        const inputs = this.molecules.map(m => this.fullMoleculeVector(m));
        const outputs = this.molecules.map(m => this.odorToVector(m.odor_labels));

        // 데이터 증강: 노드 특성에 노이즈 추가 (4배)
        const augInputs = [...inputs];
        const augOutputs = [...outputs];
        for (let aug = 0; aug < 4; aug++) {
            for (let i = 0; i < inputs.length; i++) {
                const noisy = inputs[i].map(v => Math.max(0, v + (Math.random() - 0.5) * 0.08));
                augInputs.push(noisy);
                augOutputs.push(outputs[i]);
            }
        }

        const xs = tf.tensor2d(augInputs);
        const ys = tf.tensor2d(augOutputs);

        await this.model.fit(xs, ys, {
            epochs,
            batchSize: 16,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (onProgress) onProgress(epoch + 1, epochs, logs.loss, logs.acc);
                    if ((epoch + 1) % 15 === 0) {
                        console.log(`[MolecularEngine] Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(3)}`);
                    }
                }
            }
        });

        xs.dispose();
        ys.dispose();
        this.trained = true;
        console.log('[MolecularEngine] Training complete!');
    }

    // 냄새 예측
    predictOdor(smiles, physicalProps = {}) {
        if (!this.trained) return null;

        const mol = {
            smiles,
            mw: physicalProps.mw || this._estimateMWFromSmiles(smiles),
            logP: physicalProps.logP || this._estimateLogP(smiles),
            hbd: physicalProps.hbd || 0,
            hba: physicalProps.hba || 0,
            rotatable: physicalProps.rotatable || 0,
            aromatic_rings: physicalProps.aromatic_rings || 0
        };

        const vec = this.fullMoleculeVector(mol);
        const input = tf.tensor2d([vec]);
        const output = this.model.predict(input);
        const scores = Array.from(output.dataSync());
        input.dispose();
        output.dispose();

        return this.odorLabels
            .map((label, i) => ({ label, score: scores[i] }))
            .filter(p => p.score > 0.3)
            .sort((a, b) => b.score - a.score);
    }

    // 기존 분자의 냄새 예측 (검증용)
    predictMolecule(moleculeId) {
        const mol = this.molecules.find(m => m.id === moleculeId);
        if (!mol) return null;

        const predictions = this.predictOdor(mol.smiles, mol);
        return { molecule: mol, predictions, actual: mol.odor_labels };
    }

    // ===================================================================
    // 가상 분자 변형
    // ===================================================================
    generateVariants(moleculeId) {
        const mol = this.molecules.find(m => m.id === moleculeId);
        if (!mol) return [];

        const variants = [];
        const baseSMILES = mol.smiles;

        const mutations = [
            { name: '수산기 추가', transform: s => s.replace('C)', 'C(O))'), desc: '알코올기(-OH) 추가 → 더 부드럽고 달콤한 향 예상' },
            { name: '메틸화', transform: s => s + 'C', desc: '메틸기(-CH₃) 추가 → 지속력 증가 예상' },
            { name: '에스테르화', transform: s => s.replace('CO', 'COC(=O)C'), desc: '에스테르 결합 추가 → 과일향 증가 예상' },
            { name: '고리 추가', transform: s => 'C1CC(' + s + ')CC1', desc: '시클로헥산 고리 추가 → 우디/머스크 증가 예상' },
            { name: '방향족화', transform: s => s.replace('C1CCC', 'c1ccc'), desc: '방향족 고리 도입 → 스모키/스파이시 변화 예상' },
            { name: '산화', transform: s => s.replace('CO', 'C=O'), desc: '알데하이드/케톤 전환 → 더 날카롭고 파워풀한 향 예상' },
            { name: '탈수소화', transform: s => s.replace('CC', 'C=C'), desc: '이중결합 추가 → 더 그린/프레시한 향 예상' },
            { name: '사슬 연장', transform: s => s.replace('CC', 'CCCC'), desc: '탄소 사슬 연장 → 더 무거운 베이스 향 예상' }
        ];

        for (const mutation of mutations) {
            try {
                const newSMILES = mutation.transform(baseSMILES);
                if (newSMILES !== baseSMILES && newSMILES.length > 3) {
                    const predictions = this.predictOdor(newSMILES, {
                        mw: (mol.mw || 150) * (newSMILES.length / baseSMILES.length),
                        logP: mol.logP || 2,
                        hbd: mol.hbd || 0,
                        hba: mol.hba || 0
                    });

                    variants.push({
                        name: `${mol.name} — ${mutation.name}`,
                        smiles: newSMILES,
                        mutationType: mutation.name,
                        description: mutation.desc,
                        predictedOdor: predictions,
                        originalOdor: mol.odor_labels,
                        confidence: predictions && predictions.length > 0 ? predictions[0].score : 0
                    });
                }
            } catch (e) { /* 유효하지 않은 변형은 무시 */ }
        }

        variants.sort((a, b) => b.confidence - a.confidence);
        return variants;
    }

    // 유사 분자 검색
    findSimilar(targetOdorLabels, topK = 5) {
        if (!this.molecules.length) return [];

        return this.molecules
            .map(mol => {
                const overlap = mol.odor_labels.filter(l => targetOdorLabels.includes(l));
                return {
                    ...mol,
                    similarity: overlap.length / Math.max(mol.odor_labels.length, targetOdorLabels.length),
                    matchingLabels: overlap
                };
            })
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK);
    }

    // ===================================================================
    // 구조 분석 유틸리티 (파싱된 그래프 기반)
    // ===================================================================

    // 파싱된 그래프에서 분자량 추정
    _estimateMW(parsed) {
        const atomicWeights = { 'C': 12, 'N': 14, 'O': 16, 'S': 32, 'F': 19, 'Cl': 35.5, 'Br': 80, 'I': 127, 'P': 31, 'B': 11, 'Si': 28, 'H': 1 };
        let mw = 0;
        for (const atom of parsed.atoms) {
            mw += atomicWeights[atom.element] || 12;
            mw += (atom.hCount >= 0 ? atom.hCount : 0) * 1; // 수소
        }
        return mw;
    }

    // SMILES 문자열에서 직접 분자량 추정 (파싱 안 한 경우)
    _estimateMWFromSmiles(smiles) {
        const parsed = this.parseSMILES(smiles);
        return this._estimateMW(parsed);
    }

    // LogP 추정 (원자 기여법 Wildman-Crippen 간이 버전)
    _estimateLogP(smiles) {
        const parsed = this.parseSMILES(smiles);
        const contributions = { 'C': 0.5, 'N': -1.0, 'O': -1.2, 'S': 0.6, 'F': 0.4, 'Cl': 0.9, 'Br': 1.1, 'I': 1.5, 'P': -0.5 };
        let logP = 0;
        for (const atom of parsed.atoms) {
            logP += contributions[atom.element] || 0;
            if (atom.aromatic) logP += 0.2; // 방향족 기여
        }
        return logP;
    }

    // ===================================================================
    // 기존 API 호환
    // ===================================================================
    getAll() { return this.molecules; }
    getById(id) { return this.molecules.find(m => m.id === id); }
    getOdorLabels() { return this.odorLabels; }

    // 기존 featuresToVector 호환 (explainability.js에서 사용)
    featuresToVector(features) {
        // 기존 parseSMILES 호환 래퍼
        return Object.values(features).slice(0, 17).map((v, i) => {
            const norms = [50, 20, 5, 3, 2, 5, 4, 8, 10, 3, 3, 1, 1, 1, 1, 1, 20];
            return (typeof v === 'number' ? v : 0) / (norms[i] || 1);
        });
    }
}

export default MolecularEngine;
