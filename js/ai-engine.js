// ai-engine.js - 완전체 AI 조향 파이프라인 (Level 1→5 통합)
// ============================================================
// 모든 모듈이 진짜 AI 아키텍처 사용:
// - SMILES 그래프 파서 + GNN readout
// - MPNN 메시지 패싱 + GRU 게이팅
// - Scaled Dot-Product Multi-Head Attention
// - GRU 시퀀스 모델 + 코사인 유사도 Zero-shot
// - Conv1D 초해상도
// ============================================================

import IngredientDB from './ingredient-db.js';
import HarmonyEngine from './harmony.js';
import Formulator from './formulator.js';
import Analyzer from './analyzer.js';
import NeuralNet from './neural-net.js';
import VAEGenerator from './vae-generator.js';
import MolecularEngine from './molecular-engine.js';
import GeometricGNN from './geometric-gnn.js';
import CompetitiveBinding from './competitive-binding.js';
import SensorEngine from './sensor-engine.js';
import ActiveLearner from './active-learner.js';
import Explainability from './explainability.js';

class AIEngine {
    constructor() {
        // Level 1
        this.db = new IngredientDB();
        this.harmony = null;
        this.formulator = null;
        this.analyzer = null;
        // Level 2-3
        this.neuralNet = null;
        this.vae = null;
        // Level 4
        this.molecular = new MolecularEngine();
        // Level 5 (Pillars)
        this.gnn3d = new GeometricGNN();
        this.binding = new CompetitiveBinding();
        this.sensor = new SensorEngine();
        this.activeLearner = new ActiveLearner();
        this.xai = new Explainability();
        this.primaryOdors = null;

        this.ready = false;
        this.level = 1;
    }

    async initialize(onProgress = null) {
        if (onProgress) onProgress('base', '데이터 로딩...');
        await this.db.load();
        this.harmony = new HarmonyEngine(this.db);
        this.formulator = new Formulator(this.db, this.harmony);
        this.analyzer = new Analyzer(this.db, this.harmony);
        this.ready = true;

        if (onProgress) onProgress('molecular', '분자 데이터 로딩...');
        await this.molecular.load();

        if (onProgress) onProgress('3d', '3D 분자 데이터 로딩...');
        await this.gnn3d.load();

        if (onProgress) onProgress('primary', 'Primary Odors 로딩...');
        try {
            const res = await fetch('./data/primary-odors.json');
            this.primaryOdors = await res.json();
        } catch (e) { console.warn('Primary odors load failed'); }

        console.log('[AIEngine] Level 1 Ready');
    }

    async trainModels(onProgress = null) {
        if (typeof tf === 'undefined') return;

        // L2: 신경망 (BatchNorm + 특성 기반 증강)
        if (onProgress) onProgress('neural', '신경망 학습 (Level 2)...');
        this.neuralNet = new NeuralNet(this.db);
        this.neuralNet.buildModel();
        await this.neuralNet.train(this.formulator, 30, 32, (e, t, l) => {
            if (onProgress) onProgress('neural', `신경망 ${e}/${t} (loss: ${l.toFixed(4)})`);
        });
        this.level = 2;

        // L3: VAE (KL Divergence + β-VAE)
        if (onProgress) onProgress('vae', 'VAE 생성 모델 학습 (Level 3)...');
        this.vae = new VAEGenerator(this.db);
        this.vae.buildModel();
        await this.vae.train(this.formulator, 40, (e, t, l) => {
            if (onProgress) onProgress('vae', `VAE ${e}/${t} (loss: ${l.toFixed(4)})`);
        });
        this.level = 3;

        // L4: 분자 냄새 예측 (실제 SMILES 그래프 파서)
        if (onProgress) onProgress('mol', '분자 냄새 예측 — SMILES 그래프 (Level 4)...');
        this.molecular.buildModel();
        await this.molecular.train(60, (e, t, l, a) => {
            if (onProgress) onProgress('mol', `분자 GNN-lite ${e}/${t} (acc: ${(a * 100).toFixed(1)}%)`);
        });
        this.level = 4;

        // L5-P1: 3D-GNN (MPNN 메시지 패싱)
        if (onProgress) onProgress('3dgnn', '3D-GNN MPNN 모델 (Pillar 1)...');
        this.gnn3d.buildModel();
        await this.gnn3d.train(this.molecular, 40, (e, t, l) => {
            if (onProgress) onProgress('3dgnn', `MPNN ${e}/${t} (loss: ${l?.toFixed?.(4) || '?'})`);
        });

        // L5-P2: Competitive Binding (Multi-Head Attention)
        if (onProgress) onProgress('binding', '경쟁적 결합 — Attention (Pillar 2)...');
        this.binding.buildModel();
        await this.binding.train(this.molecular, 30, (e, t, l) => {
            if (onProgress) onProgress('binding', `Attention ${e}/${t} (loss: ${l?.toFixed?.(4) || '?'})`);
        });

        // L5-P3: Sensor Super-Resolution (Conv1D)
        if (onProgress) onProgress('sensor', '센서 초해상도 — Conv1D (Pillar 3)...');
        this.sensor.buildSuperResModel();
        await this.sensor.train(this.molecular, 40, (e, t, l) => {
            if (onProgress) onProgress('sensor', `Conv1D SR ${e}/${t} (loss: ${l?.toFixed?.(6) || '?'})`);
        });

        // L5-P4: Active Learning (GRU + 임베딩 Zero-shot)
        if (onProgress) onProgress('active', '능동 학습 — GRU 화학어 모델 (Pillar 4)...');
        await this.activeLearner.train(this.molecular, 50, (e, msg) => {
            if (onProgress) onProgress('active', `${msg}`);
        });

        this.level = 5;
        if (onProgress) onProgress('complete', 'Level 5 완전체 — 전 모듈 가동!');
        console.log('[AIEngine] Level 5 FULL CAPACITY!');
    }

    // ===== L1: 규칙 기반 =====
    createPerfume(params) {
        if (!this.ready) throw new Error('Not initialized');
        const { mood, season, preferences, intensity } = params;
        let formula;
        if (this.neuralNet && this.neuralNet.trained) {
            const ruleF = this.formulator.formulate({ mood, season, preferences, intensity });
            const nnF = this.neuralNet.predict(mood, season, preferences || []);
            formula = nnF ? this._ensemble(ruleF, nnF) : ruleF;
        } else {
            formula = this.formulator.formulate({ mood, season, preferences, intensity });
        }
        const analysis = this.analyzer.analyze(formula);
        analysis.overallScore = this.analyzer.getOverallScore(analysis);

        // L5-P2: 혼합물 상호작용 분석 (Attention 기반)
        let bindingResult = null;
        if (this.binding.trained) {
            const molecules = formula.map(f => this.molecular.getAll().find(m => m.source_ingredients?.includes(f.id))).filter(Boolean);
            if (molecules.length >= 2) {
                bindingResult = this.binding.simulateBinding(molecules, this.molecular);
            }
        }

        // L5-P5: Primary Odors 분해
        let primaryDecomposition = null;
        if (this.primaryOdors) {
            primaryDecomposition = this.decomposeToPrimary(analysis);
        }

        return {
            name: this._genName(mood, season), description: this._genDesc(mood, season, analysis),
            formula, analysis, molecularInfo: this._addMolInfo(formula),
            bindingResult, primaryDecomposition,
            level: this.level, params, createdAt: new Date().toISOString()
        };
    }

    // ===== L3: 생성형 =====
    generateNovel() {
        if (!this.vae?.trained) return null;
        const formula = this.vae.generateRandom();
        const analysis = this.analyzer.analyze(formula);
        analysis.overallScore = this.analyzer.getOverallScore(analysis);
        return { name: 'AI Generated №' + Math.floor(Math.random() * 9999), description: 'VAE 잠재 공간에서 생성된 독창적 향수.', formula, analysis, level: 3, createdAt: new Date().toISOString() };
    }
    generateFromLatent(x, y) {
        if (!this.vae?.trained) return null;
        const formula = this.vae.generateFromLatent([x, y]);
        const analysis = this.analyzer.analyze(formula);
        analysis.overallScore = this.analyzer.getOverallScore(analysis);
        return { name: `Latent [${x.toFixed(1)},${y.toFixed(1)}]`, formula, analysis, level: 3, createdAt: new Date().toISOString() };
    }

    // ===== L4: 분자 (SMILES 그래프 파서 기반) =====
    exploreMolecule(id) { return this.molecular.predictMolecule(id); }
    predictSmell(smiles, props) { return this.molecular.predictOdor(smiles, props); }
    generateMoleculeVariants(id) { return this.molecular.generateVariants(id); }
    findSimilarMolecules(labels) { return this.molecular.findSimilar(labels); }
    getMolecules() { return this.molecular.getAll(); }
    getMoleculeById(id) { return this.molecular.getById(id); }

    // ===== L5-P1: 3D-GNN (MPNN) =====
    predict3D(molId) {
        const mol = this.molecular.getById(molId);
        return mol ? this.gnn3d.predict(mol, this.molecular) : null;
    }
    compareChirality(rId, sId) {
        const rMol = this.gnn3d.getAll3D().find(m => m.id === rId);
        const sMol = this.gnn3d.getAll3D().find(m => m.id === sId);
        return (rMol && sMol) ? this.gnn3d.compareChirality(rMol, sMol) : null;
    }
    getChiralPairs() { return this.gnn3d.getChiralPairs(); }
    get3DMolecules() { return this.gnn3d.getAll3D(); }

    // ===== L5-P2: 경쟁적 결합 (Multi-Head Attention) =====
    simulateMixture(moleculeIds) {
        const mols = moleculeIds.map(id => this.molecular.getById(id)).filter(Boolean);
        if (mols.length < 2) return null;
        return this.binding.simulateBinding(mols, this.molecular);
    }

    // ===== L5-P3: 센서 (Conv1D 초해상도) =====
    simulateSensor(moleculeId) {
        const mol = this.molecular.getById(moleculeId);
        if (!mol) return null;
        return this.sensor.getFullSensorProfile(mol);
    }

    // ===== L5-P4: 능동 학습 (GRU + 임베딩 Zero-shot) =====
    getUncertainMolecules() {
        return this.activeLearner.suggestNextSamples(this.molecular.getAll(), 10);
    }
    predictFromText(text) { return this.activeLearner.predictFromText(text); }

    // ===== L5-P5: XAI =====
    explainMolecule(smiles) {
        return {
            attentionMap: this.xai.computeAttentionMap(smiles, this.molecular.model, this.molecular),
            rules: this.xai.matchRules(smiles),
            counterfactuals: this.xai.generateCounterfactuals(smiles, this.molecular.model, this.molecular)
        };
    }

    // ===== Primary Odors (향의 원색) =====
    decomposeToPrimary(analysis) {
        if (!this.primaryOdors) return null;
        const basis = this.primaryOdors.basis_set;
        const categories = analysis.categoryBreakdown || [];
        return basis.map(po => {
            let weight = 0;
            for (const desc of po.descriptors) {
                const match = categories.find(c => c.name?.toLowerCase().includes(desc));
                if (match) weight += match.percentage || 5;
            }
            return { ...po, weight: Math.min(po.weight_range[1], Math.round(weight * 10) / 10) };
        }).filter(p => p.weight > 0).sort((a, b) => b.weight - a.weight);
    }
    getPrimaryOdors() { return this.primaryOdors; }

    // ===== Helpers =====
    _ensemble(rF, nnF) {
        const w = 0.6, map = new Map();
        for (const i of rF) map.set(i.id, { id: i.id, percentage: i.percentage * (1 - w) });
        for (const i of nnF) { if (map.has(i.id)) map.get(i.id).percentage += i.percentage * w; else map.set(i.id, { id: i.id, percentage: i.percentage * w }); }
        const merged = Array.from(map.values()).filter(f => f.percentage > 1).sort((a, b) => b.percentage - a.percentage).slice(0, 12);
        const total = merged.reduce((s, f) => s + f.percentage, 0);
        for (const f of merged) f.percentage = Math.round(f.percentage / total * 1000) / 10;
        const nt = merged.reduce((s, f) => s + f.percentage, 0);
        if (merged.length && nt !== 100) merged[0].percentage = Math.round((merged[0].percentage + (100 - nt)) * 10) / 10;
        return merged;
    }
    _addMolInfo(formula) {
        const mols = this.molecular.getAll();
        return formula.map(f => { const matched = mols.filter(m => m.source_ingredients?.includes(f.id)); return matched.length ? { ingredientId: f.id, molecules: matched.map(m => ({ id: m.id, name: m.name, smiles: m.smiles })) } : null; }).filter(Boolean);
    }
    _genName(mood) {
        const names = { romantic: ['Éternelle Rose', 'Amour Doux'], fresh: ['Aqua Serene', 'Crystal Clear'], elegant: ['Nuit Élégante', 'Grace Royale'], sexy: ['Velvet Sin', 'Dark Desire'], mysterious: ['Ombre Mystique', 'Luna Arcana'], cheerful: ['Soleil Joy', 'Citrus Festa'], calm: ['Zen Garden', 'Tranquil Woods'], luxurious: ['Or Imperial', 'Opulent Oud'], natural: ['Terra Verde', 'Wild Meadow'], cozy: ['Warm Embrace', 'Fireside Glow'] };
        const n = names[mood] || names.romantic; return n[Math.floor(Math.random() * n.length)];
    }
    _genDesc(mood, season, analysis) {
        const sk = { spring: '봄', summer: '여름', fall: '가을', winter: '겨울' }, mk = { romantic: '로맨틱한', fresh: '상쾌한', elegant: '우아한', sexy: '섹시한', mysterious: '신비로운', cheerful: '쾌활한', calm: '평온한', luxurious: '럭셔리한', natural: '자연스러운', cozy: '아늑한' };
        const top = analysis.notePyramid.top.map(n => n.name_ko).join(', '), mid = analysis.notePyramid.middle.map(n => n.name_ko).join(', '), base = analysis.notePyramid.base.map(n => n.name_ko).join(', ');
        let d = `${sk[season] || '봄'}의 ${mk[mood] || '감각적인'} 무드. ${top} → ${mid} → ${base}. 지속 ${analysis.longevity.hours}시간.`;
        if (this.level >= 5) d += ' [L5: MPNN + Attention + GRU + Conv1D Enhanced]';
        return d;
    }
    getIngredients() { return this.db.getAll(); }
    getCategories() { return this.db.getCategories(); }
    getLevel() { return this.level; }
}

export default AIEngine;
