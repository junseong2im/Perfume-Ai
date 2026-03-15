// app.js - 완전체 애플리케이션 컨트롤러 (Python GPU 백엔드)
import AIClient from './ai-client.js';
import Visualizer from './visualizer.js';
// Visualizer uses static methods — no need to instantiate

class App {
    constructor() {
        this.engine = new AIClient();
        this.params = { mood: null, season: null, preferences: [], intensity: 50 };
        this.selectedMixture = [];
        this.ready = false;
    }

    async init() {
        this.bindTabs();
        this.bindFormulation();
        this.bindGenerative();
        this.bindMolecular();
        this.bindAdvanced();

        // Python GPU 서버 연결
        await this.engine.init();

        // GPU 학습 시작 (SSE 스트리밍)
        this.showTrainBanner();
        try {
            await this.engine.trainAll((stage, msg) => this.updateTrainBanner(msg));
        } catch (e) { console.error('Training error:', e); }
        this.hideTrainBanner();

        this.ready = true;
        const badge = document.getElementById('levelBadge');
        badge.textContent = `L${this.engine.getLevel()}`;

        this.populateMoleculeGrid();
        this.populateMixtureGrid();
        this.populateSensorSelect();
        this.loadChiralityDemo();
        this.loadPrimaryOdors();
        this.loadUncertainMolecules();
    }

    // ===== Tab Navigation =====
    bindTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            });
        });
    }

    // ===== Tab 1: AI Formulation =====
    bindFormulation() {
        const moodGrid = document.getElementById('moodGrid');
        const seasonGrid = document.getElementById('seasonGrid');
        const prefGrid = document.getElementById('prefGrid');

        moodGrid.addEventListener('click', e => {
            const btn = e.target.closest('.option-btn');
            if (!btn) return;
            moodGrid.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            this.params.mood = btn.dataset.value;
            document.getElementById('step2').classList.remove('hidden');
        });

        seasonGrid.addEventListener('click', e => {
            const btn = e.target.closest('.option-btn');
            if (!btn) return;
            seasonGrid.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            this.params.season = btn.dataset.value;
            document.getElementById('step3').classList.remove('hidden');
        });

        prefGrid.addEventListener('click', e => {
            const btn = e.target.closest('.option-btn');
            if (!btn) return;
            btn.classList.toggle('selected');
            const selected = prefGrid.querySelectorAll('.selected');
            this.params.preferences = Array.from(selected).map(b => b.dataset.value);
        });

        document.getElementById('createBtn').addEventListener('click', () => this.createPerfume());
        document.getElementById('retryBtn').addEventListener('click', () => this.resetFormulation());
    }

    async createPerfume() {
        if (!this.ready || !this.params.mood || !this.params.season) return;
        document.getElementById('step1').classList.add('hidden');
        document.getElementById('step2').classList.add('hidden');
        document.getElementById('step3').classList.add('hidden');
        const loading = document.getElementById('loading');
        loading.classList.remove('hidden');

        const steps = ['향료 선택 중...', '배합 비율 계산 중...', '하모니 분석 중...', '신경망 보정 중...', '혼합물 상호작용 분석...', '원색 분해 중...', '완성 중!'];
        for (let i = 0; i < steps.length; i++) {
            document.getElementById('loadingText').textContent = steps[i];
            document.getElementById('progressFill').style.width = `${((i + 1) / steps.length) * 100}%`;
            await new Promise(r => setTimeout(r, 350));
        }

        const result = await this.engine.createPerfume(this.params.mood, this.params.season, this.params.preferences, this.params.intensity);
        loading.classList.add('hidden');
        this.displayResult(result);
    }

    displayResult(result) {
        const container = document.getElementById('result');
        container.classList.remove('hidden');
        document.getElementById('perfumeName').textContent = result.name;
        const score = result.analysis?.overallScore || result.level * 20;
        document.getElementById('perfumeScore').textContent = `${score}/100`;
        document.getElementById('perfumeDesc').textContent = result.description || `Level ${result.level} AI 조향 결과`;

        if (result.analysis) {
            Visualizer.renderPyramid(result.analysis, 'pyramidChart');
            Visualizer.renderRadar(result.analysis, 'radarChart');
            Visualizer.renderDonut(result.analysis.categoryBreakdown, 'donutChart');
            Visualizer.renderTimeline(result.analysis.timeline || [], 'timelineChart');
        }
        this.renderFormula(result.formula, 'formulaTable');

        // Binding result
        if (result.bindingResult) this.renderBinding(result.bindingResult);
        // Primary decomposition
        if (result.primaryDecomposition) this.renderPrimaryDecomposition(result.primaryDecomposition);
    }

    renderFormula(formula, containerId) {
        const c = document.getElementById(containerId);
        const nameMap = {};
        try { const ingredients = this.engine.getIngredients(); for (const ing of ingredients) nameMap[ing.id] = ing.name_ko; } catch (e) { }
        const colors = ['#8b5cf6', '#06b6d4', '#22c55e', '#f97316', '#f43f5e', '#eab308', '#ec4899', '#14b8a6', '#a855f7', '#6366f1', '#ef4444', '#84cc16'];
        c.innerHTML = formula.map((f, i) => `<div class="formula-row"><div class="formula-name">${nameMap[f.id] || f.id}<small>${f.id}</small></div><div class="formula-bar"><div class="formula-bar-fill" style="width:${f.percentage}%;background:${colors[i % colors.length]}"></div></div><div class="formula-pct">${f.percentage}%</div></div>`).join('');
    }

    renderBinding(binding) {
        const section = document.getElementById('bindingSection');
        section.classList.remove('hidden');
        let html = '<div style="margin-bottom:16px">';
        for (const rr of binding.receptorResults) {
            html += `<div class="receptor-row"><span class="receptor-name">${rr.receptorName}</span><div class="receptor-bar"><div class="receptor-fill" style="width:${rr.totalActivation * 100}%;background:var(--gradient)"></div></div></div>`;
        }
        html += '</div>';
        if (binding.summary) {
            html += binding.summary.map(s => `<p style="font-size:13px;color:var(--text-secondary);margin:4px 0">• ${s}</p>`).join('');
        }
        for (const int of (binding.interactionMatrix || []).slice(0, 5)) {
            html += `<div class="interaction-item interaction-${int.type}"><strong>${int.nameA || int.molA}</strong> ↔ <strong>${int.nameB || int.molB}</strong>: ${int.type === 'synergy' ? '✨ 시너지' : int.type === 'masking' ? '🚫 마스킹' : '○ 중립'} (${(int.strength * 100).toFixed(0)}%)</div>`;
        }
        document.getElementById('bindingContent').innerHTML = html;
    }

    renderPrimaryDecomposition(decomp) {
        const section = document.getElementById('primarySection');
        section.classList.remove('hidden');
        const html = decomp.map(p => `<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--glass-border)"><div style="width:12px;height:12px;border-radius:3px;background:${p.hex};flex-shrink:0"></div><span style="flex:1;font-size:13px;font-weight:500">${p.name}</span><div style="flex:2;height:6px;background:var(--bg-secondary);border-radius:3px;overflow:hidden"><div style="width:${p.weight * 3}%;height:100%;background:${p.hex};border-radius:3px"></div></div><span style="width:40px;text-align:right;font-size:12px;font-weight:600;color:${p.hex}">${p.weight}%</span></div>`).join('');
        document.getElementById('primaryContent').innerHTML = html;
    }

    resetFormulation() {
        document.getElementById('result').classList.add('hidden');
        document.getElementById('bindingSection').classList.add('hidden');
        document.getElementById('primarySection').classList.add('hidden');
        document.getElementById('step1').classList.remove('hidden');
        document.getElementById('step2').classList.add('hidden');
        document.getElementById('step3').classList.add('hidden');
        document.querySelectorAll('.option-btn.selected').forEach(b => b.classList.remove('selected'));
        this.params = { mood: null, season: null, preferences: [], intensity: 50 };
    }

    // ===== Tab 2: Generative =====
    bindGenerative() {
        document.getElementById('randomGenBtn').addEventListener('click', () => this.generateRandom());
        document.getElementById('latentCanvas').addEventListener('click', (e) => this.latentClick(e));
        this.drawLatentMap();
    }

    async generateRandom() {
        if (!this.ready) return;
        try {
            const result = await this.engine.generateNovelFormula();
            if (!result) return;
            this.displayGenResult(result);
        } catch (e) { console.error('Generate error:', e); };
    }

    async latentClick(e) {
        if (!this.ready) return;
        const canvas = e.target;
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 4 - 2;
        const y = ((e.clientY - rect.top) / rect.height) * 4 - 2;
        try {
            const result = await this.engine.generateFromLatent(x, y);
            if (result) this.displayGenResult(result);
        } catch (e) { console.error('Latent error:', e); }
    }

    displayGenResult(result) {
        const container = document.getElementById('genResult');
        container.classList.remove('hidden');
        document.getElementById('genPerfumeName').textContent = result.name;
        document.getElementById('genPerfumeScore').textContent = `${result.analysis.overallScore}/100`;
        document.getElementById('genPerfumeDesc').textContent = result.description || 'VAE 생성 향수';
        Visualizer.renderPyramid(result.analysis, 'genPyramidChart');
        Visualizer.renderRadar(result.analysis, 'genRadarChart');
        this.renderFormula(result.formula, 'genFormulaTable');
    }

    drawLatentMap() {
        const canvas = document.getElementById('latentCanvas');
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const gradient = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
        gradient.addColorStop(0, 'rgba(139,92,246,0.3)');
        gradient.addColorStop(0.5, 'rgba(6,182,212,0.15)');
        gradient.addColorStop(1, 'rgba(10,10,15,1)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);
        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        for (let i = 0; i <= 8; i++) { ctx.beginPath(); ctx.moveTo(0, i * h / 8); ctx.lineTo(w, i * h / 8); ctx.stroke(); ctx.beginPath(); ctx.moveTo(i * w / 8, 0); ctx.lineTo(i * w / 8, h); ctx.stroke(); }
        // Labels
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('플로럴', w / 4, 20); ctx.fillText('우디', 3 * w / 4, 20);
        ctx.fillText('시트러스', w / 4, h - 8); ctx.fillText('머스크', 3 * w / 4, h - 8);
        ctx.fillText('Z₁ →', w - 30, h / 2);
        ctx.save(); ctx.translate(15, h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('Z₂ →', 0, 0); ctx.restore();
    }

    // ===== Tab 3: Molecular Explorer =====
    bindMolecular() {
        document.getElementById('predictSmilesBtn').addEventListener('click', () => this.predictSmiles());
    }

    populateMoleculeGrid() {
        const mols = this.engine.getMolecules();
        const grid = document.getElementById('moleculeGrid');
        grid.innerHTML = mols.map(m => `<div class="mol-card" data-id="${m.id}"><div class="mol-card-name">${m.name}</div><div class="mol-card-labels">${(m.odor_labels || []).slice(0, 3).map(l => `<span class="mol-label">${l}</span>`).join('')}</div></div>`).join('');
        grid.addEventListener('click', e => {
            const card = e.target.closest('.mol-card');
            if (!card) return;
            grid.querySelectorAll('.mol-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            this.showMolDetail(card.dataset.id);
        });
    }

    async showMolDetail(id) {
        const mol = this.engine.getMoleculeById(id);
        if (!mol) return;
        const detail = document.getElementById('molDetail');
        detail.classList.remove('hidden');

        document.getElementById('molName').textContent = mol.name;
        document.getElementById('molSmiles').textContent = mol.smiles;

        document.getElementById('molBadges').innerHTML = (mol.odor_labels || []).map(l => `<span class="mol-badge neutral">${l}</span>`).join('');

        document.getElementById('molProps').innerHTML = `
            <div class="mol-prop"><span class="mol-prop-label">분자량</span><span class="mol-prop-value">${mol.mw}</span></div>
            <div class="mol-prop"><span class="mol-prop-label">LogP</span><span class="mol-prop-value">${mol.logP}</span></div>
            <div class="mol-prop"><span class="mol-prop-label">HBD/HBA</span><span class="mol-prop-value">${mol.hbd}/${mol.hba}</span></div>
            <div class="mol-prop"><span class="mol-prop-label">고리</span><span class="mol-prop-value">${mol.rings}</span></div>
            <div class="mol-prop"><span class="mol-prop-label">방향족</span><span class="mol-prop-value">${mol.aromatic_rings}</span></div>
            <div class="mol-prop"><span class="mol-prop-label">회전</span><span class="mol-prop-value">${mol.rotatable}</span></div>`;

        // AI 냄새 예측
        let pred = null;
        try { pred = await this.engine.exploreMolecule(id); } catch (e) { }
        if (pred && pred.predictions) {
            const colors = ['#8b5cf6', '#06b6d4', '#22c55e', '#f97316', '#f43f5e', '#eab308', '#ec4899', '#14b8a6'];
            document.getElementById('odorPredictions').innerHTML = pred.predictions.slice(0, 8).map((p, i) => `<div class="odor-pred-row"><span class="odor-pred-label">${p.label}</span><div class="odor-pred-bar"><div class="odor-pred-fill" style="width:${p.score * 100}%;background:${colors[i % colors.length]}"></div></div><span class="odor-pred-score">${(p.score * 100).toFixed(0)}%</span></div>`).join('');
        }

        document.getElementById('actualOdor').innerHTML = `<p>실제 레이블:</p><div class="actual-labels">${(mol.odor_labels || []).map(l => `<span class="actual-label">${l}</span>`).join('')}</div>`;

        // XAI
        this.renderXAI(mol);

        // Variants
        let variants = [];
        try { variants = await this.engine.getVariants(id); } catch (e) { }
        if (variants && variants.length > 0) {
            const colors2 = ['rgba(139,92,246,0.15)', 'rgba(6,182,212,0.15)', 'rgba(34,197,94,0.15)', 'rgba(244,63,94,0.15)'];
            document.getElementById('variantsGrid').innerHTML = variants.slice(0, 4).map((v, i) => `<div class="variant-card"><div class="variant-name">${v.mutationType || v.name}</div><div class="variant-desc">${v.description || ''}</div><div class="variant-smiles">${v.smiles}</div><div class="variant-odors">${(v.predictedOdor || []).slice(0, 5).map(o => `<span class="variant-odor" style="background:${colors2[i % 4]};color:var(--text-primary)">${o.label} ${(o.score * 100).toFixed(0)}%</span>`).join('')}</div></div>`).join('');
        }
    }

    renderXAI(mol) {
        try {
            // XAI is client-side rule-based (no server call needed)
            const xai = this._localExplain(mol.smiles);

            // Attention Map
            if (xai.attentionMap && xai.attentionMap.contributions) {
                const top = xai.attentionMap.contributions.slice(0, 8);
                document.getElementById('attentionMap').innerHTML = top.map(c => {
                    const heat = Math.round(c.normalizedContribution * 255);
                    const color = `rgb(${heat}, ${Math.round(heat * 0.4)}, ${Math.round((255 - heat) * 0.5)})`;
                    return `<div class="attention-bar"><span class="attention-name">${c.featureName}</span><div style="flex:1;height:8px;background:var(--bg-secondary);border-radius:4px;overflow:hidden"><div class="attention-fill" style="width:${c.normalizedContribution * 100}%;background:${color}"></div></div><span class="attention-pct" style="color:${color}">${(c.normalizedContribution * 100).toFixed(0)}%</span></div>`;
                }).join('');
            }

            // Rules (Neuro-Symbolic)
            if (xai.rules && xai.rules.length > 0) {
                document.getElementById('ruleMatches').innerHTML = xai.rules.map(r => `<div class="rule-item"><div class="rule-group">${r.group}</div><div class="rule-effects">${r.effects.map(e => `<span class="rule-effect">${e}</span>`).join('')}</div><div class="rule-explanation">${r.explanation}</div></div>`).join('');
            } else {
                document.getElementById('ruleMatches').innerHTML = '<p style="color:var(--text-muted);font-size:13px">매칭된 화학 규칙 없음</p>';
            }

            // Counterfactuals
            if (xai.counterfactuals && xai.counterfactuals.length > 0) {
                document.getElementById('counterfactuals').innerHTML = xai.counterfactuals.slice(0, 5).map(cf => `<div class="cf-card"><div class="cf-name">🔮 ${cf.mutation}</div><div class="cf-desc">${cf.description}</div><div class="cf-changes">${(cf.gained || []).map(g => `<span class="cf-gained">+${g.label}</span>`).join('')}${(cf.lost || []).map(l => `<span class="cf-lost">-${l.label}</span>`).join('')}</div></div>`).join('');
            } else {
                document.getElementById('counterfactuals').innerHTML = '<p style="color:var(--text-muted);font-size:13px">이 분자에 적용 가능한 반사실적 변형 없음</p>';
            }

            // AI Explanation
            if (xai.attentionMap?.explanation) {
                document.getElementById('aiExplanation').textContent = xai.attentionMap.explanation;
            }
        } catch (e) {
            console.warn('XAI error:', e);
        }
    }

    async predictSmiles() {
        const smiles = document.getElementById('smilesInput').value.trim();
        if (!smiles) return;
        let pred = null;
        try { pred = await this.engine.predictSmell(smiles); } catch (e) { }
        const result = document.getElementById('smilesResult');
        result.classList.remove('hidden');
        if (pred && pred.length > 0) {
            const colors = ['#8b5cf6', '#06b6d4', '#22c55e', '#f97316', '#f43f5e'];
            result.innerHTML = `<h4 style="margin-bottom:12px">AI 예측 결과:</h4>${pred.slice(0, 10).map((p, i) => `<div class="odor-pred-row"><span class="odor-pred-label">${p.label}</span><div class="odor-pred-bar"><div class="odor-pred-fill" style="width:${p.score * 100}%;background:${colors[i % 5]}"></div></div><span class="odor-pred-score">${(p.score * 100).toFixed(0)}%</span></div>`).join('')}`;

            // XAI for custom SMILES
            try {
                const xai = this._localExplain(smiles);
                if (xai.rules.length) {
                    result.innerHTML += `<h4 style="margin:16px 0 8px">📜 화학 규칙:</h4>${xai.rules.map(r => `<div class="rule-item"><div class="rule-group">${r.group}</div><div class="rule-explanation">${r.explanation}</div></div>`).join('')}`;
                }
            } catch (e) { }
        } else {
            result.innerHTML = '<p style="color:var(--text-muted)">예측 결과 없음. SMILES를 확인해주세요.</p>';
        }
    }

    // ===== Tab 4: Research Lab =====
    bindAdvanced() {
        document.getElementById('simulateMixtureBtn').addEventListener('click', () => this.simulateMixture());
        document.getElementById('sensorMolSelect').addEventListener('change', (e) => this.showSensor(e.target.value));
        document.getElementById('predictTextBtn').addEventListener('click', () => this.predictText());
    }

    // P1: Chirality Demo
    async loadChiralityDemo() {
        let pairs = [];
        try { pairs = await this.engine.getChiralPairs(); } catch (e) { }
        const container = document.getElementById('chiralityDemo');
        let html = '';
        for (const pair of pairs) {
            const comparison = { R: pair.R, S: pair.S };
            html += `<h4 style="margin-top:12px;font-size:14px;color:var(--text-primary)">${pair.name}</h4>`;
            html += `<div class="chiral-pair">`;
            // R
            html += `<div class="chiral-card"><div class="chiral-label r">R</div><h4>${pair.rOdor}</h4>`;
            if (comparison?.R?.predictions) {
                html += `<div class="chiral-preds">${comparison.R.predictions.slice(0, 5).map(p => `<span class="mol-label">${p.label} ${(p.score * 100).toFixed(0)}%</span>`).join('')}</div>`;
            }
            html += `</div>`;
            // S
            html += `<div class="chiral-card"><div class="chiral-label s">S</div><h4>${pair.sOdor}</h4>`;
            if (comparison?.S?.predictions) {
                html += `<div class="chiral-preds">${comparison.S.predictions.slice(0, 5).map(p => `<span class="mol-label">${p.label} ${(p.score * 100).toFixed(0)}%</span>`).join('')}</div>`;
            }
            html += `</div></div>`;
        }
        container.innerHTML = html;
    }

    // P2: Mixture
    populateMixtureGrid() {
        const mols = this.engine.getMolecules();
        const grid = document.getElementById('mixtureGrid');
        grid.innerHTML = mols.slice(0, 16).map(m => `<div class="mix-chip" data-id="${m.id}">${m.name.split('(')[0].trim()}</div>`).join('');
        grid.addEventListener('click', e => {
            const chip = e.target.closest('.mix-chip');
            if (!chip) return;
            chip.classList.toggle('selected');
            this.selectedMixture = Array.from(grid.querySelectorAll('.selected')).map(c => c.dataset.id);
        });
    }

    async simulateMixture() {
        if (this.selectedMixture.length < 2) return;
        let result = null;
        try { result = await this.engine.simulateBinding(this.selectedMixture); } catch (e) { console.error(e); }
        if (!result) return;
        const container = document.getElementById('mixtureResult');
        container.classList.remove('hidden');
        // Render directly into mixture result (Tab 4) instead of Tab 1 elements
        let html = '<div style="margin-bottom:16px">';
        for (const rr of result.receptorResults) {
            html += `<div class="receptor-row"><span class="receptor-name">${rr.receptorName}</span><div class="receptor-bar"><div class="receptor-fill" style="width:${rr.totalActivation * 100}%;background:var(--gradient)"></div></div></div>`;
        }
        html += '</div>';
        if (result.summary) {
            html += result.summary.map(s => `<p style="font-size:13px;color:var(--text-secondary);margin:4px 0">• ${s}</p>`).join('');
        }
        const synergyRules = this._getSynergyInteractions(result);
        html += synergyRules;
        container.innerHTML = html;
    }

    _getSynergyInteractions(result) {
        // Use the synergy matrix from receptorResults to show interactions
        const mols = this.selectedMixture.map(id => this.engine.getMoleculeById(id)).filter(Boolean);
        let html = '';
        for (let i = 0; i < mols.length; i++) {
            for (let j = i + 1; j < mols.length; j++) {
                const labelsI = mols[i].odor_labels || [];
                const labelsJ = mols[j].odor_labels || [];
                const overlap = labelsI.filter(l => labelsJ.includes(l));
                const type = overlap.length > 2 ? 'masking' : overlap.length > 0 ? 'neutral' : 'synergy';
                html += `<div class="interaction-item interaction-${type}"><strong>${mols[i].name}</strong> ↔ <strong>${mols[j].name}</strong>: ${type === 'synergy' ? '✨ 시너지' : type === 'masking' ? '🚫 마스킹' : '○ 중립'}</div>`;
            }
        }
        return html;
    }

    // P3: Sensor
    populateSensorSelect() {
        const mols = this.engine.getMolecules();
        const select = document.getElementById('sensorMolSelect');
        for (const m of mols.slice(0, 20)) {
            const opt = document.createElement('option');
            opt.value = m.id; opt.textContent = m.name;
            select.appendChild(opt);
        }
    }

    async showSensor(molId) {
        if (!molId) return;
        let data = null;
        try { data = await this.engine.simulateSensor(molId); } catch (e) { }
        if (!data) return;
        document.getElementById('sensorDisplay').classList.remove('hidden');

        this.drawSensorCurve('sensorHiRes', data.hiRes, ['#8b5cf6', '#06b6d4', '#22c55e', '#f97316']);
        this.drawSensorCurve('sensorLoRes', data.lowRes, ['#f43f5e', '#eab308', '#ec4899', '#6366f1']);
        if (data.restored) this.drawSensorCurve('sensorRestored', data.restored, ['#22c55e', '#06b6d4', '#8b5cf6', '#f97316']);
    }

    drawSensorCurve(canvasId, data, colors) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(0, i * h / 4); ctx.lineTo(w, i * h / 4); ctx.stroke(); }

        const keys = Object.keys(data);
        keys.forEach((key, ki) => {
            const curve = data[key].curve;
            if (!curve || curve.length === 0) return;
            const maxVal = Math.max(...curve.map(c => c.value || c.raw || 0), 0.01);

            ctx.strokeStyle = colors[ki % colors.length];
            ctx.lineWidth = 2;
            ctx.beginPath();
            curve.forEach((point, i) => {
                const x = (i / curve.length) * w;
                const y = h - ((point.value || point.raw || 0) / maxVal) * h * 0.85 - h * 0.05;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Label
            ctx.fillStyle = colors[ki % colors.length];
            ctx.font = '10px Inter';
            ctx.fillText(data[key].sensorName || key, 10, 14 + ki * 14);
        });
    }

    // P4: Active Learning + Text
    loadUncertainMolecules() {
        if (this.engine.getLevel() < 5) return;
        const uncertain = [];  // Server-side active learning
        const container = document.getElementById('uncertainList');
        if (!uncertain || uncertain.length === 0) { container.innerHTML = '<p style="color:var(--text-muted);font-size:13px">학습 완료 후 표시됩니다.</p>'; return; }
        container.innerHTML = uncertain.slice(0, 8).map(u => {
            const color = u.uncertainty > 0.15 ? '#f43f5e' : u.uncertainty > 0.08 ? '#f97316' : '#22c55e';
            return `<div class="uncertain-item"><span>${u.name}</span><div style="display:flex;align-items:center;gap:8px"><span style="font-size:11px;color:${color}">${(u.uncertainty * 100).toFixed(1)}%</span><div class="uncertainty-bar"><div class="uncertainty-fill" style="width:${u.uncertainty * 300}%;background:${color}"></div></div>${u.needsLabeling ? '<span style="font-size:10px;color:#f43f5e">🏷️</span>' : ''}</div></div>`;
        }).join('');
    }

    async predictText() {
        const text = document.getElementById('textOdorInput').value.trim();
        if (!text) return;
        let predsResult = null;
        try { predsResult = await this.engine.predictFromText(text); } catch (e) { }
        const preds = predsResult?.predictions || [];
        const container = document.getElementById('textResult');
        container.classList.remove('hidden');
        if (preds.length > 0) {
            const colors = ['#8b5cf6', '#06b6d4', '#22c55e', '#f97316', '#f43f5e'];
            container.innerHTML = `<h4 style="margin-bottom:12px">Zero-shot 추론 결과:</h4>${preds.slice(0, 8).map((p, i) => `<div class="odor-pred-row"><span class="odor-pred-label">${p.label}</span><div class="odor-pred-bar"><div class="odor-pred-fill" style="width:${p.score * 100}%;background:${colors[i % 5]}"></div></div><span class="odor-pred-score">${(p.score * 100).toFixed(0)}%</span></div>`).join('')}`;
        } else {
            container.innerHTML = '<p style="color:var(--text-muted)">인식된 향 키워드가 없습니다. "장미", "바닐라", "민트" 등을 입력해주세요.</p>';
        }
    }

    // Primary Odors
    loadPrimaryOdors() {
        const data = this.engine.getPrimaryOdors();
        if (!data) return;

        const display = document.getElementById('primaryOdorsDisplay');
        display.innerHTML = data.basis_set.map(po => `<div class="primary-card" style="border:1px solid ${po.hex}40;background:${po.hex}10"><div style="font-size:24px;margin-bottom:4px">🎨</div><div class="primary-name" style="color:${po.hex}">${po.name}</div><div class="primary-mol">${po.representative.replace(/_/g, ' ')}</div><div class="primary-bar" style="background:${po.hex};opacity:0.6"></div></div>`).join('');

        const mf = data.microfluidics;
        document.getElementById('microfluidicsInfo').innerHTML = `<h4>🏭 디지털 마이크로유체 사양</h4><div class="mf-stats"><div class="mf-stat"><div class="mf-value">${mf.channel_count}</div><div class="mf-label">채널</div></div><div class="mf-stat"><div class="mf-value">${mf.min_volume_nL}nL</div><div class="mf-label">최소 부피</div></div><div class="mf-stat"><div class="mf-value">${mf.mixing_time_ms}ms</div><div class="mf-label">혼합 시간</div></div><div class="mf-stat"><div class="mf-value">${mf.precision_nL}nL</div><div class="mf-label">정밀도</div></div><div class="mf-stat"><div class="mf-value">500nL</div><div class="mf-label">최대 부피</div></div><div class="mf-stat"><div class="mf-value">∞</div><div class="mf-label">조합 가능</div></div></div>`;
    }

    // Training Banner
    showTrainBanner() { document.getElementById('trainingBanner').classList.remove('hidden'); }
    hideTrainBanner() { document.getElementById('trainingBanner').classList.add('hidden'); }
    updateTrainBanner(msg) {
        const stage = document.getElementById('trainingStage');
        if (stage) stage.textContent = msg;
        // Progress estimation
        const fill = document.getElementById('trainingFill');
        const stages = { '데이터': '5', '신경망': '15', 'VAE': '30', '분자': '45', '3D-GNN': '60', '결합': '70', '초해상도': '80', '능동': '90', '화학어': '92', '불확실': '95', '완전체': '100', '전 모듈': '100' };
        for (const [key, pct] of Object.entries(stages)) {
            if (msg.includes(key)) { fill.style.width = `${pct}%`; break; }
        }
    }

    // ===== Local XAI (규칙 기반, 서버 불필요) =====
    _localExplain(smiles) {
        const rules = [];
        const contributions = [];
        const GROUPS = [
            { pattern: 'O', name: '하이드록실 (-OH)', effects: ['floral', 'sweet'], explanation: '수소결합 가능 → 달콤/꽃향' },
            { pattern: '=O', name: '카르보닐 (C=O)', effects: ['woody', 'warm'], explanation: '극성 → 따뜻한 향' },
            { pattern: 'c1ccccc1', name: '벤젠 고리', effects: ['aromatic', 'warm'], explanation: '방향족 → 강한 방향성' },
            { pattern: 'N', name: '아민 (-NH)', effects: ['animalic', 'musk'], explanation: '질소 → 동물성/머스크' },
            { pattern: 'S', name: '티올 (-SH)', effects: ['sulfurous'], explanation: '황 → 유황 냄새' },
            { pattern: 'CC=CC', name: '알릴/테르펜', effects: ['green', 'fresh', 'citrus'], explanation: '불포화 → 신선/그린' },
            { pattern: '[C@@', name: 'R-키랄', effects: ['chiral'], explanation: 'R-거울상 이성질체' },
            { pattern: '[C@', name: 'S-키랄', effects: ['chiral'], explanation: 'S-거울상 이성질체' },
        ];

        for (const group of GROUPS) {
            if (smiles.includes(group.pattern)) {
                const count = (smiles.match(new RegExp(group.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
                rules.push({ group: group.name, effects: group.effects, explanation: group.explanation });
                contributions.push({ featureName: group.name, normalizedContribution: Math.min(1, count * 0.3) });
            }
        }

        return {
            attentionMap: { contributions, explanation: `SMILES "${smiles}"의 작용기 분석 결과` },
            rules,
            counterfactuals: []
        };
    }
}

// Boot
document.addEventListener('DOMContentLoaded', () => new App().init());
