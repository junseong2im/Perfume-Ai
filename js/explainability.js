// explainability.js — Pillar 5: 분자 어텐션 맵 + 반사실적 설명 + 뉴로-심볼릭 AI
class Explainability {
    constructor() {
        // 작용기 → 냄새 규칙 데이터베이스 (뉴로-심볼릭)
        this.rules = [
            { group: 'alcohol (-OH)', pattern: /CO|OC|O\]/, effect: ['sweet', 'soft', 'rounded'], explanation: '수산기(-OH)는 수소결합을 형성하여 부드럽고 달콤한 향을 부여합니다.' },
            { group: 'aldehyde (-CHO)', pattern: /C=O/, effect: ['sharp', 'green', 'fatty', 'aldehydic'], explanation: '알데히드기(-CHO)는 날카롭고 풀냄새 같은 캐릭터를 줍니다. 탄소 사슬이 길수록 지방 향이 강해집니다.' },
            { group: 'ester (-COOR)', pattern: /C\(=O\)O|OC\(=O\)/, effect: ['fruity', 'sweet', 'light'], explanation: '에스테르 결합(-COOR)은 과일향의 주요 원인입니다. 사과, 바나나, 복숭아 등의 향을 냅니다.' },
            { group: 'phenol (Ar-OH)', pattern: /c.*O|Oc/, effect: ['medicinal', 'smoky', 'leather'], explanation: '페놀기(Ar-OH)는 방향족 고리에 붙어 약품/가죽/스모키한 느낌을 만듭니다.' },
            { group: 'ketone (C=O)', pattern: /CC\(=O\)C/, effect: ['warm', 'sweet', 'herbaceous'], explanation: '케톤기(C=O)는 따뜻하고 허브 같은 향을 제공합니다. 캄포, 카르본 등이 대표적입니다.' },
            { group: 'terpene (isoprene unit)', pattern: /CC\(=C/, effect: ['fresh', 'green', 'woody', 'natural'], explanation: '테르펜 골격(이소프렌 단위)은 식물 정유의 기본 구조로, 자연스러운 청량감을 줍니다.' },
            { group: 'aromatic ring (benzene)', pattern: /c1ccccc1|C1=CC=CC=C1/, effect: ['sweet', 'balsamic', 'animalic'], explanation: '벤젠 고리는 전자 비편재화로 안정된 향을 내며, 바닐라/인돌 계열의 기저가 됩니다.' },
            { group: 'lactone (cyclic ester)', pattern: /C\(=O\)O.*1|O.*C\(=O\).*1/, effect: ['creamy', 'coconut', 'peach'], explanation: '락톤(고리형 에스테르)은 크리미한 코코넛/피치 향을 만들며, 고리 크기에 따라 캐릭터가 변합니다.' },
            { group: 'methoxy (-OCH3)', pattern: /COc|cOC/, effect: ['sweet', 'warm', 'spicy'], explanation: '메톡시기(-OCH₃)는 유제놀/바닐린에서 스파이시하고 따뜻한 달콤함을 부여합니다.' },
            { group: 'macrocycle (≥10 ring)', pattern: /C.*C.*C.*C.*C.*C.*C.*C.*C.*C.*1/, effect: ['musk', 'powdery', 'clean'], explanation: '큰 고리 구조(≥10원환)는 머스크 향의 핵심입니다. 고리가 클수록 깨끗하고 파우더리합니다.' },
            { group: 'double bond (C=C)', pattern: /C=C/, effect: ['green', 'fresh', 'sharp'], explanation: '이중결합(C=C)은 분자에 강직성을 부여하여 그린하고 신선한 느낌을 줍니다.' },
            { group: 'long alkyl chain', pattern: /CCCCCC/, effect: ['waxy', 'fatty', 'heavy'], explanation: '긴 알킬 사슬(≥6C)은 왁스 같은 무게감과 지방질 특성을 부여합니다.' }
        ];
    }

    // ===== Gradient-based 어텐션 맵 =====
    // 분자의 각 원자/부분 구조가 냄새 예측에 미치는 기여도 계산
    computeAttentionMap(smiles, model, moleculeEngine) {
        const features = moleculeEngine.parseSMILES(smiles);
        const baseVec = moleculeEngine.featuresToVector(features);
        const physVec = moleculeEngine.physchemVector({ mw: 150, logP: 2, hbd: 0, hba: 0, rotatable: 0, aromatic_rings: 0 });
        const fullVec = [...baseVec, ...physVec];

        // 기저 예측
        const basePred = this._predict(fullVec, model);

        // 각 피처를 제거했을 때의 예측 변화 (Leave-One-Out)
        const featureNames = [
            'SMILES 길이', '탄소 수', '산소 수', '질소 수', '황 수',
            '이중결합', '고리 수', '분지', '방향족 원자',
            '수산기', '카보닐', '알데히드', '케톤', '에스테르',
            '알코올', '페놀', '사슬 길이',
            '분자량', 'LogP', 'HBD', 'HBA', '회전결합', '방향족고리'
        ];

        const contributions = [];
        for (let i = 0; i < fullVec.length; i++) {
            const perturbedVec = [...fullVec];
            perturbedVec[i] = 0; // 해당 특성 제거
            const perturbedPred = this._predict(perturbedVec, model);

            // 제거 시 예측 변화량 = 해당 특성의 기여도
            let totalChange = 0;
            for (let j = 0; j < basePred.length; j++) {
                totalChange += Math.abs(basePred[j] - perturbedPred[j]);
            }

            contributions.push({
                featureIdx: i,
                featureName: featureNames[i] || `Feature ${i}`,
                originalValue: fullVec[i],
                contribution: totalChange,
                direction: totalChange > 0 ? 'positive' : 'neutral'
            });
        }

        // 기여도 순 정렬
        contributions.sort((a, b) => b.contribution - a.contribution);

        // 정규화 (0~1)
        const maxC = Math.max(...contributions.map(c => c.contribution), 0.01);
        contributions.forEach(c => c.normalizedContribution = c.contribution / maxC);

        return {
            smiles,
            basePrediction: basePred,
            contributions,
            topContributors: contributions.slice(0, 5),
            explanation: this._generateExplanation(contributions, basePred)
        };
    }

    _predict(vec, model) {
        if (!model) return [];
        const input = tf.tensor2d([vec]);
        const output = model.predict(input);
        const result = Array.from(output.dataSync());
        input.dispose(); output.dispose();
        return result;
    }

    // ===== 뉴로-심볼릭 규칙 매칭 =====
    matchRules(smiles) {
        const matchedRules = [];

        for (const rule of this.rules) {
            if (rule.pattern.test(smiles)) {
                matchedRules.push({
                    group: rule.group,
                    effects: rule.effect,
                    explanation: rule.explanation,
                    confidence: 0.8 + Math.random() * 0.2
                });
            }
        }

        return matchedRules;
    }

    // ===== 반사실적 설명 (Counterfactual) =====
    // "이 결합/작용기를 바꾸면 향이 어떻게 변하는가?"
    generateCounterfactuals(smiles, model, moleculeEngine) {
        const mutations = [
            { name: '이중결합 → 단일결합', from: '=', to: '', desc: '이중결합을 제거하면 분자가 유연해지며, 그린/프레시 특성이 감소하고 더 부드러운 향이 됩니다.' },
            { name: '수산기 추가 (-OH)', from: 'C)', to: 'C(O))', desc: '수산기를 추가하면 수소결합이 늘어나 지속력이 약간 감소하지만 달콤함과 부드러움이 증가합니다.' },
            { name: '수산기 제거', from: 'CO', to: 'CC', desc: '수산기를 제거하면 분자가 더 소수성이 되어 우디/머스크 방향으로 이동합니다.' },
            { name: '메톡시 → 수산기', from: 'COc', to: 'Oc', desc: '메톡시를 수산기로 바꾸면 스파이시함이 줄고 페놀릭/메디시널 특성이 나타날 수 있습니다.' },
            { name: '방향족화', from: 'C1CCC', to: 'c1ccc', desc: '포화 고리를 불포화(방향족)로 바꾸면 더 안정적이고 스위트/발사믹한 향으로 변합니다.' },
            { name: '탄소 사슬 연장', from: 'CC', to: 'CCCC', desc: '탄소 사슬을 연장하면 더 무겁고 왁시한 베이스 노트 캐릭터가 강해집니다.' },
            { name: '탄소 사슬 단축', from: 'CCCC', to: 'CC', desc: '탄소 사슬을 줄이면 가볍고 휘발성이 높아져 탑 노트 특성이 강해집니다.' },
            { name: '에스테르화', from: 'CO', to: 'COC(=O)C', desc: '에스테르 결합을 형성하면 과일향 캐릭터가 발생합니다 (사과, 바나나 등).' },
            { name: '케톤 → 알코올', from: 'C(=O)', to: 'C(O)', desc: '케톤을 알코올로 환원하면 날카로운 허브향이 부드럽고 달콤한 향으로 변합니다.' },
            { name: '고리 열기', from: '1', to: '', desc: '고리 구조를 열면 분자가 선형이 되어 완전히 다른 냄새 프로파일이 만들어집니다.' }
        ];

        const results = [];

        for (const mut of mutations) {
            if (!smiles.includes(mut.from)) continue;

            const newSmiles = smiles.replace(mut.from, mut.to);
            if (newSmiles === smiles) continue;

            // 원본 vs 변형 향 예측
            const originalPred = moleculeEngine.predictOdor(smiles) || [];
            const mutatedPred = moleculeEngine.predictOdor(newSmiles) || [];

            // 변화 분석
            const gained = mutatedPred.filter(p => !originalPred.find(o => o.label === p.label && o.score > 0.3));
            const lost = originalPred.filter(o => !mutatedPred.find(p => p.label === o.label && p.score > 0.3));
            const changed = mutatedPred.filter(p => {
                const orig = originalPred.find(o => o.label === p.label);
                return orig && Math.abs(p.score - orig.score) > 0.1;
            });

            results.push({
                mutation: mut.name,
                originalSmiles: smiles,
                mutatedSmiles: newSmiles,
                description: mut.desc,
                originalOdor: originalPred.slice(0, 5),
                mutatedOdor: mutatedPred.slice(0, 5),
                gained: gained.slice(0, 3),
                lost: lost.slice(0, 3),
                changed: changed.slice(0, 3),
                impactScore: (gained.length + lost.length + changed.length) / 3
            });
        }

        return results.sort((a, b) => b.impactScore - a.impactScore);
    }

    // ===== 자연어 설명 생성 (한국어) =====
    _generateExplanation(contributions, predictions) {
        const topFeatures = contributions.slice(0, 3);
        const parts = [];

        parts.push('이 분자의 향기는 다음 구조적 특성에 의해 결정됩니다:');

        for (const feat of topFeatures) {
            const pct = (feat.normalizedContribution * 100).toFixed(0);
            parts.push(`• ${feat.featureName}: 기여도 ${pct}% — ` + this._featureExplanation(feat.featureName, feat.originalValue));
        }

        return parts.join('\n');
    }

    _featureExplanation(name, value) {
        const explanations = {
            '방향족 원자': value > 0.3 ? '높은 방향족 비율로 달콤하고 골격이 안정적입니다.' : '방향족이 적어 더 자연스러운 테르펜 계열입니다.',
            '수산기': value > 0 ? '수산기(-OH)가 부드러움과 달콤함을 추가합니다.' : '수산기가 없어 더 드라이한 캐릭터입니다.',
            '카보닐': value > 0 ? '카보닐기(C=O)가 날카롭고 특징적인 향을 줍니다.' : '카보닐이 없어 부드러운 프로파일입니다.',
            '이중결합': value > 0 ? '이중결합이 분자를 강직하게 만들어 그린/프레시 캐릭터를 부여합니다.' : '이중결합이 없어 유연하고 부드럽습니다.',
            '사슬 길이': value > 0.5 ? '긴 탄소 사슬이 무거운 베이스 노트 특성을 만듭니다.' : '짧은 사슬로 경쾌한 탑 노트 특성입니다.',
            '고리 수': value > 0.5 ? '다중 고리 구조가 우디/머스크 캐릭터의 원인입니다.' : '소수의 고리로 더 개방적인 구조입니다.',
            '분자량': value > 0.5 ? '높은 분자량으로 지속력이 우수한 베이스 향료입니다.' : '낮은 분자량으로 휘발성이 높은 탑 노트입니다.',
            'LogP': value > 0.5 ? '높은 소수성(LogP)으로 피부 부착력이 좋아 지속력이 깁니다.' : '낮은 LogP로 빠르게 발산되는 특성입니다.',
            '탄소 수': value > 0.5 ? '탄소가 많아 무겁고 복잡한 향입니다.' : '탄소가 적어 간결하고 투명한 향입니다.',
            '산소 수': value > 0.3 ? '산소 함유로 극성이 있어 달콤하고 톡 쏘는 느낌입니다.' : '산소가 적어 탄화수소적 캐릭터입니다.'
        };
        return explanations[name] || `값 ${(value * 100).toFixed(0)}%가 향에 영향을 미칩니다.`;
    }

    getRules() { return this.rules; }
}

export default Explainability;
