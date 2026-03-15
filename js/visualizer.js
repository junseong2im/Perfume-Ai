// visualizer.js - 시각화 모듈 (노트 피라미드, 차트)
class Visualizer {
    // 노트 피라미드 SVG 생성
    static renderPyramid(analysis, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const { notePyramid } = analysis;
        const w = 500, h = 400, pad = 20;

        let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px">`;

        // 피라미드 배경
        const levels = [
            { label: 'TOP', y: pad, h: h * 0.25, w1: w * 0.2, w2: w * 0.5, color: 'rgba(255,200,100,0.15)', border: 'rgba(255,200,100,0.5)', notes: notePyramid.top },
            { label: 'HEART', y: pad + h * 0.28, h: h * 0.32, w1: w * 0.35, w2: w * 0.75, color: 'rgba(255,120,150,0.15)', border: 'rgba(255,120,150,0.5)', notes: notePyramid.middle },
            { label: 'BASE', y: pad + h * 0.63, h: h * 0.32, w1: w * 0.6, w2: w * 0.95, color: 'rgba(150,120,255,0.15)', border: 'rgba(150,120,255,0.5)', notes: notePyramid.base }
        ];

        levels.forEach((lv, idx) => {
            const cx = w / 2;
            const topW = lv.w1, botW = lv.w2;
            const x1 = cx - topW / 2, x2 = cx + topW / 2;
            const x3 = cx + botW / 2, x4 = cx - botW / 2;
            const y1 = lv.y, y2 = lv.y + lv.h;

            svg += `<polygon points="${x1},${y1} ${x2},${y1} ${x3},${y2} ${x4},${y2}" 
                     fill="${lv.color}" stroke="${lv.border}" stroke-width="1.5" rx="4"/>`;
            // Level label
            svg += `<text x="${w / 2}" y="${lv.y + 18}" text-anchor="middle" fill="${lv.border}" 
                     font-size="11" font-weight="700" letter-spacing="2">${lv.label}</text>`;
            // Notes
            const notes = lv.notes;
            notes.forEach((n, i) => {
                const ny = lv.y + 32 + i * 22;
                if (ny < y2 - 5) {
                    svg += `<text x="${w / 2}" y="${ny}" text-anchor="middle" fill="rgba(255,255,255,0.9)" 
                             font-size="13">${n.name_ko} (${n.percentage}%)</text>`;
                }
            });
        });

        svg += '</svg>';
        container.innerHTML = svg;
    }

    // 카테고리 도넛 차트
    static renderDonut(categoryBreakdown, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const size = 260, cx = size / 2, cy = size / 2, r = 90, innerR = 55;
        const colors = {
            floral: '#FF6B9D', citrus: '#FFD93D', woody: '#8B6914', spicy: '#FF6347',
            fruity: '#FF69B4', amber: '#FFBF00', musk: '#DDA0DD', gourmand: '#D2691E',
            aquatic: '#00CED1', green: '#32CD32', aromatic: '#9370DB', balsamic: '#CD853F',
            animalic: '#8B4513', chypre: '#6B8E23', synthetic: '#C0C0C0'
        };

        let svg = `<svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:260px">`;
        let cumAngle = -90;

        categoryBreakdown.forEach(cat => {
            const angle = (cat.percentage / 100) * 360;
            const startRad = (cumAngle * Math.PI) / 180;
            const endRad = ((cumAngle + angle) * Math.PI) / 180;
            const largeArc = angle > 180 ? 1 : 0;

            const x1 = cx + r * Math.cos(startRad), y1 = cy + r * Math.sin(startRad);
            const x2 = cx + r * Math.cos(endRad), y2 = cy + r * Math.sin(endRad);
            const ix1 = cx + innerR * Math.cos(endRad), iy1 = cy + innerR * Math.sin(endRad);
            const ix2 = cx + innerR * Math.cos(startRad), iy2 = cy + innerR * Math.sin(startRad);

            const color = colors[cat.name] || '#888';
            svg += `<path d="M${x1},${y1} A${r},${r} 0 ${largeArc},1 ${x2},${y2} L${ix1},${iy1} A${innerR},${innerR} 0 ${largeArc},0 ${ix2},${iy2} Z"
                     fill="${color}" opacity="0.85" stroke="rgba(0,0,0,0.3)" stroke-width="1">
                     <title>${cat.name}: ${cat.percentage}%</title></path>`;

            cumAngle += angle;
        });

        // 중앙 텍스트
        svg += `<text x="${cx}" y="${cy - 5}" text-anchor="middle" fill="white" font-size="11" font-weight="600">향 계열</text>`;
        svg += `<text x="${cx}" y="${cy + 12}" text-anchor="middle" fill="rgba(255,255,255,0.6)" font-size="10">비율</text>`;
        svg += '</svg>';

        // 범례
        let legend = '<div class="chart-legend">';
        categoryBreakdown.forEach(cat => {
            const color = colors[cat.name] || '#888';
            const nameMap = {
                floral: '플로럴', citrus: '시트러스', woody: '우디', spicy: '스파이시', fruity: '프루티',
                amber: '앰버', musk: '머스크', gourmand: '구르망', aquatic: '아쿠아', green: '그린',
                aromatic: '아로마틱', balsamic: '발사믹', animalic: '애니멀릭', chypre: '시프레', synthetic: '합성'
            };
            legend += `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${nameMap[cat.name] || cat.name} ${cat.percentage}%</div>`;
        });
        legend += '</div>';

        container.innerHTML = svg + legend;
    }

    // 레이더 차트
    static renderRadar(analysis, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const size = 280, cx = size / 2, cy = size / 2, maxR = 100;
        const metrics = [
            { label: '조화도', value: analysis.harmony },
            { label: '밸런스', value: analysis.balance },
            { label: '복합성', value: analysis.complexity },
            { label: '지속력', value: analysis.longevity.score },
            { label: '확산도', value: analysis.sillage.score }
        ];
        const n = metrics.length;
        const angleStep = (2 * Math.PI) / n;

        let svg = `<svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px">`;

        // 배경 그리드
        for (let level = 1; level <= 4; level++) {
            const lr = (maxR * level) / 4;
            let points = '';
            for (let i = 0; i < n; i++) {
                const a = -Math.PI / 2 + i * angleStep;
                points += `${cx + lr * Math.cos(a)},${cy + lr * Math.sin(a)} `;
            }
            svg += `<polygon points="${points}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`;
        }

        // 축선
        for (let i = 0; i < n; i++) {
            const a = -Math.PI / 2 + i * angleStep;
            svg += `<line x1="${cx}" y1="${cy}" x2="${cx + maxR * Math.cos(a)}" y2="${cy + maxR * Math.sin(a)}" 
                     stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
        }

        // 데이터 영역
        let dataPoints = '';
        for (let i = 0; i < n; i++) {
            const a = -Math.PI / 2 + i * angleStep;
            const r = (metrics[i].value / 100) * maxR;
            dataPoints += `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)} `;
        }
        svg += `<polygon points="${dataPoints}" fill="rgba(120,200,255,0.2)" stroke="rgba(120,200,255,0.8)" stroke-width="2"/>`;

        // 점 & 라벨
        for (let i = 0; i < n; i++) {
            const a = -Math.PI / 2 + i * angleStep;
            const r = (metrics[i].value / 100) * maxR;
            const dx = cx + r * Math.cos(a), dy = cy + r * Math.sin(a);
            svg += `<circle cx="${dx}" cy="${dy}" r="4" fill="rgba(120,200,255,0.9)"/>`;

            const lx = cx + (maxR + 20) * Math.cos(a), ly = cy + (maxR + 20) * Math.sin(a);
            svg += `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" 
                     fill="rgba(255,255,255,0.8)" font-size="11">${metrics[i].label}</text>`;
            svg += `<text x="${lx}" y="${ly + 14}" text-anchor="middle" fill="rgba(120,200,255,0.9)" 
                     font-size="10" font-weight="600">${metrics[i].value}</text>`;
        }

        svg += '</svg>';
        container.innerHTML = svg;
    }

    // 타임라인 시각화
    static renderTimeline(timeline, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const colors = ['#FFD93D', '#FF6B9D', '#9B7DFF'];
        let html = '<div class="timeline">';
        timeline.forEach((phase, idx) => {
            html += `<div class="timeline-phase" style="--phase-color:${colors[idx]}">
                <div class="phase-header">
                    <span class="phase-dot" style="background:${colors[idx]}"></span>
                    <span class="phase-name">${phase.name}</span>
                </div>
                <div class="phase-notes">${phase.notes.join(', ') || '—'}</div>
            </div>`;
        });
        html += '</div>';
        container.innerHTML = html;
    }
}

export default Visualizer;
