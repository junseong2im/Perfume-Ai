'use client';
import { useRef, useEffect, memo } from 'react';

// ── Accord → Color mapping (HSL) ──
const accordColors = {
    woody: [30, 45, 35],
    floral: [330, 50, 50],
    fresh: [185, 40, 50],
    citrus: [48, 60, 55],
    oriental: [15, 55, 40],
    spicy: [12, 50, 38],
    sweet: [340, 45, 50],
    musky: [280, 25, 40],
    amber: [35, 60, 42],
    vanilla: [42, 50, 52],
    leather: [22, 35, 28],
    powdery: [300, 30, 55],
    aromatic: [155, 35, 40],
    fruity: [350, 55, 52],
    smoky: [0, 15, 28],
    green: [135, 35, 40],
    oud: [28, 50, 30],
    aquatic: [205, 45, 48],
    gourmand: [22, 55, 40],
    balsamic: [18, 40, 32],
    earthy: [42, 30, 30],
    tobacco: [32, 35, 30],
    rose: [345, 50, 48],
    warm: [38, 55, 48],
    saffron: [38, 60, 48],
    iris: [270, 35, 50],
    lavender: [262, 40, 50],
    jasmine: [58, 45, 52],
    patchouli: [28, 35, 30],
    sandalwood: [38, 35, 38],
    bergamot: [82, 45, 48],
    vetiver: [115, 25, 32],
};

function getAccordColor(accord) {
    const key = accord.toLowerCase();
    if (accordColors[key]) return accordColors[key];
    for (const [k, v] of Object.entries(accordColors)) {
        if (key.includes(k) || k.includes(key)) return v;
    }
    return [0, 10, 30];
}

// Seeded PRNG for deterministic galaxies
function seededRandom(seed) {
    let s = 0;
    for (let i = 0; i < seed.length; i++) {
        s = ((s << 5) - s + seed.charCodeAt(i)) | 0;
    }
    return function () {
        s = (s * 16807 + 0) % 2147483647;
        return (s & 0x7fffffff) / 0x7fffffff;
    };
}

function drawGalaxy(canvas, accords, name) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const rng = seededRandom(name || 'default');

    ctx.clearRect(0, 0, w, h);

    // Get 2-3 colors from accords
    const colors = (accords || []).slice(0, 3).map(getAccordColor);
    if (colors.length === 0) colors.push([0, 10, 30]);
    if (colors.length === 1) colors.push([colors[0][0] + 30, colors[0][1], colors[0][2] - 5]);

    const r = Math.min(cx, cy) * 0.85;

    // ── Background: faint radial glow ──
    const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    bgGrad.addColorStop(0, `hsla(${colors[0][0]}, ${colors[0][1]}%, ${colors[0][2]}%, 0.15)`);
    bgGrad.addColorStop(0.5, `hsla(${colors[1 % colors.length][0]}, ${colors[1 % colors.length][1]}%, ${colors[1 % colors.length][2] - 10}%, 0.06)`);
    bgGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // ── Nebula layers: soft elliptical clouds ──
    const numClouds = 5 + Math.floor(rng() * 4);
    for (let i = 0; i < numClouds; i++) {
        const color = colors[i % colors.length];
        const angle = rng() * Math.PI * 2;
        const dist = rng() * r * 0.5;
        const cloudX = cx + Math.cos(angle) * dist;
        const cloudY = cy + Math.sin(angle) * dist;
        const cloudR = r * (0.2 + rng() * 0.35);
        const opacity = 0.04 + rng() * 0.06;

        const grad = ctx.createRadialGradient(cloudX, cloudY, 0, cloudX, cloudY, cloudR);
        grad.addColorStop(0, `hsla(${color[0]}, ${color[1] + 10}%, ${color[2] + 15}%, ${opacity})`);
        grad.addColorStop(0.6, `hsla(${color[0] + 10}, ${color[1]}%, ${color[2]}%, ${opacity * 0.4})`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
    }

    // ── Spiral arms ──
    const numArms = 2 + Math.floor(rng() * 2);
    const armOffset = rng() * Math.PI * 2;
    const spiralTightness = 2.5 + rng() * 1.5;

    for (let arm = 0; arm < numArms; arm++) {
        const baseAngle = armOffset + (arm * Math.PI * 2) / numArms;
        const armColor = colors[arm % colors.length];

        // Draw particles along spiral arm
        const numParticles = 60 + Math.floor(rng() * 40);
        for (let j = 0; j < numParticles; j++) {
            const t = j / numParticles;
            const spiralAngle = baseAngle + t * spiralTightness;
            const spiralR = t * r * 0.9;

            // Add spread perpendicular to the spiral
            const spread = (rng() - 0.5) * r * 0.15 * (1 - t * 0.5);
            const perpAngle = spiralAngle + Math.PI / 2;

            const px = cx + Math.cos(spiralAngle) * spiralR + Math.cos(perpAngle) * spread;
            const py = cy + Math.sin(spiralAngle) * spiralR + Math.sin(perpAngle) * spread;

            const size = 0.3 + rng() * 1.2;
            const opacity = 0.08 + rng() * 0.15 * (1 - t);

            ctx.beginPath();
            ctx.arc(px, py, size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${armColor[0] + (rng() - 0.5) * 20}, ${armColor[1] + 15}%, ${armColor[2] + 20}%, ${opacity})`;
            ctx.fill();
        }
    }

    // ── Core glow ──
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.2);
    coreGrad.addColorStop(0, `hsla(${colors[0][0]}, ${colors[0][1]}%, ${Math.min(80, colors[0][2] + 30)}%, 0.25)`);
    coreGrad.addColorStop(0.5, `hsla(${colors[0][0]}, ${colors[0][1] - 10}%, ${colors[0][2] + 10}%, 0.08)`);
    coreGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = coreGrad;
    ctx.fillRect(0, 0, w, h);

    // ── Scattered stars ──
    const numStars = 30 + Math.floor(rng() * 25);
    for (let i = 0; i < numStars; i++) {
        const sx = rng() * w;
        const sy = rng() * h;
        const dist = Math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2);
        if (dist > r * 1.1) continue;

        const starSize = 0.2 + rng() * 0.6;
        const starOpacity = 0.15 + rng() * 0.4;

        ctx.beginPath();
        ctx.arc(sx, sy, starSize, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(0, 0%, 100%, ${starOpacity})`;
        ctx.fill();
    }

    // ── Bright core stars ──
    const numBright = 3 + Math.floor(rng() * 4);
    for (let i = 0; i < numBright; i++) {
        const angle = rng() * Math.PI * 2;
        const dist = rng() * r * 0.3;
        const bx = cx + Math.cos(angle) * dist;
        const by = cy + Math.sin(angle) * dist;

        // Star glow
        const starGlow = ctx.createRadialGradient(bx, by, 0, bx, by, 3);
        starGlow.addColorStop(0, `hsla(0, 0%, 100%, 0.5)`);
        starGlow.addColorStop(1, 'transparent');
        ctx.fillStyle = starGlow;
        ctx.fillRect(bx - 3, by - 3, 6, 6);

        ctx.beginPath();
        ctx.arc(bx, by, 0.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fill();
    }
}

const ScentGalaxy = memo(function ScentGalaxy({ accords, name, size = 140 }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        canvas.style.width = `${size}px`;
        canvas.style.height = `${size}px`;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        // Reset scale to logical pixels
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        ctx.scale(dpr, dpr);
        drawGalaxy(canvas, accords, name);
    }, [accords, name, size]);

    return (
        <canvas
            ref={canvasRef}
            className="scent-galaxy"
            style={{ width: size, height: size }}
        />
    );
});

ScentGalaxy.displayName = 'ScentGalaxy';
export default ScentGalaxy;
