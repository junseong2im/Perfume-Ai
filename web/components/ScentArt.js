'use client';
import { useEffect, useRef, useMemo } from 'react';

/**
 * ScentArt — generative procedural art for each perfume.
 * Maps accords to HSL colors, notes determine shapes/flow.
 * Every perfume gets a unique, deterministic visual fingerprint.
 */

const ACCORD_COLORS = {
    warm_spicy: [15, 70, 45],
    sweet: [340, 55, 50],
    woody: [30, 40, 35],
    musky: [280, 25, 40],
    powdery: [320, 40, 55],
    amber: [35, 65, 45],
    floral: [330, 60, 55],
    citrus: [55, 70, 55],
    fresh_spicy: [160, 45, 45],
    aromatic: [140, 40, 45],
    green: [120, 45, 40],
    fruity: [20, 65, 50],
    leather: [20, 35, 30],
    smoky: [0, 10, 30],
    oud: [25, 50, 28],
    vanilla: [40, 55, 55],
    balsamic: [15, 45, 35],
    earthy: [45, 30, 30],
    aquatic: [200, 50, 50],
    fresh: [180, 50, 50],
    tobacco: [28, 40, 32],
    rose: [345, 60, 50],
    iris: [260, 35, 50],
    jasmine: [50, 50, 55],
    patchouli: [35, 35, 30],
    sandalwood: [38, 45, 42],
    lavender: [255, 40, 50],
    cinnamon: [18, 65, 40],
    vetiver: [110, 30, 35],
    bergamot: [65, 60, 50],
    saffron: [30, 75, 45],
    incense: [270, 30, 35],
    animalic: [10, 30, 28],
};

function hashStr(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
        h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    }
    return Math.abs(h);
}

function seededRandom(seed) {
    let s = seed;
    return () => {
        s = (s * 16807 + 0) % 2147483647;
        return (s - 1) / 2147483646;
    };
}

function getAccordColor(accord) {
    const key = accord.toLowerCase().replace(/[\s-]/g, '_');
    if (ACCORD_COLORS[key]) return ACCORD_COLORS[key];
    // fallback: derive from string hash
    const h = hashStr(accord);
    return [h % 360, 30 + (h % 30), 35 + (h % 20)];
}

export default function ScentArt({ accords = [], notes = [], name = '', size = 200, className = '' }) {
    const canvasRef = useRef(null);
    const seed = useMemo(() => hashStr(name + accords.join(',') + notes.join(',')), [name, accords, notes]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = size;
        const h = size;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        const rand = seededRandom(seed);
        const cx = w / 2;
        const cy = h / 2;

        // Background — very dark, almost matches page
        ctx.fillStyle = '#0d0d0d';
        ctx.fillRect(0, 0, w, h);

        // Get colors from accords
        const colors = accords.slice(0, 5).map(a => getAccordColor(a));
        if (colors.length === 0) colors.push([0, 0, 30]);

        // Layer 1: Radial gradient base from primary accord
        const [h1, s1, l1] = colors[0];
        const grad = ctx.createRadialGradient(
            cx + (rand() - 0.5) * w * 0.3,
            cy + (rand() - 0.5) * h * 0.3,
            0,
            cx, cy, w * 0.55
        );
        grad.addColorStop(0, `hsla(${h1}, ${s1}%, ${l1}%, 0.25)`);
        grad.addColorStop(0.6, `hsla(${h1}, ${s1 * 0.6}%, ${l1 * 0.5}%, 0.08)`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Layer 2: Flowing curves from notes
        const totalNotes = notes.length || 3;
        const curveCount = Math.min(totalNotes + 2, 8);

        for (let i = 0; i < curveCount; i++) {
            const colorIdx = i % colors.length;
            const [ch, cs, cl] = colors[colorIdx];

            ctx.beginPath();
            ctx.strokeStyle = `hsla(${ch}, ${cs}%, ${cl + 10}%, ${0.12 + rand() * 0.1})`;
            ctx.lineWidth = 0.5 + rand() * 1.5;

            const startX = rand() * w;
            const startY = rand() * h;
            ctx.moveTo(startX, startY);

            const segments = 3 + Math.floor(rand() * 4);
            for (let j = 0; j < segments; j++) {
                const cpx1 = rand() * w;
                const cpy1 = rand() * h;
                const cpx2 = rand() * w;
                const cpy2 = rand() * h;
                const endX = rand() * w;
                const endY = rand() * h;
                ctx.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, endX, endY);
            }
            ctx.stroke();
        }

        // Layer 3: Orbital circles — one per top accord
        for (let i = 0; i < Math.min(colors.length, 4); i++) {
            const [ch, cs, cl] = colors[i];
            const angle = (i / colors.length) * Math.PI * 2 + rand() * 0.5;
            const dist = w * 0.15 + rand() * w * 0.2;
            const ox = cx + Math.cos(angle) * dist;
            const oy = cy + Math.sin(angle) * dist;
            const radius = w * 0.08 + rand() * w * 0.12;

            const orbGrad = ctx.createRadialGradient(ox, oy, 0, ox, oy, radius);
            orbGrad.addColorStop(0, `hsla(${ch}, ${cs}%, ${cl + 15}%, 0.18)`);
            orbGrad.addColorStop(1, 'transparent');

            ctx.beginPath();
            ctx.fillStyle = orbGrad;
            ctx.arc(ox, oy, radius, 0, Math.PI * 2);
            ctx.fill();
        }

        // Layer 4: Fine dots — particle scatter
        const dotCount = 20 + Math.floor(rand() * 30);
        for (let i = 0; i < dotCount; i++) {
            const colorIdx = Math.floor(rand() * colors.length);
            const [ch, cs, cl] = colors[colorIdx];
            const dx = rand() * w;
            const dy = rand() * h;
            const dr = 0.3 + rand() * 1.2;

            ctx.beginPath();
            ctx.fillStyle = `hsla(${ch}, ${cs}%, ${cl + 20}%, ${0.15 + rand() * 0.2})`;
            ctx.arc(dx, dy, dr, 0, Math.PI * 2);
            ctx.fill();
        }

        // Layer 5: Thin ring — signature element
        const ringColor = colors[colors.length > 1 ? 1 : 0];
        const [rh, rs, rl] = ringColor;
        const ringRadius = w * 0.25 + rand() * w * 0.1;
        ctx.beginPath();
        ctx.strokeStyle = `hsla(${rh}, ${rs}%, ${rl + 5}%, 0.08)`;
        ctx.lineWidth = 0.5;
        ctx.arc(cx + (rand() - 0.5) * 20, cy + (rand() - 0.5) * 20, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

    }, [seed, size, accords, notes]);

    return (
        <canvas
            ref={canvasRef}
            className={`scent-art ${className}`}
            style={{ width: size, height: size }}
        />
    );
}
