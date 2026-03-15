'use client';
import { useState, useMemo, useEffect, useRef } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import MobileMenu from '@/components/MobileMenu';
import perfumes from '@/data/famous_perfumes.json';

const ScentGalaxy = dynamic(() => import('@/components/ScentGalaxy'), { ssr: false });

// ── Accord → Color mapping (HSL) ──
const accordColors = {
    woody: { h: 30, s: 35, l: 30 },
    floral: { h: 330, s: 40, l: 45 },
    fresh: { h: 180, s: 30, l: 40 },
    citrus: { h: 45, s: 50, l: 45 },
    oriental: { h: 15, s: 45, l: 35 },
    spicy: { h: 10, s: 40, l: 30 },
    sweet: { h: 340, s: 35, l: 40 },
    musky: { h: 280, s: 15, l: 35 },
    amber: { h: 35, s: 50, l: 35 },
    vanilla: { h: 40, s: 40, l: 45 },
    leather: { h: 20, s: 30, l: 25 },
    powdery: { h: 300, s: 20, l: 50 },
    aromatic: { h: 150, s: 25, l: 35 },
    fruity: { h: 350, s: 45, l: 45 },
    smoky: { h: 0, s: 10, l: 25 },
    green: { h: 130, s: 25, l: 35 },
    oud: { h: 25, s: 40, l: 25 },
    aquatic: { h: 200, s: 35, l: 40 },
    gourmand: { h: 20, s: 45, l: 35 },
    balsamic: { h: 15, s: 35, l: 30 },
    earthy: { h: 40, s: 25, l: 28 },
    tobacco: { h: 30, s: 30, l: 28 },
    rose: { h: 345, s: 40, l: 42 },
    warm: { h: 35, s: 45, l: 40 },
    saffron: { h: 35, s: 55, l: 40 },
    iris: { h: 270, s: 25, l: 45 },
    lavender: { h: 260, s: 30, l: 42 },
    jasmine: { h: 55, s: 35, l: 45 },
    patchouli: { h: 25, s: 30, l: 28 },
    sandalwood: { h: 35, s: 30, l: 35 },
    bergamot: { h: 80, s: 35, l: 42 },
    vetiver: { h: 110, s: 20, l: 30 },
};

function getOrbColors(accords) {
    if (!accords || accords.length === 0) return ['hsla(0,0%,25%,0.4)', 'hsla(0,0%,15%,0.15)'];

    const colors = accords.slice(0, 3).map(accord => {
        const key = accord.toLowerCase();
        let c = accordColors[key];
        if (!c) {
            for (const [k, v] of Object.entries(accordColors)) {
                if (key.includes(k) || k.includes(key)) { c = v; break; }
            }
        }
        if (!c) c = { h: 0, s: 10, l: 30 };
        return c;
    });

    if (colors.length === 1) {
        const c = colors[0];
        return [
            `hsla(${c.h}, ${c.s}%, ${c.l + 10}%, 0.45)`,
            `hsla(${c.h + 20}, ${c.s - 5}%, ${c.l - 5}%, 0.15)`,
        ];
    }

    return [
        `hsla(${colors[0].h}, ${colors[0].s}%, ${colors[0].l + 8}%, 0.45)`,
        `hsla(${colors[1].h}, ${colors[1].s}%, ${colors[1].l}%, 0.18)`,
    ];
}

// ── ScentOrb: Soft CSS gradient circle ──
function ScentOrb({ accords, size = 120 }) {
    const [c1, c2] = getOrbColors(accords);
    return (
        <div
            className="scent-orb"
            style={{
                width: size,
                height: size,
                background: `radial-gradient(circle at 40% 35%, ${c1}, ${c2}, transparent 72%)`,
            }}
        />
    );
}

// ── One-line emotional tagline from accords ──
const accordMoods = {
    woody: 'warm earth and ancient forests',
    floral: 'petals unfurling at dawn',
    fresh: 'morning dew on cool skin',
    citrus: 'sunlit groves and zest',
    oriental: 'silk and incense smoke',
    spicy: 'warmth against cold night air',
    sweet: 'honey in slow amber light',
    musky: 'the trace someone leaves behind',
    amber: 'molten gold at sunset',
    vanilla: 'comfort in every breath',
    leather: 'rain on old leather bindings',
    powdery: 'soft light through lace curtains',
    aromatic: 'wild herbs on a cliff',
    fruity: 'summer caught in glass',
    smoky: 'embers and whispered stories',
    green: 'crushed stems after a storm',
    oud: 'sacred wood and silent prayer',
    aquatic: 'salt spray and endless horizons',
    gourmand: 'warmth and sugar',
    balsamic: 'resinous warmth',
    earthy: 'forest floor after rainfall',
    tobacco: 'old books and quiet evenings',
    rose: 'a single bloom in an empty room',
    warm: 'golden hour on bare skin',
};

function getTagline(p) {
    if (!p.accords || p.accords.length === 0) return '';
    for (const accord of p.accords) {
        const key = accord.toLowerCase();
        if (accordMoods[key]) return accordMoods[key];
        for (const [k, v] of Object.entries(accordMoods)) {
            if (key.includes(k) || k.includes(key)) return v;
        }
    }
    return '';
}

// ── Data ──
const catalog = perfumes.reduce((acc, p) => {
    const key = `${p.brand}-${p.name}`;
    if (!acc.map.has(key)) {
        acc.map.set(key, true);
        acc.list.push(p);
    }
    return acc;
}, { map: new Map(), list: [] }).list;

const ALL_STYLES = [...new Set(catalog.map(p => p.style))].sort();
const ALL_BRANDS = [...new Set(catalog.map(p => p.brand))].sort();

function slugify(brand, name) {
    return `${brand}-${name}`.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

const PAGE_SIZE = 30;

export default function BoutiquePage() {
    const [style, setStyle] = useState('all');
    const [brand, setBrand] = useState('all');
    const [sort, setSort] = useState('rating');
    const [search, setSearch] = useState('');
    const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
    const sentinelRef = useRef(null);

    const filtered = useMemo(() => {
        let items = catalog;
        if (style !== 'all') items = items.filter(p => p.style === style);
        if (brand !== 'all') items = items.filter(p => p.brand === brand);
        if (search.trim()) {
            const q = search.toLowerCase();
            items = items.filter(p =>
                p.name.toLowerCase().includes(q) ||
                p.brand.toLowerCase().includes(q) ||
                (p.accords || []).some(a => a.toLowerCase().includes(q))
            );
        }
        items = [...items].sort((a, b) => {
            if (sort === 'rating') return (b.rating || 0) - (a.rating || 0);
            if (sort === 'year') return (b.year || 0) - (a.year || 0);
            if (sort === 'longevity') return (b.longevity || 0) - (a.longevity || 0);
            return a.name.localeCompare(b.name);
        });
        return items;
    }, [style, brand, sort, search]);

    useEffect(() => { setVisibleCount(PAGE_SIZE); }, [style, brand, sort, search]);

    useEffect(() => {
        const sentinel = sentinelRef.current;
        if (!sentinel) return;
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setVisibleCount(prev => Math.min(prev + PAGE_SIZE, filtered.length));
                }
            },
            { rootMargin: '200px' }
        );
        observer.observe(sentinel);
        return () => observer.disconnect();
    }, [filtered.length]);

    const visible = filtered.slice(0, visibleCount);

    return (
        <>
            <div className="film-grain" />

            <nav className="nav">
                <Link href="/" className="nav-logo">Synesthesia</Link>
                <ul className="nav-links">
                    <li><Link href="/bespoke" className="nav-link">Bespoke</Link></li>
                    <li><Link href="/boutique" className="nav-link" style={{ color: 'var(--text-dim)' }}>Boutique</Link></li>
                    <li><Link href="/vault" className="nav-link">Vault</Link></li>
                </ul>
                <MobileMenu current="boutique" />
            </nav>

            <main className="boutique-page">
                <header className="boutique-header">
                    <h1 className="boutique-title">Boutique</h1>
                </header>

                <div className="boutique-filters">
                    <input
                        type="text"
                        className="filter-search"
                        placeholder="Search..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                    <select className="filter-select" value={style} onChange={e => setStyle(e.target.value)}>
                        <option value="all">All Styles</option>
                        {ALL_STYLES.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <select className="filter-select" value={brand} onChange={e => setBrand(e.target.value)}>
                        <option value="all">All Brands</option>
                        {ALL_BRANDS.map(b => <option key={b} value={b}>{b}</option>)}
                    </select>
                    <select className="filter-select" value={sort} onChange={e => setSort(e.target.value)}>
                        <option value="rating">Rating</option>
                        <option value="year">Newest</option>
                        <option value="longevity">Longevity</option>
                        <option value="name">A → Z</option>
                    </select>
                </div>

                <div className="boutique-grid">
                    {filtered.length === 0 && (
                        <div className="boutique-empty">
                            <p className="boutique-empty-text">No fragrances found</p>
                        </div>
                    )}
                    {visible.map((p, i) => (
                        <CardReveal key={`${p.brand}-${p.name}-${i}`} index={i}>
                            <Link
                                href={`/boutique/${slugify(p.brand, p.name)}`}
                                className="perfume-card"
                            >
                                <div className="card-art-wrapper">
                                    <ScentGalaxy
                                        accords={p.accords || []}
                                        name={`${p.brand}-${p.name}`}
                                        size={130}
                                    />
                                </div>
                                <h2 className="card-name">{p.name}</h2>
                                <p className="card-brand">{p.brand}</p>
                                {getTagline(p) && (
                                    <p className="card-tagline">{getTagline(p)}</p>
                                )}
                            </Link>
                        </CardReveal>
                    ))}
                </div>

                {visibleCount < filtered.length && (
                    <div ref={sentinelRef} className="load-sentinel">
                        <span className="load-text">Loading more...</span>
                    </div>
                )}

                <footer className="footer">
                    <div className="footer-inner">
                        <span className="footer-brand">SYNESTHESIA</span>
                        <div className="footer-links">
                            <Link href="/bespoke">Bespoke</Link>
                            <Link href="/boutique">Boutique</Link>
                            <Link href="/vault">Vault</Link>
                        </div>
                        <span className="footer-copy">© 2026 · Seoul</span>
                    </div>
                </footer>
            </main>
        </>
    );
}


function CardReveal({ children, index }) {
    const ref = useRef(null);
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setVisible(true);
                    observer.unobserve(el);
                }
            },
            { threshold: 0.1, rootMargin: '50px' }
        );
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    return (
        <div
            ref={ref}
            className={`card-reveal ${visible ? 'revealed' : ''}`}
            style={{ transitionDelay: `${(index % 3) * 80}ms` }}
        >
            {children}
        </div>
    );
}
