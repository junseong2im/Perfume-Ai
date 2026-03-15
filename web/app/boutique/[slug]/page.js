'use client';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import MobileMenu from '@/components/MobileMenu';
import perfumes from '@/data/famous_perfumes.json';

const ScentGalaxy = dynamic(() => import('@/components/ScentGalaxy'), { ssr: false });

function slugify(brand, name) {
    return `${brand}-${name}`.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

function formatNote(note) {
    return note.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/** Find similar perfumes by accord overlap */
function findSimilar(target, allPerfumes, count = 4) {
    const targetAccords = new Set(target.accords || []);
    if (targetAccords.size === 0) return [];

    return allPerfumes
        .filter(p => !(p.brand === target.brand && p.name === target.name))
        .map(p => {
            const pAccords = new Set(p.accords || []);
            let overlap = 0;
            for (const a of targetAccords) { if (pAccords.has(a)) overlap++; }
            const union = new Set([...targetAccords, ...pAccords]).size;
            return { ...p, similarity: union > 0 ? overlap / union : 0 };
        })
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, count);
}

// Accord mood taglines
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

export default function PerfumeDetailPage() {
    const params = useParams();
    const slug = params.slug;
    const [saved, setSaved] = useState(false);

    const perfume = perfumes.find(p => slugify(p.brand, p.name) === slug);
    const similar = perfume ? findSimilar(perfume, perfumes) : [];

    useEffect(() => {
        if (!perfume) return;
        const vault = JSON.parse(localStorage.getItem('synesthesia-vault') || '[]');
        setSaved(vault.some(v => v.slug === slug));
    }, [slug, perfume]);

    const toggleSave = () => {
        const vault = JSON.parse(localStorage.getItem('synesthesia-vault') || '[]');
        if (saved) {
            const updated = vault.filter(v => v.slug !== slug);
            localStorage.setItem('synesthesia-vault', JSON.stringify(updated));
            setSaved(false);
        } else {
            vault.push({ slug, name: perfume.name, brand: perfume.brand, year: perfume.year, rating: perfume.rating, style: perfume.style, accords: perfume.accords, concentration: perfume.concentration });
            localStorage.setItem('synesthesia-vault', JSON.stringify(vault));
            setSaved(true);
        }
    };

    if (!perfume) {
        return (
            <>
                <nav className="nav">
                    <Link href="/" className="nav-logo">Synesthesia</Link>
                    <ul className="nav-links">
                        <li><Link href="/boutique" className="nav-link">← Boutique</Link></li>
                    </ul>
                    <MobileMenu current="" />
                </nav>
                <main className="detail-page">
                    <div className="detail-empty">
                        <h2>Fragrance not found</h2>
                        <Link href="/boutique" className="cta-text">Browse Boutique →</Link>
                    </div>
                </main>
            </>
        );
    }

    const allNotes = [...(perfume.top_notes || []), ...(perfume.middle_notes || []), ...(perfume.base_notes || [])];
    const tagline = getTagline(perfume);

    return (
        <>
            <div className="film-grain" />

            <nav className="nav">
                <Link href="/" className="nav-logo">Synesthesia</Link>
                <ul className="nav-links">
                    <li><Link href="/boutique" className="nav-link">← Boutique</Link></li>
                    <li><Link href="/vault" className="nav-link">Vault</Link></li>
                </ul>
                <MobileMenu current="" />
            </nav>

            <main className="detail-page">
                {/* Hero — Galaxy + Title */}
                <header className="detail-hero">
                    <div className="detail-galaxy">
                        <ScentGalaxy
                            accords={perfume.accords || []}
                            name={`${perfume.brand}-${perfume.name}`}
                            size={220}
                        />
                    </div>

                    <p className="detail-brand-year">{perfume.brand} — {perfume.year}</p>
                    <h1 className="detail-name">{perfume.name}</h1>
                    {tagline && <p className="detail-tagline">{tagline}</p>}

                    <div className="detail-meta-pills">
                        <span className="detail-pill">{perfume.concentration}</span>
                        <span className="detail-pill">{perfume.style}</span>
                    </div>
                </header>

                {/* Scores — Minimal row */}
                <section className="detail-scores">
                    <div className="detail-stat">
                        <span className="detail-stat-value">{perfume.rating?.toFixed(1)}</span>
                        <span className="detail-stat-label">Rating</span>
                    </div>
                    <div className="detail-stat-divider" />
                    <div className="detail-stat">
                        <span className="detail-stat-value">{perfume.longevity}</span>
                        <span className="detail-stat-label">Longevity</span>
                    </div>
                    <div className="detail-stat-divider" />
                    <div className="detail-stat">
                        <span className="detail-stat-value">{perfume.sillage}</span>
                        <span className="detail-stat-label">Projection</span>
                    </div>
                </section>

                {/* Note Pyramid */}
                <section className="detail-notes">
                    <h2 className="detail-section-title">Note Pyramid</h2>

                    <div className="note-layer">
                        <div className="note-layer-label">
                            <span className="note-layer-num">01</span>
                            <span className="note-layer-name">Top</span>
                        </div>
                        <div className="note-tags">
                            {perfume.top_notes?.map((n, i) => (
                                <span key={i} className="note-tag">{formatNote(n)}</span>
                            ))}
                        </div>
                    </div>

                    <div className="note-layer">
                        <div className="note-layer-label">
                            <span className="note-layer-num">02</span>
                            <span className="note-layer-name">Heart</span>
                        </div>
                        <div className="note-tags">
                            {perfume.middle_notes?.map((n, i) => (
                                <span key={i} className="note-tag">{formatNote(n)}</span>
                            ))}
                        </div>
                    </div>

                    <div className="note-layer">
                        <div className="note-layer-label">
                            <span className="note-layer-num">03</span>
                            <span className="note-layer-name">Base</span>
                        </div>
                        <div className="note-tags">
                            {perfume.base_notes?.map((n, i) => (
                                <span key={i} className="note-tag">{formatNote(n)}</span>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Accords */}
                <section className="detail-accords">
                    <h2 className="detail-section-title">Accords</h2>
                    <div className="accord-tags">
                        {perfume.accords?.map((a, i) => (
                            <span key={i} className="accord-tag">{a}</span>
                        ))}
                    </div>
                </section>

                {/* Save */}
                <section className="detail-save">
                    <button className={`btn-vault ${saved ? 'saved' : ''}`} onClick={toggleSave}>
                        {saved ? '✓ In Your Vault' : '+ Add to Vault'}
                    </button>
                </section>

                {/* Similar */}
                {similar.length > 0 && (
                    <section className="detail-similar">
                        <h2 className="detail-section-title">You May Also Like</h2>
                        <div className="similar-grid">
                            {similar.map((s, i) => (
                                <Link
                                    href={`/boutique/${slugify(s.brand, s.name)}`}
                                    key={`${s.brand}-${s.name}-${i}`}
                                    className="similar-card"
                                >
                                    <div className="similar-galaxy">
                                        <ScentGalaxy
                                            accords={s.accords || []}
                                            name={`${s.brand}-${s.name}`}
                                            size={80}
                                        />
                                    </div>
                                    <h3 className="similar-name">{s.name}</h3>
                                    <p className="similar-brand">{s.brand}</p>
                                </Link>
                            ))}
                        </div>
                    </section>
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
