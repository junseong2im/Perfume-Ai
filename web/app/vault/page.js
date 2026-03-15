'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import MobileMenu from '@/components/MobileMenu';

const ScentArt = dynamic(() => import('@/components/ScentArt'), { ssr: false });

export default function VaultPage() {
    const [saved, setSaved] = useState([]);
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        const vault = JSON.parse(localStorage.getItem('synesthesia-vault') || '[]');
        setSaved(vault);
        setLoaded(true);
    }, []);

    const removeItem = (slug) => {
        const updated = saved.filter(v => v.slug !== slug);
        localStorage.setItem('synesthesia-vault', JSON.stringify(updated));
        setSaved(updated);
    };

    return (
        <>
            <div className="film-grain" />

            <nav className="nav">
                <Link href="/" className="nav-logo">Synesthesia</Link>
                <ul className="nav-links">
                    <li><Link href="/bespoke" className="nav-link">Bespoke</Link></li>
                    <li><Link href="/boutique" className="nav-link">Boutique</Link></li>
                    <li><Link href="/vault" className="nav-link" style={{ color: 'var(--text-dim)' }}>Vault</Link></li>
                </ul>
                <MobileMenu current="vault" />
            </nav>

            <main className="vault-page">
                <header className="vault-header">
                    <span className="section-label">Your Collection</span>
                    <h1 className="vault-title">Vault</h1>
                    {loaded && <p className="vault-count">{saved.length} fragrance{saved.length !== 1 ? 's' : ''} saved</p>}
                </header>

                {loaded && saved.length === 0 && (
                    <div className="vault-empty">
                        <p className="vault-empty-text">
                            Your vault is empty.<br />
                            Every collection begins with a single scent.
                        </p>
                        <div className="vault-empty-actions">
                            <Link href="/boutique" className="cta-text">Explore Boutique →</Link>
                            <Link href="/bespoke" className="cta-text">Create Bespoke →</Link>
                        </div>
                    </div>
                )}

                {loaded && saved.length > 0 && (
                    <div className="vault-grid">
                        {saved.map((item, i) => (
                            <div key={item.slug || i} className="vault-card">
                                <Link href={`/boutique/${item.slug}`} className="vault-card-link">
                                    <div className="vault-art">
                                        <ScentArt
                                            accords={item.accords || []}
                                            notes={[]}
                                            name={`${item.brand}-${item.name}`}
                                            size={120}
                                        />
                                    </div>
                                    <h2 className="card-name">{item.name}</h2>
                                    <p className="card-brand">{item.brand} — {item.year}</p>
                                    <div className="card-stats">
                                        <span className="card-rating">★ {item.rating?.toFixed(1)}</span>
                                        <span className="card-meta">{item.concentration}</span>
                                    </div>
                                </Link>
                                <button className="vault-remove" onClick={() => removeItem(item.slug)}>Remove</button>
                            </div>
                        ))}
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
                        <span className="footer-copy">© 2026</span>
                    </div>
                </footer>
            </main>
        </>
    );
}
