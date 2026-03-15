'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

/**
 * MobileMenu — fullscreen overlay menu for small screens.
 * Hamburger icon → slide-in overlay with staggered link reveals.
 */
export default function MobileMenu({ current = '' }) {
    const [open, setOpen] = useState(false);

    useEffect(() => {
        if (open) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => { document.body.style.overflow = ''; };
    }, [open]);

    return (
        <>
            <button
                className="mobile-burger"
                onClick={() => setOpen(!open)}
                aria-label="Menu"
            >
                <span className={`burger-line ${open ? 'open' : ''}`} />
                <span className={`burger-line ${open ? 'open' : ''}`} />
            </button>

            <div className={`mobile-overlay ${open ? 'active' : ''}`}>
                <nav className="mobile-nav-inner">
                    <Link
                        href="/"
                        className={`mobile-nav-item ${current === 'home' ? 'current' : ''}`}
                        onClick={() => setOpen(false)}
                    >
                        Home
                    </Link>
                    <Link
                        href="/bespoke"
                        className={`mobile-nav-item ${current === 'bespoke' ? 'current' : ''}`}
                        onClick={() => setOpen(false)}
                    >
                        Bespoke
                    </Link>
                    <Link
                        href="/boutique"
                        className={`mobile-nav-item ${current === 'boutique' ? 'current' : ''}`}
                        onClick={() => setOpen(false)}
                    >
                        Boutique
                    </Link>
                    <Link
                        href="/vault"
                        className={`mobile-nav-item ${current === 'vault' ? 'current' : ''}`}
                        onClick={() => setOpen(false)}
                    >
                        Vault
                    </Link>
                </nav>
                <span className="mobile-nav-brand">Synesthesia</span>
            </div>
        </>
    );
}
