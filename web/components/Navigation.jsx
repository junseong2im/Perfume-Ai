'use client';
import Link from 'next/link';

export default function Navigation() {
    return (
        <nav className="nav">
            <Link href="/" className="nav-logo">Synesthesia</Link>
            <ul className="nav-links">
                <li><Link href="/bespoke" className="nav-link">Bespoke</Link></li>
                <li><Link href="/boutique" className="nav-link">Boutique</Link></li>
                <li><Link href="/vault" className="nav-link">Scent Vault</Link></li>
            </ul>
        </nav>
    );
}
