'use client';
import { useEffect, useRef, useCallback, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import MobileMenu from '@/components/MobileMenu';
import { useStore } from '@/lib/store';

const RecipeVisualization = dynamic(() => import('@/components/RecipeVisualization'), { ssr: false });

// Mock recipe for demo (since backend may not be running)
const DEMO_RECIPE = {
    name: 'Midnight Vetiver',
    theme: '비 오는 새벽, 젖은 흙과 차가운 공기',
    scores: { hedonic: 0.843, longevity: '8.0h', synergy: 0.372, harmony: 0.822 },
    ingredients: [
        { name: 'Bulgarian Juniper', category: 'woody', note_type: 'top', concentration: 5.6, relevance: 0.94 },
        { name: 'Dihydromyrcenol', category: 'fresh', note_type: 'top', concentration: 4.9, relevance: 0.98 },
        { name: 'Ethyl Butyrate', category: 'fruity', note_type: 'top', concentration: 3.9, relevance: 0.83 },
        { name: 'Nerolidol', category: 'floral', note_type: 'middle', concentration: 8.1, relevance: 0.94 },
        { name: 'Eugenyl Acetate', category: 'spicy', note_type: 'middle', concentration: 8.0, relevance: 0.93 },
        { name: 'Terpineol', category: 'herbal', note_type: 'middle', concentration: 12.4, relevance: 0.83 },
        { name: 'Hawaiian Tuberose', category: 'floral', note_type: 'middle', concentration: 5.1, relevance: 0.95 },
        { name: 'Galaxolide', category: 'musk', note_type: 'base', concentration: 17.6, relevance: 0.98 },
        { name: 'Ethylene Brassylate', category: 'musk', note_type: 'base', concentration: 14.0, relevance: 0.97 },
        { name: 'Habanolide', category: 'musk', note_type: 'base', concentration: 10.2, relevance: 0.99 },
        { name: 'Clearwood', category: 'woody', note_type: 'base', concentration: 9.6, relevance: 0.94 },
        { name: 'Civetone', category: 'animalic', note_type: 'base', concentration: 5.6, relevance: 0.91 },
    ],
};

// Generate poetic name from theme
function generateName(theme) {
    const names = [
        'Midnight Petrichor', 'Velvet Ember', 'Liquid Dusk',
        'Silent Thunder', 'Amber Reverie', 'Phantom Wood',
        'Iron & Moss', 'Obsidian Rain', 'Cold Fire',
    ];
    return names[Math.floor(Math.random() * names.length)];
}

export default function BespokePage() {
    const { phase, setPhase, themeText, setThemeText, detectedKeywords, recipe, setRecipe, reset } = useStore();
    const textareaRef = useRef(null);
    const [brewProgress, setBrewProgress] = useState(0);

    // Auto-focus textarea
    useEffect(() => {
        if (phase === 'idle' && textareaRef.current) {
            textareaRef.current.focus();
        }
    }, [phase]);

    const handleBrew = useCallback(async () => {
        if (!themeText.trim()) return;
        setPhase('brewing');

        // Simulate brewing progress
        let p = 0;
        const interval = setInterval(() => {
            p += Math.random() * 15;
            if (p > 95) p = 95;
            setBrewProgress(p);
        }, 200);

        // Try real API first, fall back to demo
        try {
            const res = await fetch('http://localhost:8000/formulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ theme: themeText }),
            });
            if (res.ok) {
                const data = await res.json();
                clearInterval(interval);
                setBrewProgress(100);
                setTimeout(() => setRecipe(data), 500);
                return;
            }
        } catch (e) {
            // Backend not running — use demo
        }

        // Demo mode: wait 3 seconds then show demo recipe
        setTimeout(() => {
            clearInterval(interval);
            setBrewProgress(100);
            const demoResult = {
                ...DEMO_RECIPE,
                name: generateName(themeText),
                theme: themeText,
            };
            setTimeout(() => setRecipe(demoResult), 500);
        }, 2500);
    }, [themeText, setPhase, setRecipe]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleBrew();
        }
    };

    return (
        <>
            <div className="film-grain" />

            <nav className="nav">
                <Link href="/" className="nav-logo">Synesthesia</Link>
                <ul className="nav-links">
                    <li><Link href="/bespoke" className="nav-link" style={{ color: 'var(--text-dim)' }}>Bespoke</Link></li>
                    <li><Link href="/boutique" className="nav-link">Boutique</Link></li>
                    <li><Link href="/vault" className="nav-link">Vault</Link></li>
                </ul>
                <MobileMenu current="bespoke" />
            </nav>

            <main className="alchemy-page">
                {phase === 'idle' && (
                    <>
                        <div className="alchemy-header">
                            <span className="alchemy-label">The Alchemy Canvas</span>
                            <h2 style={{ fontStyle: 'italic', fontWeight: 300 }}>
                                당신의 기억을 향으로
                            </h2>
                        </div>

                        <div className="memory-container">
                            <textarea
                                ref={textareaRef}
                                className="memory-textarea"
                                placeholder="비 오는 새벽 4시, 젖은 흙냄새와 차가운 공기..."
                                value={themeText}
                                onChange={(e) => setThemeText(e.target.value)}
                                onKeyDown={handleKeyDown}
                                rows={5}
                            />
                            <p className="memory-hint">
                                감정, 기억, 분위기를 자유롭게 적어주세요. Enter로 조향 시작.
                            </p>

                            {detectedKeywords.length > 0 && (
                                <div className="memory-keywords">
                                    {detectedKeywords.map((kw, i) => (
                                        <span key={i} className="keyword-tag">{kw}</span>
                                    ))}
                                </div>
                            )}

                            <div className="alchemy-actions">
                                <button className="btn-primary" onClick={handleBrew} disabled={!themeText.trim()}>
                                    조향 시작
                                </button>
                            </div>
                        </div>
                    </>
                )}

                {phase === 'brewing' && (
                    <div className="brewing">
                        <p className="brewing-text">연금술사가 향을 빚고 있습니다...</p>
                        <div className="brewing-bar">
                            <div className="brewing-fill" style={{ width: `${brewProgress}%`, animation: 'none', transition: 'width 0.3s ease' }} />
                        </div>
                        <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '1rem', letterSpacing: '0.15em' }}>
                            {brewProgress < 30 ? '테마 분석 중...' : brewProgress < 60 ? '원료 조합 중...' : brewProgress < 90 ? '심사관 평가 중...' : '완성 중...'}
                        </p>
                    </div>
                )}

                {phase === 'result' && recipe && (
                    <div className="result-section">
                        <div className="result-header">
                            <h2 className="result-name">{recipe.name}</h2>
                            <p className="result-theme">{recipe.theme}</p>
                        </div>

                        {/* Scores */}
                        <div className="result-scores">
                            <div className="score-card">
                                <div className="score-value">{typeof recipe.scores?.hedonic === 'number' ? recipe.scores.hedonic.toFixed(2) : recipe.scores?.hedonic || '—'}</div>
                                <div className="score-label">쾌적도</div>
                            </div>
                            <div className="score-card">
                                <div className="score-value">{recipe.scores?.longevity || '—'}</div>
                                <div className="score-label">지속력</div>
                            </div>
                            <div className="score-card">
                                <div className="score-value">{typeof recipe.scores?.synergy === 'number' ? recipe.scores.synergy.toFixed(2) : recipe.scores?.synergy || '—'}</div>
                                <div className="score-label">시너지</div>
                            </div>
                            <div className="score-card">
                                <div className="score-value">{typeof recipe.scores?.harmony === 'number' ? recipe.scores.harmony.toFixed(2) : recipe.scores?.harmony || '—'}</div>
                                <div className="score-label">조화도</div>
                            </div>
                        </div>

                        {/* 3D Visualization */}
                        <div className="viz-3d">
                            <RecipeVisualization ingredients={recipe.ingredients} />
                        </div>

                        {/* Notes Table */}
                        <div className="notes-section">
                            {['top', 'middle', 'base'].map(noteType => {
                                const noteLabel = { top: '🔺 Top Notes', middle: '🔷 Middle Notes', base: '🔻 Base Notes' };
                                const items = recipe.ingredients.filter(i => i.note_type === noteType);
                                if (items.length === 0) return null;
                                return (
                                    <div key={noteType} className="note-group">
                                        <div className={`note-group-title ${noteType}`}>{noteLabel[noteType]}</div>
                                        {items.map((item, i) => (
                                            <div key={i} className="note-item">
                                                <span className="note-name">{item.name}</span>
                                                <span className="note-category">{item.category}</span>
                                                <span className="note-percent">{item.concentration.toFixed(1)}%</span>
                                                <div className="note-bar-wrapper">
                                                    <div
                                                        className={`note-bar ${noteType}`}
                                                        style={{ width: `${Math.min(100, item.concentration * 5)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                );
                            })}
                        </div>

                        {/* Retry */}
                        <div style={{ textAlign: 'center', marginTop: 'var(--space-lg)' }}>
                            <button className="btn-ghost" onClick={reset}>
                                ← 다시 조향하기
                            </button>
                        </div>
                    </div>
                )}
            </main>
        </>
    );
}
