'use client';
import { useEffect, useRef, useState } from 'react';

export default function IntroSequence({ onComplete }) {
    const [phase, setPhase] = useState('loading'); // loading → reveal → done
    const containerRef = useRef(null);

    useEffect(() => {
        const init = async () => {
            const gsapModule = await import('gsap');
            const gsap = gsapModule.default || gsapModule.gsap || gsapModule;

            const tl = gsap.timeline({
                onComplete: () => {
                    setPhase('done');
                    if (onComplete) onComplete();
                }
            });

            // Phase 1: Letters appear one by one
            tl.from('.intro-letter', {
                y: 80,
                opacity: 0,
                rotateX: -90,
                stagger: 0.06,
                duration: 0.8,
                ease: 'power3.out',
            })

                // Phase 2: Hold
                .to({}, { duration: 0.6 })

                // Phase 3: Counter increment
                .from('.intro-counter', {
                    textContent: 0,
                    duration: 1.2,
                    ease: 'power2.out',
                    snap: { textContent: 1 },
                    onUpdate: function () {
                        const el = document.querySelector('.intro-counter');
                        if (el) el.textContent = Math.round(this.targets()[0].textContent);
                    }
                }, '-=0.4')

                // Phase 4: Everything slides up and fades
                .to('.intro-content', {
                    y: -60,
                    opacity: 0,
                    duration: 0.8,
                    ease: 'power3.inOut',
                })
                .to('.intro-overlay', {
                    clipPath: 'inset(0 0 100% 0)',
                    duration: 1,
                    ease: 'power4.inOut',
                }, '-=0.4');
        };

        init();
    }, [onComplete]);

    if (phase === 'done') return null;

    return (
        <div ref={containerRef} className="intro-overlay">
            <div className="intro-content">
                <div className="intro-brand">
                    {'SYNESTHESIA'.split('').map((letter, i) => (
                        <span key={i} className="intro-letter">{letter}</span>
                    ))}
                </div>
                <div className="intro-sub">
                    <span className="intro-line" />
                    <span className="intro-counter">100</span>
                </div>
            </div>
        </div>
    );
}
