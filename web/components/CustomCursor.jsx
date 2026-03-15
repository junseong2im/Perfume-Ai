'use client';
import { useEffect, useRef } from 'react';

export default function CustomCursor() {
    const cursorRef = useRef(null);
    const glowRef = useRef(null);
    const pos = useRef({ x: 0, y: 0 });
    const target = useRef({ x: 0, y: 0 });

    useEffect(() => {
        const cursor = cursorRef.current;
        const glow = glowRef.current;
        if (!cursor || !glow) return;

        const onMove = (e) => {
            target.current = { x: e.clientX, y: e.clientY };
        };

        const onEnterInteractive = () => {
            cursor.classList.add('cursor-hover');
            glow.classList.add('glow-hover');
        };
        const onLeaveInteractive = () => {
            cursor.classList.remove('cursor-hover');
            glow.classList.remove('glow-hover');
        };

        window.addEventListener('mousemove', onMove);

        // Observe interactive elements
        const observer = new MutationObserver(() => {
            document.querySelectorAll('a, button, .interactive, .cta-text').forEach(el => {
                el.removeEventListener('mouseenter', onEnterInteractive);
                el.removeEventListener('mouseleave', onLeaveInteractive);
                el.addEventListener('mouseenter', onEnterInteractive);
                el.addEventListener('mouseleave', onLeaveInteractive);
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Initial bind
        document.querySelectorAll('a, button, .interactive, .cta-text').forEach(el => {
            el.addEventListener('mouseenter', onEnterInteractive);
            el.addEventListener('mouseleave', onLeaveInteractive);
        });

        // Smooth follow loop
        let raf;
        const animate = () => {
            pos.current.x += (target.current.x - pos.current.x) * 0.12;
            pos.current.y += (target.current.y - pos.current.y) * 0.12;

            cursor.style.transform = `translate(${pos.current.x}px, ${pos.current.y}px)`;
            glow.style.transform = `translate(${pos.current.x}px, ${pos.current.y}px)`;

            raf = requestAnimationFrame(animate);
        };
        animate();

        return () => {
            window.removeEventListener('mousemove', onMove);
            cancelAnimationFrame(raf);
            observer.disconnect();
        };
    }, []);

    return (
        <>
            {/* Large ambient glow — the "light" */}
            <div ref={glowRef} className="cursor-glow" />
            {/* Small precise dot */}
            <div ref={cursorRef} className="cursor-dot" />
        </>
    );
}
