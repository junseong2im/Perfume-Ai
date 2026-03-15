'use client';
import { useEffect, useRef } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import MobileMenu from '@/components/MobileMenu';
import Marquee from '@/components/Marquee';

export default function LandingPage() {
  const mainRef = useRef(null);

  useEffect(() => {

    const init = async () => {
      try {
        const gsapModule = await import('gsap');
        const { ScrollTrigger } = await import('gsap/ScrollTrigger');
        const gsap = gsapModule.default || gsapModule.gsap || gsapModule;
        gsap.registerPlugin(ScrollTrigger);

        // Hero title entrance — use set() first to ensure visibility
        gsap.set('.hero-title-line', { opacity: 1 });
        gsap.fromTo('.hero-title-line',
          { y: 60, opacity: 0 },
          { y: 0, opacity: 1, stagger: 0.15, duration: 1.2, ease: 'power3.out', delay: 0.2 }
        );
        gsap.fromTo('.hero-marquee-container',
          { opacity: 0 },
          { opacity: 1, duration: 1, delay: 0.8 }
        );

        // ── Storytelling sections: pin & reveal ──
        const sections = gsap.utils.toArray('.story-section');
        sections.forEach((section) => {
          const content = section.querySelector('.story-content');
          const number = section.querySelector('.story-num');

          const tl = gsap.timeline({
            scrollTrigger: {
              trigger: section,
              start: 'top 80%',
              end: 'top 30%',
              scrub: 0.8,
            }
          });
          tl.fromTo(number, { x: -30, opacity: 0 }, { x: 0, opacity: 1, duration: 1 }, 0);
          tl.fromTo(content, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 1 }, 0.1);
        });

        // Manifesto word reveal
        gsap.utils.toArray('.m-word').forEach((word) => {
          gsap.fromTo(word,
            { opacity: 0.1 },
            {
              opacity: 1,
              duration: 0.6,
              scrollTrigger: {
                trigger: word,
                start: 'top 85%',
                end: 'top 55%',
                scrub: 0.5,
              }
            }
          );
        });

        // Final CTA
        gsap.fromTo('.final-cta h2',
          { y: 50, opacity: 0 },
          { y: 0, opacity: 1, duration: 1.2, scrollTrigger: { trigger: '.final-cta', start: 'top 75%' } }
        );
        gsap.fromTo('.final-cta .cta-text',
          { y: 30, opacity: 0 },
          { y: 0, opacity: 1, duration: 1, delay: 0.2, scrollTrigger: { trigger: '.final-cta', start: 'top 65%' } }
        );

        // Horizontal rules
        gsap.utils.toArray('.h-rule').forEach(el => {
          gsap.fromTo(el,
            { scaleX: 0 },
            { scaleX: 1, duration: 1.5, ease: 'power2.out', scrollTrigger: { trigger: el, start: 'top 90%' } }
          );
        });
      } catch (e) {
        console.warn('GSAP failed to load, showing content directly:', e);
        document.querySelectorAll('.hero-title-line, .hero-marquee-container, .story-num, .story-content, .m-word, .final-cta h2, .final-cta .cta-text').forEach(el => {
          el.style.opacity = '1';
          el.style.transform = 'none';
        });
      }
    };

    init();
  }, []);

  return (
    <>
      <div className="film-grain" />

      {/* ── Navigation ── */}
      <nav className="nav">
        <Link href="/" className="nav-logo">Synesthesia</Link>
        <ul className="nav-links">
          <li><Link href="/bespoke" className="nav-link">Bespoke</Link></li>
          <li><Link href="/boutique" className="nav-link">Boutique</Link></li>
          <li><Link href="/vault" className="nav-link">Vault</Link></li>
        </ul>
        <MobileMenu current="home" />
      </nav>

      {/* ── Main content: fades in after intro ── */}
      <main ref={mainRef} className="main-content visible">

        {/* ════════ HERO ════════ */}
        <section className="hero">
          <div className="hero-inner">
            <div className="hero-title-block">
              <div className="hero-overflow">
                <h1 className="hero-title-line hero-thin-kr">
                  보이지 않는 것이
                </h1>
              </div>
              <div className="hero-overflow">
                <h1 className="hero-title-line hero-title-large">
                  가장 오래 남는다
                </h1>
              </div>
              <div className="hero-overflow">
                <p className="hero-title-line hero-en-sub">The invisible lingers longest</p>
              </div>
            </div>

            <div className="hero-meta">
              <span className="hero-tag">Seoul · 2026</span>
              <span className="hero-tag">AI Bespoke Perfumery</span>
            </div>
          </div>

          {/* Marquee band */}
          <div className="hero-marquee-container">
            <Marquee speed={25}>
              <span className="marquee-item">INVISIBLE MADE VISIBLE</span>
              <span className="marquee-sep">✦</span>
              <span className="marquee-item">향을 보이는 예술로</span>
              <span className="marquee-sep">✦</span>
              <span className="marquee-item">AI BESPOKE PERFUMERY</span>
              <span className="marquee-sep">✦</span>
            </Marquee>
          </div>

          <div className="hero-scroll-cue">
            <div className="scroll-bar" />
          </div>
        </section>

        {/* ════════ MANIFESTO — 정(情) ════════ */}
        <section className="manifesto-section">
          <div className="manifesto-inner">
            <span className="section-label">情 · Philosophy</span>
            <p className="manifesto-text">
              {[
                '우리는', '향을', '만들지',
                '않습니다.', '당신이', '잊었던',
                '순간을', '되살립니다.',
                '할머니', '집', '마루의',
                '나무', '냄새,',
                '여름', '소나기', '뒤의',
                '흙', '향기,',
                '그', '사람의', '목덜미에서',
                '나던', '온기.',
                '인공지능은', '당신의',
                '정(情)을', '읽고,',
                '분자로', '번역합니다.'
              ].map((word, i) => (
                <span key={i} className="m-word">{word} </span>
              ))}
            </p>
          </div>
        </section>

        <div className="h-rule" />

        {/* ════════ PROCESS STORYTELLING ════════ */}
        <section className="story-container">
          <div className="story-header">
            <span className="section-label">風流 · The Process</span>
          </div>

          <div className="story-section">
            <span className="story-num">01</span>
            <div className="story-content">
              <h3 className="story-title">餘白 — The Memory</h3>
              <p className="story-desc">
                비 오는 새벽의 젖은 흙냄새,<br />
                할머니 댁 장작불의 잔향,<br />
                여름밤 풀 위의 이슬—
              </p>
              <p className="story-sub">당신이 적는 한 줄의 기억이 조향의 시작입니다.</p>
            </div>
          </div>

          <div className="story-section">
            <span className="story-num">02</span>
            <div className="story-content">
              <h3 className="story-title">共感 — The Alchemy</h3>
              <p className="story-desc">
                1,136개의 실제 원료.<br />
                Graph Neural Network가 분자를 읽고,<br />
                4명의 AI 심사관이 진화시킵니다.
              </p>
              <p className="story-sub">분자 수준에서 당신의 감정을 번역합니다.</p>
            </div>
          </div>

          <div className="story-section">
            <span className="story-num">03</span>
            <div className="story-content">
              <h3 className="story-title">恨 — The Scent</h3>
              <p className="story-desc">
                Top, Middle, Base—<br />
                세 개의 층위가 시간에 따라<br />
                피었다 사라지는 이야기를 남깁니다.
              </p>
              <p className="story-sub">이별 뒤에도 남는 것이 있습니다.</p>
            </div>
          </div>
        </section>

        <div className="h-rule" />

        {/* ════════ FINAL CTA ════════ */}
        <section className="final-cta">
          <h2>
            당신의 정(情)은<br />
            어떤 향기인가요
          </h2>
          <Link href="/bespoke" className="cta-text">
            Begin
            <span className="cta-arrow">→</span>
          </Link>
        </section>

        {/* ════════ FOOTER ════════ */}
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
