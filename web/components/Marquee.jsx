'use client';

export default function Marquee({ children, speed = 30, reverse = false, className = '' }) {
    const direction = reverse ? 'reverse' : 'normal';

    return (
        <div className={`marquee-wrapper ${className}`}>
            <div
                className="marquee-track"
                style={{
                    animationDuration: `${speed}s`,
                    animationDirection: direction,
                }}
            >
                <span className="marquee-content">{children}</span>
                <span className="marquee-content" aria-hidden="true">{children}</span>
                <span className="marquee-content" aria-hidden="true">{children}</span>
            </div>
        </div>
    );
}
