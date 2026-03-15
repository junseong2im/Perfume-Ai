import './globals.css';
import './pages.css';

export const metadata = {
  title: 'Synesthesia — AI Bespoke Perfumery',
  description: '보이지 않는 향을 시각화하는 AI 조향 경험. 당신의 기억과 감정으로 세상에 하나뿐인 향수를 만드세요. AI-crafted fragrances from your memories.',
  keywords: ['perfume', 'AI', 'bespoke', 'fragrance', 'synesthesia', '향수', '조향', 'scent'],
  openGraph: {
    title: 'Synesthesia — AI Bespoke Perfumery',
    description: 'The art of scent, reimagined through AI. Create your one-of-a-kind fragrance from memories and emotions.',
    siteName: 'Synesthesia',
    locale: 'ko_KR',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Synesthesia — AI Bespoke Perfumery',
    description: 'The art of scent, reimagined through AI.',
  },
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
