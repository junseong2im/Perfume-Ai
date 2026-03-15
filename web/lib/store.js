import { create } from 'zustand';

// Keyword → Mood color mapping
const MOOD_COLORS = {
    // Korean keywords
    '비': { h: 220, s: 60, l: 25, speed: 0.3 },
    '바다': { h: 200, s: 70, l: 30, speed: 0.5 },
    '꽃': { h: 330, s: 50, l: 35, speed: 0.4 },
    '불': { h: 10, s: 80, l: 30, speed: 0.8 },
    '숲': { h: 140, s: 40, l: 20, speed: 0.3 },
    '밤': { h: 260, s: 50, l: 15, speed: 0.2 },
    '새벽': { h: 270, s: 30, l: 20, speed: 0.25 },
    '흙': { h: 30, s: 40, l: 20, speed: 0.3 },
    '우디': { h: 25, s: 50, l: 25, speed: 0.35 },
    '머스크': { h: 280, s: 30, l: 22, speed: 0.3 },
    '달콤': { h: 340, s: 55, l: 30, speed: 0.4 },
    '시원': { h: 190, s: 60, l: 30, speed: 0.6 },
    '따뜻': { h: 25, s: 60, l: 28, speed: 0.35 },
    '차가운': { h: 210, s: 50, l: 25, speed: 0.5 },
    '상쾌': { h: 160, s: 55, l: 28, speed: 0.7 },
    // English keywords
    'rain': { h: 220, s: 60, l: 25, speed: 0.3 },
    'ocean': { h: 200, s: 70, l: 30, speed: 0.5 },
    'flower': { h: 330, s: 50, l: 35, speed: 0.4 },
    'fire': { h: 10, s: 80, l: 30, speed: 0.8 },
    'forest': { h: 140, s: 40, l: 20, speed: 0.3 },
    'night': { h: 260, s: 50, l: 15, speed: 0.2 },
    'dawn': { h: 270, s: 30, l: 20, speed: 0.25 },
    'sweet': { h: 340, s: 55, l: 30, speed: 0.4 },
    'fresh': { h: 160, s: 55, l: 28, speed: 0.7 },
    'warm': { h: 25, s: 60, l: 28, speed: 0.35 },
    'cold': { h: 210, s: 50, l: 25, speed: 0.5 },
    'woody': { h: 25, s: 50, l: 25, speed: 0.35 },
    'musk': { h: 280, s: 30, l: 22, speed: 0.3 },
    'leather': { h: 15, s: 45, l: 20, speed: 0.3 },
    'spicy': { h: 5, s: 65, l: 28, speed: 0.5 },
};

const DEFAULT_FLUID = { h: 250, s: 30, l: 12, speed: 0.2 };

export const useStore = create((set, get) => ({
    // Phase: idle, inputting, brewing, result
    phase: 'idle',
    setPhase: (phase) => set({ phase }),

    // Theme text
    themeText: '',
    setThemeText: (text) => {
        const keywords = extractKeywords(text);
        const fluidMood = computeFluidMood(keywords);
        set({ themeText: text, detectedKeywords: keywords, fluidMood });
    },

    // Detected keywords
    detectedKeywords: [],

    // Fluid background mood
    fluidMood: DEFAULT_FLUID,

    // Recipe result
    recipe: null,
    setRecipe: (recipe) => set({ recipe, phase: 'result' }),

    // Reset
    reset: () => set({
        phase: 'idle',
        themeText: '',
        detectedKeywords: [],
        fluidMood: DEFAULT_FLUID,
        recipe: null,
    }),
}));

function extractKeywords(text) {
    const lower = text.toLowerCase();
    const found = [];
    for (const keyword of Object.keys(MOOD_COLORS)) {
        if (lower.includes(keyword)) {
            found.push(keyword);
        }
    }
    return found;
}

function computeFluidMood(keywords) {
    if (keywords.length === 0) return DEFAULT_FLUID;

    let totalH = 0, totalS = 0, totalL = 0, totalSpeed = 0;
    for (const kw of keywords) {
        const mood = MOOD_COLORS[kw];
        if (mood) {
            totalH += mood.h;
            totalS += mood.s;
            totalL += mood.l;
            totalSpeed += mood.speed;
        }
    }
    const n = keywords.length;
    return {
        h: Math.round(totalH / n),
        s: Math.round(totalS / n),
        l: Math.round(totalL / n),
        speed: totalSpeed / n,
    };
}
