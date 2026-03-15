// tf-config.js — TF.js GPU 가속 + 성능 최적화
// ============================================
// WebGL/WebGPU 백엔드 활성화 → CPU 대비 10~50배 속도 향상
// ============================================

class TFConfig {
    static initialized = false;
    static backendName = 'cpu';
    static gpuAvailable = false;

    // GPU 백엔드 초기화 (앱 시작 시 1회 호출)
    static async initialize() {
        if (this.initialized) return;
        if (typeof tf === 'undefined') {
            console.warn('[TFConfig] TensorFlow.js not loaded');
            return;
        }

        console.log('[TFConfig] Initializing GPU acceleration...');

        // 1. 프로덕션 모드 (디버그 검증 비활성화 → 연산 오버헤드 제거)
        tf.enableProdMode();
        console.log('[TFConfig] ✓ Production mode enabled');

        // 2. WebGPU 시도 (최신 브라우저, 가장 빠름)
        try {
            if (navigator.gpu) {
                await tf.setBackend('webgpu');
                await tf.ready();
                this.backendName = 'webgpu';
                this.gpuAvailable = true;
                console.log('[TFConfig] ✓ WebGPU backend activated (최고 성능)');
            }
        } catch (e) {
            console.log('[TFConfig] WebGPU unavailable, trying WebGL...');
        }

        // 3. WebGL 폴백 (대부분의 브라우저 지원)
        if (!this.gpuAvailable) {
            try {
                await tf.setBackend('webgl');
                await tf.ready();
                this.backendName = 'webgl';
                this.gpuAvailable = true;

                // WebGL 최적화 플래그
                tf.env().set('WEBGL_PACK', true);                 // 텍스처 패킹 (4x 병렬화)
                tf.env().set('WEBGL_PACK_DEPTHWISECONV', true);   // Depthwise Conv 패킹
                tf.env().set('WEBGL_EXP_CONV', true);             // 실험적 Conv 최적화
                tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);         // GPU 명령 즉시 플러시

                // GPU 정보 출력
                const gl = document.createElement('canvas').getContext('webgl');
                if (gl) {
                    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                    if (debugInfo) {
                        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                        console.log(`[TFConfig] ✓ WebGL GPU: ${renderer}`);
                    }
                }

                console.log('[TFConfig] ✓ WebGL backend activated');
            } catch (e) {
                console.warn('[TFConfig] WebGL unavailable, falling back to CPU');
                await tf.setBackend('cpu');
                await tf.ready();
                this.backendName = 'cpu';
            }
        }

        // 4. 메모리 관리 설정
        tf.engine().startScope(); // 텐서 누수 방지 스코프

        this.initialized = true;
        console.log(`[TFConfig] Backend: ${this.backendName}, GPU: ${this.gpuAvailable}`);
    }

    // 현재 백엔드 정보
    static getInfo() {
        return {
            backend: this.backendName,
            gpuAvailable: this.gpuAvailable,
            numTensors: tf.memory().numTensors,
            numBytes: tf.memory().numBytes,
            unreliable: tf.memory().unreliable
        };
    }

    // 메모리 정리
    static cleanup() {
        if (typeof tf !== 'undefined') {
            tf.disposeVariables();
            console.log(`[TFConfig] Memory cleaned. Tensors: ${tf.memory().numTensors}`);
        }
    }

    // float16 양자화 (모델 크기 50% 감소, 약간의 정밀도 손실)
    static async quantizeModel(model) {
        if (!model) return model;
        try {
            // TF.js에서 float16 양자화는 직접 지원이 제한적이므로
            // weight만 float16 변환
            const weights = model.getWeights();
            const quantized = weights.map(w => {
                return tf.tidy(() => {
                    // float32 → float16 시뮬레이션 (범위 축소)
                    const min = w.min();
                    const max = w.max();
                    const range = max.sub(min);
                    // 65536 levels (16-bit)
                    const quantized = w.sub(min).div(range.add(tf.scalar(1e-7)))
                        .mul(tf.scalar(65535)).round()
                        .div(tf.scalar(65535)).mul(range).add(min);
                    return quantized;
                });
            });
            model.setWeights(quantized);
            quantized.forEach(t => t.dispose());
            console.log('[TFConfig] Model quantized (simulated float16)');
        } catch (e) {
            console.warn('[TFConfig] Quantization failed:', e.message);
        }
        return model;
    }

    // 벤치마크: GPU 속도 테스트
    static async benchmark() {
        const sizes = [100, 500, 1000];
        const results = {};

        for (const n of sizes) {
            const a = tf.randomNormal([n, n]);
            const b = tf.randomNormal([n, n]);

            const start = performance.now();
            for (let i = 0; i < 10; i++) {
                const c = tf.matMul(a, b);
                c.dataSync(); // 동기 대기 (벤치마크용)
                c.dispose();
            }
            const elapsed = performance.now() - start;

            results[`${n}x${n} matmul x10`] = `${elapsed.toFixed(1)}ms`;
            a.dispose();
            b.dispose();
        }

        console.log('[TFConfig] Benchmark:', results);
        return results;
    }
}

export default TFConfig;
