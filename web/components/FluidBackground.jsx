'use client';
import { useRef, useEffect } from 'react';
import { useStore } from '@/lib/store';

const VERTEX_SHADER = `
  attribute vec2 position;
  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  precision highp float;
  
  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec2 uMouse;
  uniform vec3 uColor1;
  uniform vec3 uColor2;
  uniform float uSpeed;
  
  // Simplex noise helpers
  vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
  
  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }
  
  // Fractional Brownian Motion
  float fbm(vec2 p) {
    float f = 0.0;
    float w = 0.5;
    for (int i = 0; i < 5; i++) {
      f += w * snoise(p);
      p *= 2.0;
      w *= 0.5;
    }
    return f;
  }
  
  void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec2 p = uv * 3.0;
    
    float t = uTime * uSpeed * 0.15;
    
    // Mouse influence — creates ripple trail
    vec2 mouseUV = uMouse / uResolution;
    float mouseDist = length(uv - mouseUV);
    float mouseInfluence = smoothstep(0.4, 0.0, mouseDist) * 0.3;
    
    // Layered noise for fluid motion
    float n1 = fbm(p + vec2(t * 0.7, t * 0.4));
    float n2 = fbm(p * 1.5 + vec2(-t * 0.3, t * 0.6) + n1 * 0.5);
    float n3 = fbm(p * 0.8 + vec2(t * 0.2, -t * 0.5) + n2 * 0.3);
    
    // Combine with mouse
    float noise = n1 * 0.4 + n2 * 0.35 + n3 * 0.25 + mouseInfluence;
    
    // Color mixing — lifted base for visible fluid texture
    vec3 col = mix(uColor1 * 1.6 + 0.03, uColor2 * 1.4 + 0.02, noise * 0.5 + 0.5);
    
    // Golden dust specks — tiny bright spots scattered in fluid
    float dust = smoothstep(0.65, 0.7, snoise(p * 4.0 + t * 0.3));
    col += vec3(0.25, 0.18, 0.06) * dust * 0.15;
    
    // Mouse trail — warm golden light bloom
    col += vec3(0.22, 0.16, 0.05) * mouseInfluence * 4.0;
    
    // Center glow — soft radial warmth
    float centerDist = length(uv - 0.5);
    float centerGlow = smoothstep(0.6, 0.0, centerDist) * 0.04;
    col += vec3(0.15, 0.1, 0.04) * centerGlow;
    
    // Softer vignette — less crushing
    float vig = 1.0 - smoothstep(0.5, 1.4, centerDist * 1.3);
    col *= vig;
    
    // Film grain
    float grain = (fract(sin(dot(uv * uTime, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) * 0.015;
    col += grain;
    
    gl_FragColor = vec4(col, 1.0);
  }
`;

function hslToRgb(h, s, l) {
  h /= 360; s /= 100; l /= 100;
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [r, g, b];
}

export default function FluidBackground() {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const programRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const targetColorRef = useRef([0.12, 0.08, 0.18]);
  const currentColorRef = useRef([0.12, 0.08, 0.18]);
  const targetColor2Ref = useRef([0.06, 0.10, 0.16]);
  const currentColor2Ref = useRef([0.06, 0.10, 0.16]);
  const targetSpeedRef = useRef(0.2);
  const currentSpeedRef = useRef(0.2);
  const startTimeRef = useRef(Date.now());

  const fluidMood = useStore(s => s.fluidMood);

  // Update target colors from store
  useEffect(() => {
    const { h, s, l, speed } = fluidMood;
    const rgb1 = hslToRgb(h, s, l);
    const rgb2 = hslToRgb((h + 40) % 360, Math.max(0, s - 15), Math.max(0, l - 5));
    targetColorRef.current = rgb1;
    targetColor2Ref.current = rgb2;
    targetSpeedRef.current = speed;
  }, [fluidMood]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!gl) return;
    glRef.current = gl;

    // Compile shaders
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, VERTEX_SHADER);
    gl.compileShader(vs);

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, FRAGMENT_SHADER);
    gl.compileShader(fs);

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.useProgram(program);
    programRef.current = program;

    // Full-screen quad
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Resize
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener('resize', resize);

    // Mouse
    const onMouse = (e) => {
      mouseRef.current = { x: e.clientX, y: canvas.height - e.clientY };
    };
    window.addEventListener('mousemove', onMouse);

    // Render loop
    let raf;
    const render = () => {
      const t = (Date.now() - startTimeRef.current) / 1000;

      // Smooth color transition
      const lerp = (a, b, t) => a + (b - a) * t;
      for (let i = 0; i < 3; i++) {
        currentColorRef.current[i] = lerp(currentColorRef.current[i], targetColorRef.current[i], 0.02);
        currentColor2Ref.current[i] = lerp(currentColor2Ref.current[i], targetColor2Ref.current[i], 0.02);
      }
      currentSpeedRef.current = lerp(currentSpeedRef.current, targetSpeedRef.current, 0.02);

      gl.uniform1f(gl.getUniformLocation(program, 'uTime'), t);
      gl.uniform2f(gl.getUniformLocation(program, 'uResolution'), canvas.width, canvas.height);
      gl.uniform2f(gl.getUniformLocation(program, 'uMouse'), mouseRef.current.x, mouseRef.current.y);
      gl.uniform3f(gl.getUniformLocation(program, 'uColor1'), ...currentColorRef.current);
      gl.uniform3f(gl.getUniformLocation(program, 'uColor2'), ...currentColor2Ref.current);
      gl.uniform1f(gl.getUniformLocation(program, 'uSpeed'), currentSpeedRef.current);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      raf = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMouse);
    };
  }, []);

  return <canvas ref={canvasRef} className="fluid-canvas" />;
}
