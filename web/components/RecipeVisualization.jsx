'use client';
import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Float, Environment } from '@react-three/drei';
import * as THREE from 'three';

// Category → Color mapping
const CATEGORY_COLORS = {
    woody: '#8B6914',
    fresh: '#4ECDC4',
    fruity: '#FF6B6B',
    floral: '#C77DBA',
    spicy: '#E85D04',
    musk: '#9B8EC0',
    herbal: '#6B8E23',
    synthetic: '#7DA5C7',
    animalic: '#8B4513',
    aldehyde: '#DAA520',
    amber: '#D4A373',
    citrus: '#FFD93D',
    gourmand: '#CD853F',
    aquatic: '#4A90D9',
    aromatic: '#6B8E23',
};

function getColor(category) {
    return CATEGORY_COLORS[category] || '#888888';
}

// Note layering positions
const NOTE_Y = { top: 3, middle: 0, base: -3 };
const NOTE_LABEL = { top: 'TOP', middle: 'MIDDLE', base: 'BASE' };

function IngredientSphere({ ingredient, index, total, noteType, onClick, isHovered }) {
    const meshRef = useRef();
    const color = useMemo(() => getColor(ingredient.category), [ingredient.category]);
    const size = useMemo(() => 0.3 + ingredient.concentration * 0.04, [ingredient.concentration]);

    // Spread ingredients in a circle at their note level
    const noteItems = useMemo(() => total, [total]);
    const angle = (index / noteItems) * Math.PI * 2 + Math.PI / 4;
    const radius = 2.5 + index * 0.3;
    const baseY = NOTE_Y[noteType] || 0;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;

    useFrame((state) => {
        if (meshRef.current) {
            // Gentle float
            meshRef.current.position.y = baseY + Math.sin(state.clock.elapsedTime * 0.5 + index) * 0.15;
            // Pulse on hover
            if (isHovered) {
                const s = size * (1 + Math.sin(state.clock.elapsedTime * 3) * 0.1);
                meshRef.current.scale.setScalar(s / size);
            } else {
                meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
            }
        }
    });

    return (
        <group position={[x, baseY, z]}>
            <mesh
                ref={meshRef}
                onClick={onClick}
                onPointerEnter={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer'; }}
                onPointerLeave={() => { document.body.style.cursor = 'default'; }}
            >
                <sphereGeometry args={[size, 64, 64]} />
                <meshPhysicalMaterial
                    color={color}
                    transmission={0.92}
                    opacity={1}
                    metalness={0}
                    roughness={0.05}
                    ior={1.5}
                    thickness={size * 1.2}
                    envMapIntensity={1.5}
                    emissive={color}
                    emissiveIntensity={isHovered ? 0.3 : 0.05}
                    transparent={false}
                    attenuationColor={color}
                    attenuationDistance={0.5}
                />
            </mesh>
            {/* Label */}
            <Text
                position={[0, size + 0.3, 0]}
                fontSize={0.2}
                color="white"
                anchorX="center"
                anchorY="bottom"
                font="/fonts/Inter-Regular.woff"
                fillOpacity={isHovered ? 1 : 0.5}
            >
                {ingredient.name}
            </Text>
            <Text
                position={[0, size + 0.1, 0]}
                fontSize={0.14}
                color="#c9a96e"
                anchorX="center"
                anchorY="bottom"
                font="/fonts/Inter-Regular.woff"
                fillOpacity={isHovered ? 1 : 0.4}
            >
                {ingredient.concentration.toFixed(1)}%
            </Text>
        </group>
    );
}

function NoteLabel({ noteType }) {
    const y = NOTE_Y[noteType];
    return (
        <Text
            position={[-5, y, 0]}
            fontSize={0.3}
            color={noteType === 'top' ? '#d4a373' : noteType === 'middle' ? '#c77dba' : '#7da5c7'}
            anchorX="right"
            anchorY="middle"
            font="/fonts/Inter-Regular.woff"
            fillOpacity={0.6}
            letterSpacing={0.15}
        >
            {NOTE_LABEL[noteType]}
        </Text>
    );
}

function NotePlane({ noteType }) {
    const y = NOTE_Y[noteType];
    const color = noteType === 'top' ? '#d4a373' : noteType === 'middle' ? '#c77dba' : '#7da5c7';
    return (
        <mesh position={[0, y, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[1.8, 5, 64]} />
            <meshBasicMaterial
                color={color}
                transparent
                opacity={0.03}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
}

function Scene({ ingredients }) {
    const [hoveredIdx, setHoveredIdx] = useState(null);

    const grouped = useMemo(() => {
        const groups = { top: [], middle: [], base: [] };
        ingredients.forEach((ing) => {
            const nt = ing.note_type || 'middle';
            if (groups[nt]) groups[nt].push(ing);
        });
        return groups;
    }, [ingredients]);

    let globalIdx = 0;

    return (
        <>
            {/* Lighting — brighter for glass transmission */}
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1.5} color="#f0ede6" />
            <pointLight position={[-10, -5, 5]} intensity={0.6} color="#7da5c7" />
            <pointLight position={[0, 5, -10]} intensity={0.5} color="#c9a96e" />
            <pointLight position={[5, -5, 8]} intensity={0.3} color="#c77dba" />
            <Environment preset="night" />

            {/* Note layers */}
            {['top', 'middle', 'base'].map(nt => (
                <group key={nt}>
                    <NoteLabel noteType={nt} />
                    <NotePlane noteType={nt} />
                    {grouped[nt].map((ing, i) => {
                        const idx = globalIdx++;
                        return (
                            <Float key={i} speed={1} rotationIntensity={0} floatIntensity={0.3}>
                                <IngredientSphere
                                    ingredient={ing}
                                    index={i}
                                    total={grouped[nt].length}
                                    noteType={nt}
                                    isHovered={hoveredIdx === idx}
                                    onClick={() => setHoveredIdx(hoveredIdx === idx ? null : idx)}
                                />
                            </Float>
                        );
                    })}
                </group>
            ))}

            {/* Connecting lines between layers */}
            <mesh position={[0, 0, 0]}>
                <cylinderGeometry args={[0.005, 0.005, 6, 8]} />
                <meshBasicMaterial color="#c9a96e" transparent opacity={0.1} />
            </mesh>
        </>
    );
}

export default function RecipeVisualization({ ingredients = [] }) {
    if (!ingredients || ingredients.length === 0) {
        return (
            <div style={{
                width: '100%', height: '100%', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontStyle: 'italic'
            }}>
                레시피를 기다리고 있습니다...
            </div>
        );
    }

    return (
        <Canvas
            camera={{ position: [0, 2, 10], fov: 50 }}
            style={{ background: 'transparent' }}
            gl={{ alpha: true, antialias: true }}
        >
            <Scene ingredients={ingredients} />
            <OrbitControls
                enableDamping
                dampingFactor={0.05}
                rotateSpeed={0.5}
                enableZoom={true}
                minDistance={5}
                maxDistance={20}
                enablePan={false}
                autoRotate
                autoRotateSpeed={0.3}
            />
        </Canvas>
    );
}
