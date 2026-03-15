"""
SelfPlayRL 학습 + 학습된 모델로 고객 레시피 생성
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import biophysics_simulator as biophys

def on_progress(gen, total, avg, best):
    bar = "█" * int(gen/total*30) + "░" * (30 - int(gen/total*30))
    print(f"\r  [{bar}] {gen}/{total} | avg={avg:.3f} best={best:.3f}", end="", flush=True)

def main():
    print("=" * 60)
    print("  🧠 SelfPlayRL PPO 학습")
    print("=" * 60)
    
    # 에이전트 로딩
    rl = biophys.get_rl()
    print(f"\n  원료: {rl.n_ingredients}개")
    print(f"  디바이스: {rl.device}")
    print(f"  초기 generation: {rl.generation}")
    
    # 학습 전 레시피 생성 (비교용)
    print(f"\n  📊 학습 전 레시피:")
    before = rl.generate_recipe(n_ingredients=8)
    if before:
        ev = before['evaluation']
        print(f"     reward={ev['reward']:.3f} hedonic={ev['hedonic']['hedonic_score']:.2f} "
              f"longevity={ev['thermodynamics']['longevity_hours']}h")
    
    # PPO 학습 — 200세대 × 20 population
    GENS = 200
    POP = 20
    print(f"\n  🏋️ 학습 시작: {GENS}세대 × {POP}개/세대 = {GENS*POP}회 시뮬레이션")
    print(f"  (예상 소요: 5~10분)\n")
    
    start = time.time()
    result = rl.evolve(generations=GENS, population=POP, on_progress=on_progress)
    elapsed = time.time() - start
    
    print(f"\n\n  ✅ 학습 완료! ({elapsed:.0f}초)")
    print(f"     세대: {result['generations']}")
    print(f"     최고 점수: {result['best_score']:.4f}")
    print(f"     마지막 10세대 평균: {result['final_avg']:.4f}")
    
    # 학습 후 레시피 생성 (비교용)
    print(f"\n  📊 학습 후 레시피:")
    after = rl.generate_recipe(n_ingredients=8)
    if after:
        ev = after['evaluation']
        print(f"     reward={ev['reward']:.3f} hedonic={ev['hedonic']['hedonic_score']:.2f} "
              f"longevity={ev['thermodynamics']['longevity_hours']}h")
        print(f"     원료:")
        for ing in after['ingredients']:
            name = ing.get('name_ko', '') or ing.get('category', '')
            print(f"       • {name:20s} [{ing['category']:8s}] {ing['percentage']:.1f}%")
    
    # 학습 히스토리 저장
    history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'rl_training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generations': result['generations'],
            'best_score': result['best_score'],
            'final_avg': result['final_avg'],
            'elapsed_seconds': round(elapsed, 1),
            'history': rl.history,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 학습 히스토리 저장: {history_path}")
    
    # 이제 고객 요청 3개 테스트
    print(f"\n{'=' * 60}")
    print(f"  🧴 학습된 모델로 고객 레시피 생성")
    print(f"{'=' * 60}")
    
    from generate_custom_recipe import generate_for_customer
    
    demos = [
        "여름에 어울리는 시원하고 상쾌한 향수",
        "따뜻하고 달콤한 겨울 밤 향수",
        "고급스러운 남성용 우디 향수",
    ]
    for demo in demos:
        generate_for_customer(demo, n_results=1, n_candidates=15)
        print()

if __name__ == '__main__':
    main()
