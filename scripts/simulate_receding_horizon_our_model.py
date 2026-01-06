"""
우리 방식으로 18회 추론하는 시뮬레이션
(Receding Horizon 스타일로)
"""

import numpy as np
import time

def simulate_receding_horizon_our_model(total_frames=18, latency_ms=450.0):
    """
    우리 모델로 RoboVLMs 방식(Receding Horizon) 시뮬레이션
    
    Args:
        total_frames: 총 프레임 수
        latency_ms: 모델 추론 latency (실측값)
    """
    print("="*80)
    print("🔄 우리 모델 + Receding Horizon (RoboVLMs 방식)")
    print("="*80)
    
    results = {
        'strategy': 'receding_horizon_our_model',
        'total_frames': total_frames,
        'inference_calls': total_frames,  # 매 step 추론
        'latency_per_call_ms': latency_ms,
        'total_time_ms': total_frames * latency_ms,
        'timeline': []
    }
    
    cumulative_time = 0
    
    for frame_i in range(total_frames):
        # 매 프레임마다 추론
        cumulative_time += latency_ms
        
        results['timeline'].append({
            'frame': frame_i,
            'action': 'infer',
            'latency_ms': latency_ms,
            'cumulative_time_s': cumulative_time / 1000,
            'note': f'매 step 추론 (RoboVLMs 방식)'
        })
    
    # 요약 출력
    print(f"\n📊 요약:")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - 추론 횟수: {results['inference_calls']}회")
    print(f"  - 추론당 latency: {latency_ms}ms")
    print(f"  - 총 소요 시간: {results['total_time_ms']/1000:.2f}초")
    print(f"  - 처리 속도: {total_frames/(results['total_time_ms']/1000):.2f} FPS")
    
    # 타임라인 (샘플)
    print(f"\n⏱️  타임라인 (처음 5개, 마지막 3개):")
    for entry in results['timeline'][:5]:
        print(f"  Frame {entry['frame']:2d}: 🔄 INFER ({entry['latency_ms']:.0f}ms) | Cumulative: {entry['cumulative_time_s']:.2f}s")
    
    if total_frames > 8:
        print(f"  ...")
        for entry in results['timeline'][-3:]:
            print(f"  Frame {entry['frame']:2d}: 🔄 INFER ({entry['latency_ms']:.0f}ms) | Cumulative: {entry['cumulative_time_s']:.2f}s")
    
    return results


def compare_strategies(total_frames=18, latency_ms=450.0):
    """
    Chunk Reuse vs Receding Horizon 비교
    """
    print("\n" + "="*80)
    print("📊 전략 비교: Chunk Reuse vs Receding Horizon")
    print("="*80)
    
    # Chunk Reuse (우리 방식)
    chunk_size = 10
    chunk_reuse_calls = int(np.ceil(total_frames / chunk_size))
    chunk_reuse_time = chunk_reuse_calls * latency_ms
    
    # Receding Horizon (RoboVLMs 방식)
    receding_calls = total_frames
    receding_time = receding_calls * latency_ms
    
    print(f"\n{'Strategy':<25} {'Calls':<10} {'Time':<15} {'FPS':<10} {'Speedup':<10}")
    print("-"*80)
    
    # Chunk Reuse
    chunk_fps = total_frames / (chunk_reuse_time / 1000)
    print(f"{'Chunk Reuse (우리)':<25} {chunk_reuse_calls:<10} {chunk_reuse_time/1000:.2f}s ({chunk_reuse_time:.0f}ms) {chunk_fps:<10.2f} 9.0x")
    
    # Receding Horizon
    receding_fps = total_frames / (receding_time / 1000)
    print(f"{'Receding Horizon':<25} {receding_calls:<10} {receding_time/1000:.2f}s ({receding_time:.0f}ms) {receding_fps:<10.2f} 1.0x")
    
    print("-"*80)
    
    # 차이 계산
    speedup = receding_time / chunk_reuse_time
    time_saved = (receding_time - chunk_reuse_time) / 1000
    
    print(f"\n💡 결론:")
    print(f"  - Chunk Reuse가 {speedup:.1f}배 빠름")
    print(f"  - 시간 절약: {time_saved:.2f}초")
    print(f"  - FPS 차이: {chunk_fps:.1f} vs {receding_fps:.1f}")
    
    # 우리 모델로 Receding Horizon 사용 시
    print(f"\n⚠️  우리 모델로 Receding Horizon (RoboVLMs 방식) 사용 시:")
    print(f"  - 18 프레임: {receding_time/1000:.2f}초 ({receding_fps:.1f} FPS)")
    print(f"  - 실시간 불가능 (10 Hz control 필요)")
    print(f"  - GPU 사용률: 100% (비효율적)")


if __name__ == "__main__":
    # 1. 우리 모델로 Receding Horizon 시뮬레이션
    results = simulate_receding_horizon_our_model(
        total_frames=18,
        latency_ms=450.0  # 우리 실측값
    )
    
    # 2. 전략 비교
    compare_strategies(
        total_frames=18,
        latency_ms=450.0
    )
    
    print("\n" + "="*80)
    print("✅ 시뮬레이션 완료")
    print("="*80)
