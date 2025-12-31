#!/usr/bin/env python3
"""
Chunk 성능 비교
Chunk 크기(5 vs 10)에 따른 API 호출 빈도 및 성능 분석
"""

import json
import sys
from pathlib import Path
from typing import Dict


def analyze_chunk_strategy(
    chunk_size: int,
    duration: float = 18.0,
    action_interval: float = 0.3
) -> Dict:
    """
    Chunk 전략 분석
    
    Args:
        chunk_size: action chunk 크기 (5 or 10)
        duration: 총 주행 시간 (초)
        action_interval: action 간격 (초, 기본 300ms)
    
    Returns:
        분석 결과 딕셔너리
    """
    
    # Chunk 지속 시간
    chunk_duration = chunk_size * action_interval
    
    # 총 호출 횟수
    total_calls = int(duration / chunk_duration) + 1
    
    # 호출 빈도 (Hz)
    call_frequency = 1.0 / chunk_duration
    
    # 총 action 수
    total_actions = int(duration / action_interval)
    
    # 사용되는 action과 폐기되는 action
    actions_used = total_calls * chunk_size
    actions_wasted = actions_used - total_actions
    efficiency = (total_actions / actions_used) * 100 if actions_used > 0 else 0
    
    return {
        "chunk_size": chunk_size,
        "action_interval_ms": int(action_interval * 1000),
        "chunk_duration_s": round(chunk_duration, 2),
        "total_calls": total_calls,
        "call_frequency_hz": round(call_frequency, 2),
        "total_duration_s": duration,
        "total_actions_needed": total_actions,
        "actions_generated": actions_used,
        "actions_wasted": actions_wasted,
        "efficiency_percent": round(efficiency, 1)
    }


def compare_strategies(duration=18.0, action_interval=0.3):
    """Chunk 5 vs 10 비교"""
    
    chunk5 = analyze_chunk_strategy(5, duration, action_interval)
    chunk10 = analyze_chunk_strategy(10, duration, action_interval)
    
    # 비교 지표
    call_ratio = chunk5['total_calls'] / chunk10['total_calls']
    
    comparison = {
        "chunk_5": chunk5,
        "chunk_10": chunk10,
        "comparison": {
            "call_ratio": round(call_ratio, 2),
            "call_reduction_percent": round((1 - 1/call_ratio) * 100, 1) if call_ratio > 1 else 0,
            "efficiency_difference": round(chunk10['efficiency_percent'] - chunk5['efficiency_percent'], 1)
        }
    }
    
    return comparison


def print_comparison(comparison):
    """비교 결과 출력"""
    
    chunk5 = comparison['chunk_5']
    chunk10 = comparison['chunk_10']
    comp = comparison['comparison']
    
    print("="*70)
    print("  Chunk 전략 성능 비교")
    print("="*70)
    print()
    
    print("📊 Chunk 5:")
    print(f"   Chunk 지속 시간: {chunk5['chunk_duration_s']}초")
    print(f"   호출 빈도: {chunk5['call_frequency_hz']} Hz")
    print(f"   총 호출 횟수: {chunk5['total_calls']}회")
    print(f"   생성 action: {chunk5['actions_generated']}개")
    print(f"   폐기 action: {chunk5['actions_wasted']}개")
    print(f"   효율성: {chunk5['efficiency_percent']}%")
    print()
    
    print("📊 Chunk 10:")
    print(f"   Chunk 지속 시간: {chunk10['chunk_duration_s']}초")
    print(f"   호출 빈도: {chunk10['call_frequency_hz']} Hz")
    print(f"   총 호출 횟수: {chunk10['total_calls']}회")
    print(f"   생성 action: {chunk10['actions_generated']}개")
    print(f"   폐기 action: {chunk10['actions_wasted']}개")
    print(f"   효율성: {chunk10['efficiency_percent']}%")
    print()
    
    print("🔄 비교:")
    print(f"   호출 비율: Chunk 5는 Chunk 10의 {comp['call_ratio']:.1f}배")
    print(f"   호출 감소: Chunk 10은 {comp['call_reduction_percent']:.1f}% 감소")
    print(f"   효율성 차이: {comp['efficiency_difference']:+.1f}%p")
    print()
    
    # 권장사항
    print("💡 권장사항:")
    if chunk10['total_calls'] < chunk5['total_calls']:
        print(f"   - Chunk 10은 호출을 {chunk10['total_calls']}회로 줄여 네트워크/API 부하 감소")
        print(f"   - 효율성은 {chunk10['efficiency_percent']:.1f}%로 약간 낮지만 허용 범위")
        print(f"   - 추론 간격이 {chunk10['chunk_duration_s']}초로 늘어나 자연스러움은 약간 감소 가능")
    
    print()
    print("="*70)


def save_comparison(comparison, output_path=None):
    """비교 결과 저장"""
    
    if output_path is None:
        output_path = Path("logs/chunk_performance_comparison.json")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"💾 결과 저장: {output_path}")
    return output_path


def main():
    """메인 함수"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Chunk 성능 비교')
    parser.add_argument('--duration', type=float, default=18.0, help='총 주행 시간 (초)')
    parser.add_argument('--interval', type=float, default=0.3, help='Action 간격 (초)')
    args = parser.parse_args()
    
    print("🚀 Chunk 전략 성능 비교 시작...")
    print(f"   주행 시간: {args.duration}초")
    print(f"   Action 간격: {args.interval}초")
    print()
    
    comparison = compare_strategies(args.duration, args.interval)
    
    print_comparison(comparison)
    
    save_comparison(comparison)
    
    print("✅ 비교 완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
