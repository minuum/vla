#!/usr/bin/env python3
"""
📏 거리별 특화 학습 결과 분석
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

def analyze_distance_aware_performance():
    """거리별 특화 학습 성과 분석"""
    print("🎯 거리별 특화 학습 결과 분석")
    print("=" * 60)
    
    # 최종 성능 (로그에서 추출)
    final_results = {
        'train_loss': 0.0937,
        'train_mae': 0.2855,
        'val_loss': 0.0816,
        'val_mae': 0.2602,
        'distance_mae': {
            'close': 0.2617,
            'medium': 0.2017,
            'far': 0.3373
        }
    }
    
    print("📊 최종 성능:")
    print(f"   훈련 Loss: {final_results['train_loss']:.4f}")
    print(f"   훈련 MAE: {final_results['train_mae']:.4f}")
    print(f"   검증 Loss: {final_results['val_loss']:.4f}")
    print(f"   검증 MAE: {final_results['val_mae']:.4f}")
    print()
    
    print("📏 거리별 성능:")
    for distance, mae in final_results['distance_mae'].items():
        print(f"   {distance.capitalize()}: MAE {mae:.4f}")
    print()
    
    # 이전 결과와 비교
    print("🔄 이전 결과와 비교:")
    print("   이전 (일반 증강): MAE ≈ 0.442")
    print(f"   현재 (거리별 특화): MAE = {final_results['val_mae']:.4f}")
    improvement = ((0.442 - final_results['val_mae']) / 0.442) * 100
    print(f"   개선도: {improvement:.1f}%")
    print()
    
    # 거리별 특화 효과 분석
    print("🎯 거리별 특화 효과:")
    best_distance = min(final_results['distance_mae'], key=final_results['distance_mae'].get)
    worst_distance = max(final_results['distance_mae'], key=final_results['distance_mae'].get)
    
    print(f"   최고 성능: {best_distance.capitalize()} (MAE: {final_results['distance_mae'][best_distance]:.4f})")
    print(f"   최저 성능: {worst_distance.capitalize()} (MAE: {final_results['distance_mae'][worst_distance]:.4f})")
    
    performance_gap = final_results['distance_mae'][worst_distance] - final_results['distance_mae'][best_distance]
    print(f"   성능 차이: {performance_gap:.4f}")
    print()
    
    # 실제 액션 예측 정확도 해석
    print("🎮 실제 액션 예측 정확도:")
    action_range = 2.3  # 실제 사용된 액션 범위
    
    for distance, mae in final_results['distance_mae'].items():
        accuracy = max(0, (1 - mae / action_range)) * 100
        print(f"   {distance.capitalize()}: {accuracy:.1f}% 정확도")
    
    overall_accuracy = max(0, (1 - final_results['val_mae'] / action_range)) * 100
    print(f"   전체 평균: {overall_accuracy:.1f}% 정확도")
    print()
    
    # 거리별 특화 전략 평가
    print("💡 거리별 특화 전략 평가:")
    print("   ✅ Medium 거리에서 최고 성능 (MAE: 0.2017)")
    print("   ✅ Close 거리에서 안정적 성능 (MAE: 0.2617)")
    print("   ⚠️  Far 거리에서 개선 필요 (MAE: 0.3373)")
    print()
    
    print("🎉 거리별 특화 증강이 성공적으로 작동!")
    print("   - Medium 거리: 표준 패턴 다양화 효과")
    print("   - Close 거리: 정밀 조작 강화 효과")
    print("   - Far 거리: 추가 개선 여지 있음")

def create_performance_visualization():
    """성능 시각화 생성"""
    # 거리별 MAE 데이터
    distances = ['Close', 'Medium', 'Far']
    mae_values = [0.2617, 0.2017, 0.3373]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 거리별 MAE 막대 그래프
    bars = ax1.bar(distances, mae_values, color=colors, alpha=0.8)
    ax1.set_ylabel('MAE')
    ax1.set_title('거리별 성능 비교')
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 정확도 파이 차트
    action_range = 2.3
    accuracies = [max(0, (1 - mae / action_range)) * 100 for mae in mae_values]
    
    wedges, texts, autotexts = ax2.pie(accuracies, labels=distances, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax2.set_title('거리별 예측 정확도')
    
    plt.tight_layout()
    plt.savefig('distance_aware_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_previous_approaches():
    """이전 접근법과 비교"""
    print("📈 접근법별 성능 비교")
    print("=" * 60)
    
    approaches = {
        '초기 (무증강)': 1.2,
        '일반 증강': 0.442,
        '거리별 특화': 0.2602
    }
    
    print("📊 MAE 비교:")
    for approach, mae in approaches.items():
        print(f"   {approach}: {mae:.4f}")
    
    print()
    print("🚀 개선 효과:")
    initial_mae = approaches['초기 (무증강)']
    for approach, mae in approaches.items():
        if approach != '초기 (무증강)':
            improvement = ((initial_mae - mae) / initial_mae) * 100
            print(f"   {approach}: {improvement:.1f}% 개선")
    
    print()
    print("💡 결론:")
    print("   - 거리별 특화 증강이 가장 효과적")
    print("   - 일반 증강 대비 41% 추가 개선")
    print("   - 초기 대비 78% 개선")

def main():
    """메인 분석"""
    analyze_distance_aware_performance()
    print("\n" + "=" * 60)
    compare_with_previous_approaches()
    
    # 시각화 생성
    create_performance_visualization()
    
    print("\n🎯 분석 완료!")
    print("📁 생성된 파일:")
    print("   - distance_aware_performance.png (성능 시각화)")

if __name__ == "__main__":
    main()
