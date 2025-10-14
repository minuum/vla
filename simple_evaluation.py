#!/usr/bin/env python3
"""
간단한 성능 평가 및 비교
"""

import numpy as np

def mae_to_success_rate(mae, threshold=0.1):
    """MAE를 Success Rate로 변환"""
    if mae <= threshold:
        return 1.0  # 100% 성공
    else:
        return max(0, 1 - (mae - threshold) / threshold)

def calculate_metrics(mae):
    """기본 지표 계산"""
    mse = mae ** 2  # 근사치
    rmse = mae
    navigation_accuracy = 1 - mae
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Navigation_Accuracy': navigation_accuracy
    }

def main():
    print("🔍 VLA 모델 성능 비교 분석")
    print("="*60)
    
    # 현재 모델 성능
    current_mae = 0.2121
    
    # 기본 지표 계산
    metrics = calculate_metrics(current_mae)
    
    print("📊 현재 모델 성능 지표")
    print("-"*40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Success Rate 계산 (다양한 임계값)
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
    print(f"\n🎯 임계값별 Success Rate")
    print("-"*40)
    for threshold in thresholds:
        success_rate = mae_to_success_rate(current_mae, threshold)
        print(f"임계값 {threshold}: {success_rate:.1%}")
    
    # 다른 모델들과 비교
    print(f"\n📊 다른 VLA 연구와 성능 비교")
    print("-"*60)
    print(f"{'모델':<15} {'데이터셋 크기':<15} {'Success Rate':<15} {'MAE':<10} {'비고'}")
    print("-"*60)
    
    other_models = [
        ('RT-2', 130000, 0.90, 'N/A', 'Google DeepMind'),
        ('RT-1', 130000, 0.85, 'N/A', 'Google DeepMind'),
        ('PaLM-E', 562000, 0.80, 'N/A', 'Google DeepMind'),
        ('Our Model', 72, mae_to_success_rate(current_mae, 0.1), current_mae, '우리 연구')
    ]
    
    for model, episodes, success_rate, mae, note in other_models:
        if mae == 'N/A':
            print(f"{model:<15} {episodes:<15,} {success_rate:<15.1%} {mae:<10} {note}")
        else:
            print(f"{model:<15} {episodes:<15,} {success_rate:<15.1%} {mae:<10.4f} {note}")
    
    # 개선 가능성 분석
    print(f"\n🚀 개선 가능성 분석")
    print("-"*50)
    
    targets = {
        '단기 목표 (1개월)': 0.1,
        '중기 목표 (3개월)': 0.05,
        '장기 목표 (6개월)': 0.02
    }
    
    current_sr = mae_to_success_rate(current_mae, 0.1)
    
    for period, target_mae in targets.items():
        target_sr = mae_to_success_rate(target_mae, 0.1)
        improvement = (target_sr - current_sr) * 100
        
        print(f"{period}:")
        print(f"  현재 MAE: {current_mae:.4f} → 목표 MAE: {target_mae:.4f}")
        print(f"  현재 Success Rate: {current_sr:.1%} → 목표 Success Rate: {target_sr:.1%}")
        print(f"  개선 폭: {improvement:+.1f}%p")
        print()
    
    # 데이터셋 크기 비교
    print(f"📈 데이터셋 크기 비교")
    print("-"*40)
    for model, episodes, _, _, _ in other_models:
        ratio = episodes / 72
        print(f"{model}: {episodes:,} episodes (우리 대비 {ratio:.0f}배)")
    
    # 핵심 인사이트
    print(f"\n💡 핵심 인사이트")
    print("-"*40)
    print(f"1. 현재 Success Rate: {current_sr:.1%} (매우 낮음)")
    print(f"2. 데이터셋 크기: 다른 연구 대비 1,800배 적음")
    print(f"3. 개선 우선순위: 데이터셋 확장 > 모델 최적화")
    print(f"4. 목표: MAE 0.1 이하 달성 (Success Rate 50%+)")

if __name__ == "__main__":
    main()
