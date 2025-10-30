"""
액션 데이터 분석 및 MAE 성능 해석
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.mobile_dataset import MobileVLADataset

def analyze_action_distribution():
    """액션 데이터의 분포를 분석합니다."""
    
    # 데이터셋 로드
    dataset = MobileVLADataset(
        data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/",
        sequence_length=18,
        image_size=(224, 224),
        normalize_actions=True
    )
    
    # 모든 액션 데이터 수집
    all_actions = []
    
    print("📊 액션 데이터 수집 중...")
    for i in range(len(dataset)):
        sample = dataset[i]
        actions = sample['actions']  # [seq_len, action_dim]
        # 2D 액션만 사용 (linear_x, linear_y)
        actions_2d = actions[:, :2]  # [seq_len, 2]
        all_actions.append(actions_2d)
        
        if (i + 1) % 10 == 0:
            print(f"   - {i + 1}/{len(dataset)} 샘플 처리 완료")
    
    # 모든 액션을 하나로 합치기
    all_actions = np.concatenate(all_actions, axis=0)  # [total_frames, 2]
    
    print(f"\n📊 액션 데이터 분석:")
    print(f"   - 총 프레임 수: {len(all_actions):,}")
    print(f"   - 액션 차원: {all_actions.shape[1]} (linear_x, linear_y)")
    
    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(f"   - linear_x 범위: [{all_actions[:, 0].min():.4f}, {all_actions[:, 0].max():.4f}]")
    print(f"   - linear_y 범위: [{all_actions[:, 1].min():.4f}, {all_actions[:, 1].max():.4f}]")
    print(f"   - linear_x 평균: {all_actions[:, 0].mean():.4f}")
    print(f"   - linear_y 평균: {all_actions[:, 1].mean():.4f}")
    print(f"   - linear_x 표준편차: {all_actions[:, 0].std():.4f}")
    print(f"   - linear_y 표준편차: {all_actions[:, 1].std():.4f}")
    
    # 액션 크기 분석
    action_magnitudes = np.sqrt(all_actions[:, 0]**2 + all_actions[:, 1]**2)
    print(f"\n🎯 액션 크기 분석:")
    print(f"   - 평균 액션 크기: {action_magnitudes.mean():.4f}")
    print(f"   - 최대 액션 크기: {action_magnitudes.max():.4f}")
    print(f"   - 액션 크기 표준편차: {action_magnitudes.std():.4f}")
    
    # MAE 성능 해석
    mae_values = [0.212, 0.222]
    
    print(f"\n🎯 MAE 성능 해석:")
    for mae in mae_values:
        print(f"\n   MAE {mae}:")
        
        # 액션 크기 대비 오차 비율
        avg_magnitude = action_magnitudes.mean()
        error_ratio = mae / avg_magnitude * 100
        print(f"     - 평균 액션 크기 대비 오차: {error_ratio:.1f}%")
        
        # 표준편차 대비 오차 비율
        std_magnitude = action_magnitudes.std()
        error_ratio_std = mae / std_magnitude * 100
        print(f"     - 액션 크기 표준편차 대비 오차: {error_ratio_std:.1f}%")
        
        # 정확도 추정 (임계값 기반)
        thresholds = [0.1, 0.2, 0.3, 0.5]
        for threshold in thresholds:
            accurate_predictions = np.sum(action_magnitudes <= threshold + mae)
            accuracy = accurate_predictions / len(action_magnitudes) * 100
            print(f"     - {threshold:.1f} 이내 정확도: {accuracy:.1f}%")
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 액션 분포
    plt.subplot(2, 3, 1)
    plt.scatter(all_actions[:, 0], all_actions[:, 1], alpha=0.5, s=1)
    plt.xlabel('linear_x')
    plt.ylabel('linear_y')
    plt.title('액션 분포 (linear_x vs linear_y)')
    plt.grid(True)
    
    # 2. linear_x 히스토그램
    plt.subplot(2, 3, 2)
    plt.hist(all_actions[:, 0], bins=50, alpha=0.7, color='blue')
    plt.xlabel('linear_x')
    plt.ylabel('빈도')
    plt.title('linear_x 분포')
    plt.grid(True)
    
    # 3. linear_y 히스토그램
    plt.subplot(2, 3, 3)
    plt.hist(all_actions[:, 1], bins=50, alpha=0.7, color='red')
    plt.xlabel('linear_y')
    plt.ylabel('빈도')
    plt.title('linear_y 분포')
    plt.grid(True)
    
    # 4. 액션 크기 분포
    plt.subplot(2, 3, 4)
    plt.hist(action_magnitudes, bins=50, alpha=0.7, color='green')
    plt.xlabel('액션 크기')
    plt.ylabel('빈도')
    plt.title('액션 크기 분포')
    plt.grid(True)
    
    # 5. MAE 성능 비교
    plt.subplot(2, 3, 5)
    models = ['Simple CLIP LSTM', 'Simple LSTM']
    colors = ['red', 'blue']
    bars = plt.bar(models, mae_values, color=colors, alpha=0.7)
    plt.ylabel('MAE')
    plt.title('모델 성능 비교')
    plt.ylim(0, 0.3)
    
    # 값 표시
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{mae:.3f}', ha='center', va='bottom')
    
    # 6. 정확도 비교
    plt.subplot(2, 3, 6)
    thresholds = [0.1, 0.2, 0.3, 0.5]
    clip_accuracies = []
    lstm_accuracies = []
    
    for threshold in thresholds:
        # Simple CLIP LSTM (MAE 0.212)
        accurate_clip = np.sum(action_magnitudes <= threshold + 0.212)
        accuracy_clip = accurate_clip / len(action_magnitudes) * 100
        clip_accuracies.append(accuracy_clip)
        
        # Simple LSTM (MAE 0.222)
        accurate_lstm = np.sum(action_magnitudes <= threshold + 0.222)
        accuracy_lstm = accurate_lstm / len(action_magnitudes) * 100
        lstm_accuracies.append(accuracy_lstm)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    plt.bar(x - width/2, clip_accuracies, width, label='Simple CLIP LSTM', alpha=0.7)
    plt.bar(x + width/2, lstm_accuracies, width, label='Simple LSTM', alpha=0.7)
    
    plt.xlabel('임계값')
    plt.ylabel('정확도 (%)')
    plt.title('임계값별 정확도 비교')
    plt.xticks(x, [f'{t:.1f}' for t in thresholds])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('action_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 저장
    results = {
        'total_frames': len(all_actions),
        'action_dimensions': all_actions.shape[1],
        'linear_x_range': [float(all_actions[:, 0].min()), float(all_actions[:, 0].max())],
        'linear_y_range': [float(all_actions[:, 1].min()), float(all_actions[:, 1].max())],
        'linear_x_mean': float(all_actions[:, 0].mean()),
        'linear_y_mean': float(all_actions[:, 1].mean()),
        'linear_x_std': float(all_actions[:, 0].std()),
        'linear_y_std': float(all_actions[:, 1].std()),
        'action_magnitude_mean': float(action_magnitudes.mean()),
        'action_magnitude_max': float(action_magnitudes.max()),
        'action_magnitude_std': float(action_magnitudes.std()),
        'mae_analysis': {
            '0.212': {
                'error_ratio_mean': float(mae_values[0] / action_magnitudes.mean() * 100),
                'error_ratio_std': float(mae_values[0] / action_magnitudes.std() * 100),
                'accuracies': {f'threshold_{t}': float(np.sum(action_magnitudes <= t + mae_values[0]) / len(action_magnitudes) * 100) for t in thresholds}
            },
            '0.222': {
                'error_ratio_mean': float(mae_values[1] / action_magnitudes.mean() * 100),
                'error_ratio_std': float(mae_values[1] / action_magnitudes.std() * 100),
                'accuracies': {f'threshold_{t}': float(np.sum(action_magnitudes <= t + mae_values[1]) / len(action_magnitudes) * 100) for t in thresholds}
            }
        }
    }
    
    with open('action_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 분석 결과가 'action_analysis_results.json'에 저장되었습니다.")
    print(f"📊 시각화가 'action_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    analyze_action_distribution()
