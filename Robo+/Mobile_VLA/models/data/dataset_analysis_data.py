#!/usr/bin/env python3
"""
📊 데이터셋 분석 및 태스크 특성 파악
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from PIL import Image
import cv2

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset

def analyze_dataset():
    """데이터셋 상세 분석"""
    print("🔍 데이터셋 상세 분석 시작...")
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"📊 전체 에피소드 수: {len(dataset)}")
    
    # 1. 액션 분포 분석
    print("\n📈 액션 분포 분석...")
    all_actions = []
    action_ranges = {'linear_x': [], 'linear_y': [], 'angular_z': []}
    episode_lengths = []
    
    for i in range(len(dataset)):
        episode = dataset[i]
        actions = episode['actions']
        
        if isinstance(actions, np.ndarray):
            all_actions.append(actions)
            episode_lengths.append(len(actions))
            
            # 각 축별 범위 계산
            action_ranges['linear_x'].extend([actions[:, 0].min(), actions[:, 0].max()])
            action_ranges['linear_y'].extend([actions[:, 1].min(), actions[:, 1].max()])
            action_ranges['angular_z'].extend([actions[:, 2].min(), actions[:, 2].max()])
    
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"   총 액션 프레임 수: {len(all_actions)}")
    print(f"   에피소드 길이 범위: {min(episode_lengths)} ~ {max(episode_lengths)}")
    print(f"   평균 에피소드 길이: {np.mean(episode_lengths):.1f}")
    
    # 2. 액션 통계
    print("\n📊 액션 통계:")
    for i, axis in enumerate(['linear_x', 'linear_y', 'angular_z']):
        values = all_actions[:, i]
        print(f"   {axis}:")
        print(f"     평균: {np.mean(values):.4f}")
        print(f"     표준편차: {np.std(values):.4f}")
        print(f"     범위: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"     중앙값: {np.median(values):.4f}")
        print(f"     제로 비율: {(values == 0).sum() / len(values) * 100:.1f}%")
    
    # 3. 액션 패턴 분석
    print("\n🔄 액션 패턴 분석...")
    
    # 연속된 같은 값 패턴 찾기
    consecutive_patterns = defaultdict(int)
    direction_changes = {'linear_x': 0, 'linear_y': 0, 'angular_z': 0}
    
    for episode_actions in all_actions.reshape(-1, len(all_actions) // len(dataset), 3):
        for i in range(1, len(episode_actions)):
            # 방향 변화 감지
            for j, axis in enumerate(['linear_x', 'linear_y', 'angular_z']):
                if (episode_actions[i-1, j] * episode_actions[i, j]) < 0:  # 부호 변화
                    direction_changes[axis] += 1
    
    print("   방향 변화 횟수:")
    for axis, count in direction_changes.items():
        print(f"     {axis}: {count}회")
    
    # 4. 이미지 특성 분석
    print("\n🖼️ 이미지 특성 분석...")
    
    # 샘플 이미지들 분석
    sample_images = []
    brightness_values = []
    contrast_values = []
    
    for i in range(min(10, len(dataset))):  # 처음 10개 에피소드만
        episode = dataset[i]
        images = episode['images']
        
        for j, img in enumerate(images[:5]):  # 각 에피소드의 처음 5장만
            if isinstance(img, Image.Image):
                img_array = np.array(img)
                sample_images.append(img_array)
                
                # 밝기 계산
                brightness = np.mean(img_array)
                brightness_values.append(brightness)
                
                # 대비 계산
                contrast = np.std(img_array)
                contrast_values.append(contrast)
    
    print(f"   분석한 이미지 수: {len(sample_images)}")
    print(f"   평균 밝기: {np.mean(brightness_values):.1f}")
    print(f"   평균 대비: {np.mean(contrast_values):.1f}")
    
    # 5. 태스크 특성 추론
    print("\n🎯 태스크 특성 추론...")
    
    # Z축이 모두 0인지 확인
    z_all_zero = np.all(all_actions[:, 2] == 0)
    print(f"   Z축 모두 0: {z_all_zero}")
    
    # 주요 이동 방향 분석
    x_dominant = np.abs(all_actions[:, 0]).mean() > np.abs(all_actions[:, 1]).mean()
    print(f"   X축 우세 이동: {x_dominant}")
    
    # 정지 상태 비율
    stationary_ratio = np.mean(np.all(np.abs(all_actions) < 0.01, axis=1))
    print(f"   정지 상태 비율: {stationary_ratio * 100:.1f}%")
    
    # 6. 시각화
    print("\n📊 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 액션 분포 히스토그램
    for i, axis in enumerate(['linear_x', 'linear_y', 'angular_z']):
        axes[0, i].hist(all_actions[:, i], bins=50, alpha=0.7)
        axes[0, i].set_title(f'{axis} 분포')
        axes[0, i].set_xlabel('값')
        axes[0, i].set_ylabel('빈도')
    
    # 에피소드 길이 분포
    axes[1, 0].hist(episode_lengths, bins=20, alpha=0.7)
    axes[1, 0].set_title('에피소드 길이 분포')
    axes[1, 0].set_xlabel('길이')
    axes[1, 0].set_ylabel('에피소드 수')
    
    # 액션 크기 분포
    action_magnitudes = np.linalg.norm(all_actions[:, :2], axis=1)  # linear_x, linear_y만
    axes[1, 1].hist(action_magnitudes, bins=50, alpha=0.7)
    axes[1, 1].set_title('이동 크기 분포')
    axes[1, 1].set_xlabel('크기')
    axes[1, 1].set_ylabel('빈도')
    
    # 시간에 따른 액션 변화 (샘플)
    sample_episode = all_actions[:episode_lengths[0]]
    axes[1, 2].plot(sample_episode[:, 0], label='linear_x', alpha=0.7)
    axes[1, 2].plot(sample_episode[:, 1], label='linear_y', alpha=0.7)
    axes[1, 2].plot(sample_episode[:, 2], label='angular_z', alpha=0.7)
    axes[1, 2].set_title('샘플 에피소드 액션 변화')
    axes[1, 2].set_xlabel('시간')
    axes[1, 2].set_ylabel('액션 값')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 시각화 저장됨: dataset_analysis.png")
    
    # 7. 증강 전략 제안
    print("\n💡 맞춤형 증강 전략 제안...")
    
    augmentation_strategy = {
        'task_type': 'mobile_navigation',
        'action_characteristics': {
            'z_axis_zero': z_all_zero,
            'x_dominant': x_dominant,
            'stationary_ratio': stationary_ratio,
            'avg_episode_length': np.mean(episode_lengths)
        },
        'recommended_augmentations': []
    }
    
    # Z축이 모두 0이면 2D 이동에 집중
    if z_all_zero:
        augmentation_strategy['recommended_augmentations'].append({
            'type': 'horizontal_flip',
            'reason': 'Z축이 0이므로 좌우 대칭이 물리적으로 타당',
            'probability': 0.5
        })
    
    # 정지 상태가 많으면 정지-이동 전환 증강
    if stationary_ratio > 0.3:
        augmentation_strategy['recommended_augmentations'].append({
            'type': 'start_stop_patterns',
            'reason': '정지 상태가 많으므로 시작-정지 패턴 증강',
            'probability': 0.3
        })
    
    # 에피소드 길이가 다양하면 길이 조정
    if np.std(episode_lengths) > np.mean(episode_lengths) * 0.5:
        augmentation_strategy['recommended_augmentations'].append({
            'type': 'temporal_scaling',
            'reason': '에피소드 길이가 다양하므로 시간적 스케일링',
            'probability': 0.4
        })
    
    # 기본 증강들
    augmentation_strategy['recommended_augmentations'].extend([
        {
            'type': 'action_noise',
            'reason': '센서 노이즈 시뮬레이션',
            'std': 0.005,
            'probability': 1.0
        },
        {
            'type': 'brightness_variation',
            'reason': '조명 조건 변화',
            'range': [0.8, 1.2],
            'probability': 0.3
        },
        {
            'type': 'temporal_jitter',
            'reason': '시간적 변동성 증가',
            'max_shift': 2,
            'probability': 0.2
        }
    ])
    
    # 결과 저장
    analysis_results = {
        'dataset_info': {
            'total_episodes': len(dataset),
            'total_frames': len(all_actions),
            'avg_episode_length': np.mean(episode_lengths),
            'episode_length_std': np.std(episode_lengths)
        },
        'action_statistics': {
            'linear_x': {
                'mean': float(np.mean(all_actions[:, 0])),
                'std': float(np.std(all_actions[:, 0])),
                'min': float(np.min(all_actions[:, 0])),
                'max': float(np.max(all_actions[:, 0])),
                'zero_ratio': float((all_actions[:, 0] == 0).sum() / len(all_actions))
            },
            'linear_y': {
                'mean': float(np.mean(all_actions[:, 1])),
                'std': float(np.std(all_actions[:, 1])),
                'min': float(np.min(all_actions[:, 1])),
                'max': float(np.max(all_actions[:, 1])),
                'zero_ratio': float((all_actions[:, 1] == 0).sum() / len(all_actions))
            },
            'angular_z': {
                'mean': float(np.mean(all_actions[:, 2])),
                'std': float(np.std(all_actions[:, 2])),
                'min': float(np.min(all_actions[:, 2])),
                'max': float(np.max(all_actions[:, 2])),
                'zero_ratio': float((all_actions[:, 2] == 0).sum() / len(all_actions))
            }
        },
        'task_characteristics': {
            'z_axis_zero': bool(z_all_zero),
            'x_dominant_movement': bool(x_dominant),
            'stationary_ratio': float(stationary_ratio),
            'direction_changes': {k: int(v) for k, v in direction_changes.items()}
        },
        'augmentation_strategy': augmentation_strategy
    }
    
    with open('dataset_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("   💾 분석 결과 저장됨: dataset_analysis_results.json")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_dataset()
    print("\n✅ 데이터셋 분석 완료!")
