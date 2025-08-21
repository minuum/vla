#!/usr/bin/env python3
"""
🎯 태스크 특성 기반 맞춤형 증강
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import cv2

class TaskSpecificAugmentation:
    """데이터셋 분석 결과를 바탕으로 한 맞춤형 증강"""
    
    def __init__(self):
        # 데이터셋 분석 결과 기반 설정
        self.task_characteristics = {
            'z_axis_zero': True,           # Z축이 모두 0
            'x_dominant': True,            # X축 우세 이동
            'y_zero_ratio': 0.536,         # Y축 53.6%가 0
            'episode_length': 18,          # 고정 길이
            'stationary_ratio': 0.056      # 정지 상태 5.6%
        }
        
        # 증강 확률 설정
        self.augmentation_probs = {
            'horizontal_flip': 0.5,        # 좌우 대칭 (2D 이동이므로 타당)
            'forward_backward_flip': 0.3,  # 전진/후진 뒤집기
            'action_noise': 0.8,           # 액션 노이즈
            'brightness_variation': 0.4,   # 밝기 변화
            'temporal_jitter': 0.2,        # 시간적 지터
            'speed_variation': 0.3,        # 속도 변화
            'start_stop_patterns': 0.2     # 시작-정지 패턴
        }
        
        # 노이즈 강도 설정
        self.noise_levels = {
            'action_noise_std': 0.005,     # 액션 노이즈
            'brightness_range': [0.8, 1.2], # 밝기 변화 범위
            'speed_scale_range': [0.8, 1.2] # 속도 스케일 범위
        }
        
        print("🎯 태스크 특성 기반 증강 초기화 완료")
        print(f"   Z축 0: {self.task_characteristics['z_axis_zero']}")
        print(f"   X축 우세: {self.task_characteristics['x_dominant']}")
        print(f"   Y축 0 비율: {self.task_characteristics['y_zero_ratio']:.1%}")
    
    def augment_episode(self, images, actions):
        """에피소드 전체 증강"""
        augmented_images = images.copy()
        augmented_actions = actions.copy()
        
        # 1. 좌우 대칭 (2D 이동이므로 물리적으로 타당)
        if random.random() < self.augmentation_probs['horizontal_flip']:
            augmented_images, augmented_actions = self._horizontal_flip(
                augmented_images, augmented_actions
            )
        
        # 2. 전진/후진 뒤집기 (X축 우세이므로 효과적)
        if random.random() < self.augmentation_probs['forward_backward_flip']:
            augmented_images, augmented_actions = self._forward_backward_flip(
                augmented_images, augmented_actions
            )
        
        # 3. 액션 노이즈 (센서 노이즈 시뮬레이션)
        if random.random() < self.augmentation_probs['action_noise']:
            augmented_actions = self._add_action_noise(augmented_actions)
        
        # 4. 밝기 변화 (조명 조건 변화)
        if random.random() < self.augmentation_probs['brightness_variation']:
            augmented_images = self._brightness_variation(augmented_images)
        
        # 5. 시간적 지터 (시간적 변동성)
        if random.random() < self.augmentation_probs['temporal_jitter']:
            augmented_images, augmented_actions = self._temporal_jitter(
                augmented_images, augmented_actions
            )
        
        # 6. 속도 변화 (다양한 속도로 이동)
        if random.random() < self.augmentation_probs['speed_variation']:
            augmented_actions = self._speed_variation(augmented_actions)
        
        # 7. 시작-정지 패턴 (정지 상태가 적으므로 학습)
        if random.random() < self.augmentation_probs['start_stop_patterns']:
            augmented_actions = self._start_stop_patterns(augmented_actions)
        
        return augmented_images, augmented_actions
    
    def _horizontal_flip(self, images, actions):
        """좌우 대칭 (Y축 부호 변경)"""
        flipped_images = []
        flipped_actions = actions.copy()
        
        for img in images:
            if isinstance(img, Image.Image):
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(img, np.ndarray):
                flipped_img = np.fliplr(img)
            else:
                flipped_img = img
            flipped_images.append(flipped_img)
        
        # Y축 액션 부호 변경 (좌우 대칭)
        flipped_actions[:, 1] = -flipped_actions[:, 1]
        
        return flipped_images, flipped_actions
    
    def _forward_backward_flip(self, images, actions):
        """전진/후진 뒤집기 (X축 부호 변경)"""
        flipped_images = list(reversed(images))
        flipped_actions = actions.copy()
        
        # X축 액션 부호 변경 (전진/후진 뒤집기)
        flipped_actions[:, 0] = -flipped_actions[:, 0]
        
        return flipped_images, flipped_actions
    
    def _add_action_noise(self, actions):
        """액션에 노이즈 추가"""
        noisy_actions = actions.copy()
        
        # X축 (주요 이동축)에 작은 노이즈
        x_noise = np.random.normal(0, self.noise_levels['action_noise_std'], actions.shape[0])
        noisy_actions[:, 0] += x_noise
        
        # Y축 (보조 이동축)에 더 작은 노이즈
        y_noise = np.random.normal(0, self.noise_levels['action_noise_std'] * 0.5, actions.shape[0])
        noisy_actions[:, 1] += y_noise
        
        # Z축은 0이므로 노이즈 추가하지 않음
        
        # 범위 제한
        noisy_actions = np.clip(noisy_actions, -1.15, 1.15)
        
        return noisy_actions
    
    def _brightness_variation(self, images):
        """밝기 변화"""
        brightness_factor = random.uniform(*self.noise_levels['brightness_range'])
        
        brightened_images = []
        for img in images:
            if isinstance(img, Image.Image):
                # PIL 이미지 밝기 조정
                enhancer = transforms.ColorJitter(brightness=brightness_factor)
                brightened_img = enhancer(img)
            elif isinstance(img, np.ndarray):
                # numpy 배열 밝기 조정
                brightened_img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
            else:
                brightened_img = img
            brightened_images.append(brightened_img)
        
        return brightened_images
    
    def _temporal_jitter(self, images, actions):
        """시간적 지터 (인접 프레임 순서 약간 변경)"""
        jittered_images = images.copy()
        jittered_actions = actions.copy()
        
        # 인접한 2-3개 프레임의 순서를 약간 섞기
        for i in range(1, len(images) - 1, 3):
            if random.random() < 0.3:  # 30% 확률로 순서 변경
                # 3개 프레임 순환
                jittered_images[i:i+3] = jittered_images[i+1:i+4] + [jittered_images[i]]
                jittered_actions[i:i+3] = jittered_actions[i+1:i+4] + [jittered_actions[i]]
        
        return jittered_images, jittered_actions
    
    def _speed_variation(self, actions):
        """속도 변화 (전체 시퀀스 속도 스케일링)"""
        speed_scale = random.uniform(*self.noise_levels['speed_scale_range'])
        
        scaled_actions = actions.copy()
        # X축 (주요 이동축)만 스케일링
        scaled_actions[:, 0] *= speed_scale
        
        # 범위 제한
        scaled_actions = np.clip(scaled_actions, -1.15, 1.15)
        
        return scaled_actions
    
    def _start_stop_patterns(self, actions):
        """시작-정지 패턴 (정지 상태 학습)"""
        pattern_actions = actions.copy()
        
        # 에피소드 시작 부분에 정지 패턴 추가
        if random.random() < 0.5:
            # 처음 2-3프레임을 정지 상태로
            stop_frames = random.randint(1, 3)
            pattern_actions[:stop_frames] = 0
        
        # 에피소드 중간에 짧은 정지 추가
        if random.random() < 0.3:
            mid_point = len(actions) // 2
            pattern_actions[mid_point] = 0
        
        return pattern_actions
    
    def create_augmented_dataset(self, original_dataset, augmentation_factor=3):
        """원본 데이터셋을 증강하여 새로운 데이터셋 생성"""
        print(f"🔄 데이터셋 증강 시작 (증강 배수: {augmentation_factor})")
        
        augmented_episodes = []
        
        for i, episode in enumerate(original_dataset):
            # 원본 에피소드 추가
            augmented_episodes.append(episode)
            
            # 증강된 에피소드들 추가
            for j in range(augmentation_factor):
                images = episode['images']
                actions = episode['actions']
                
                # 액션을 numpy 배열로 변환
                if isinstance(actions, list):
                    actions = np.array(actions)
                
                # 증강 적용
                aug_images, aug_actions = self.augment_episode(images, actions)
                
                # 증강된 에피소드 생성
                augmented_episode = {
                    'images': aug_images,
                    'actions': aug_actions,
                    'episode_id': f"{episode.get('episode_id', i)}_aug_{j}"
                }
                
                augmented_episodes.append(augmented_episode)
            
            if (i + 1) % 10 == 0:
                print(f"   진행률: {i+1}/{len(original_dataset)} 에피소드 처리 완료")
        
        print(f"✅ 증강 완료: {len(original_dataset)} → {len(augmented_episodes)} 에피소드")
        
        return augmented_episodes

def test_augmentation():
    """증강 테스트"""
    print("🧪 증강 테스트 시작...")
    
    # 테스트 데이터 생성
    test_images = [Image.new('RGB', (224, 224), color=(100, 150, 200)) for _ in range(5)]
    test_actions = np.array([
        [1.0, 0.0, 0.0],  # 전진
        [0.8, 0.2, 0.0],  # 전진 + 측면
        [0.5, 0.0, 0.0],  # 느린 전진
        [0.0, 0.5, 0.0],  # 측면 이동
        [0.0, 0.0, 0.0]   # 정지
    ])
    
    # 증강기 생성
    augmenter = TaskSpecificAugmentation()
    
    # 증강 테스트
    aug_images, aug_actions = augmenter.augment_episode(test_images, test_actions)
    
    print("📊 증강 결과:")
    print(f"   원본 액션:\n{test_actions}")
    print(f"   증강 액션:\n{aug_actions}")
    print(f"   이미지 개수: {len(aug_images)}")
    
    return augmenter

if __name__ == "__main__":
    test_augmentation()
