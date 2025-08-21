#!/usr/bin/env python3
"""
🔄 Mobile VLA 데이터 증강 스크립트

기존 72개 에피소드를 다양한 방법으로 증강하여 데이터셋을 확장합니다.
"""

import numpy as np
import h5py
import cv2
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import json
from datetime import datetime

class MobileVLADataAugmenter:
    """Mobile VLA 데이터 증강 클래스"""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "augmented"
        self.output_dir.mkdir(exist_ok=True)
        
        # 증강 파라미터
        self.augmentation_config = {
            'brightness': [0.7, 0.8, 0.9, 1.1, 1.2, 1.3],  # 밝기 변화
            'contrast': [0.8, 0.9, 1.1, 1.2],              # 대비 변화
            'saturation': [0.8, 0.9, 1.1, 1.2],           # 채도 변화
            'noise_levels': [0.01, 0.02, 0.03],           # 노이즈 레벨
            'blur_levels': [0.5, 1.0, 1.5],               # 블러 레벨
            'rotation_angles': [-5, -3, -1, 1, 3, 5],     # 회전 각도
            'crop_ratios': [0.9, 0.95, 1.05, 1.1],        # 크롭 비율
        }
        
        print(f"🔄 Mobile VLA 데이터 증강 초기화")
        print(f"   입력 디렉토리: {self.data_dir}")
        print(f"   출력 디렉토리: {self.output_dir}")
    
    def load_episode(self, file_path: Path) -> Dict:
        """에피소드 로드"""
        with h5py.File(file_path, 'r') as f:
            episode = {
                'images': f['images'][:],
                'actions': f['actions'][:],
                'scenario': file_path.stem.split('_')[2:4],  # 예: ['left', 'vertical']
                'episode_id': file_path.stem
            }
        return episode
    
    def save_episode(self, episode: Dict, filename: str):
        """증강된 에피소드 저장"""
        output_path = self.output_dir / filename
        
        with h5py.File(output_path, 'w') as f:
            # 이미지 데이터 생성
            f.create_dataset('images', data=episode['images'])
            
            # 액션 데이터 생성
            f.create_dataset('actions', data=episode['actions'])
            
            # 메타데이터 추가
            f.attrs['scenario'] = '_'.join(episode['scenario'])
            f.attrs['original_episode'] = episode['episode_id']
            f.attrs['augmentation_type'] = episode.get('aug_type', 'original')
            f.attrs['augmentation_params'] = json.dumps(episode.get('aug_params', {}))
    
    def augment_image(self, image: np.ndarray, aug_type: str, params: Dict) -> np.ndarray:
        """이미지 증강"""
        pil_image = Image.fromarray(image)
        
        if aug_type == 'brightness':
            factor = params['factor']
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)
            
        elif aug_type == 'contrast':
            factor = params['factor']
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)
            
        elif aug_type == 'saturation':
            factor = params['factor']
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)
            
        elif aug_type == 'noise':
            # 가우시안 노이즈 추가
            noise_level = params['level']
            image_array = np.array(pil_image, dtype=np.float32) / 255.0
            noise = np.random.normal(0, noise_level, image_array.shape)
            image_array = np.clip(image_array + noise, 0, 1)
            pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
            
        elif aug_type == 'blur':
            # 가우시안 블러
            blur_radius = params['radius']
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        elif aug_type == 'rotation':
            # 회전 (작은 각도만)
            angle = params['angle']
            pil_image = pil_image.rotate(angle, fillcolor=(128, 128, 128))
            
        elif aug_type == 'crop_resize':
            # 크롭 후 리사이즈
            crop_ratio = params['ratio']
            w, h = pil_image.size
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            
            # 중앙 크롭
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            right = left + new_w
            bottom = top + new_h
            
            pil_image = pil_image.crop((left, top, right, bottom))
            pil_image = pil_image.resize((w, h), Image.LANCZOS)
        
        return np.array(pil_image)
    
    def augment_actions(self, actions: np.ndarray, aug_type: str, params: Dict) -> np.ndarray:
        """액션 증강 (제한적으로)"""
        augmented_actions = actions.copy()
        
        if aug_type == 'noise':
            # 액션에 작은 노이즈 추가
            noise_level = params['level']
            noise = np.random.normal(0, noise_level, actions.shape)
            augmented_actions = np.clip(actions + noise, -1, 1)
            
        elif aug_type == 'scale':
            # 액션 스케일링 (제한적으로)
            scale_factor = params['factor']
            augmented_actions = np.clip(actions * scale_factor, -1, 1)
            
        return augmented_actions
    
    def create_augmented_episode(self, episode: Dict, aug_type: str, params: Dict) -> Dict:
        """증강된 에피소드 생성"""
        augmented_episode = episode.copy()
        
        # 이미지 증강
        augmented_images = []
        for image in episode['images']:
            aug_image = self.augment_image(image, aug_type, params)
            augmented_images.append(aug_image)
        
        augmented_episode['images'] = np.array(augmented_images)
        
        # 액션 증강 (선택적으로)
        if aug_type in ['noise', 'scale']:
            augmented_episode['actions'] = self.augment_actions(episode['actions'], aug_type, params)
        
        # 메타데이터 업데이트
        augmented_episode['aug_type'] = aug_type
        augmented_episode['aug_params'] = params
        
        return augmented_episode
    
    def generate_augmentations(self, target_multiplier: int = 5) -> List[Tuple[str, Dict]]:
        """증강 방법들 생성"""
        augmentations = []
        
        # 밝기 증강
        for factor in self.augmentation_config['brightness']:
            augmentations.append(('brightness', {'factor': factor}))
        
        # 대비 증강
        for factor in self.augmentation_config['contrast']:
            augmentations.append(('contrast', {'factor': factor}))
        
        # 채도 증강
        for factor in self.augmentation_config['saturation']:
            augmentations.append(('saturation', {'factor': factor}))
        
        # 노이즈 증강
        for level in self.augmentation_config['noise_levels']:
            augmentations.append(('noise', {'level': level}))
        
        # 블러 증강
        for radius in self.augmentation_config['blur_levels']:
            augmentations.append(('blur', {'radius': radius}))
        
        # 회전 증강
        for angle in self.augmentation_config['rotation_angles']:
            augmentations.append(('rotation', {'angle': angle}))
        
        # 크롭 증강
        for ratio in self.augmentation_config['crop_ratios']:
            augmentations.append(('crop_resize', {'ratio': ratio}))
        
        # 액션 노이즈 (제한적)
        for level in [0.01, 0.02]:
            augmentations.append(('noise', {'level': level}))
        
        # 액션 스케일링 (제한적)
        for factor in [0.95, 1.05]:
            augmentations.append(('scale', {'factor': factor}))
        
        # 목표 배수에 맞게 조정
        if len(augmentations) > target_multiplier:
            augmentations = random.sample(augmentations, target_multiplier)
        
        return augmentations
    
    def augment_dataset(self, target_multiplier: int = 5):
        """전체 데이터셋 증강"""
        print(f"🔄 데이터셋 증강 시작 (목표: {target_multiplier}배)")
        
        # 원본 파일들 로드
        original_files = list(self.data_dir.glob("*.h5"))
        print(f"   원본 에피소드: {len(original_files)}개")
        
        # 증강 방법들 생성
        augmentations = self.generate_augmentations(target_multiplier)
        print(f"   증강 방법: {len(augmentations)}개")
        
        total_episodes = 0
        
        for file_path in original_files:
            print(f"   처리 중: {file_path.name}")
            
            # 원본 에피소드 로드
            episode = self.load_episode(file_path)
            
            # 원본 저장 (이미 존재하지만 메타데이터 추가)
            original_filename = f"original_{file_path.name}"
            episode['aug_type'] = 'original'
            episode['aug_params'] = {}
            self.save_episode(episode, original_filename)
            total_episodes += 1
            
            # 증강된 에피소드들 생성
            for i, (aug_type, params) in enumerate(augmentations):
                try:
                    augmented_episode = self.create_augmented_episode(episode, aug_type, params)
                    
                    # 파일명 생성
                    aug_filename = f"aug_{aug_type}_{i}_{file_path.name}"
                    
                    # 저장
                    self.save_episode(augmented_episode, aug_filename)
                    total_episodes += 1
                    
                except Exception as e:
                    print(f"     증강 실패 ({aug_type}): {e}")
                    continue
        
        print(f"✅ 증강 완료!")
        print(f"   원본: {len(original_files)}개")
        print(f"   증강: {total_episodes - len(original_files)}개")
        print(f"   총합: {total_episodes}개")
        print(f"   증강 배수: {(total_episodes / len(original_files)):.1f}배")
        
        # 증강 통계 저장
        stats = {
            'original_count': len(original_files),
            'augmented_count': total_episodes - len(original_files),
            'total_count': total_episodes,
            'multiplier': total_episodes / len(original_files),
            'augmentation_types': [aug[0] for aug in augmentations],
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'augmentation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return total_episodes

def main():
    """메인 실행 함수"""
    data_dir = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"
    output_dir = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset_augmented"
    
    augmenter = MobileVLADataAugmenter(data_dir, output_dir)
    
    # 5배 증강 실행
    total_episodes = augmenter.augment_dataset(target_multiplier=5)
    
    print(f"\n🎉 데이터 증강 완료!")
    print(f"📊 결과: {total_episodes}개 에피소드")
    print(f"📁 저장 위치: {output_dir}")

if __name__ == "__main__":
    main()
