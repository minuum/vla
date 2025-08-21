#!/usr/bin/env python3
"""
🚀 Case 2: 단기 적용 - 고급 데이터 증강
목표: MAE 0.5 → 0.3, 정확도 15% → 35%
특징: 시간적/공간적 증강 + 고급 변형
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
import random
import logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAugmentationDataset(Dataset):
    """
    고급 데이터 증강 데이터셋
    - 시간적 증강 (프레임 순서 변경)
    - 공간적 증강 (회전, 크롭, 왜곡)
    - 고급 이미지 처리 (블러, 샤프닝, 채도 조정)
    - 액션 시퀀스 변형
    """
    
    def __init__(self, data_path, processor, frame_selection='random',
                 spatial_aug_prob=0.6, temporal_aug_prob=0.4, 
                 advanced_aug_prob=0.3, action_noise_std=0.08):
        self.data_path = data_path
        self.processor = processor
        self.frame_selection = frame_selection
        self.spatial_aug_prob = spatial_aug_prob
        self.temporal_aug_prob = temporal_aug_prob
        self.advanced_aug_prob = advanced_aug_prob
        self.action_noise_std = action_noise_std
        
        # 데이터 로드
        self.data = self._load_data()
        
        logger.info(f"✅ Enhanced Augmentation Dataset 초기화 완료:")
        logger.info(f"   - 데이터 경로: {data_path}")
        logger.info(f"   - 샘플 수: {len(self.data)}")
        logger.info(f"   - 공간적 증강 확률: {spatial_aug_prob}")
        logger.info(f"   - 시간적 증강 확률: {temporal_aug_prob}")
        logger.info(f"   - 고급 증강 확률: {advanced_aug_prob}")
        logger.info(f"   - 액션 노이즈 표준편차: {action_noise_std}")
    
    def _load_data(self):
        """H5 파일들에서 데이터 로드"""
        data = []
        data_path = Path(self.data_path)
        
        # H5 파일들 찾기
        h5_files = list(data_path.glob("*.h5"))
        logger.info(f"📁 H5 파일 수: {len(h5_files)}")
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]
                        actions = f['actions'][:]
                        
                        # 각 프레임을 샘플로 추가 (시간적 증강을 위해 여러 프레임 저장)
                        if self.frame_selection == 'all':
                            # 모든 프레임 사용
                            for frame_idx in range(len(images)):
                                data.append({
                                    'image': images[frame_idx],
                                    'action': actions[frame_idx][:2],
                                    'episode_id': len(data),
                                    'frame_id': frame_idx,
                                    'all_images': images,  # 시간적 증강용
                                    'all_actions': actions  # 시간적 증강용
                                })
                        else:
                            # 첫 프레임만 사용
                            frame_idx = 0 if self.frame_selection == 'first' else random.randint(0, len(images) - 1)
                            data.append({
                                'image': images[frame_idx],
                                'action': actions[frame_idx][:2],
                                'episode_id': len(data),
                                'frame_id': frame_idx,
                                'all_images': images,
                                'all_actions': actions
                            })
                            
            except Exception as e:
                logger.error(f"❌ {h5_file} 로드 오류: {e}")
                continue
        
        logger.info(f"📊 로드된 샘플 수: {len(data)}")
        return data
    
    def _spatial_augmentation(self, image):
        """공간적 증강"""
        if random.random() > self.spatial_aug_prob:
            return image
        
        # PIL 이미지로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 1. 작은 회전 (-5도 ~ 5도)
        if random.random() < 0.4:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        
        # 2. 크롭 및 리사이즈 (90~100% 크기)
        if random.random() < 0.3:
            w, h = image.size
            crop_ratio = random.uniform(0.9, 1.0)
            new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h))
            image = image.resize((w, h), Image.BILINEAR)
        
        # 3. 수평 플립 (로봇 제어에 적합하도록 낮은 확률)
        if random.random() < 0.2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        return image
    
    def _temporal_augmentation(self, item):
        """시간적 증강 - 인접 프레임 사용"""
        if random.random() > self.temporal_aug_prob:
            return item['image'], item['action']
        
        all_images = item['all_images']
        all_actions = item['all_actions']
        current_frame = item['frame_id']
        
        # 인접 프레임 선택 (±1~2 프레임)
        max_offset = min(2, len(all_images) - 1)
        if max_offset > 0:
            offset = random.randint(-max_offset, max_offset)
            new_frame = max(0, min(len(all_images) - 1, current_frame + offset))
            
            return all_images[new_frame], all_actions[new_frame][:2]
        
        return item['image'], item['action']
    
    def _advanced_augmentation(self, image):
        """고급 이미지 증강"""
        if random.random() > self.advanced_aug_prob:
            return image
        
        # PIL 이미지로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 1. 채도 조정
        if random.random() < 0.4:
            enhancer = ImageEnhance.Color(image)
            factor = random.uniform(0.8, 1.3)
            image = enhancer.enhance(factor)
        
        # 2. 샤프닝/블러
        if random.random() < 0.3:
            if random.random() < 0.5:
                # 샤프닝
                enhancer = ImageEnhance.Sharpness(image)
                factor = random.uniform(1.0, 1.5)
                image = enhancer.enhance(factor)
            else:
                # 블러
                radius = random.uniform(0.5, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # 3. 밝기 조정 (미세)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
        
        return image
    
    def _action_augmentation(self, action):
        """액션 증강"""
        # 가우시안 노이즈 추가
        if random.random() < 0.7:
            noise = np.random.normal(0, self.action_noise_std, action.shape)
            action = action + noise
        
        # 액션 스케일링 (미세 조정)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.95, 1.05)
            action = action * scale_factor
        
        # 액션 범위 제한
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        item = self.data[idx]
        
        # 시간적 증강 적용
        image, action = self._temporal_augmentation(item)
        
        # 공간적 증강 적용
        image = self._spatial_augmentation(image)
        
        # 고급 증강 적용
        image = self._advanced_augmentation(image)
        
        # PIL 이미지로 변환 (최종)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # 액션 증강 적용
        action = self._action_augmentation(action)
        
        # 텍스트 (다양한 프롬프트)
        prompts = [
            "Navigate the robot to the target location.",
            "Move the robot forward to reach the goal.",
            "Control the robot to complete the task.",
            "Guide the robot to the destination.",
            "Navigate to the target point."
        ]
        text = random.choice(prompts)
        
        return {
            'image': image,
            'action': torch.tensor(action, dtype=torch.float32),
            'text': text,
            'episode_id': item['episode_id']
        }
    
    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    """PIL 이미지를 처리하는 커스텀 collate 함수"""
    images = [item['image'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    texts = [item['text'] for item in batch]
    episode_ids = [item['episode_id'] for item in batch]
    
    return {
        'image': images,  # PIL 이미지 리스트
        'action': actions,
        'text': texts,
        'episode_id': episode_ids
    }

def create_enhanced_data_loaders(data_path, processor, batch_size=2, 
                                train_split=0.7, val_split=0.15, test_split=0.15):
    """고급 증강 데이터 로더 생성"""
    
    # 전체 데이터셋 생성
    full_dataset = EnhancedAugmentationDataset(
        data_path=data_path,
        processor=processor,
        frame_selection='random',  # 랜덤 프레임 사용
        spatial_aug_prob=0.6,
        temporal_aug_prob=0.4,
        advanced_aug_prob=0.3,
        action_noise_std=0.08
    )
    
    # 데이터셋 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    logger.info(f"📊 Enhanced Dataset 분할:")
    logger.info(f"   - 전체: {total_size}")
    logger.info(f"   - 훈련: {train_size}")
    logger.info(f"   - 검증: {val_size}")
    logger.info(f"   - 테스트: {test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성 (custom_collate_fn 사용)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"✅ Enhanced Data Loaders 생성 완료")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 테스트 코드
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터셋 생성
    dataset = EnhancedAugmentationDataset(
        data_path="../../../../ROS_action/mobile_vla_dataset/",
        processor=processor,
        frame_selection='first'
    )
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(
        data_path="../../../../ROS_action/mobile_vla_dataset/",
        processor=processor,
        batch_size=2
    )
    
    # 배치 테스트
    for batch in train_loader:
        logger.info(f"📦 Enhanced 배치 정보:")
        logger.info(f"   - 이미지: {batch['image'][0].size}")
        logger.info(f"   - 액션: {batch['action'].shape}")
        logger.info(f"   - 텍스트: {len(batch['text'])}")
        break
