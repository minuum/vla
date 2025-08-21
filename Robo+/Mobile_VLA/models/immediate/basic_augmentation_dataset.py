#!/usr/bin/env python3
"""
🎯 Case 1: 즉시 적용 - 기본 데이터 증강
목표: MAE 0.8 → 0.5, 정확도 0% → 15%
특징: 기본적인 이미지/액션 증강으로 데이터 다양성 증가
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicAugmentationDataset(Dataset):
    """
    기본 데이터 증강 데이터셋
    - 이미지 밝기/대비 조정
    - 액션 노이즈 추가
    - 기본적인 변형으로 데이터 다양성 증가
    """
    
    def __init__(self, data_path, processor, frame_selection='random', 
                 brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 action_noise_std=0.05, augmentation_prob=0.7):
        self.data_path = data_path
        self.processor = processor
        self.frame_selection = frame_selection
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.action_noise_std = action_noise_std
        self.augmentation_prob = augmentation_prob
        
        # 데이터 로드
        self.data = self._load_data()
        
        # 기본 이미지 변환
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"✅ Basic Augmentation Dataset 초기화 완료:")
        logger.info(f"   - 데이터 경로: {data_path}")
        logger.info(f"   - 샘플 수: {len(self.data)}")
        logger.info(f"   - 밝기 범위: {brightness_range}")
        logger.info(f"   - 대비 범위: {contrast_range}")
        logger.info(f"   - 액션 노이즈: {action_noise_std}")
        logger.info(f"   - 증강 확률: {augmentation_prob}")
    
    def _load_data(self):
        """데이터 로드"""
        data = []
        
        if isinstance(self.data_path, str):
            data_paths = [self.data_path]
        else:
            data_paths = self.data_path
        
        for path in data_paths:
            if path.endswith('.h5'):
                # H5 파일 처리
                with h5py.File(path, 'r') as f:
                    images = f['images'][:]
                    actions = f['actions'][:]
                    
                    for i in range(len(images)):
                        if self.frame_selection == 'first':
                            frame_idx = 0
                        elif self.frame_selection == 'random':
                            frame_idx = random.randint(0, len(images[i]) - 1)
                        else:
                            frame_idx = 0
                        
                        data.append({
                            'image': images[i][frame_idx],
                            'action': actions[i][frame_idx][:2],  # 2D 액션만
                            'episode_id': i
                        })
            elif Path(path).is_dir():
                # 폴더 내의 H5 파일들도 처리
                h5_files = list(Path(path).glob("*.h5"))
                for h5_file in h5_files:
                    with h5py.File(h5_file, 'r') as f:
                        images = f['images'][:]
                        actions = f['actions'][:]
                        
                        for i in range(len(images)):
                            if self.frame_selection == 'first':
                                frame_idx = 0
                            elif self.frame_selection == 'random':
                                frame_idx = random.randint(0, len(images[i]) - 1)
                            else:
                                frame_idx = 0
                            
                                                    # 인덱스 범위 확인
                        if frame_idx < len(images[i]) and frame_idx < len(actions[i]):
                            data.append({
                                'image': images[i][frame_idx],
                                'action': actions[i][frame_idx][:2],  # 2D 액션만
                                'episode_id': len(data)  # 고유한 ID
                            })
                
                # 폴더 구조 처리 (기존 코드)
                # 폴더 구조 처리
                path = Path(path)
                for episode_dir in path.iterdir():
                    if episode_dir.is_dir():
                        # 에피소드 ID를 문자열에서 숫자로 변환 (실패하면 인덱스 사용)
                        try:
                            episode_id = int(episode_dir.name)
                        except ValueError:
                            # 문자열인 경우 해시값을 사용하거나 인덱스 사용
                            episode_id = hash(episode_dir.name) % 10000  # 해시값을 4자리 숫자로 변환
                        
                        # 이미지 파일들
                        image_files = sorted(list(episode_dir.glob('*.png')))
                        action_file = episode_dir / 'actions.npy'
                        
                        if len(image_files) > 0 and action_file.exists():
                            actions = np.load(action_file)
                            
                            for frame_idx, img_file in enumerate(image_files):
                                if frame_idx < len(actions):
                                    data.append({
                                        'image_path': str(img_file),
                                        'action': actions[frame_idx][:2],  # 2D 액션만
                                        'episode_id': episode_id
                                    })
        
        logger.info(f"📊 로드된 데이터: {len(data)} 샘플")
        return data
    
    def _augment_image(self, image):
        """이미지 증강"""
        if random.random() > self.augmentation_prob:
            return image
        
        # PIL 이미지로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 밝기 조정
        brightness_factor = random.uniform(*self.brightness_range)
        image = transforms.functional.adjust_brightness(image, brightness_factor)
        
        # 대비 조정
        contrast_factor = random.uniform(*self.contrast_range)
        image = transforms.functional.adjust_contrast(image, contrast_factor)
        
        # 노이즈 추가 (가우시안)
        if random.random() < 0.3:
            noise_std = random.uniform(0.01, 0.05)
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        return image
    
    def _augment_action(self, action):
        """액션 증강"""
        if random.random() > self.augmentation_prob:
            return action
        
        # 가우시안 노이즈 추가
        noise = np.random.normal(0, self.action_noise_std, action.shape)
        augmented_action = action + noise
        
        # 액션 범위 제한 (선형 속도 제한)
        augmented_action = np.clip(augmented_action, -1.0, 1.0)
        
        return augmented_action
    
    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        item = self.data[idx]
        
        # 이미지 처리
        if 'image_path' in item:
            # 폴더 구조에서 이미지 로드
            image = Image.open(item['image_path']).convert('RGB')
        else:
            # H5 파일에서 이미지 로드
            image = Image.fromarray(item['image']).convert('RGB')
        
        # 이미지 증강
        image = self._augment_image(image)
        
        # 액션 증강
        action = self._augment_action(item['action'])
        
        # 텍스트 (기본 프롬프트)
        text = "Navigate the robot to the target location."
        
        return {
            'image': image,
            'action': torch.tensor(action, dtype=torch.float32),
            'text': text,
            'episode_id': item['episode_id']
        }
    
    def __len__(self):
        return len(self.data)

def create_basic_augmentation_data_loaders(data_path, processor, batch_size=2, 
                                          train_split=0.7, val_split=0.15, test_split=0.15):
    """
    기본 증강 데이터 로더 생성
    - batch_size: 4 → 2 (적은 데이터에 맞게 감소)
    - train/val/test split: 70/15/15 (기존 80/20에서 조정)
    """
    
    # 전체 데이터셋 생성
    full_dataset = BasicAugmentationDataset(
        data_path=data_path,
        processor=processor,
        frame_selection='random',
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        action_noise_std=0.05,
        augmentation_prob=0.7
    )
    
    # 데이터셋 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"✅ Basic Augmentation Data Loaders 생성 완료:")
    logger.info(f"   - 전체 샘플: {total_size}")
    logger.info(f"   - 훈련 샘플: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    logger.info(f"   - 검증 샘플: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
    logger.info(f"   - 테스트 샘플: {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    logger.info(f"   - 배치 크기: {batch_size}")
    
    return train_loader, val_loader, test_loader

def analyze_augmentation_effects(dataset, num_samples=10):
    """증강 효과 분석"""
    logger.info(f"🔍 증강 효과 분석 (샘플 {num_samples}개):")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # 원본과 증강된 데이터 비교
        original_action = sample['action']
        
        logger.info(f"   샘플 {i+1}:")
        logger.info(f"     - 액션: {original_action.numpy()}")
        logger.info(f"     - 이미지 크기: {sample['image'].size}")
        logger.info(f"     - 에피소드 ID: {sample['episode_id']}")

if __name__ == "__main__":
    # 테스트 코드
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 경로 설정
    data_path = "../../../../ROS_action/mobile_vla_dataset/"
    
    # 데이터셋 생성
    dataset = BasicAugmentationDataset(
        data_path=data_path,
        processor=processor,
        frame_selection='random',
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        action_noise_std=0.05,
        augmentation_prob=0.7
    )
    
    # 증강 효과 분석
    analyze_augmentation_effects(dataset, num_samples=5)
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_basic_augmentation_data_loaders(
        data_path=data_path,
        processor=processor,
        batch_size=2,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # 배치 테스트
    for batch in train_loader:
        logger.info(f"📦 배치 정보:")
        logger.info(f"   - 이미지: {batch['image'].shape}")
        logger.info(f"   - 액션: {batch['action'].shape}")
        logger.info(f"   - 텍스트: {len(batch['text'])}")
        break
