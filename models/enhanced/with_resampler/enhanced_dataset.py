"""
📊 Enhanced Dataset for Vision Resampler
Vision Resampler를 사용하는 완전한 데이터셋 클래스
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Enhanced2DActionDataset(Dataset):
    """Enhanced 2D Action Dataset with Vision Resampler support"""
    
    def __init__(self, data_path, processor, split='train', frame_selection='random', 
                 use_vision_resampler=True):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.frame_selection = frame_selection
        self.use_vision_resampler = use_vision_resampler
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        logger.info(f"📊 {split} Enhanced 2D 액션 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        logger.info(f"   - 프레임 선택: {frame_selection}")
        logger.info(f"   - Z축 제외: True")
        logger.info(f"   - 비전 리샘플러: {use_vision_resampler}")
    
    def _load_episodes(self):
        """에피소드 로드 (Z축 제외, 2D 액션만)"""
        if os.path.isdir(self.data_path):
            h5_files = list(Path(self.data_path).glob("*.h5"))
        else:
            h5_files = [self.data_path]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]  # [18, H, W, 3]
                        actions = f['actions'][:]  # [18, 3]
                        
                        # 첫 프레임 제외 (프레임 1-17만 사용)
                        valid_frames = list(range(1, 18))  # 1, 2, 3, ..., 17
                        
                        if self.frame_selection == 'random':
                            frame_idx = np.random.choice(valid_frames)
                        elif self.frame_selection == 'middle':
                            frame_idx = valid_frames[len(valid_frames)//2]  # 9
                        elif self.frame_selection == 'all':
                            for frame_idx in valid_frames:
                                single_image = images[frame_idx]  # [H, W, 3]
                                single_action = actions[frame_idx]  # [3]
                                
                                # 2D 액션으로 변환 (Z축 제외)
                                action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                                
                                episode_data = {
                                    'image': single_image,
                                    'action': action_2d,  # 2D 액션
                                    'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                                    'frame_idx': frame_idx,
                                    'original_file': h5_file.name
                                }
                                
                                self.episodes.append(episode_data)
                            continue
                        
                        # 단일 프레임 선택
                        single_image = images[frame_idx]  # [H, W, 3]
                        single_action = actions[frame_idx]  # [3]
                        
                        # 2D 액션으로 변환 (Z축 제외)
                        action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                        
                        episode_data = {
                            'image': single_image,
                            'action': action_2d,  # 2D 액션
                            'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                            'frame_idx': frame_idx,
                            'original_file': h5_file.name
                        }
                        
                        self.episodes.append(episode_data)
                        
            except Exception as e:
                logger.error(f"❌ {h5_file} 로드 실패: {e}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지: [H, W, 3] → [3, H, W] (PyTorch 형식)
        image = episode['image']  # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        # 이미지 전처리
        image = Image.fromarray(image.transpose(1, 2, 0))
        inputs = self.processor(images=image, return_tensors="pt")
        image_tensor = inputs['pixel_values'].squeeze(0)  # [3, H, W]
        
        # 액션
        action = torch.FloatTensor(episode['action'])  # [2]
        
        # 텍스트 (더미)
        text = "로봇을 제어하세요"
        
        result = {
            'image': image_tensor,
            'action': action,
            'text': text,
            'episode_id': episode['episode_id']
        }
        
        return result

def create_enhanced_data_loaders(data_path, processor, batch_size=4, train_split=0.8, 
                                frame_selection='random', use_vision_resampler=True):
    """Create enhanced data loaders with Vision Resampler support"""
    
    # Load full dataset
    full_dataset = Enhanced2DActionDataset(
        data_path, processor, 'full', frame_selection, use_vision_resampler
    )
    
    # Train/validation split
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    logger.info(f"📊 Enhanced data loaders created:")
    logger.info(f"   - Train: {len(train_dataset)} episodes")
    logger.info(f"   - Validation: {len(val_dataset)} episodes")
    logger.info(f"   - Batch size: {batch_size}")
    logger.info(f"   - Action dimension: 2D (Z-axis excluded)")
    logger.info(f"   - Vision resampler: {use_vision_resampler}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test dataset
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    data_path = "path/to/your/h5/data"
    
    # Create dataset
    dataset = Enhanced2DActionDataset(
        data_path=data_path,
        processor=processor,
        split='train',
        frame_selection='random',
        use_vision_resampler=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single item
    if len(dataset) > 0:
        item = dataset[0]
        print(f"Image shape: {item['image'].shape}")
        print(f"Action shape: {item['action'].shape}")
        print(f"Text: {item['text']}")
        print(f"Episode ID: {item['episode_id']}")

