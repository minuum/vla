#!/usr/bin/env python3
"""
간단한 데이터 로더 - H5 파일만 처리
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """간단한 데이터셋 - H5 파일만 처리"""
    
    def __init__(self, data_path, processor, frame_selection='first'):
        self.data_path = data_path
        self.processor = processor
        self.frame_selection = frame_selection
        
        # 데이터 로드
        self.data = self._load_data()
        
        logger.info(f"✅ Simple Dataset 초기화 완료:")
        logger.info(f"   - 데이터 경로: {data_path}")
        logger.info(f"   - 샘플 수: {len(self.data)}")
    
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
                        
                        # 단일 에피소드의 각 프레임을 샘플로 추가
                        # images shape: (18, 720, 1280, 3)
                        # actions shape: (18, 3)
                        
                        for frame_idx in range(len(images)):
                            if self.frame_selection == 'first' and frame_idx != 0:
                                continue
                            elif self.frame_selection == 'random' and frame_idx != random.randint(0, len(images) - 1):
                                continue
                            
                            data.append({
                                'image': images[frame_idx],
                                'action': actions[frame_idx][:2],  # 2D 액션만
                                'episode_id': len(data),  # 고유 ID
                                'frame_id': frame_idx
                            })
                            
                            # first 모드에서는 첫 프레임만 추가
                            if self.frame_selection == 'first':
                                break
                                    
            except Exception as e:
                logger.error(f"❌ {h5_file} 로드 오류: {e}")
                continue
        
        logger.info(f"📊 로드된 샘플 수: {len(data)}")
        return data
    
    def __getitem__(self, idx):
        """데이터 샘플 반환"""
        item = self.data[idx]
        
        # 이미지 처리
        image = Image.fromarray(item['image']).convert('RGB')
        
        # 액션
        action = torch.tensor(item['action'], dtype=torch.float32)
        
        # 텍스트 (기본 프롬프트)
        text = "Navigate the robot to the target location."
        
        return {
            'image': image,
            'action': action,
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

def create_simple_data_loaders(data_path, processor, batch_size=2, 
                              train_split=0.7, val_split=0.15, test_split=0.15):
    """간단한 데이터 로더 생성"""
    
    # 전체 데이터셋 생성
    full_dataset = SimpleDataset(
        data_path=data_path,
        processor=processor,
        frame_selection='random'  # 랜덤 프레임 사용
    )
    
    # 데이터셋 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    logger.info(f"📊 데이터셋 분할:")
    logger.info(f"   - 전체: {total_size}")
    logger.info(f"   - 훈련: {train_size}")
    logger.info(f"   - 검증: {val_size}")
    logger.info(f"   - 테스트: {test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성 (커스텀 collate 함수 사용)
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
    
    logger.info(f"✅ Simple Data Loaders 생성 완료")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 테스트 코드
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터셋 생성
    dataset = SimpleDataset(
        data_path="../../../../ROS_action/mobile_vla_dataset/",
        processor=processor,
        frame_selection='first'
    )
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_simple_data_loaders(
        data_path="../../../../ROS_action/mobile_vla_dataset/",
        processor=processor,
        batch_size=2
    )
    
    # 배치 테스트
    for batch in train_loader:
        logger.info(f"📦 배치 정보:")
        logger.info(f"   - 이미지: {batch['image'][0].size}")
        logger.info(f"   - 액션: {batch['action'].shape}")
        logger.info(f"   - 텍스트: {len(batch['text'])}")
        break
