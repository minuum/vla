"""
🚀 Fixed RoboVLMs Style Training Script
완전히 수정된 RoboVLMs 스타일 모델로 훈련
모든 Claw Matrix, Hierarchical Planning, Advanced Attention 기능 포함
원본 72개 데이터셋으로 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from tqdm import tqdm
import json

from fixed_robovlms_model import (
    FixedRoboVLMStyleSingleImageModel,
    train_fixed_robovlms_model
)

class FixedRoboVLMStyleDataset(Dataset):
    """수정된 RoboVLMs 스타일 데이터셋 (단일 이미지 → 단일 액션)"""
    
    def __init__(self, data_path, processor, split='train'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
    
    def _load_episodes(self):
        """에피소드 로드"""
        if os.path.isdir(self.data_path):
            # 폴더 기반 데이터
            h5_files = list(Path(self.data_path).glob("*.h5"))
        else:
            # 단일 파일
            h5_files = [self.data_path]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # H5 파일 내부 구조 확인
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]  # [18, H, W, 3]
                        actions = f['actions'][:]  # [18, 3]
                        
                        # 첫 프레임만 사용 (단일 이미지)
                        single_image = images[0]  # [H, W, 3] - 첫 프레임만
                        single_action = actions[0]  # [3] - 첫 프레임 액션만
                        
                        self.episodes.append({
                            'image': single_image,  # [H, W, 3] - 단일 이미지
                            'action': single_action,  # [3] - 단일 액션
                            'episode_id': f"{h5_file.stem}"
                        })
                    else:
                        print(f"⚠️ {h5_file}에 images/actions 키가 없습니다")
            except Exception as e:
                print(f"❌ {h5_file} 로드 실패: {e}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지: [H, W, 3] → [3, H, W] (PyTorch 형식)
        image = episode['image']  # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        # 0-1 범위로 정규화
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # 액션: [3]
        action = episode['action']  # [3]
        
        return {
            'image': image,  # [3, H, W]
            'action': action,  # [3]
            'episode_id': episode['episode_id']
        }

def create_data_loaders(data_path, processor, batch_size=4, train_split=0.8):
    """데이터 로더 생성 (배치 크기 줄임)"""
    
    # 전체 데이터셋 로드
    full_dataset = FixedRoboVLMStyleDataset(data_path, processor, 'full')
    
    # 훈련/검증 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 데이터 로더 생성 (배치 크기 줄임)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 줄임
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,  # 줄임
        pin_memory=True
    )
    
    print(f"📊 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    
    return train_loader, val_loader

def main():
    """메인 훈련 함수"""
    
    # 설정
    config = {
        'data_path': '../../ROS_action/mobile_vla_dataset',  # 원본 72개 데이터셋
        'batch_size': 4,  # 배치 크기 줄임 (메모리 절약)
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'dropout': 0.2,
        'z_axis_weight': 0.05,
        'use_claw_matrix': True,  # 활성화
        'use_hierarchical': True,  # 활성화
        'use_advanced_attention': True,  # 활성화
        'early_stopping_patience': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🚀 Fixed RoboVLMs Style Training 시작!")
    print(f"📊 설정: {json.dumps(config, indent=2)}")
    
    # 프로세서 로드
    print("🔧 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 로더 생성
    print("📊 데이터 로더 생성 중...")
    train_loader, val_loader = create_data_loaders(
        config['data_path'],
        processor,
        batch_size=config['batch_size']
    )
    
    # 모델 초기화
    print("🤖 모델 초기화 중...")
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=config['dropout'],
        use_claw_matrix=config['use_claw_matrix'],
        use_hierarchical=config['use_hierarchical'],
        use_advanced_attention=config['use_advanced_attention'],
        z_axis_weight=config['z_axis_weight']
    )
    
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 활성화된 고급 기능:")
    print(f"   - Fixed Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Fixed Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Fixed Advanced Attention: {model.use_advanced_attention}")
    
    # 훈련 실행
    print("🎯 훈련 시작!")
    try:
        trained_model = train_fixed_robovlms_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            early_stopping_patience=config['early_stopping_patience'],
            device=config['device']
        )
        
        print("✅ Fixed RoboVLMs Style Training 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 결과 저장
    results = {
        'model_type': 'Fixed_RoboVLMs_Style_Single_Image',
        'input_type': 'single_image',
        'output_type': 'single_action',
        'data_size': len(train_loader.dataset) + len(val_loader.dataset),
        'config': config,
        'advanced_features': {
            'fixed_claw_matrix': config['use_claw_matrix'],
            'fixed_hierarchical_planning': config['use_hierarchical'],
            'fixed_advanced_attention': config['use_advanced_attention']
        },
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_status': 'completed'
    }
    
    with open('fixed_robovlms_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 결과 저장 완료: fixed_robovlms_training_results.json")
    
    # 모델 상태 확인
    if os.path.exists('fixed_robovlms_model_best.pth'):
        checkpoint = torch.load('fixed_robovlms_model_best.pth', map_location='cpu')
        print(f"📊 최고 모델 성능:")
        print(f"   - 에포크: {checkpoint['epoch']}")
        print(f"   - 훈련 손실: {checkpoint['train_loss']:.4f}")
        print(f"   - 검증 손실: {checkpoint['val_loss']:.4f}")

if __name__ == "__main__":
    main()
