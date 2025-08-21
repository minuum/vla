#!/usr/bin/env python3
"""
🔍 DataLoader 배치 구조 디버깅
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset
from torch.utils.data import DataLoader

def debug_dataloader_batch():
    """DataLoader 배치 구조 분석"""
    print("🔍 DataLoader 배치 구조 디버깅")
    print("=" * 50)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 기본 collate_fn
    def default_collate_fn(batch):
        print(f"기본 collate_fn 호출됨")
        print(f"배치 타입: {type(batch)}")
        print(f"배치 길이: {len(batch)}")
        print(f"배치[0] 타입: {type(batch[0])}")
        return batch[0]
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=default_collate_fn
    )
    
    print("\n📊 첫 번째 배치 분석:")
    for i, batch in enumerate(dataloader):
        print(f"\n배치 {i}:")
        print(f"   배치 타입: {type(batch)}")
        print(f"   배치 키: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}")
        
        if isinstance(batch, dict):
            actions = batch['actions']
            print(f"   액션 타입: {type(actions)}")
            print(f"   액션 shape: {getattr(actions, 'shape', 'N/A')}")
            print(f"   액션 길이: {len(actions) if hasattr(actions, '__len__') else 'N/A'}")
        
        if i >= 2:  # 처음 3개만 확인
            break
    
    print("\n🔧 수정된 collate_fn 테스트:")
    
    def modified_collate_fn(batch):
        episode = batch[0]
        print(f"수정된 collate_fn - 에피소드 키: {list(episode.keys())}")
        
        # 액션 확인
        actions = episode['actions']
        print(f"   액션 타입: {type(actions)}")
        if isinstance(actions, np.ndarray):
            print(f"   액션 shape: {actions.shape}")
            print(f"   액션 dtype: {actions.dtype}")
        
        return episode
    
    # 수정된 DataLoader
    modified_dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=modified_collate_fn
    )
    
    print("\n📊 수정된 DataLoader 첫 번째 배치:")
    for i, batch in enumerate(modified_dataloader):
        print(f"\n수정된 배치 {i}:")
        print(f"   배치 타입: {type(batch)}")
        print(f"   배치 키: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}")
        
        if isinstance(batch, dict):
            actions = batch['actions']
            print(f"   액션 타입: {type(actions)}")
            print(f"   액션 shape: {getattr(actions, 'shape', 'N/A')}")
        
        if i >= 1:  # 첫 번째만 확인
            break

if __name__ == "__main__":
    debug_dataloader_batch()
