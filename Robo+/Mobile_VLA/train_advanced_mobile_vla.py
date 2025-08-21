#!/usr/bin/env python3
"""
🚀 Advanced Mobile VLA Model Training Script
Claw Matrix + Hierarchical Planning + Advanced Attention 훈련
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import json
import os
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# RoboVLMs 모듈 추가
sys.path.append(str(Path(__file__).parent / "robovlms" / "models"))
from advanced_mobile_vla_model import AdvancedMobileVLAModel, test_advanced_model

class MobileVLADataset(Dataset):
    """Mobile VLA 데이터셋 로더"""
    def __init__(self, data_path, max_episodes=None):
        self.data_path = data_path
        self.episodes = []
        
        # H5 파일들 찾기
        h5_files = list(Path(data_path).glob("*.h5"))
        if max_episodes:
            h5_files = h5_files[:max_episodes]
        
        print(f"📁 데이터셋 로딩: {len(h5_files)}개 에피소드")
        
        for h5_file in tqdm(h5_files, desc="데이터 로딩"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    # 데이터 구조 확인 - images와 actions 사용
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]
                        actions = f['actions'][:]
                        
                        # 데이터 검증
                        if len(images) > 0 and len(actions) > 0:
                            self.episodes.append({
                                'images': images,
                                'actions': actions,
                                'file': str(h5_file)
                            })
            except Exception as e:
                print(f"⚠️ 파일 로딩 실패 {h5_file}: {e}")
        
        print(f"✅ 로딩 완료: {len(self.episodes)}개 유효한 에피소드")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지와 액션 데이터 추출
        images = episode['images']
        actions = episode['actions']
        
        # 데이터 형태 정규화
        if len(images.shape) == 3:
            images = images.reshape(1, *images.shape)
        
        # 액션 데이터 형태 확인 및 조정
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        return {
            'images': torch.FloatTensor(images),
            'actions': torch.FloatTensor(actions),
            'episode_id': idx
        }

def collate_fn(batch):
    """배치 데이터 정리"""
    images = [item['images'] for item in batch]
    actions = [item['actions'] for item in batch]
    episode_ids = [item['episode_id'] for item in batch]
    
    # 패딩으로 배치 크기 맞추기
    max_images = max(f.shape[0] for f in images)
    max_actions = max(a.shape[0] for a in actions)
    
    padded_images = []
    padded_actions = []
    
    for image, action in zip(images, actions):
        # 이미지 패딩
        if image.shape[0] < max_images:
            pad_size = max_images - image.shape[0]
            padded_image = torch.cat([image, torch.zeros(pad_size, *image.shape[1:])], dim=0)
        else:
            padded_image = image[:max_images]
        padded_images.append(padded_image)
        
        # 액션 패딩
        if action.shape[0] < max_actions:
            pad_size = max_actions - action.shape[0]
            padded_action = torch.cat([action, torch.zeros(pad_size, *action.shape[1:])], dim=0)
        else:
            padded_action = action[:max_actions]
        padded_actions.append(padded_action)
    
    return {
        'images': torch.stack(padded_images),
        'actions': torch.stack(padded_actions),
        'episode_ids': episode_ids
    }

def train_advanced_mobile_vla():
    """고급 Mobile VLA 모델 훈련"""
    print("🚀 Advanced Mobile VLA Model 훈련 시작")
    
    # 설정
    config = {
        'data_path': '../../ROS_action/mobile_vla_dataset',
        'batch_size': 1,  # 이전 코드와 동일
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'save_interval': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_episodes': None,  # 제한 없음 - 전체 데이터셋 사용
        'vision_dim': 768,
        'language_dim': 768,
        'action_dim': 3,
        'fusion_dim': 512,
        'plan_dim': 256,
        'num_claw_layers': 3,
        'num_subgoals': 6,
        'frames_per_subgoal': 3,
        'use_claw_matrix': True,
        'use_hierarchical': True,
        'use_advanced_attention': True
    }
    
    print(f"🔧 설정: {config}")
    print(f"💻 디바이스: {config['device']}")
    
    # 데이터셋 로드
    print("📊 데이터셋 로딩 중...")
    dataset = MobileVLADataset(config['data_path'], config['max_episodes'])
    
    if len(dataset) == 0:
        print("❌ 유효한 데이터가 없습니다!")
        return
    
    # 데이터로더 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # 모델 초기화
    print("🏗️ 모델 초기화 중...")
    from transformers import AutoProcessor, AutoModel
    
    # Kosmos-2 모델 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    vision_model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 고급 모델 생성
    model = AdvancedMobileVLAModel(
        processor=processor,
        vision_dim=config['vision_dim'],
        language_dim=config['language_dim'],
        action_dim=config['action_dim'],
        fusion_dim=config['fusion_dim'],
        plan_dim=config['plan_dim'],
        num_claw_layers=config['num_claw_layers'],
        num_subgoals=config['num_subgoals'],
        frames_per_subgoal=config['frames_per_subgoal'],
        use_claw_matrix=config['use_claw_matrix'],
        use_hierarchical=config['use_hierarchical'],
        use_advanced_attention=config['use_advanced_attention']
    )
    
    model = model.to(config['device'])
    
    # 손실 함수와 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    
    print("🎯 훈련 시작!")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['images'].to(config['device'])
                actions = batch['actions'].to(config['device'])
                
                # 모델 순전파
                optimizer.zero_grad()
                
                # 거리 라벨 생성 (더미 데이터)
                batch_size = images.shape[0]
                distance_labels = torch.randint(0, 3, (batch_size,), device=config['device'])
                
                # 모델 호출 (distance_labels 포함)
                predicted_actions = model(
                    images=images,
                    distance_labels=distance_labels
                )
                
                # 손실 계산 (액션 형태 맞추기)
                target_actions = actions[:, :predicted_actions.shape[1], :predicted_actions.shape[2]]
                
                loss = criterion(predicted_actions, target_actions)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 진행률 업데이트
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        # 에포크 완료
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_loss)
        
        print(f"📊 Epoch {epoch+1} 완료 - 평균 손실: {avg_loss:.4f}")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 모델 저장
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = f"advanced_mobile_vla_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, save_path)
            print(f"💾 모델 저장: {save_path}")
    
    # 최종 모델 저장
    final_save_path = "advanced_mobile_vla_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses
    }, final_save_path)
    
    # 훈련 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Advanced Mobile VLA Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('advanced_mobile_vla_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 저장
    results = {
        'final_loss': train_losses[-1] if train_losses else float('inf'),
        'best_loss': min(train_losses) if train_losses else float('inf'),
        'epochs_trained': len(train_losses),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('advanced_mobile_vla_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 훈련 완료!")
    print(f"📈 최종 손실: {results['final_loss']:.4f}")
    print(f"🏆 최고 성능: {results['best_loss']:.4f}")
    print(f"💾 모델 저장: {final_save_path}")
    print(f"📊 결과 저장: advanced_mobile_vla_training_results.json")

if __name__ == "__main__":
    train_advanced_mobile_vla()
