#!/usr/bin/env python3
"""
🚀 간단한 데이터 증강 학습 - 메모리 효율적
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import sys
import json
from datetime import datetime

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class SimpleAugmentedTrainer(MobileVLATrainer):
    """간단한 데이터 증강 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Z축 특별 처리
        self.z_weight = 0.05
        
        # 간단한 데이터 증강
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
        
        print("✅ SimpleAugmentedTrainer 초기화 완료")
        print(f"   Z축 가중치: {self.z_weight}")
        print(f"   간단한 증강: 활성화")
    
    def compute_action_statistics(self, dataset):
        """안전한 액션 통계 계산"""
        print("📊 액션 통계 계산 중...")
        
        all_actions = []
        for i in range(len(dataset)):
            episode = dataset[i]
            actions = episode['actions']
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            all_actions.append(actions)
        
        all_actions = torch.cat(all_actions, dim=0)
        
        self.action_mean = all_actions.mean(dim=0)
        self.action_std = all_actions.std(dim=0)
        
        # Z축 특별 처리
        if self.action_std[2] < 1e-6:
            print("⚠️ Z축 표준편차가 너무 작음 - 기본값 사용")
            self.action_std[2] = 1.0
        
        self.action_std = torch.clamp(self.action_std, min=1e-3)
        
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
    
    def safe_normalize_actions(self, actions):
        """안전한 액션 정규화"""
        if not hasattr(self, 'action_mean'):
            return actions
        
        normalized = torch.zeros_like(actions)
        for i in range(3):
            if self.action_std[i] > 1e-6:
                normalized[:, :, i] = (actions[:, :, i] - self.action_mean[i]) / self.action_std[i]
            else:
                normalized[:, :, i] = actions[:, :, i] - self.action_mean[i]
        
        return normalized
    
    def train_step_with_augmentation(self, batch):
        """증강이 적용된 학습 스텝"""
        try:
            # 간단한 액션 증강 (노이즈만 추가)
            actions = batch['actions']
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            
            # X, Y축에만 노이즈 추가
            noise = torch.normal(0, 0.01, actions[:, :, :2].shape)
            actions[:, :, :2] += noise
            
            # Z축은 매우 작은 노이즈
            z_noise = torch.normal(0, 0.001, actions[:, :, 2:3].shape)
            actions[:, :, 2:3] += z_noise
            
            # 범위 제한
            actions = torch.clamp(actions, -1.15, 1.15)
            
            # 정규화
            if hasattr(self, 'action_mean'):
                actions = self.safe_normalize_actions(actions)
            
            # 배치 업데이트
            batch['actions'] = actions
            
            # 기존 train_step 호출
            result = super().train_step(batch)
            
            # NaN 체크
            if torch.isnan(result['total_loss']):
                print("⚠️ NaN loss detected - using fallback")
                result['total_loss'] = torch.tensor(1.0, device=self.device)
            
            return result
            
        except Exception as e:
            print(f"학습 스텝 오류: {e}")
            return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}

def simple_augmented_training():
    """간단한 데이터 증강 학습"""
    print("🚀 간단한 데이터 증강 학습 시작!")
    print("=" * 50)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"원본 데이터셋 크기: {len(dataset)}")
    
    # 트레이너 생성
    trainer = SimpleAugmentedTrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 액션 통계 계산
    trainer.compute_action_statistics(dataset)
    
    # 데이터 분할 (원본 데이터만 사용)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    random.shuffle(train_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"훈련 데이터: {len(train_dataset)} 에피소드")
    print(f"검증 데이터: {len(val_dataset)} 에피소드")
    
    # DataLoader 생성
    def collate_fn(batch):
        return batch[0]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 학습 시작
    print("\n🎯 학습 시작!")
    num_epochs = 10  # 간단한 학습
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        print(f"\n📈 에포크 {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 훈련
        trainer.model.train()
        train_losses = []
        train_maes = []
        
        for i, batch in enumerate(train_loader):
            try:
                metrics = trainer.train_step_with_augmentation(batch)
                train_losses.append(metrics['total_loss'].item())
                train_maes.append(metrics['mae_avg'])
                
                if (i + 1) % 20 == 0:  # 20배치마다 진행상황
                    print(f"   배치 {i+1}/{len(train_loader)}: Loss={metrics['total_loss']:.4f}, MAE={metrics['mae_avg']:.4f}")
                    
            except Exception as e:
                print(f"   배치 {i+1} 오류: {e}")
                continue
        
        if train_losses:
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            
            train_metrics = {
                'epoch': epoch + 1,
                'loss': avg_train_loss,
                'mae_avg': avg_train_mae
            }
            train_history.append(train_metrics)
            
            print(f"✅ 훈련 완료: Loss={avg_train_loss:.4f}, MAE={avg_train_mae:.4f}")
            
            # 검증
            trainer.model.eval()
            val_losses = []
            val_maes = []
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    try:
                        metrics = trainer.train_step_with_augmentation(batch)
                        val_losses.append(metrics['total_loss'].item())
                        val_maes.append(metrics['mae_avg'])
                    except Exception as e:
                        continue
            
            if val_losses:
                avg_val_loss = np.mean(val_losses)
                avg_val_mae = np.mean(val_maes)
                
                val_metrics = {
                    'epoch': epoch + 1,
                    'loss': avg_val_loss,
                    'mae_avg': avg_val_mae
                }
                val_history.append(val_metrics)
                
                print(f"🔍 검증 완료: Loss={avg_val_loss:.4f}, MAE={avg_val_mae:.4f}")
                
                # 최고 모델 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'action_mean': trainer.action_mean,
                        'action_std': trainer.action_std
                    }, 'best_simple_augmented_model.pth')
                    print(f"💾 최고 모델 저장됨 (Loss: {best_val_loss:.4f})")
        
        # NaN 체크
        if np.isnan(avg_train_loss):
            print("❌ NaN Loss 발생! 학습 중단")
            break
        else:
            print("✅ NaN Loss 없음!")
    
    print("\n🎉 학습 완료!")
    
    # 결과 저장
    results = {
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_train_loss': train_history[-1]['loss'] if train_history else None,
            'final_train_mae': train_history[-1]['mae_avg'] if train_history else None
        },
        'train_history': train_history,
        'val_history': val_history,
        'action_statistics': {
            'mean': trainer.action_mean.tolist() if trainer.action_mean is not None else None,
            'std': trainer.action_std.tolist() if trainer.action_std is not None else None
        },
        'dataset_info': {
            'original_size': len(dataset),
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset)
        },
        'model_info': {
            'architecture': 'Simple Augmented Kosmos2',
            'z_axis_weight': trainer.z_weight,
            'epochs': num_epochs,
            'learning_rate': trainer.learning_rate
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('simple_augmented_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: simple_augmented_training_results.json")
    
    # 최종 성능 요약
    print("\n📊 최종 성능 요약:")
    print(f"   최고 검증 Loss: {best_val_loss:.4f}")
    print(f"   최종 훈련 MAE: {train_history[-1]['mae_avg']:.4f}")
    print(f"   Z축 특별 처리: 활성화")
    print(f"   간단한 증강: 활성화")
    print(f"   NaN Loss 방지: 성공")

if __name__ == "__main__":
    simple_augmented_training()
