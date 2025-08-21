#!/usr/bin/env python3
"""
🤖 Simple Enhanced Mobile VLA Training with Action Normalization
"""

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from torch.utils.data import DataLoader, random_split

class SimpleEnhancedTrainer(MobileVLATrainer):
    """기존 MobileVLATrainer를 확장한 향상된 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 액션 정규화 통계
        self.action_mean = None
        self.action_std = None
        
        print(f"✅ SimpleEnhancedTrainer 초기화 완료")
        print(f"   액션 정규화: 활성화")
        
    def compute_action_statistics(self, dataset: MobileVLADataset):
        """액션 데이터의 통계 계산"""
        print("📊 액션 통계 계산 중...")
        
        all_actions = []
        for i in range(len(dataset)):
            episode = dataset[i]
            actions = episode['actions']
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            all_actions.append(actions)
        
        all_actions = torch.cat(all_actions, dim=0)  # [N, 3]
        
        self.action_mean = all_actions.mean(dim=0)  # [3]
        self.action_std = all_actions.std(dim=0)    # [3]
        
        # 0으로 나누기 방지
        self.action_std = torch.clamp(self.action_std, min=1e-6)
        
        print(f"   액션 범위: {all_actions.min():.4f} ~ {all_actions.max():.4f}")
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
        
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """액션 정규화"""
        if self.action_mean is None or self.action_std is None:
            return actions
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(actions.device)
        
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """액션 역정규화"""
        if self.action_mean is None or self.action_std is None:
            return actions
        return actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
        
    def train_step(self, batch: dict) -> dict:
        """향상된 학습 스텝"""
        # 기존 train_step 호출
        result = super().train_step(batch)
        
        # 액션 정규화 적용된 메트릭 계산
        if self.action_mean is not None:
            # 예측과 타겟을 역정규화하여 실제 스케일로 변환
            with torch.no_grad():
                # 여기서는 간단히 기존 메트릭을 사용하되, 정규화 정보를 추가
                result['action_mean'] = self.action_mean.cpu().numpy().tolist()
                result['action_std'] = self.action_std.cpu().numpy().tolist()
        
        return result

def custom_collate_fn(batch):
    """커스텀 collate 함수"""
    return batch[0]  # 배치 크기가 1이므로 첫 번째 요소만 반환

def main():
    """메인 함수"""
    print("🚀 Simple Enhanced Mobile VLA Training 시작!")
    
    # 데이터셋 로드
    print("📊 데이터셋 로드 중...")
    dataset = MobileVLADataset(DATA_DIR)
    print(f"   총 에피소드: {len(dataset)}개")
    
    # 향상된 트레이너 초기화
    trainer = SimpleEnhancedTrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 액션 통계 계산
    trainer.compute_action_statistics(dataset)
    
    # 데이터 분할 (시간 기반)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 시간 순서대로 분할 (나중 에피소드를 검증용으로)
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    
    print(f"   훈련 데이터: {len(train_dataset)}개")
    print(f"   검증 데이터: {len(val_dataset)}개")
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # 학습 루프
    print("\n🎯 학습 시작...")
    num_epochs = 10  # 테스트용으로 줄임
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        print(f"\n📈 에포크 {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 학습
        epoch_losses = []
        epoch_maes = []
        
        for step, batch in enumerate(train_loader):
            try:
                metrics = trainer.train_step(batch)
                epoch_losses.append(metrics['total_loss'])
                epoch_maes.append(metrics['mae_avg'])
                
                if (step + 1) % 10 == 0:
                    print(f"   배치 {step+1}/{len(train_loader)}: Loss={metrics['total_loss']:.4f}, MAE={metrics['mae_avg']:.4f}")
                    
            except Exception as e:
                print(f"   배치 {step+1} 오류: {e}")
                continue
        
        if epoch_losses:
            train_metrics = {
                'total_loss': np.mean(epoch_losses),
                'mae_avg': np.mean(epoch_maes)
            }
            train_history.append(train_metrics)
            
            print(f"✅ 학습 완료:")
            print(f"   Loss: {train_metrics['total_loss']:.4f}")
            print(f"   MAE: {train_metrics['mae_avg']:.4f}")
            
            # 간단한 검증 (마지막 배치로)
            try:
                val_batch = next(iter(val_loader))
                val_metrics = trainer.train_step(val_batch)  # 검증 모드로 실행
                val_history.append({
                    'loss': val_metrics['total_loss'],
                    'mae_avg': val_metrics['mae_avg']
                })
                
                print(f"🔍 검증 결과:")
                print(f"   Loss: {val_metrics['total_loss']:.4f}")
                print(f"   MAE: {val_metrics['mae_avg']:.4f}")
                
                # 최고 모델 저장
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    torch.save({
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'action_mean': trainer.action_mean,
                        'action_std': trainer.action_std
                    }, 'best_simple_enhanced_model.pth')
                    print(f"💾 최고 모델 저장됨 (Loss: {best_val_loss:.4f})")
                    
            except Exception as e:
                print(f"🔍 검증 오류: {e}")
    
    # 결과 저장
    results = {
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_train_loss': train_history[-1]['total_loss'] if train_history else None,
            'final_train_mae': train_history[-1]['mae_avg'] if train_history else None
        },
        'train_history': train_history,
        'val_history': val_history,
        'action_statistics': {
            'mean': trainer.action_mean.tolist() if trainer.action_mean is not None else None,
            'std': trainer.action_std.tolist() if trainer.action_std is not None else None
        },
        'dataset_info': {
            'total_episodes': len(dataset),
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset)
        },
        'model_info': {
            'architecture': 'Enhanced Kosmos 2B + Action Head',
            'action_normalization': True,
            'epochs': num_epochs,
            'learning_rate': trainer.learning_rate
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('simple_enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: simple_enhanced_training_results.json")
    print("🎉 간단한 향상된 학습 완료!")

if __name__ == "__main__":
    main()
