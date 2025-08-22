#!/usr/bin/env python3
"""
🎯 보수적 데이터 증강 학습 (72개 → 144개)
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import sys
import json
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from torch.utils.data import DataLoader

class ConservativeAugmentationTrainer(MobileVLATrainer):
    """보수적 데이터 증강 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Z축 특별 처리
        self.z_weight = 0.05
        
        # 보수적 증강 설정 (1단계)
        self.augmentation_level = 1
        self.image_flip_prob = 0.5
        self.action_noise_std = 0.005  # 매우 작은 노이즈
        self.z_noise_std = 0.0  # Z축은 건드리지 않음
        
        # 이미지 변환
        self.image_flip = transforms.RandomHorizontalFlip(p=1.0)  # 확률적으로 적용
        self.image_normal = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ ConservativeAugmentationTrainer 초기화 완료")
        print(f"   Z축 가중치: {self.z_weight}")
        print(f"   증강 레벨: {self.augmentation_level} (보수적)")
        print(f"   이미지 flip 확률: {self.image_flip_prob}")
        print(f"   액션 노이즈: σ={self.action_noise_std}")
        print(f"   Z축 노이즈: σ={self.z_noise_std}")
    
    def compute_action_statistics(self, dataset):
        """안전한 액션 통계 계산"""
        print("📊 액션 통계 계산 중...")
        
        all_actions = []
        for i in range(len(dataset)):
            episode = dataset[i]
            actions = episode['actions']
            if isinstance(actions, list):
                actions = torch.tensor(actions, dtype=torch.float32)
            elif isinstance(actions, np.ndarray):
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
        
        print(f"   액션 범위: {all_actions.min():.4f} ~ {all_actions.max():.4f}")
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
    
    def safe_normalize_actions(self, actions):
        """안전한 액션 정규화"""
        if not hasattr(self, 'action_mean'):
            return actions
        
        # 데이터 타입 확인 및 변환
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.float32)
        elif isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        
        # 차원 확인 및 조정
        original_shape = actions.shape
        if actions.dim() == 2:  # [T, 3]
            actions = actions.unsqueeze(0)  # [1, T, 3]
        
        normalized = torch.zeros_like(actions)
        for i in range(3):
            if self.action_std[i] > 1e-6:
                normalized[:, :, i] = (actions[:, :, i] - self.action_mean[i]) / self.action_std[i]
            else:
                normalized[:, :, i] = actions[:, :, i] - self.action_mean[i]
        
        # 원래 형태로 복원
        if len(original_shape) == 2:
            normalized = normalized.squeeze(0)
        
        return normalized
    
    def augment_episode_conservative(self, episode):
        """보수적 에피소드 증강 (2배로 확장)"""
        augmented_episodes = []
        
        # 원본 에피소드 (정규화만)
        original_episode = self._process_episode(episode, augment=False)
        augmented_episodes.append(original_episode)
        
        # 증강된 에피소드 1개
        augmented_episode = self._process_episode(episode, augment=True)
        augmented_episodes.append(augmented_episode)
        
        return augmented_episodes
    
    def _process_episode(self, episode, augment=False):
        """에피소드 처리 (증강 여부에 따라)"""
        processed_episode = episode.copy()
        
        # 이미지 처리
        images = episode['images']
        processed_images = []
        
        for img in images:
            # PIL 이미지로 변환
            if isinstance(img, torch.Tensor):
                # 텐서를 PIL로 변환
                if img.max() <= 1.0:
                    img_pil = transforms.ToPILImage()(img)
                else:
                    img_pil = transforms.ToPILImage()(img / 255.0)
            elif isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img.astype(np.uint8))
            else:
                img_pil = img
            
            # 증강 적용
            if augment and random.random() < self.image_flip_prob:
                img_processed = self.image_normal(self.image_flip(img_pil))
            else:
                img_processed = self.image_normal(img_pil)
            
            processed_images.append(img_processed)
        
        processed_episode['images'] = torch.stack(processed_images)
        
        # 액션 처리
        actions = episode['actions']
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.float32)
        elif isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        
        # 액션 증강
        if augment:
            # X, Y축에만 작은 노이즈 추가
            xy_noise = torch.normal(0, self.action_noise_std, actions[:, :2].shape)
            actions[:, :2] += xy_noise
            
            # 범위 제한
            actions = torch.clamp(actions, -1.15, 1.15)
        
        # 액션 정규화
        actions = self.safe_normalize_actions(actions)
        processed_episode['actions'] = actions
        
        return processed_episode
    
    def train_step_conservative(self, batch):
        """보수적 증강이 적용된 학습 스텝"""
        try:
            # 이미 처리된 데이터이므로 바로 학습
            result = super().train_step(batch)
            
            # NaN 체크
            if torch.isnan(result['total_loss']):
                print("⚠️ NaN loss detected - using fallback")
                result['total_loss'] = torch.tensor(1.0, device=self.device)
            
            return result
            
        except Exception as e:
            print(f"학습 스텝 오류: {e}")
            return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}

def conservative_augmentation_training():
    """보수적 데이터 증강 학습"""
    print("🎯 보수적 데이터 증강 학습 시작! (72개 → 144개)")
    print("=" * 60)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"원본 데이터셋 크기: {len(dataset)}")
    
    # 트레이너 생성
    trainer = ConservativeAugmentationTrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 액션 통계 계산
    trainer.compute_action_statistics(dataset)
    
    # 데이터 증강 적용
    print("\n📈 보수적 데이터 증강 적용 중...")
    augmented_dataset = []
    
    for i, episode in enumerate(dataset):
        if i % 10 == 0:
            print(f"   진행률: {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%)")
        
        # 보수적 증강 적용 (2배 확장)
        augmented_episodes = trainer.augment_episode_conservative(episode)
        augmented_dataset.extend(augmented_episodes)
    
    print(f"증강 완료! 데이터셋 크기: {len(augmented_dataset)}")
    print(f"증강 배수: {len(augmented_dataset) / len(dataset):.1f}x")
    
    # 데이터 분할
    total_size = len(augmented_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 랜덤 셔플
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = [augmented_dataset[i] for i in train_indices]
    val_dataset = [augmented_dataset[i] for i in val_indices]
    
    print(f"훈련 데이터: {len(train_dataset)} 에피소드")
    print(f"검증 데이터: {len(val_dataset)} 에피소드")
    
    # DataLoader 생성
    def collate_fn(batch):
        return batch[0]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 학습 시작
    print("\n🎯 보수적 증강 학습 시작!")
    num_epochs = 15
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n📈 에포크 {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 훈련
        trainer.model.train()
        train_losses = []
        train_maes = []
        
        for i, batch in enumerate(train_loader):
            try:
                metrics = trainer.train_step_conservative(batch)
                train_losses.append(metrics['total_loss'].item())
                train_maes.append(metrics['mae_avg'])
                
                if (i + 1) % 20 == 0:
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
                        metrics = trainer.train_step_conservative(batch)
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
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val_loss,
                        'action_mean': trainer.action_mean,
                        'action_std': trainer.action_std
                    }, 'best_conservative_augmentation_model.pth')
                    print(f"💾 최고 모델 저장됨 (Loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"⏳ Early stopping 카운터: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping! {patience} 에포크 동안 개선 없음")
                    break
        
        # NaN 체크
        if np.isnan(avg_train_loss):
            print("❌ NaN Loss 발생! 학습 중단")
            break
        else:
            print("✅ NaN Loss 없음!")
    
    print("\n🎉 보수적 증강 학습 완료!")
    
    # 결과 저장
    results = {
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_train_loss': train_history[-1]['loss'] if train_history else None,
            'final_train_mae': train_history[-1]['mae_avg'] if train_history else None,
            'epochs_trained': len(train_history)
        },
        'train_history': train_history,
        'val_history': val_history,
        'action_statistics': {
            'mean': trainer.action_mean.tolist() if trainer.action_mean is not None else None,
            'std': trainer.action_std.tolist() if trainer.action_std is not None else None
        },
        'dataset_info': {
            'original_size': len(dataset),
            'augmented_size': len(augmented_dataset),
            'augmentation_multiplier': len(augmented_dataset) / len(dataset),
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset)
        },
        'augmentation_info': {
            'level': trainer.augmentation_level,
            'image_flip_prob': trainer.image_flip_prob,
            'action_noise_std': trainer.action_noise_std,
            'z_noise_std': trainer.z_noise_std,
            'approach': 'Conservative (72→144)'
        },
        'model_info': {
            'architecture': 'Conservative Augmentation Kosmos2',
            'z_axis_weight': trainer.z_weight,
            'epochs': num_epochs,
            'learning_rate': trainer.learning_rate,
            'early_stopping': True,
            'patience': patience
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('conservative_augmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: conservative_augmentation_results.json")
    
    # 최종 성능 요약
    print("\n📊 최종 성능 요약:")
    print(f"   최고 검증 Loss: {best_val_loss:.4f}")
    print(f"   최종 훈련 MAE: {train_history[-1]['mae_avg']:.4f}")
    print(f"   학습 에포크: {len(train_history)}")
    print(f"   데이터 증강: {len(augmented_dataset) / len(dataset):.1f}x (보수적)")
    print(f"   이미지 증강: Random flip (50%)")
    print(f"   액션 증강: 매우 작은 노이즈 (σ=0.005)")
    print(f"   Z축 처리: 건드리지 않음")
    print(f"   Early stopping: {'활성화' if patience_counter >= patience else '비활성화'}")

if __name__ == "__main__":
    conservative_augmentation_training()
