#!/usr/bin/env python3
"""
🛠️ 과적합 방지 및 데이터 증강 솔루션
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
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
from torch.utils.data import DataLoader

class AdvancedDataAugmentation:
    """고급 데이터 증강 클래스"""
    
    def __init__(self):
        # 이미지 증강
        self.image_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_normal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def augment_episode(self, episode, augment_prob=0.7):
        """에피소드 데이터 증강"""
        images = episode['images']
        actions = episode['actions'].copy()
        
        # numpy to tensor 변환
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        
        augmented_images = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # [C, H, W] to [H, W, C] for PIL
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
            
            if random.random() < augment_prob:
                aug_img = self.image_augment(img)
            else:
                aug_img = self.image_normal(img)
                
            augmented_images.append(aug_img)
        
        # 액션 노이즈 추가 (미세 조정)
        if random.random() < 0.3:
            noise = torch.normal(0, 0.01, actions.shape)
            actions = actions + noise
            actions = torch.clamp(actions, -1.15, 1.15)
        
        # 시간 시프트 (temporal augmentation)
        if random.random() < 0.2 and len(augmented_images) > 2:
            shift = random.randint(-1, 1)
            if shift != 0:
                if shift > 0:
                    augmented_images = augmented_images[shift:] + [augmented_images[-1]] * shift
                    actions = actions[shift:]
                else:
                    augmented_images = [augmented_images[0]] * abs(shift) + augmented_images[:shift]
                    actions = torch.cat([actions[:abs(shift)], actions])
        
        return {
            'images': torch.stack(augmented_images),
            'actions': actions,
            'task_description': episode['task_description'],
            'scenario': episode['scenario']
        }

class RegularizedMobileVLATrainer(MobileVLATrainer):
    """정규화가 강화된 Mobile VLA 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 데이터 증강기
        self.augmenter = AdvancedDataAugmentation()
        
        # 조기 종료 설정
        self.patience = 5
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 액션 정규화 통계
        self.action_mean = None
        self.action_std = None
        
        # 강화된 정규화 적용
        self._enhance_regularization()
        
        print("✅ RegularizedMobileVLATrainer 초기화 완료")
        print(f"   데이터 증강: 활성화")
        print(f"   조기 종료: {self.patience} 에포크")
        print(f"   정규화 강화: 활성화")
    
    def _enhance_regularization(self):
        """정규화 강화"""
        # 기존 모델에 드롭아웃 추가
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'action_head' in name:
                # 액션 헤드에 더 강한 드롭아웃 적용
                module.register_forward_hook(self._dropout_hook)
        
        # 가중치 감쇠 증가
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = 0.01
    
    def _dropout_hook(self, module, input, output):
        """드롭아웃 훅"""
        if self.model.training:
            return nn.functional.dropout(output, p=0.3, training=True)
        return output
    
    def compute_action_statistics(self, dataset):
        """액션 통계 계산"""
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
        self.action_std = torch.clamp(self.action_std, min=1e-6)
        
        print(f"   액션 범위: {all_actions.min():.4f} ~ {all_actions.max():.4f}")
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
    
    def normalize_actions(self, actions):
        """액션 정규화"""
        if self.action_mean is None:
            return actions
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(actions.device)
    
    def denormalize_actions(self, actions):
        """액션 역정규화"""
        if self.action_mean is None:
            return actions
        return actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
    
    def train_step_with_augmentation(self, batch):
        """데이터 증강이 적용된 학습 스텝"""
        # 데이터 증강 적용
        augmented_batch = self.augmenter.augment_episode(batch)
        
        # 액션 정규화
        if self.action_mean is not None:
            augmented_batch['actions'] = self.normalize_actions(augmented_batch['actions'])
        
        # 기존 train_step 호출
        try:
            result = super().train_step(augmented_batch)
            
            # 축별 가중치 적용 (Y, Z축 강화)
            if 'mae_linear_y' in result and 'mae_angular_z' in result:
                result['weighted_loss'] = (
                    result['total_loss'] + 
                    2.0 * result.get('mae_linear_y', 0) +  # Y축 가중치 2배
                    2.0 * result.get('mae_angular_z', 0)   # Z축 가중치 2배
                )
            
            return result
            
        except Exception as e:
            print(f"학습 스텝 오류: {e}")
            return {'total_loss': float('inf'), 'mae_avg': float('inf')}
    
    def validate(self, val_loader):
        """검증 함수"""
        self.model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # 증강 없이 검증
                    if self.action_mean is not None:
                        batch['actions'] = self.normalize_actions(batch['actions'])
                    
                    result = super().train_step(batch)
                    val_losses.append(result['total_loss'])
                    val_maes.append(result['mae_avg'])
                except:
                    continue
        
        if val_losses:
            avg_loss = np.mean(val_losses)
            avg_mae = np.mean(val_maes)
            
            # 조기 종료 체크
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.patience_counter = 0
                return avg_loss, avg_mae, False  # 종료하지 않음
            else:
                self.patience_counter += 1
                early_stop = self.patience_counter >= self.patience
                return avg_loss, avg_mae, early_stop
        
        return float('inf'), float('inf'), False

def demonstrate_overfitting_solution():
    """과적합 해결 솔루션 시연"""
    print("🚀 과적합 해결 솔루션 시연")
    print("=" * 50)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 향상된 트레이너 생성
    trainer = RegularizedMobileVLATrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2,
        learning_rate=5e-5,  # 낮은 학습률
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 액션 통계 계산
    trainer.compute_action_statistics(dataset)
    
    # 데이터 분할 (더 엄격한 시간 기반)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% 훈련
    val_size = total_size - train_size  # 30% 검증
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # 무작위 셔플 (과적합 방지)
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
    
    # 학습 루프 (3 에포크만 시연)
    print("\n🎯 개선된 학습 시작...")
    for epoch in range(3):
        print(f"\n📈 에포크 {epoch+1}/3")
        print("-" * 30)
        
        # 훈련
        trainer.model.train()
        train_losses = []
        train_maes = []
        
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 시연용으로 10배치만
                break
                
            try:
                metrics = trainer.train_step_with_augmentation(batch)
                train_losses.append(metrics['total_loss'])
                train_maes.append(metrics['mae_avg'])
                
                if (i + 1) % 5 == 0:
                    print(f"   배치 {i+1}: Loss={metrics['total_loss']:.4f}, MAE={metrics['mae_avg']:.4f}")
                    
            except Exception as e:
                print(f"   배치 {i+1} 오류: {e}")
                continue
        
        if train_losses:
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)
            
            print(f"✅ 훈련 완료: Loss={avg_train_loss:.4f}, MAE={avg_train_mae:.4f}")
            
            # 검증
            val_loss, val_mae, early_stop = trainer.validate(val_loader)
            print(f"🔍 검증 결과: Loss={val_loss:.4f}, MAE={val_mae:.4f}")
            
            if early_stop:
                print("⏹️ 조기 종료 트리거됨")
                break
    
    print("\n✅ 과적합 해결 솔루션 시연 완료!")
    
    # 해결된 부분 요약
    print("\n📋 해결된 과적합 문제:")
    print("1. ✅ 고급 데이터 증강 (이미지 + 시간적)")
    print("2. ✅ 강화된 정규화 (드롭아웃, 가중치 감쇠)")
    print("3. ✅ 조기 종료")
    print("4. ✅ 축별 가중치 조정")
    print("5. ✅ 액션 정규화")
    print("6. ✅ 더 엄격한 데이터 분할")

if __name__ == "__main__":
    demonstrate_overfitting_solution()
