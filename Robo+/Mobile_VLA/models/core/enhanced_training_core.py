#!/usr/bin/env python3
"""
🤖 Enhanced Mobile VLA Training with Data Augmentation & Action Normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoProcessor, AutoModel
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import gc

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

import sys
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

class EnhancedMobileVLATrainer:
    """향상된 Mobile VLA 트레이너 - 데이터 증강, 액션 정규화, 개선된 검증 포함"""
    
    def __init__(
        self,
        model_name: str = "microsoft/kosmos-2-patch14-224",
        action_dim: int = 3,
        window_size: int = 8,
        chunk_size: int = 2,
        learning_rate: float = 1e-4,
        num_epochs: int = 20,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        augmentation_multiplier: float = 3.0
    ):
        self.model_name = model_name
        self.action_dim = action_dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.augmentation_multiplier = augmentation_multiplier
        
        # 액션 정규화 통계 (나중에 계산)
        self.action_mean = None
        self.action_std = None
        
        # 모델 초기화
        self._initialize_model()
        
        # 데이터 증강 설정
        self._setup_augmentation()
        
        print(f"✅ EnhancedMobileVLATrainer 초기화 완료")
        print(f"   디바이스: {self.device}")
        print(f"   데이터 증강 배수: {self.augmentation_multiplier}x")
        print(f"   액션 차원: {self.action_dim}")
        
    def _initialize_model(self):
        """모델 초기화"""
        # Kosmos 2B 모델 로드
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.vision_model = AutoModel.from_pretrained(self.model_name)
        
        # 액션 예측 헤드 (Kosmos2는 text_config.hidden_size 사용)
        try:
            hidden_size = self.vision_model.config.hidden_size
        except AttributeError:
            try:
                hidden_size = self.vision_model.config.text_config.hidden_size
            except AttributeError:
                hidden_size = 2048  # 기본값
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.action_dim)
        )
        
        # 모델을 디바이스로 이동
        self.vision_model.to(self.device)
        self.action_head.to(self.device)
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.AdamW([
            {'params': self.vision_model.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.action_head.parameters(), 'lr': self.learning_rate}
        ], weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01
        )
        
        # 손실 함수 (가중 Huber Loss)
        self.criterion = nn.HuberLoss(delta=0.1)
        
    def _setup_augmentation(self):
        """데이터 증강 설정"""
        self.augmentation_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def compute_action_statistics(self, dataset: MobileVLADataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """액션 데이터의 통계 계산"""
        print("📊 액션 통계 계산 중...")
        
        all_actions = []
        for i in range(len(dataset)):
            episode = dataset[i]
            actions = episode['actions']
            # numpy 배열을 tensor로 변환
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            all_actions.append(actions)
        
        all_actions = torch.cat(all_actions, dim=0)  # [N, 3]
        
        action_mean = all_actions.mean(dim=0)  # [3]
        action_std = all_actions.std(dim=0)    # [3]
        
        # 0으로 나누기 방지
        action_std = torch.clamp(action_std, min=1e-6)
        
        print(f"   액션 범위: {all_actions.min():.4f} ~ {all_actions.max():.4f}")
        print(f"   액션 평균: {action_mean}")
        print(f"   액션 표준편차: {action_std}")
        
        return action_mean, action_std
        
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
        
    def augment_batch(self, batch: Dict) -> Dict:
        """배치 데이터 증강"""
        images = batch['images']  # [T, C, H, W] 또는 [B, T, C, H, W]
        actions = batch['actions']  # [T, 3] 또는 [B, T, 3]
        
        # numpy 배열을 tensor로 변환
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        
        # 배치 차원이 없는 경우 추가
        if len(images.shape) == 4:  # [T, C, H, W]
            images = images.unsqueeze(0)  # [1, T, C, H, W]
            actions = actions.unsqueeze(0)  # [1, T, 3]
        
        batch_size, seq_len = images.shape[:2]
        
        # 증강된 이미지 생성
        augmented_images = []
        for b in range(batch_size):
            seq_images = []
            for t in range(seq_len):
                img = images[b, t]  # [C, H, W]
                
                # 이미지 차원 확인 및 수정
                if len(img.shape) == 2:  # [H, W] -> [C, H, W]
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                elif len(img.shape) == 3 and img.shape[0] == 1:  # [1, H, W] -> [C, H, W]
                    img = img.repeat(3, 1, 1)
                
                # 증강 적용
                if torch.rand(1) < 0.7:  # 70% 확률로 증강
                    img = self.augmentation_transforms(img)
                else:
                    img = self.test_transforms(img)
                seq_images.append(img)
            augmented_images.append(torch.stack(seq_images))
        
        augmented_images = torch.stack(augmented_images)
        
        return {
            'images': augmented_images,
            'actions': actions,
            'task_description': batch['task_description'],
            'scenario': batch['scenario']
        }
        
    def train_step(self, batch: Dict) -> Dict:
        """학습 스텝"""
        self.vision_model.train()
        self.action_head.train()
        
        # 데이터 증강
        batch = self.augment_batch(batch)
        
        # 데이터 준비
        images = batch['images'].to(self.device)  # [B, T, C, H, W]
        actions = batch['actions'].to(self.device)  # [B, T, 3]
        
        # 액션 정규화
        actions_normalized = self.normalize_actions(actions)
        
        batch_size, seq_len = images.shape[:2]
        
        # Window/Chunk 분할
        if seq_len >= self.window_size + self.chunk_size:
            window_images = images[:, :self.window_size]  # [B, W, C, H, W]
            target_actions = actions_normalized[:, self.window_size:self.window_size + self.chunk_size]  # [B, C, 3]
        else:
            window_images = images[:, :min(seq_len, self.window_size)]
            target_actions = actions_normalized[:, -self.chunk_size:] if seq_len >= self.chunk_size else actions_normalized
        
        # 예측
        predictions = []
        for t in range(window_images.shape[1]):
            img = window_images[:, t]  # [B, C, H, W]
            
            # Vision 모델 (더미 텍스트 입력 추가)
            with torch.no_grad():
                batch_size = img.shape[0]
                dummy_text = ["<image>"] * batch_size  # 더미 텍스트
                
                # 텍스트 토크나이징
                text_inputs = self.processor(
                    text=dummy_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                vision_outputs = self.vision_model(
                    pixel_values=img,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                )
                vision_features = vision_outputs.last_hidden_state[:, 0]  # [B, hidden_size]
            
            # 액션 예측
            action_pred = self.action_head(vision_features)  # [B, 3]
            predictions.append(action_pred)
        
        # 시퀀스 평균
        predictions = torch.stack(predictions, dim=1)  # [B, T, 3]
        predictions = predictions.mean(dim=1, keepdim=True).expand(-1, target_actions.shape[1], -1)  # [B, C, 3]
        
        # 손실 계산
        loss = self.criterion(predictions, target_actions)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.vision_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 메트릭 계산
        with torch.no_grad():
            predictions_denorm = self.denormalize_actions(predictions)
            target_actions_denorm = self.denormalize_actions(target_actions)
            
            mae = torch.mean(torch.abs(predictions_denorm - target_actions_denorm))
            mae_linear_x = torch.mean(torch.abs(predictions_denorm[:, :, 0] - target_actions_denorm[:, :, 0]))
            mae_linear_y = torch.mean(torch.abs(predictions_denorm[:, :, 1] - target_actions_denorm[:, :, 1]))
            mae_angular_z = torch.mean(torch.abs(predictions_denorm[:, :, 2] - target_actions_denorm[:, :, 2]))
        
        return {
            'total_loss': loss.item(),
            'mae_avg': mae.item(),
            'mae_linear_x': mae_linear_x.item(),
            'mae_linear_y': mae_linear_y.item(),
            'mae_angular_z': mae_angular_z.item()
        }
        
    def evaluate_model(self, dataloader: DataLoader) -> Tuple[Dict, List, List]:
        """모델 평가"""
        self.vision_model.eval()
        self.action_head.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 데이터 준비 (증강 없음)
                images = batch['images'].to(self.device)
                actions = batch['actions'].to(self.device)
                actions_normalized = self.normalize_actions(actions)
                
                batch_size, seq_len = images.shape[:2]
                
                # Window/Chunk 분할
                if seq_len >= self.window_size + self.chunk_size:
                    window_images = images[:, :self.window_size]
                    target_actions = actions_normalized[:, self.window_size:self.window_size + self.chunk_size]
                else:
                    window_images = images[:, :min(seq_len, self.window_size)]
                    target_actions = actions_normalized[:, -self.chunk_size:] if seq_len >= self.chunk_size else actions_normalized
                
                # 예측
                predictions = []
                for t in range(window_images.shape[1]):
                    img = window_images[:, t]
                    
                    # 더미 텍스트 입력 추가
                    batch_size = img.shape[0]
                    dummy_text = ["<image>"] * batch_size
                    
                    # 텍스트 토크나이징
                    text_inputs = self.processor(
                        text=dummy_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    vision_outputs = self.vision_model(
                        pixel_values=img,
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs['attention_mask']
                    )
                    vision_features = vision_outputs.last_hidden_state[:, 0]
                    action_pred = self.action_head(vision_features)
                    predictions.append(action_pred)
                
                predictions = torch.stack(predictions, dim=1)
                predictions = predictions.mean(dim=1, keepdim=True).expand(-1, target_actions.shape[1], -1)
                
                # 손실 계산
                loss = self.criterion(predictions, target_actions)
                
                # 역정규화
                predictions_denorm = self.denormalize_actions(predictions)
                target_actions_denorm = self.denormalize_actions(target_actions)
                
                # 메트릭
                mae = torch.mean(torch.abs(predictions_denorm - target_actions_denorm))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
                
                # 예측/타겟 저장
                all_predictions.extend(predictions_denorm.cpu().numpy().reshape(-1, 3))
                all_targets.extend(target_actions_denorm.cpu().numpy().reshape(-1, 3))
        
        # 최종 메트릭 계산
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        # R² 계산
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        r2_linear_x = r2_score(all_targets[:, 0], all_predictions[:, 0])
        r2_linear_y = r2_score(all_targets[:, 1], all_predictions[:, 1])
        r2_angular_z = r2_score(all_targets[:, 2], all_predictions[:, 2])
        
        # 임계값 정확도
        threshold_0_1 = np.mean(np.all(np.abs(all_predictions - all_targets) < 0.1, axis=1))
        
        return {
            'loss': avg_loss,
            'mae_avg': avg_mae,
            'r2_linear_x': r2_linear_x,
            'r2_linear_y': r2_linear_y,
            'r2_angular_z': r2_angular_z,
            'threshold_0.1': threshold_0_1
        }, all_predictions, all_targets
        
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': {
                'vision_model': self.vision_model.state_dict(),
                'action_head': self.action_head.state_dict()
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'metrics': metrics,
            'config': {
                'model_name': self.model_name,
                'action_dim': self.action_dim,
                'window_size': self.window_size,
                'chunk_size': self.chunk_size,
                'learning_rate': self.learning_rate
            }
        }, path)
        
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.vision_model.load_state_dict(checkpoint['model_state_dict']['vision_model'])
        self.action_head.load_state_dict(checkpoint['model_state_dict']['action_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']
        
        return checkpoint['epoch'], checkpoint['metrics']

def custom_collate_fn(batch):
    """PIL 이미지를 텐서로 변환하는 커스텀 collate 함수"""
    # 배치 크기가 1이므로 첫 번째 요소만 처리
    sample = batch[0]
    
    # 이미지를 텐서로 변환
    if 'images' in sample:
        images = sample['images']
        if isinstance(images, list) and len(images) > 0 and hasattr(images[0], 'convert'):
            # PIL 이미지 리스트를 텐서로 변환
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            tensor_images = []
            for img in images:
                if hasattr(img, 'convert'):
                    img = img.convert('RGB')
                tensor_images.append(to_tensor(img))
            sample['images'] = torch.stack(tensor_images)
    
    return sample

def main():
    """메인 함수"""
    print("🚀 Enhanced Mobile VLA Training 시작!")
    
    # 데이터셋 로드
    print("📊 데이터셋 로드 중...")
    dataset = MobileVLADataset(DATA_DIR)
    print(f"   총 에피소드: {len(dataset)}개")
    
    # 액션 통계 계산
    trainer = EnhancedMobileVLATrainer()
    trainer.action_mean, trainer.action_std = trainer.compute_action_statistics(dataset)
    
    # 데이터 분할 (시간 기반)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 시간 순서대로 분할 (나중 에피소드를 검증용으로)
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    
    print(f"   훈련 데이터: {len(train_dataset)}개")
    print(f"   검증 데이터: {len(val_dataset)}개")
    
    # DataLoader 생성 (커스텀 collate_fn 사용)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # 학습 루프
    print("\n🎯 학습 시작...")
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(trainer.num_epochs):
        print(f"\n📈 에포크 {epoch+1}/{trainer.num_epochs}")
        print("-" * 40)
        
        # 학습
        epoch_losses = []
        epoch_maes = []
        
        for step, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics['total_loss'])
            epoch_maes.append(metrics['mae_avg'])
            
            if (step + 1) % 10 == 0:
                print(f"   배치 {step+1}/{len(train_loader)}: Loss={metrics['total_loss']:.4f}, MAE={metrics['mae_avg']:.4f}")
        
        train_metrics = {
            'total_loss': np.mean(epoch_losses),
            'mae_avg': np.mean(epoch_maes)
        }
        train_history.append(train_metrics)
        
        # 검증
        val_metrics, _, _ = trainer.evaluate_model(val_loader)
        val_history.append(val_metrics)
        
        # 결과 출력
        print(f"✅ 학습 완료:")
        print(f"   Loss: {train_metrics['total_loss']:.4f}")
        print(f"   MAE: {train_metrics['mae_avg']:.4f}")
        
        print(f"🔍 검증 결과:")
        print(f"   Loss: {val_metrics['loss']:.4f}")
        print(f"   MAE: {val_metrics['mae_avg']:.4f}")
        print(f"   R² Linear X: {val_metrics['r2_linear_x']:.4f}")
        print(f"   R² Linear Y: {val_metrics['r2_linear_y']:.4f}")
        print(f"   R² Angular Z: {val_metrics['r2_angular_z']:.4f}")
        print(f"   임계값 정확도 (0.1): {val_metrics['threshold_0.1']:.4f}")
        
        # 학습률 스케줄링
        trainer.scheduler.step()
        
        # 최고 모델 저장
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_checkpoint('best_enhanced_model.pth', epoch + 1, val_metrics)
            print(f"💾 최고 모델 저장됨 (Loss: {best_val_loss:.4f})")
    
    # 최종 평가
    print("\n🎯 최종 평가...")
    try:
        trainer.load_checkpoint('best_enhanced_model.pth')
        final_metrics, final_preds, final_targets = trainer.evaluate_model(val_loader)
        
        print(f"\n🏆 최종 성능:")
        print(f"   전체 MAE: {final_metrics['mae_avg']:.4f}")
        print(f"   임계값 정확도 (0.1): {final_metrics['threshold_0.1']:.4f}")
        print(f"   Linear X R²: {final_metrics['r2_linear_x']:.4f}")
        print(f"   Linear Y R²: {final_metrics['r2_linear_y']:.4f}")
        print(f"   Angular Z R²: {final_metrics['r2_angular_z']:.4f}")
        
    except FileNotFoundError:
        print("❌ 최고 모델 파일을 찾을 수 없습니다.")
        final_metrics = val_metrics
    
    # 결과 저장
    results = {
        'final_metrics': final_metrics,
        'train_history': train_history,
        'val_history': val_history,
        'action_statistics': {
            'mean': trainer.action_mean.tolist() if trainer.action_mean is not None else None,
            'std': trainer.action_std.tolist() if trainer.action_std is not None else None
        },
        'dataset_info': {
            'total_episodes': len(dataset),
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset),
            'augmentation_multiplier': trainer.augmentation_multiplier
        },
        'model_info': {
            'architecture': 'Enhanced Kosmos 2B + Action Head',
            'loss_function': 'Huber Loss',
            'optimizer': 'AdamW with Cosine Annealing',
            'epochs': trainer.num_epochs,
            'learning_rate': trainer.learning_rate,
            'action_normalization': True,
            'data_augmentation': True
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('enhanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: enhanced_training_results.json")
    print("🎉 향상된 학습 완료!")

if __name__ == "__main__":
    main()
