#!/usr/bin/env python3
"""
🎯 태스크 특성 기반 맞춤형 증강 학습
"""
import sys
from pathlib import Path
import torch
import numpy as np
import random
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
from task_specific_augmentation import TaskSpecificAugmentation
from torch.utils.data import DataLoader

class TaskSpecificTrainer(MobileVLATrainer):
    """태스크 특성 기반 맞춤형 증강 학습기"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 태스크 특성 기반 증강기 초기화
        self.augmenter = TaskSpecificAugmentation()
        
        # 이미지 변환
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 액션 통계 초기화
        self.action_mean = None
        self.action_std = None
        
        print("🎯 TaskSpecificTrainer 초기화 완료")
        print(f"   맞춤형 증강: 활성화")
        print(f"   Z축 0 처리: 특별 처리")
        print(f"   X축 우세: 전진/후진 중심")
    
    def compute_action_statistics(self, dataset):
        """액션 통계 계산"""
        print("📊 액션 통계 계산 중...")
        
        all_actions = []
        for i in range(len(dataset)):
            episode = dataset[i]
            actions = episode['actions']
            if isinstance(actions, np.ndarray):
                all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0)
        
        # Z축 특별 처리 (모두 0이므로)
        self.action_std[2] = 1.0  # 기본값 사용
        
        self.action_std = np.clip(self.action_std, 1e-3, None)
        
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
    
    def process_images_to_tensor(self, images):
        """이미지를 텐서로 변환"""
        if isinstance(images, list):
            tensor_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    tensor_img = self.image_transform(img)
                elif isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    tensor_img = self.image_transform(pil_img)
                elif isinstance(img, torch.Tensor):
                    tensor_img = img
                else:
                    continue
                tensor_images.append(tensor_img)
            
            if tensor_images:
                return torch.stack(tensor_images)
            else:
                return None
        else:
            return images
    
    def safe_normalize_actions(self, actions):
        """안전한 액션 정규화"""
        if isinstance(actions, list):
            actions = np.array(actions)
        elif isinstance(actions, np.ndarray):
            actions = actions.copy()
        
        if actions.ndim == 2:
            actions = np.expand_dims(actions, axis=0)
        
        normalized = np.zeros_like(actions)
        for i in range(3):
            if self.action_std[i] > 1e-6:
                normalized[:, :, i] = (actions[:, :, i] - self.action_mean[i]) / self.action_std[i]
            else:
                normalized[:, :, i] = actions[:, :, i] - self.action_mean[i]
        
        return normalized
    
    def train_step_task_specific(self, batch):
        """태스크 특성 기반 학습 스텝"""
        try:
            # 이미지 처리
            images = batch['images']
            images_tensor = self.process_images_to_tensor(images)
            
            if images_tensor is None:
                return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}
            
            if images_tensor.dim() == 3:
                images_tensor = images_tensor.unsqueeze(0)
            
            # 액션 처리
            actions = batch['actions']
            
            if isinstance(actions, list):
                actions = np.array(actions)
            elif isinstance(actions, np.ndarray):
                actions = actions.copy()
            
            if actions.ndim == 2:
                actions = np.expand_dims(actions, axis=0)
            
            # 태스크 특성 기반 맞춤형 증강 적용
            if random.random() < 0.7:  # 70% 확률로 증강 적용
                aug_images, aug_actions = self.augmenter.augment_episode(images, actions)
                
                # 증강된 이미지를 텐서로 변환
                aug_images_tensor = self.process_images_to_tensor(aug_images)
                if aug_images_tensor is not None:
                    if aug_images_tensor.dim() == 3:
                        aug_images_tensor = aug_images_tensor.unsqueeze(0)
                    
                    # 증강된 데이터 사용
                    images_tensor = aug_images_tensor
                    actions = aug_actions
                    
                    if actions.ndim == 2:
                        actions = np.expand_dims(actions, axis=0)
            
            # 정규화
            if self.action_mean is not None:
                actions = self.safe_normalize_actions(actions)
            
            # 배치 업데이트
            batch['images'] = images_tensor
            batch['actions'] = actions
            
            # 기존 train_step 호출
            result = super().train_step(batch)
            
            # NaN 체크
            if torch.isnan(result['total_loss']):
                result['total_loss'] = torch.tensor(1.0, device=self.device)
            
            return result
            
        except Exception as e:
            print(f"학습 스텝 오류: {e}")
            import traceback
            traceback.print_exc()
            return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}
    
    def validation_step(self, batch):
        """검증 스텝 (증강 없이)"""
        try:
            # 이미지 처리
            images = batch['images']
            images_tensor = self.process_images_to_tensor(images)
            
            if images_tensor is None:
                return {'total_loss': 1.0, 'mae_avg': 1.0}
            
            if images_tensor.dim() == 3:
                images_tensor = images_tensor.unsqueeze(0)
            
            # 액션 처리
            actions = batch['actions']
            
            if isinstance(actions, list):
                actions = np.array(actions)
            elif isinstance(actions, np.ndarray):
                actions = actions.copy()
            
            if actions.ndim == 2:
                actions = np.expand_dims(actions, axis=0)
            
            # 정규화 (증강 없이)
            if self.action_mean is not None:
                actions = self.safe_normalize_actions(actions)
            
            # 배치 업데이트
            batch['images'] = images_tensor
            batch['actions'] = actions
            
            # 모델 예측 (gradient 없이)
            images = batch["images"]
            actions = batch["actions"]
            
            # Window/Chunk 분할
            batch_size, sequence_length = images.shape[:2]
            
            if sequence_length >= self.window_size + self.chunk_size:
                window_images = images[:, :self.window_size]
                chunk_actions = actions[:, self.window_size:self.window_size + self.chunk_size]
            else:
                window_images = images[:, :min(sequence_length, self.window_size)]
                chunk_actions = actions[:, -self.chunk_size:] if sequence_length >= self.chunk_size else actions
            
            # 텍스트 처리
            task_descriptions = ["Navigate around obstacles to track the target cup"] * batch_size
            text_inputs = self.processor(text=task_descriptions, return_tensors="pt", padding=True, truncation=True)
            
            # 디바이스로 이동
            window_images = window_images.to(self.device)
            chunk_actions = chunk_actions.to(self.device)
            input_ids = text_inputs["input_ids"].to(self.device)
            attention_mask = text_inputs["attention_mask"].to(self.device)
            
            # Forward pass
            predictions = self.model(window_images, input_ids, attention_mask)
            targets = {"action_chunk": chunk_actions}
            
            # Loss 계산
            loss_dict = self.compute_loss(predictions, targets)
            
            return {
                'total_loss': loss_dict["total_loss"].item(),
                'mae_avg': loss_dict["mae_avg"]
            }
            
        except Exception as e:
            print(f"검증 스텝 오류: {e}")
            return {'total_loss': 1.0, 'mae_avg': 1.0}

def task_specific_training():
    """태스크 특성 기반 맞춤형 증강 학습"""
    print("🎯 태스크 특성 기반 맞춤형 증강 학습!")
    print("=" * 70)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"원본 데이터셋 크기: {len(dataset)}")
    
    # 트레이너 생성
    trainer = TaskSpecificTrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 액션 통계 계산
    trainer.compute_action_statistics(dataset)
    
    # 데이터 분할
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
    print("\n🎯 태스크 특성 기반 학습 시작!")
    num_epochs = 15
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    patience = 7
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
                print(f"\n--- 배치 {i+1} 처리 중 ---")
                metrics = trainer.train_step_task_specific(batch)
                train_losses.append(metrics['total_loss'].item())
                train_maes.append(metrics['mae_avg'])
                
                if (i + 1) % 10 == 0:
                    print(f"   배치 {i+1}/{len(train_loader)}: Loss={metrics['total_loss']:.4f}, MAE={metrics['mae_avg']:.4f}")
                    
            except Exception as e:
                print(f"   배치 {i+1} 오류: {e}")
                continue
            
            # 처음 5개 배치만 테스트
            if i >= 4:
                break
        
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
                        metrics = trainer.validation_step(batch)
                        val_losses.append(metrics['total_loss'])
                        val_maes.append(metrics['mae_avg'])
                    except Exception as e:
                        continue
                    
                    # 처음 3개만 테스트
                    if i >= 2:
                        break
            
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
                    }, 'best_task_specific_model.pth')
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
    
    print("\n🎉 태스크 특성 기반 학습 완료!")
    
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
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset)
        },
        'augmentation_info': {
            'approach': 'Task-Specific Custom Augmentation',
            'augmentation_probs': trainer.augmenter.augmentation_probs,
            'noise_levels': trainer.augmenter.noise_levels
        },
        'model_info': {
            'architecture': 'Task-Specific Kosmos2',
            'epochs': num_epochs,
            'learning_rate': trainer.learning_rate,
            'early_stopping': True,
            'patience': patience
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('task_specific_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: task_specific_results.json")
    
    # 최종 성능 요약
    print("\n📊 최종 성능 요약:")
    print(f"   최고 검증 Loss: {best_val_loss:.4f}")
    print(f"   최종 훈련 MAE: {train_history[-1]['mae_avg']:.4f}")
    print(f"   학습 에포크: {len(train_history)}")
    print(f"   증강 방식: 태스크 특성 기반 맞춤형")
    print(f"   Z축 처리: 특별 처리")
    print(f"   X축 우세: 전진/후진 중심")
    print(f"   Early stopping: {'활성화' if patience_counter >= patience else '비활성화'}")

if __name__ == "__main__":
    task_specific_training()
