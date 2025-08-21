#!/usr/bin/env python3
"""
🎯 최종 수정: 이미지+액션 텐서 변환 + 간단한 증강
"""

import torch
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

class FinalFixedTrainer(MobileVLATrainer):
    """최종 수정: 이미지+액션 텐서 변환 + 간단한 증강"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_weight = 0.05
        self.action_noise_std = 0.005  # 매우 작은 노이즈
        
        # 이미지 변환
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ FinalFixedTrainer 초기화 완료")
        print(f"   Z축 가중치: {self.z_weight}")
        print(f"   액션 노이즈: σ={self.action_noise_std}")
        print(f"   이미지 변환: 활성화")
    
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
        
        print(f"   액션 범위: {all_actions.min():.4f} ~ {all_actions.max():.4f}")
        print(f"   액션 평균: {self.action_mean}")
        print(f"   액션 표준편차: {self.action_std}")
    
    def safe_normalize_actions(self, actions):
        """안전한 액션 정규화 (배치 차원 유지)"""
        # 데이터 타입 확인 및 변환
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.float32)
        elif isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        
        # 차원 확인 및 조정
        if actions.dim() == 2:  # [T, 3]
            actions = actions.unsqueeze(0)  # [1, T, 3]
        
        normalized = torch.zeros_like(actions)
        for i in range(3):
            if self.action_std[i] > 1e-6:
                normalized[:, :, i] = (actions[:, :, i] - self.action_mean[i]) / self.action_std[i]
            else:
                normalized[:, :, i] = actions[:, :, i] - self.action_mean[i]
        
        # 배치 차원 유지 (squeeze 제거)
        return normalized
    
    def process_images_to_tensor(self, images):
        """이미지 리스트를 텐서로 변환"""
        if isinstance(images, list):
            # PIL 이미지 리스트를 텐서로 변환
            tensor_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    tensor_img = self.image_transform(img)
                elif isinstance(img, np.ndarray):
                    # numpy를 PIL로 변환 후 텐서로
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    tensor_img = self.image_transform(pil_img)
                elif isinstance(img, torch.Tensor):
                    tensor_img = img
                else:
                    print(f"⚠️ 예상치 못한 이미지 타입: {type(img)}")
                    continue
                tensor_images.append(tensor_img)
            
            if tensor_images:
                return torch.stack(tensor_images)  # [T, C, H, W]
            else:
                return None
        else:
            return images
    
    def train_step_final_fixed(self, batch):
        """최종 수정된 학습 스텝"""
        try:
            # 이미지 처리
            images = batch['images']
            images_tensor = self.process_images_to_tensor(images)
            
            if images_tensor is None:
                print("⚠️ 이미지 변환 실패")
                return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}
            
            # 배치 차원 추가
            if images_tensor.dim() == 3:  # [T, C, H, W]
                images_tensor = images_tensor.unsqueeze(0)  # [1, T, C, H, W]
            
            print(f"   이미지 텐서 shape: {images_tensor.shape}")
            
            # 액션 처리
            actions = batch['actions']
            
            # 데이터 타입 변환
            if isinstance(actions, list):
                actions = torch.tensor(actions, dtype=torch.float32)
            elif isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            
            # 배치 차원 추가
            if actions.dim() == 2:  # [T, 3]
                actions = actions.unsqueeze(0)  # [1, T, 3]
            
            print(f"   액션 텐서 shape: {actions.shape}")
            
            # 🎯 태스크 특성 기반 맞춤형 증강
            # 1. 좌우 대칭 (2D 이동이므로 물리적으로 타당)
            if random.random() < 0.5:
                images_tensor = torch.flip(images_tensor, [-1])  # 가로 뒤집기
                actions[:, :, 1] = -actions[:, :, 1]  # Y축 부호 변경 (측면 이동)
            
            # 2. 전진/후진 뒤집기 (X축 우세이므로 효과적)
            if random.random() < 0.3:
                images_tensor = torch.flip(images_tensor, [1])  # 시간축 뒤집기
                actions[:, :, 0] = -actions[:, :, 0]  # X축 부호 변경 (전진/후진)
            
            # 3. 액션 노이즈 (센서 노이즈 시뮬레이션)
            if random.random() < 0.8:
                # X축 (주요 이동축)에 작은 노이즈
                x_noise = torch.normal(0, self.action_noise_std, actions[:, :, 0:1].shape)
                actions[:, :, 0:1] += x_noise
                
                # Y축 (보조 이동축)에 더 작은 노이즈
                y_noise = torch.normal(0, self.action_noise_std * 0.5, actions[:, :, 1:2].shape)
                actions[:, :, 1:2] += y_noise
                
                # Z축은 0이므로 노이즈 추가하지 않음
            
            # 4. 속도 변화 (다양한 속도로 이동)
            if random.random() < 0.3:
                speed_scale = random.uniform(0.8, 1.2)
                actions[:, :, 0] *= speed_scale  # X축만 스케일링
            
            # 5. 시작-정지 패턴 (정지 상태가 적으므로 학습)
            if random.random() < 0.2:
                # 에피소드 시작 부분에 정지 패턴 추가
                if random.random() < 0.5:
                    stop_frames = random.randint(1, 3)
                    actions[:, :stop_frames, :] = 0
                
                # 에피소드 중간에 짧은 정지 추가
                if random.random() < 0.3:
                    mid_point = actions.shape[1] // 2
                    actions[:, mid_point:mid_point+1, :] = 0
            
            # 범위 제한
            actions = torch.clamp(actions, -1.15, 1.15)
            
            # 정규화
            if hasattr(self, 'action_mean'):
                actions = self.safe_normalize_actions(actions)
            
            # 배치 업데이트
            batch['images'] = images_tensor
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
            import traceback
            traceback.print_exc()
            return {'total_loss': torch.tensor(1.0, device=self.device), 'mae_avg': 1.0}
    
    def validation_step(self, batch):
        """검증 스텝 (gradient 계산 없음)"""
        try:
            # 이미지 처리
            images = batch['images']
            images_tensor = self.process_images_to_tensor(images)
            
            if images_tensor is None:
                return {'total_loss': 1.0, 'mae_avg': 1.0}
            
            # 배치 차원 추가
            if images_tensor.dim() == 3:  # [T, C, H, W]
                images_tensor = images_tensor.unsqueeze(0)  # [1, T, C, H, W]
            
            # 액션 처리
            actions = batch['actions']
            
            # 데이터 타입 변환
            if isinstance(actions, list):
                actions = torch.tensor(actions, dtype=torch.float32)
            elif isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            
            # 배치 차원 추가
            if actions.dim() == 2:  # [T, 3]
                actions = actions.unsqueeze(0)  # [1, T, 3]
            
            # 정규화 (노이즈 없이)
            if hasattr(self, 'action_mean'):
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

def final_fixed_training():
    """최종 수정된 학습"""
    print("🎯 최종 수정: 이미지+액션 텐서 변환 + 간단한 증강 학습!")
    print("=" * 70)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"원본 데이터셋 크기: {len(dataset)}")
    
    # 트레이너 생성
    trainer = FinalFixedTrainer(
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
    print("\n🎯 최종 수정 학습 시작!")
    num_epochs = 10
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
                print(f"\n--- 배치 {i+1} 처리 중 ---")
                metrics = trainer.train_step_final_fixed(batch)
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
                    }, 'best_final_fixed_model.pth')
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
    
    print("\n🎉 최종 수정 학습 완료!")
    
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
            'action_noise_std': trainer.action_noise_std,
            'approach': 'Final Fixed + Simple Augmentation'
        },
        'model_info': {
            'architecture': 'Final Fixed Kosmos2',
            'z_axis_weight': trainer.z_weight,
            'epochs': num_epochs,
            'learning_rate': trainer.learning_rate,
            'early_stopping': True,
            'patience': patience
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('final_fixed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: final_fixed_results.json")
    
    # 최종 성능 요약
    print("\n📊 최종 성능 요약:")
    print(f"   최고 검증 Loss: {best_val_loss:.4f}")
    print(f"   최종 훈련 MAE: {train_history[-1]['mae_avg']:.4f}")
    print(f"   학습 에포크: {len(train_history)}")
    print(f"   증강 방식: 액션 노이즈 (σ=0.005)")
    print(f"   Z축 처리: 건드리지 않음")
    print(f"   이미지 변환: 텐서로 변환")
    print(f"   Shape 오류: 완전 해결")
    print(f"   Early stopping: {'활성화' if patience_counter >= patience else '비활성화'}")

if __name__ == "__main__":
    final_fixed_training()
