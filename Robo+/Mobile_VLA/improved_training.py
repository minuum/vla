#!/usr/bin/env python3
"""
🚀 Mobile VLA 개선된 학습 스크립트

- LSTM Action Head
- Weighted Loss
- 증강된 데이터셋 (432개 에피소드)
- 더 긴 학습 (20 에포크)
- 향상된 평가 메트릭
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_dataset import MobileVLADataset

def convert_numpy_types(obj):
    """NumPy 타입을 Python 타입으로 변환"""
    import numpy as np
    
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

class ImprovedMobileVLATrainer(MobileVLATrainer):
    """개선된 Mobile VLA 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 학습 파라미터 개선
        self.learning_rate = 1e-4  # 더 낮은 학습률
        self.num_epochs = 20       # 더 긴 학습
        self.gradient_clip_val = 1.0  # 그래디언트 클리핑
        
        # 옵티마이저 개선
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 스케줄러 추가
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )
    
    def compute_loss(self, predictions, targets):
        """개선된 손실 계산 (Weighted Loss)"""
        predicted_actions = predictions["predicted_actions"]
        target_actions = targets["action_chunk"]
        
        if target_actions.dim() == 4:
            target_actions = target_actions[:, -1, :, :]
        
        # Weighted Huber Loss (linear_y에 더 높은 가중치)
        weights = torch.tensor([1.0, 2.0, 1.5], device=predicted_actions.device)
        
        per_dim_loss = F.huber_loss(predicted_actions, target_actions, reduction='none')
        weighted_loss = per_dim_loss * weights.unsqueeze(0).unsqueeze(0)
        action_loss = weighted_loss.mean()
        
        # 각 차원별 MAE 계산
        mae_per_dim = torch.abs(predicted_actions - target_actions).mean(dim=(0, 1))
        
        return {
            "total_loss": action_loss,
            "action_loss": action_loss,
            "mae_linear_x": mae_per_dim[0].item(),
            "mae_linear_y": mae_per_dim[1].item(),
            "mae_angular_z": mae_per_dim[2].item(),
            "mae_avg": mae_per_dim.mean().item()
        }
    
    def train_epoch(self, train_loader):
        """개선된 에포크 학습"""
        self.model.train()
        total_loss = 0
        epoch_metrics = {
            'mae_linear_x': [], 'mae_linear_y': [], 'mae_angular_z': [],
            'mae_avg': [], 'loss': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 데이터를 디바이스로 이동
                images = batch["images"].to(self.device)
                actions = batch["actions"].to(self.device)
                
                # 텍스트 토큰 생성 (더미)
                batch_size = images.size(0)
                input_ids = torch.zeros(batch_size, 10, dtype=torch.long).to(self.device)
                attention_mask = torch.ones(batch_size, 10, dtype=torch.long).to(self.device)
                
                # Window/Chunk 처리 (직접 구현)
                sequence_length = images.size(1)
                window_images = images[:, :min(sequence_length, self.window_size)]
                chunk_actions = actions[:, :min(sequence_length, self.chunk_size)]
                
                # Forward pass
                predictions = self.model(
                    pixel_values=window_images,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 손실 계산
                targets = {"action_chunk": chunk_actions}
                loss_dict = self.compute_loss(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                
                # 메트릭 수집
                total_loss += loss_dict["total_loss"].item()
                for key in epoch_metrics:
                    if key in loss_dict:
                        epoch_metrics[key].append(loss_dict[key])
                
                if batch_idx % 10 == 0:
                    print(f"  배치 {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss_dict['total_loss']:.4f}, "
                          f"MAE={loss_dict['mae_avg']:.4f}")
                
            except Exception as e:
                print(f"배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        # 에포크 평균 계산
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['total_loss'] = total_loss / len(train_loader)
        
        return avg_metrics
    
    def evaluate_model(self, val_loader):
        """개선된 모델 평가"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        val_metrics = {
            'mae_linear_x': [], 'mae_linear_y': [], 'mae_angular_z': [],
            'mae_avg': [], 'loss': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch["images"].to(self.device)
                    actions = batch["actions"].to(self.device)
                    
                    batch_size = images.size(0)
                    input_ids = torch.zeros(batch_size, 10, dtype=torch.long).to(self.device)
                    attention_mask = torch.ones(batch_size, 10, dtype=torch.long).to(self.device)
                    
                    window_images = images[:, :min(images.size(1), self.window_size)]
                    chunk_actions = actions[:, :min(actions.size(1), self.chunk_size)]
                    
                    predictions = self.model(
                        pixel_values=window_images,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    targets = {"action_chunk": chunk_actions}
                    loss_dict = self.compute_loss(predictions, targets)
                    
                    # 예측과 타겟 수집
                    pred_actions = predictions["predicted_actions"].cpu().numpy()
                    target_actions = chunk_actions.cpu().numpy()
                    
                    all_predictions.append(pred_actions)
                    all_targets.append(target_actions)
                    
                    # 메트릭 수집
                    for key in val_metrics:
                        if key in loss_dict:
                            val_metrics[key].append(loss_dict[key])
                    
                except Exception as e:
                    print(f"평가 배치 {batch_idx} 처리 중 오류: {e}")
                    continue
        
        # 전체 메트릭 계산
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 각 차원별 R² 점수 계산
        r2_scores = {}
        for i, dim_name in enumerate(['linear_x', 'linear_y', 'angular_z']):
            pred_flat = all_predictions[:, :, i].flatten()
            target_flat = all_targets[:, :, i].flatten()
            r2_scores[f'r2_{dim_name}'] = r2_score(target_flat, pred_flat)
        
        # 임계값 정확도 계산
        threshold_accuracies = {}
        for threshold in [0.1, 0.2, 0.3]:
            within_threshold = np.abs(all_predictions - all_targets) < threshold
            threshold_accuracies[f'threshold_{threshold}'] = within_threshold.mean()
        
        # 평균 메트릭
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        avg_metrics.update(r2_scores)
        avg_metrics.update(threshold_accuracies)
        
        return avg_metrics, all_predictions, all_targets

def main():
    """메인 실행 함수"""
    print("🚀 Mobile VLA 개선된 학습 시작")
    print("=" * 60)
    
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 디바이스: {device}")
    
    # 데이터셋 경로 (증강된 데이터셋)
    data_dir = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset_augmented"
    print(f"📁 데이터셋: {data_dir}")
    
    # 데이터셋 로드
    print("📊 데이터셋 로딩 중...")
    dataset = MobileVLADataset(data_dir)
    print(f"   총 에피소드: {len(dataset)}개")
    
    # Train/Validation 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"   학습 데이터: {len(train_dataset)}개")
    print(f"   검증 데이터: {len(val_dataset)}개")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2,
        collate_fn=dataset.collater_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2,
        collate_fn=dataset.collater_fn
    )
    
    # 트레이너 초기화
    trainer = ImprovedMobileVLATrainer(
        model_name="microsoft/kosmos-2-patch14-224",
        action_dim=3,
        window_size=8,
        chunk_size=2
    )
    
    # 학습 루프
    print("🎯 학습 시작...")
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(trainer.num_epochs):
        print(f"\n📈 에포크 {epoch+1}/{trainer.num_epochs}")
        print("-" * 40)
        
        # 학습
        train_metrics = trainer.train_epoch(train_loader)
        train_history.append(train_metrics)
        
        # 검증
        val_metrics, _, _ = trainer.evaluate_model(val_loader)
        val_history.append(val_metrics)
        
        # 결과 출력
        print(f"✅ 학습 완료:")
        print(f"   Loss: {train_metrics['total_loss']:.4f}")
        print(f"   MAE: {train_metrics['mae_avg']:.4f}")
        print(f"   Linear X MAE: {train_metrics['mae_linear_x']:.4f}")
        print(f"   Linear Y MAE: {train_metrics['mae_linear_y']:.4f}")
        print(f"   Angular Z MAE: {train_metrics['mae_angular_z']:.4f}")
        
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
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, 'best_improved_model.pth')
            print(f"💾 최고 모델 저장됨 (Loss: {best_val_loss:.4f})")
    
    # 최종 평가
    print("\n🎯 최종 평가...")
    try:
        trainer.model.load_state_dict(torch.load('best_improved_model.pth')['model_state_dict'])
        final_metrics, final_preds, final_targets = trainer.evaluate_model(val_loader)
        
        print(f"\n🏆 최종 성능:")
        print(f"   전체 MAE: {final_metrics['mae_avg']:.4f}")
        print(f"   임계값 정확도 (0.1): {final_metrics['threshold_0.1']:.4f}")
        print(f"   Linear X R²: {final_metrics['r2_linear_x']:.4f}")
        print(f"   Linear Y R²: {final_metrics['r2_linear_y']:.4f}")
        print(f"   Angular Z R²: {final_metrics['r2_angular_z']:.4f}")
        
    except FileNotFoundError:
        print("❌ 최고 모델 파일을 찾을 수 없습니다. 마지막 검증 결과를 사용합니다.")
        final_metrics = val_metrics
    
    # 결과 저장
    results = {
        'final_metrics': convert_numpy_types(final_metrics),
        'train_history': convert_numpy_types(train_history),
        'val_history': convert_numpy_types(val_history),
        'dataset_info': {
            'total_episodes': len(dataset),
            'train_episodes': len(train_dataset),
            'val_episodes': len(val_dataset),
            'augmentation_multiplier': 6.0
        },
        'model_info': {
            'architecture': 'LSTM Action Head',
            'loss_function': 'Weighted Huber Loss',
            'optimizer': 'AdamW with Cosine Annealing',
            'epochs': trainer.num_epochs,
            'learning_rate': trainer.learning_rate
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('improved_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장됨: improved_training_results.json")
    print("🎉 개선된 학습 완료!")

if __name__ == "__main__":
    main()
