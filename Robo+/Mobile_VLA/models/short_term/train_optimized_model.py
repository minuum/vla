#!/usr/bin/env python3
"""
🚀 Case 2: 단기 적용 - 최적화된 모델 훈련
목표: MAE 0.5 → 0.3, 정확도 15% → 35%
특징: Vision Resampler 최적화 + CLIP Normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 로컬 모듈 임포트
import sys
sys.path.append('..')
from clip_normalized_model_v2 import CLIPNormalized2DActionModelV2, CLIPNormalizedTrainerV2
from enhanced_dataset import create_enhanced_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def train_optimized_model(data_path, output_dir, num_epochs=50, batch_size=2, 
                         learning_rate=3e-5, weight_decay=1e-3, patience=5,
                         use_clip_normalization=True, use_vision_resampler=True):
    """
    최적화된 모델 훈련
    - CLIP Normalization 적용
    - Vision Resampler 최적화
    - 고급 데이터 증강
    """
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Case 2 훈련 시작 - 디바이스: {device}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 프로세서 로드
    logger.info("📥 Kosmos2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 로더 생성
    logger.info("📊 고급 데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(
        data_path=data_path,
        processor=processor,
        batch_size=batch_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # 모델 및 훈련기 생성
    logger.info("🤖 CLIP Normalized 모델 생성 중...")
    model = CLIPNormalized2DActionModelV2(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,  # Kosmos2 text 모델 출력 차원
        action_dim=2,
        hidden_dim=256,  # Case 1과 동일
        dropout=0.3,     # 0.4 → 0.3 (약간 감소)
        use_vision_resampler=use_vision_resampler,
        use_clip_normalization=use_clip_normalization
    )
    
    trainer = CLIPNormalizedTrainerV2(
        model=model,
        device=device,
        learning_rate=learning_rate,  # 3e-5 (더 낮은 학습률)
        weight_decay=weight_decay
    )
    
    # 조기 종료 설정
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    val_maes = []
    best_mae = float('inf')
    
    logger.info(f"🎯 Case 2 훈련 설정:")
    logger.info(f"   - 에포크: {num_epochs}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 학습률: {learning_rate}")
    logger.info(f"   - Weight Decay: {weight_decay}")
    logger.info(f"   - CLIP Normalization: {use_clip_normalization}")
    logger.info(f"   - Vision Resampler: {use_vision_resampler}")
    logger.info(f"   - 조기 종료 인내심: {patience}")
    
    # 훈련 루프
    for epoch in range(num_epochs):
        logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in train_pbar:
            loss = trainer.train_step(batch)
            train_loss += loss
            train_batches += 1
            train_pbar.set_postfix({'Loss': f'{loss:.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        with torch.no_grad():
            for batch in val_pbar:
                loss, mae = trainer.validate_step(batch)
                val_loss += loss
                val_mae += mae
                val_batches += 1
                val_pbar.set_postfix({'Loss': f'{loss:.6f}', 'MAE': f'{mae:.6f}'})
        
        avg_val_loss = val_loss / val_batches
        avg_val_mae = val_mae / val_batches
        val_losses.append(avg_val_loss)
        val_maes.append(avg_val_mae)
        
        # 스케줄러 업데이트
        trainer.scheduler.step()
        
        # 로그 출력
        logger.info(f"   📊 훈련 손실: {avg_train_loss:.6f}")
        logger.info(f"   📊 검증 손실: {avg_val_loss:.6f}")
        logger.info(f"   📊 검증 MAE: {avg_val_mae:.6f}")
        logger.info(f"   📊 학습률: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # 최고 성능 체크포인트 저장
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            best_checkpoint_path = output_path / f"best_case2_model_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(best_checkpoint_path, epoch+1, avg_val_loss, avg_val_mae)
            logger.info(f"   🏆 새로운 최고 성능! MAE: {best_mae:.6f}")
        
        # 정기 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"case2_model_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(checkpoint_path, epoch+1, avg_val_loss, avg_val_mae)
        
        # 조기 종료 체크
        if early_stopping(avg_val_mae, model):
            logger.info(f"   ⏹️ 조기 종료! {patience} 에포크 동안 개선 없음")
            break
    
    # 최종 모델 저장
    final_checkpoint_path = output_path / "final_case2_model.pth"
    trainer.save_checkpoint(final_checkpoint_path, epoch+1, avg_val_loss, avg_val_mae)
    
    # 훈련 결과 시각화
    plot_training_results(train_losses, val_losses, val_maes, output_path)
    
    # 훈련 결과 저장
    save_training_results(train_losses, val_losses, val_maes, best_mae, output_path, 
                         use_clip_normalization, use_vision_resampler)
    
    # 최종 성능 평가
    evaluate_final_performance(model, test_loader, device, output_path)
    
    logger.info(f"✅ Case 2 훈련 완료!")
    logger.info(f"   - 최고 MAE: {best_mae:.6f}")
    logger.info(f"   - 목표 달성: {'✅' if best_mae < 0.5 else '❌'} (목표: < 0.5)")
    logger.info(f"   - 최종 에포크: {epoch+1}")
    logger.info(f"   - 결과 저장: {output_path}")
    
    return model, trainer

def plot_training_results(train_losses, val_losses, val_maes, output_path):
    """훈련 결과 시각화"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Case 2: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE 그래프 + 목표선
    ax2.plot(epochs, val_maes, 'g-', label='Validation MAE', linewidth=2)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Case 1 Target (0.5)', alpha=0.7)
    ax2.axhline(y=0.3, color='red', linestyle='--', label='Case 2 Target (0.3)', alpha=0.7)
    ax2.set_title('Case 2: Validation MAE Progress', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'case2_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Case 2 훈련 결과 시각화 저장: {output_path / 'case2_training_results.png'}")

def save_training_results(train_losses, val_losses, val_maes, best_mae, output_path, 
                         use_clip_norm, use_vision_resampler):
    """훈련 결과 저장"""
    
    results = {
        'case': 'Case 2: 단기 적용',
        'target': 'MAE 0.5 → 0.3, 정확도 15% → 35%',
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_mae': best_mae,
        'final_epoch': len(train_losses),
        'target_achieved': best_mae < 0.3,
        'training_config': {
            'model_type': 'CLIPNormalized2DActionModel',
            'hidden_dim': 256,
            'action_dim': 2,
            'dropout': 0.3,
            'learning_rate': 3e-5,
            'weight_decay': 1e-3,
            'batch_size': 2,
            'use_clip_normalization': use_clip_norm,
            'use_vision_resampler': use_vision_resampler
        },
        'features': {
            'optimized_vision_resampler': 'latents 64→16, heads 8→4',
            'clip_normalization': 'feature alignment 개선',
            'enhanced_augmentation': '시간적/공간적/고급 증강',
            'improved_hyperparameters': '학습률 최적화'
        }
    }
    
    with open(output_path / 'case2_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Case 2 훈련 결과 저장: {output_path / 'case2_training_results.json'}")

def evaluate_final_performance(model, test_loader, device, output_path):
    """최종 성능 평가"""
    
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_batches = 0
    all_predictions = []
    all_targets = []
    
    logger.info("🔍 Case 2 최종 성능 평가 중...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Case 2"):
            images = batch['image']  # PIL 이미지 리스트
            actions = batch['action'].to(device)
            texts = batch['text']
            
            predicted_actions = model(images, texts)
            
            # 손실 계산
            criterion = nn.HuberLoss(delta=0.1)
            loss = criterion(predicted_actions, actions)
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            test_loss += loss.item()
            test_mae += mae.item()
            test_batches += 1
            
            # 예측 결과 저장
            all_predictions.extend(predicted_actions.cpu().numpy())
            all_targets.extend(actions.cpu().numpy())
    
    avg_test_loss = test_loss / test_batches
    avg_test_mae = test_mae / test_batches
    
    # 정확도 계산 (실제 로봇 제어 관점)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 실제 로봇 제어에 적합한 임계값 (더 관대)
    thresholds = [0.3, 0.2, 0.15]  # 0.3m/s, 0.2m/s, 0.15m/s
    accuracies = {}
    
    for threshold in thresholds:
        # 전체 정확도 (모든 축이 임계값 내)
        all_axes_success = np.all(np.abs(all_predictions - all_targets) < threshold, axis=1)
        accuracies[f'accuracy_{threshold}'] = np.mean(all_axes_success) * 100
        
        # 개별 축 정확도
        for i, axis_name in enumerate(['linear_x', 'linear_y']):
            axis_success = np.abs(all_predictions[:, i] - all_targets[:, i]) < threshold
            accuracies[f'{axis_name}_{threshold}'] = np.mean(axis_success) * 100
    
    # 실제 로봇 제어 성능 지표 추가
    # 1. 추적 성능 (목표 지점 근처 도달률)
    tracking_threshold = 0.5  # 0.5m/s 이내면 성공
    tracking_success = np.all(np.abs(all_predictions - all_targets) < tracking_threshold, axis=1)
    tracking_accuracy = np.mean(tracking_success) * 100
    
    # 2. 방향 정확도 (부호가 맞는지)
    direction_correct_x = np.sign(all_predictions[:, 0]) == np.sign(all_targets[:, 0])
    direction_correct_y = np.sign(all_predictions[:, 1]) == np.sign(all_targets[:, 1])
    direction_accuracy_x = np.mean(direction_correct_x) * 100
    direction_accuracy_y = np.mean(direction_correct_y) * 100
    
    # 3. 크기 순서 정확도 (상대적 크기가 맞는지)
    magnitude_order_correct = (
        (all_predictions[:, 0] > all_predictions[:, 1]) == (all_targets[:, 0] > all_targets[:, 1])
    )
    magnitude_order_accuracy = np.mean(magnitude_order_correct) * 100
    
    # 결과 출력
    logger.info(f"📊 Case 2 최종 성능 결과:")
    logger.info(f"   - 테스트 손실: {avg_test_loss:.6f}")
    logger.info(f"   - 테스트 MAE: {avg_test_mae:.6f}")
    logger.info(f"   - 목표 달성: {'✅' if avg_test_mae < 0.3 else '❌'} (목표: < 0.3)")
    
    # 실제 로봇 제어 성능
    logger.info(f"   - 추적 성능 (0.5m/s): {tracking_accuracy:.2f}%")
    logger.info(f"   - 방향 정확도:")
    logger.info(f"     - linear_x: {direction_accuracy_x:.2f}%")
    logger.info(f"     - linear_y: {direction_accuracy_y:.2f}%")
    logger.info(f"   - 크기 순서 정확도: {magnitude_order_accuracy:.2f}%")
    
    for threshold in thresholds:
        logger.info(f"   - 정확도 ({threshold}): {accuracies[f'accuracy_{threshold}']:.2f}%")
        logger.info(f"     - linear_x: {accuracies[f'linear_x_{threshold}']:.2f}%")
        logger.info(f"     - linear_y: {accuracies[f'linear_y_{threshold}']:.2f}%")
    
    # 결과 저장
    test_results = {
        'test_loss': avg_test_loss,
        'test_mae': avg_test_mae,
        'target_achieved': avg_test_mae < 0.3,
        'accuracies': accuracies,
        'tracking_accuracy': tracking_accuracy,
        'direction_accuracy': {
            'linear_x': direction_accuracy_x,
            'linear_y': direction_accuracy_y
        },
        'magnitude_order_accuracy': magnitude_order_accuracy,
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist()
    }
    
    with open(output_path / 'case2_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"💾 Case 2 테스트 결과 저장: {output_path / 'case2_test_results.json'}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Train Case 2: Optimized 2D Action Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='case2_results', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--no_clip_norm', action='store_true', help='Disable CLIP normalization')
    parser.add_argument('--no_vision_resampler', action='store_true', help='Disable Vision Resampler')
    
    args = parser.parse_args()
    
    # 훈련 실행
    model, trainer = train_optimized_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_clip_normalization=not args.no_clip_norm,
        use_vision_resampler=not args.no_vision_resampler
    )

if __name__ == "__main__":
    main()
