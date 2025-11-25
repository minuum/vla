#!/usr/bin/env python3
"""
Case 3 훈련 스크립트 - Case 1 기반
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor
import numpy as np
from sklearn.metrics import r2_score
import json
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_case3_model import create_simple_case3_model
from simple_case3_dataset import SimpleCase3Dataset

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_simple_case3_data_loaders(data_path, processor, batch_size=2,
                                    train_split=0.7, val_split=0.15, test_split=0.15):
    """Case 3 데이터 로더 생성"""
    logger.info("📊 간단한 데이터 로더 생성 중...")
    
    # 전체 데이터셋 생성
    full_dataset = SimpleCase3Dataset(
        data_path=data_path,
        processor=processor,
        frame_selection='random'  # 랜덤 프레임 사용
    )
    
    # 데이터셋 분할
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    logger.info("✅ Simple Case 3 Data Loaders 생성 완료")
    return train_loader, val_loader, test_loader

def custom_collate_fn(batch):
    """커스텀 배치 함수"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    
    return {
        'image': images,
        'text': texts,
        'action': actions
    }

def train_simple_case3_model(data_path, output_dir, num_epochs=5, batch_size=2, 
                            learning_rate=5e-5, weight_decay=1e-3, patience=5):
    """Case 3 모델 훈련"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Case 3 훈련 시작 - 디바이스: {device}")
    
    # Kosmos2 프로세서 로드
    logger.info("📥 Kosmos2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_simple_case3_data_loaders(
        data_path, processor, batch_size
    )
    
    # 모델 및 훈련기 생성
    logger.info("🤖 간단한 모델 및 훈련기 생성 중...")
    model, trainer = create_simple_case3_model(processor, device)
    
    # 훈련 설정
    logger.info("🎯 훈련 설정:")
    logger.info(f"   - 에포크: {num_epochs}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 학습률: {learning_rate}")
    logger.info(f"   - Weight Decay: {weight_decay}")
    logger.info(f"   - 조기 종료 인내심: {patience}")
    
    # 훈련 루프
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # 훈련
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss = trainer.train_step(batch)
                if loss is not None:
                    train_losses.append(loss)
            except Exception as e:
                logger.error(f"❌ 훈련 배치 오류: {e}")
                continue
        
        # 검증
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    loss, mae = trainer.validate_step(batch)
                    if loss is not None and mae is not None:
                        val_losses.append(loss)
                        val_maes.append(mae)
                except Exception as e:
                    logger.error(f"❌ 검증 배치 오류: {e}")
                    continue
        
        # 평균 계산
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        avg_val_mae = np.mean(val_maes) if val_maes else float('inf')
        
        # 결과 출력
        logger.info(f"📊 Epoch {epoch+1} 결과:")
        logger.info(f"   - 훈련 손실: {avg_train_loss:.6f}")
        logger.info(f"   - 검증 손실: {avg_val_loss:.6f}")
        logger.info(f"   - 검증 MAE: {avg_val_mae:.6f}")
        logger.info(f"   - 학습률: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # 조기 종료 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 최고 모델 저장
            checkpoint_path = os.path.join(output_dir, f"best_case3_model.pth")
            trainer.save_checkpoint(checkpoint_path, epoch+1, avg_val_loss, avg_val_mae)
        else:
            patience_counter += 1
            logger.info(f"⏳ 조기 종료 카운터: {patience_counter}/{patience}")
        
        # 학습률 스케줄링
        trainer.scheduler.step()
        
        # 조기 종료
        if patience_counter >= patience:
            logger.info("🛑 조기 종료!")
            break
    
    # 최종 성능 평가
    logger.info("📊 최종 성능 평가 중...")
    test_loss, test_mae, test_results = evaluate_final_performance(model, test_loader, device)
    
    # 결과 저장
    results = {
        'case': 'Case 3',
        'model_type': 'Simple Case 3 Model',
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'patience': patience
        },
        'final_performance': {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'target_achieved': test_mae < 0.3 if test_mae != float('inf') else False
        },
        'test_results': test_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(output_dir, 'case3_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 결과 저장: {results_path}")
    logger.info("✅ Case 3 훈련 완료!")
    
    return model, trainer

def evaluate_final_performance(model, test_loader, device):
    """최종 성능 평가"""
    model.eval()
    all_predictions = []
    all_targets = []
    test_losses = []
    test_maes = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                images = batch['image']
                texts = batch['text']
                targets = batch['action'].to(device)
                
                predictions = model(images, texts)
                
                # 손실 계산
                criterion = nn.HuberLoss(delta=0.1)
                loss = criterion(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))
                
                test_losses.append(loss.item())
                test_maes.append(mae.item())
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
            except Exception as e:
                logger.error(f"❌ 테스트 배치 오류: {e}")
                continue
    
    if not test_losses:
        logger.error("❌ 성공적인 테스트 배치가 없습니다")
        return float('inf'), float('inf'), {}
    
    # 평균 계산
    avg_test_loss = np.mean(test_losses)
    avg_test_mae = np.mean(test_maes)
    
    # 예측 결과 분석
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 정확도 계산 (실제 로봇 제어 관점)
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
    
    # R² 점수 계산 (모델의 예측 능력 측정)
    r2_scores = {}
    for i, axis_name in enumerate(['linear_x', 'linear_y']):
        r2_scores[f'{axis_name}_r2'] = r2_score(all_targets[:, i], all_predictions[:, i])
    
    # 상관관계 계산
    correlations = {}
    for i, axis_name in enumerate(['linear_x', 'linear_y']):
        correlations[f'{axis_name}_corr'] = np.corrcoef(all_targets[:, i], all_predictions[:, i])[0, 1]
    
    # 결과 출력
    logger.info(f"📊 Case 3 최종 성능 결과:")
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
    
    # R² 점수 출력
    logger.info(f"   - R² 점수:")
    logger.info(f"     - linear_x: {r2_scores['linear_x_r2']:.4f}")
    logger.info(f"     - linear_y: {r2_scores['linear_y_r2']:.4f}")
    
    # 상관관계 출력
    logger.info(f"   - 상관관계:")
    logger.info(f"     - linear_x: {correlations['linear_x_corr']:.4f}")
    logger.info(f"     - linear_y: {correlations['linear_y_corr']:.4f}")
    
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
        'r2_scores': r2_scores,
        'correlations': correlations,
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist()
    }
    
    return avg_test_loss, avg_test_mae, test_results

def main():
    parser = argparse.ArgumentParser(description='Case 3 훈련 스크립트')
    parser.add_argument('--data_path', type=str, required=True, help='데이터 경로')
    parser.add_argument('--output_dir', type=str, required=True, help='출력 디렉토리')
    parser.add_argument('--num_epochs', type=int, default=5, help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=2, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='조기 종료 인내심')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 실행
    model, trainer = train_simple_case3_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
