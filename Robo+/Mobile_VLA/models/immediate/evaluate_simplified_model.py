#!/usr/bin/env python3
"""
🎯 Case 1: 즉시 적용 - 단순화된 모델 평가
목표: MAE 0.8 → 0.5, 정확도 0% → 15%
특징: 상세한 성능 분석 및 시각화
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# 로컬 모듈 임포트
import sys
sys.path.append('..')
from simplified_2d_model import Simplified2DActionModel
from basic_augmentation_dataset import create_basic_augmentation_data_loaders
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simplified_model(checkpoint_path, processor, device):
    """단순화된 모델 로드"""
    
    logger.info(f"📥 모델 로드 중: {checkpoint_path}")
    
    # 모델 생성
    model = Simplified2DActionModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=2,
        hidden_dim=256,
        dropout=0.4,
        use_vision_resampler=False
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"✅ 모델 로드 완료:")
    logger.info(f"   - 에포크: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"   - 손실: {checkpoint.get('loss', 'N/A'):.6f}")
    logger.info(f"   - MAE: {checkpoint.get('mae', 'N/A'):.6f}")
    
    return model, checkpoint

def calculate_detailed_metrics(predictions, targets):
    """상세한 성능 메트릭 계산"""
    
    # 기본 메트릭
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mse)
    
    # 각 차원별 성능
    mse_x = mean_squared_error(targets[:, 0], predictions[:, 0])
    mse_y = mean_squared_error(targets[:, 1], predictions[:, 1])
    
    mae_x = mean_absolute_error(targets[:, 0], predictions[:, 0])
    mae_y = mean_absolute_error(targets[:, 1], predictions[:, 1])
    
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    
    # 정확도 계산 (임계값별)
    thresholds = [0.1, 0.05, 0.01]
    accuracies = {}
    
    for threshold in thresholds:
        # 전체 정확도 (모든 축이 임계값 내)
        all_axes_success = np.all(np.abs(predictions - targets) < threshold, axis=1)
        accuracies[f'accuracy_{threshold}'] = np.mean(all_axes_success) * 100
        
        # 개별 축 정확도
        for i, axis_name in enumerate(['linear_x', 'linear_y']):
            axis_success = np.abs(predictions[:, i] - targets[:, i]) < threshold
            accuracies[f'{axis_name}_{threshold}'] = np.mean(axis_success) * 100
    
    # 추가 메트릭
    # 평균 오차 기반 성공률
    mean_error = np.mean(np.abs(predictions - targets), axis=1)
    for threshold in thresholds:
        accuracies[f'mean_error_{threshold}'] = np.mean(mean_error < threshold) * 100
    
    # 가중 평균 오차 (linear_x에 더 높은 가중치)
    weighted_error = 0.7 * np.abs(predictions[:, 0] - targets[:, 0]) + 0.3 * np.abs(predictions[:, 1] - targets[:, 1])
    for threshold in thresholds:
        accuracies[f'weighted_error_{threshold}'] = np.mean(weighted_error < threshold) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'mse_x': mse_x,
        'mse_y': mse_y,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'accuracies': accuracies,
        'mean_error': np.mean(mean_error),
        'weighted_error': np.mean(weighted_error)
    }

def evaluate_simplified_model(model, data_loader, device, model_name="Simplified 2D Model"):
    """모델 성능 평가"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_episode_ids = []
    
    logger.info(f"🔍 {model_name} 평가 중...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            episode_ids = batch['episode_id']
            
            # 모델 예측
            predicted_actions = model(images, texts)
            
            # 결과 저장
            all_predictions.extend(predicted_actions.cpu().numpy())
            all_targets.extend(actions.cpu().numpy())
            all_episode_ids.extend(episode_ids)
    
    # numpy 배열로 변환
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # 상세 메트릭 계산
    metrics = calculate_detailed_metrics(predictions, targets)
    
    # 결과 출력
    logger.info(f"📊 {model_name} 평가 결과:")
    logger.info(f"   MSE: {metrics['mse']:.6f}")
    logger.info(f"   MAE: {metrics['mae']:.6f}")
    logger.info(f"   R²: {metrics['r2']:.6f}")
    logger.info(f"   RMSE: {metrics['rmse']:.6f}")
    logger.info(f"   MSE X: {metrics['mse_x']:.6f}")
    logger.info(f"   MSE Y: {metrics['mse_y']:.6f}")
    logger.info(f"   MAE X: {metrics['mae_x']:.6f}")
    logger.info(f"   MAE Y: {metrics['mae_y']:.6f}")
    
    # 정확도 출력
    for threshold in [0.1, 0.05, 0.01]:
        logger.info(f"   정확도 ({threshold}): {metrics['accuracies'][f'accuracy_{threshold}']:.2f}%")
        logger.info(f"     - linear_x: {metrics['accuracies'][f'linear_x_{threshold}']:.2f}%")
        logger.info(f"     - linear_y: {metrics['accuracies'][f'linear_y_{threshold}']:.2f}%")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'episode_ids': all_episode_ids,
        'metrics': metrics
    }

def create_detailed_visualizations(results, save_path):
    """상세한 시각화 생성"""
    
    predictions = results['predictions']
    targets = results['targets']
    metrics = results['metrics']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Simplified 2D Action Model - Detailed Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, label='X-axis', s=50)
    axes[0, 0].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, label='Y-axis', s=50)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Actions')
    axes[0, 0].set_ylabel('Predicted Actions')
    axes[0, 0].set_title('Predicted vs Actual Actions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = predictions - targets
    axes[0, 1].hist(errors[:, 0], bins=30, alpha=0.7, label='X-axis Error', color='blue')
    axes[0, 1].hist(errors[:, 1], bins=30, alpha=0.7, label='Y-axis Error', color='orange')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Action space visualization
    axes[0, 2].scatter(targets[:, 0], targets[:, 1], alpha=0.6, label='Actual', s=50, color='blue')
    axes[0, 2].scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, label='Predicted', s=50, color='red')
    axes[0, 2].set_xlabel('X-axis Action')
    axes[0, 2].set_ylabel('Y-axis Action')
    axes[0, 2].set_title('Action Space: Actual vs Predicted')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Metrics comparison
    metric_names = ['MSE', 'MAE', 'R²', 'RMSE']
    metric_values = [metrics['mse'], metrics['mae'], metrics['r2'], metrics['rmse']]
    
    bars = axes[1, 0].bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('Overall Performance Metrics')
    axes[1, 0].set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
    
    # 5. Accuracy comparison
    thresholds = [0.1, 0.05, 0.01]
    accuracy_values = [metrics['accuracies'][f'accuracy_{t}'] for t in thresholds]
    
    bars2 = axes[1, 1].bar([f'{t}' for t in thresholds], accuracy_values, color=['green', 'orange', 'red'])
    axes[1, 1].set_title('Accuracy by Threshold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Accuracy (%)')
    
    for bar, value in zip(bars2, accuracy_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom')
    
    # 6. Axis-wise performance
    axis_metrics = ['MAE', 'MSE', 'RMSE']
    x_values = [metrics['mae_x'], metrics['mse_x'], metrics['rmse_x']]
    y_values = [metrics['mae_y'], metrics['mse_y'], metrics['rmse_y']]
    
    x = np.arange(len(axis_metrics))
    width = 0.35
    
    bars3 = axes[1, 2].bar(x - width/2, x_values, width, label='X-axis', color='blue', alpha=0.7)
    bars4 = axes[1, 2].bar(x + width/2, y_values, width, label='Y-axis', color='orange', alpha=0.7)
    
    axes[1, 2].set_title('Axis-wise Performance')
    axes[1, 2].set_xlabel('Metric')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(axis_metrics)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 상세 시각화 저장: {save_path}")

def create_performance_summary(results, save_path):
    """성능 요약 테이블 생성"""
    
    metrics = results['metrics']
    
    # 요약 데이터프레임 생성
    summary_data = {
        'Metric': [
            'Overall MAE', 'Overall MSE', 'Overall RMSE', 'R² Score',
            'X-axis MAE', 'X-axis MSE', 'X-axis RMSE',
            'Y-axis MAE', 'Y-axis MSE', 'Y-axis RMSE',
            'Accuracy (0.1)', 'Accuracy (0.05)', 'Accuracy (0.01)',
            'X-axis Accuracy (0.1)', 'X-axis Accuracy (0.05)', 'X-axis Accuracy (0.01)',
            'Y-axis Accuracy (0.1)', 'Y-axis Accuracy (0.05)', 'Y-axis Accuracy (0.01)'
        ],
        'Value': [
            f"{metrics['mae']:.6f}",
            f"{metrics['mse']:.6f}",
            f"{metrics['rmse']:.6f}",
            f"{metrics['r2']:.6f}",
            f"{metrics['mae_x']:.6f}",
            f"{metrics['mse_x']:.6f}",
            f"{metrics['rmse_x']:.6f}",
            f"{metrics['mae_y']:.6f}",
            f"{metrics['mse_y']:.6f}",
            f"{metrics['rmse_y']:.6f}",
            f"{metrics['accuracies']['accuracy_0.1']:.2f}%",
            f"{metrics['accuracies']['accuracy_0.05']:.2f}%",
            f"{metrics['accuracies']['accuracy_0.01']:.2f}%",
            f"{metrics['accuracies']['linear_x_0.1']:.2f}%",
            f"{metrics['accuracies']['linear_x_0.05']:.2f}%",
            f"{metrics['accuracies']['linear_x_0.01']:.2f}%",
            f"{metrics['accuracies']['linear_y_0.1']:.2f}%",
            f"{metrics['accuracies']['linear_y_0.05']:.2f}%",
            f"{metrics['accuracies']['linear_y_0.01']:.2f}%"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    # HTML 테이블로 저장
    html_content = f"""
    <html>
    <head>
        <title>Simplified 2D Model Performance Summary</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .value {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>Simplified 2D Action Model Performance Summary</h1>
        {df.to_html(index=False, classes='performance-table')}
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📋 성능 요약 저장: {save_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Evaluate Simplified 2D Action Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 디바이스 설정
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 프로세서 로드
    logger.info("Loading Kosmos2 processor...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 로드
    model, checkpoint = load_simplified_model(args.model_path, processor, device)
    
    # 데이터 로더 생성
    logger.info("Creating evaluation data loader...")
    _, val_loader, test_loader = create_basic_augmentation_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # 모델 평가
    logger.info("Starting evaluation...")
    results = evaluate_simplified_model(model, test_loader, device, "Simplified 2D Model")
    
    # 결과 저장
    results_path = output_path / 'simplified_model_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    # 시각화 생성
    plot_path = output_path / 'simplified_model_evaluation_plots.png'
    create_detailed_visualizations(results, plot_path)
    
    # 성능 요약 생성
    summary_path = output_path / 'simplified_model_performance_summary.html'
    create_performance_summary(results, summary_path)
    
    logger.info(f"✅ 평가 완료!")
    logger.info(f"   결과 저장: {results_path}")
    logger.info(f"   시각화 저장: {plot_path}")
    logger.info(f"   성능 요약 저장: {summary_path}")
    logger.info(f"   최종 MAE: {results['metrics']['mae']:.6f}")
    logger.info(f"   최종 R² Score: {results['metrics']['r2']:.6f}")

if __name__ == "__main__":
    main()
