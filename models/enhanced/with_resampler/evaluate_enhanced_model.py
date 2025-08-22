#!/usr/bin/env python3
"""
🚀 Enhanced 2D Action Model with Vision Resampler - Performance Evaluation
Vision Resampler를 포함한 향상된 2D 모델의 성능 평가
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_2d_model_complete import Enhanced2DActionModel
from enhanced_dataset import Enhanced2DActionDataset, create_enhanced_data_loaders
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(predictions, targets):
    """성능 메트릭을 계산합니다."""
    # MSE (Mean Squared Error)
    mse = mean_squared_error(targets, predictions)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(targets, predictions)
    
    # R² Score
    r2 = r2_score(targets, predictions)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # 각 차원별 성능
    mse_x = mean_squared_error(targets[:, 0], predictions[:, 0])
    mse_y = mean_squared_error(targets[:, 1], predictions[:, 1])
    
    mae_x = mean_absolute_error(targets[:, 0], predictions[:, 0])
    mae_y = mean_absolute_error(targets[:, 1], predictions[:, 1])
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'mse_x': mse_x,
        'mse_y': mse_y,
        'mae_x': mae_x,
        'mae_y': mae_y
    }

def evaluate_model(model, data_loader, device, model_name="Enhanced 2D Model"):
    """모델 성능을 정확히 평가합니다."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_episode_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            episode_ids = batch['episode_id']
            
            # 모델 예측
            predicted_actions = model(images, texts)
            
            # 결과 저장
            all_predictions.append(predicted_actions.cpu().numpy())
            all_targets.append(actions.cpu().numpy())
            all_episode_ids.extend(episode_ids)
    
    # 결과 합치기
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 성능 메트릭 계산
    metrics = calculate_metrics(predictions, targets)
    
    logger.info(f"📊 {model_name} 평가 결과:")
    logger.info(f"   MSE: {metrics['mse']:.6f}")
    logger.info(f"   MAE: {metrics['mae']:.6f}")
    logger.info(f"   R²: {metrics['r2']:.6f}")
    logger.info(f"   RMSE: {metrics['rmse']:.6f}")
    logger.info(f"   MSE X: {metrics['mse_x']:.6f}")
    logger.info(f"   MSE Y: {metrics['mse_y']:.6f}")
    logger.info(f"   MAE X: {metrics['mae_x']:.6f}")
    logger.info(f"   MAE Y: {metrics['mae_y']:.6f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'episode_ids': all_episode_ids,
        'metrics': metrics
    }

def plot_results(results, save_path):
    """결과를 시각화합니다."""
    predictions = results['predictions']
    targets = results['targets']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced 2D Action Model with Vision Resampler - Evaluation Results', fontsize=16)
    
    # 1. Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, label='X-axis')
    axes[0, 0].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, label='Y-axis')
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Actions')
    axes[0, 0].set_ylabel('Predicted Actions')
    axes[0, 0].set_title('Predicted vs Actual Actions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    errors = predictions - targets
    axes[0, 1].hist(errors[:, 0], bins=30, alpha=0.7, label='X-axis Error')
    axes[0, 1].hist(errors[:, 1], bins=30, alpha=0.7, label='Y-axis Error')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Action space visualization
    axes[1, 0].scatter(targets[:, 0], targets[:, 1], alpha=0.6, label='Actual', s=50)
    axes[1, 0].scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, label='Predicted', s=50)
    axes[1, 0].set_xlabel('X-axis Action')
    axes[1, 0].set_ylabel('Y-axis Action')
    axes[1, 0].set_title('Action Space: Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics summary
    metrics = results['metrics']
    metric_names = ['MSE', 'MAE', 'R²', 'RMSE']
    metric_values = [metrics['mse'], metrics['mae'], metrics['r2'], metrics['rmse']]
    
    bars = axes[1, 1].bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 시각화 결과 저장: {save_path}")

def main():
    """메인 평가 함수"""
    parser = argparse.ArgumentParser(description='Evaluate Enhanced 2D Action Model with Vision Resampler')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Device setup
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load processor
    logger.info("Loading Kosmos2 processor...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = Enhanced2DActionModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=2,
        hidden_dim=512,
        dropout=0.2,
        use_vision_resampler=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create data loader
    logger.info("Creating evaluation data loader...")
    _, val_loader = create_enhanced_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=args.batch_size,
        train_split=0.8,
        frame_selection='random',
        use_vision_resampler=True
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(model, val_loader, device, "Enhanced 2D Model with Vision Resampler")
    
    # Save results
    results_path = Path(args.output_dir) / 'enhanced_model_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    # Plot results
    plot_path = Path(args.output_dir) / 'enhanced_model_evaluation_plots.png'
    plot_results(results, plot_path)
    
    logger.info(f"✅ 평가 완료!")
    logger.info(f"   결과 저장: {results_path}")
    logger.info(f"   시각화 저장: {plot_path}")
    logger.info(f"   최종 R² Score: {results['metrics']['r2']:.6f}")

if __name__ == "__main__":
    main()
