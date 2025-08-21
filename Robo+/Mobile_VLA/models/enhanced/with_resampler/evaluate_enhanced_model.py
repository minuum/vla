"""
📊 Evaluate Enhanced 2D Model with Vision Resampler
Vision Resampler를 포함한 향상된 2D 모델 평가 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_2d_model_complete import Enhanced2DActionModel
from enhanced_dataset import create_enhanced_data_loaders

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_accuracy(predictions, targets, threshold=0.1):
    """Calculate accuracy based on error threshold"""
    errors = torch.abs(predictions - targets)
    correct = (errors < threshold).all(dim=1)
    return correct.float().mean().item()

def calculate_individual_accuracy(predictions, targets, threshold=0.1):
    """Calculate accuracy for each action dimension"""
    errors = torch.abs(predictions - targets)
    accuracies = {}
    
    for i, dim_name in enumerate(['linear_x', 'linear_y']):
        correct = errors[:, i] < threshold
        accuracies[dim_name] = correct.float().mean().item()
    
    return accuracies

def evaluate_enhanced_2d_model(
    model,
    test_loader,
    device='cuda',
    save_dir='evaluation_results'
):
    """Evaluate enhanced 2D action model with Vision Resampler"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Evaluation metrics
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_accuracy_01 = 0
    total_accuracy_005 = 0
    total_accuracy_001 = 0
    
    # Individual dimension metrics
    individual_accuracies = {
        'linear_x': {'0.1': 0, '0.05': 0, '0.01': 0},
        'linear_y': {'0.1': 0, '0.05': 0, '0.01': 0}
    }
    
    # Store predictions and targets for detailed analysis
    all_predictions = []
    all_targets = []
    all_episode_ids = []
    
    num_batches = 0
    
    logger.info("🔍 Starting evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            episode_ids = batch['episode_id']
            
            # Forward pass
            predictions = model(images, texts)
            
            # Calculate loss
            loss = criterion(predictions, actions)
            total_loss += loss.item()
            
            # Calculate MAE and RMSE
            mae = torch.mean(torch.abs(predictions - actions)).item()
            rmse = torch.sqrt(torch.mean((predictions - actions) ** 2)).item()
            total_mae += mae
            total_rmse += rmse
            
            # Calculate accuracy at different thresholds
            acc_01 = calculate_accuracy(predictions, actions, threshold=0.1)
            acc_005 = calculate_accuracy(predictions, actions, threshold=0.05)
            acc_001 = calculate_accuracy(predictions, actions, threshold=0.01)
            
            total_accuracy_01 += acc_01
            total_accuracy_005 += acc_005
            total_accuracy_001 += acc_001
            
            # Calculate individual dimension accuracy
            ind_acc = calculate_individual_accuracy(predictions, actions, threshold=0.1)
            for dim, acc in ind_acc.items():
                individual_accuracies[dim]['0.1'] += acc
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(actions.cpu().numpy())
            all_episode_ids.extend(episode_ids)
            
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_accuracy_01 = total_accuracy_01 / num_batches
    avg_accuracy_005 = total_accuracy_005 / num_batches
    avg_accuracy_001 = total_accuracy_001 / num_batches
    
    # Calculate individual dimension averages
    for dim in individual_accuracies:
        for threshold in individual_accuracies[dim]:
            individual_accuracies[dim][threshold] /= num_batches
    
    # Combine all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate detailed metrics
    detailed_metrics = {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'accuracy_0.1': avg_accuracy_01,
        'accuracy_0.05': avg_accuracy_005,
        'accuracy_0.01': avg_accuracy_001,
        'individual_accuracies': individual_accuracies,
        'num_samples': len(all_predictions),
        'model_config': {
            'use_vision_resampler': model.use_vision_resampler,
            'action_dim': model.action_dim,
            'hidden_dim': model.hidden_dim,
            'dropout': model.dropout
        }
    }
    
    # Save detailed results
    results_path = os.path.join(save_dir, 'enhanced_2d_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2, default=str)
    
    # Create detailed analysis DataFrame
    df_results = pd.DataFrame({
        'episode_id': all_episode_ids,
        'pred_linear_x': all_predictions[:, 0],
        'pred_linear_y': all_predictions[:, 1],
        'true_linear_x': all_targets[:, 0],
        'true_linear_y': all_targets[:, 1],
        'error_linear_x': np.abs(all_predictions[:, 0] - all_targets[:, 0]),
        'error_linear_y': np.abs(all_predictions[:, 1] - all_targets[:, 1])
    })
    
    # Save detailed DataFrame
    df_path = os.path.join(save_dir, 'enhanced_2d_detailed_results.csv')
    df_results.to_csv(df_path, index=False)
    
    # Create visualizations
    create_evaluation_plots(df_results, save_dir)
    
    # Print results
    logger.info("📊 Enhanced 2D Model Evaluation Results:")
    logger.info(f"   - Loss: {avg_loss:.6f}")
    logger.info(f"   - MAE: {avg_mae:.6f}")
    logger.info(f"   - RMSE: {avg_rmse:.6f}")
    logger.info(f"   - Accuracy (0.1): {avg_accuracy_01:.4f}")
    logger.info(f"   - Accuracy (0.05): {avg_accuracy_005:.4f}")
    logger.info(f"   - Accuracy (0.01): {avg_accuracy_001:.4f}")
    logger.info(f"   - Individual Accuracies (0.1):")
    for dim, acc in individual_accuracies.items():
        logger.info(f"     * {dim}: {acc['0.1']:.4f}")
    
    logger.info(f"   - Results saved to: {save_dir}")
    
    return detailed_metrics

def create_evaluation_plots(df_results, save_dir):
    """Create evaluation plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced 2D Model with Vision Resampler - Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs Target scatter plot
    axes[0, 0].scatter(df_results['true_linear_x'], df_results['pred_linear_x'], alpha=0.6, label='Linear X')
    axes[0, 0].scatter(df_results['true_linear_y'], df_results['pred_linear_y'], alpha=0.6, label='Linear Y')
    axes[0, 0].plot([df_results['true_linear_x'].min(), df_results['true_linear_x'].max()], 
                    [df_results['true_linear_x'].min(), df_results['true_linear_x'].max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Prediction vs Target')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution
    axes[0, 1].hist(df_results['error_linear_x'], bins=30, alpha=0.7, label='Linear X Error', density=True)
    axes[0, 1].hist(df_results['error_linear_y'], bins=30, alpha=0.7, label='Linear Y Error', density=True)
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error by episode
    episode_errors = df_results.groupby('episode_id')[['error_linear_x', 'error_linear_y']].mean()
    axes[1, 0].plot(episode_errors.index, episode_errors['error_linear_x'], label='Linear X', alpha=0.7)
    axes[1, 0].plot(episode_errors.index, episode_errors['error_linear_y'], label='Linear Y', alpha=0.7)
    axes[1, 0].set_xlabel('Episode ID')
    axes[1, 0].set_ylabel('Mean Error')
    axes[1, 0].set_title('Error by Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of errors
    error_data = [df_results['error_linear_x'], df_results['error_linear_y']]
    axes[1, 1].boxplot(error_data, labels=['Linear X', 'Linear Y'])
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error Distribution (Box Plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'enhanced_2d_evaluation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Evaluation plots saved to: {plot_path}")

def compare_with_previous_models(evaluation_results, save_dir):
    """Compare with previous model results"""
    
    # Load previous results if available
    previous_results_path = "../../accurate_2d_action_evaluation_results.json"
    
    if os.path.exists(previous_results_path):
        with open(previous_results_path, 'r') as f:
            previous_results = json.load(f)
        
        # Create comparison
        comparison = {
            'enhanced_2d_with_resampler': evaluation_results,
            'previous_2d_model': previous_results
        }
        
        # Save comparison
        comparison_path = os.path.join(save_dir, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Print comparison
        logger.info("📊 Model Comparison:")
        logger.info("   Enhanced 2D with Vision Resampler:")
        logger.info(f"     - MAE: {evaluation_results['mae']:.6f}")
        logger.info(f"     - RMSE: {evaluation_results['rmse']:.6f}")
        logger.info(f"     - Accuracy (0.1): {evaluation_results['accuracy_0.1']:.4f}")
        
        logger.info("   Previous 2D Model:")
        logger.info(f"     - MAE: {previous_results['mae']:.6f}")
        logger.info(f"     - RMSE: {previous_results['rmse']:.6f}")
        logger.info(f"     - Accuracy (0.1): {previous_results['accuracy_0.1']:.4f}")
        
        # Calculate improvements
        mae_improvement = (previous_results['mae'] - evaluation_results['mae']) / previous_results['mae'] * 100
        rmse_improvement = (previous_results['rmse'] - evaluation_results['rmse']) / previous_results['rmse'] * 100
        acc_improvement = (evaluation_results['accuracy_0.1'] - previous_results['accuracy_0.1']) / previous_results['accuracy_0.1'] * 100
        
        logger.info("   Improvements:")
        logger.info(f"     - MAE: {mae_improvement:+.2f}%")
        logger.info(f"     - RMSE: {rmse_improvement:+.2f}%")
        logger.info(f"     - Accuracy: {acc_improvement:+.2f}%")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Enhanced 2D Model with Vision Resampler')
    parser.add_argument('--model_path', type=str, default='checkpoints/enhanced_2d_model_best.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/', 
                       help='Path to H5 data directory')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', 
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Device setup
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load processor
    logger.info("Loading Kosmos2 processor...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # Create enhanced model
    logger.info("Creating enhanced 2D action model...")
    model = Enhanced2DActionModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=2,
        hidden_dim=512,
        dropout=0.2,
        use_vision_resampler=True
    )
    
    # Load trained weights (handle dynamic adapters)
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Filter out dynamic adapter keys
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('language_adapter.') and not key.startswith('fusion_adapter.'):
                filtered_state_dict[key] = value
        
        # Load filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"Model loaded successfully (Epoch: {checkpoint.get('epoch', 'Unknown')})")
        if missing_keys:
            logger.info(f"Missing keys (dynamic adapters): {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys: {unexpected_keys}")
    else:
        logger.warning(f"Model checkpoint not found: {args.model_path}")
        logger.info("Using untrained model for evaluation")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    _, test_loader = create_enhanced_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=args.batch_size,
        train_split=0.8,
        frame_selection='random',
        use_vision_resampler=True
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    evaluation_results = evaluate_enhanced_2d_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir
    )
    
    # Compare with previous models
    compare_with_previous_models(evaluation_results, args.save_dir)
    
    logger.info("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
