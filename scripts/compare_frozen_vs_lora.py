#!/usr/bin/env python3
"""
Context Vector 및 Latent Space 추출 스크립트
============================================
VLM Frozen vs LoRA 비교를 위한 의미 벡터 추출

목적:
1. Case 3 (Frozen) context vector 추출
2. Case 4 (LoRA) context vector 추출 (학습 후)
3. LSTM latent space 추출
4. 유사도 비교 (Cosine, Euclidean, Correlation)
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import json
import sys
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "RoboVLMs_upstream")


def load_model(checkpoint_path, device='cuda'):
    """
    모델 로드 (Frozen 또는 LoRA)
    """
    print(f"\n{'='*70}")
    print(f"모델 로드: {Path(checkpoint_path).name}")
    print(f"{'='*70}")
    
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = MobileVLATrainer.load_from_checkpoint(
        checkpoint_path, 
        map_location='cpu'
    )
    model.eval()
    model.to(device)
    
    print(f"✅ 모델 로드 완료")
    print(f"   Device: {device}")
    
    return model


def load_sample_images(num_samples=50, seed=42):
    """
    샘플 이미지 로드 (재현 가능하도록)
    
    Returns:
        tensor: (num_samples, 8, 3, 224, 224)
    """
    print(f"\n{'='*70}")
    print(f"샘플 이미지 로드 (seed={seed})")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    h5_files = sorted(list(Path("ROS_action/mobile_vla_dataset").glob("episode_*.h5")))
    
    # 균형있게 샘플링
    left_files = [f for f in h5_files if 'left' in str(f)]
    right_files = [f for f in h5_files if 'right' in str(f)]
    
    # Random sampling
    selected_left = np.random.choice(left_files, num_samples//2, replace=False)
    selected_right = np.random.choice(right_files, num_samples//2, replace=False)
    selected_files = list(selected_left) + list(selected_right)
    
    print(f"  샘플링: {len(selected_left)} left + {len(selected_right)} right")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    sample_images = []
    metadata = []
    
    for h5_file in selected_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                frames = []
                for i in range(min(8, len(f['images']))):
                    img = Image.fromarray(f['images'][i].astype(np.uint8))
                    frames.append(transform(img))
                
                while len(frames) < 8:
                    frames.append(torch.zeros(3, 224, 224))
                
                sample_images.append(torch.stack(frames))
                
                # Metadata
                metadata.append({
                    'file': h5_file.name,
                    'direction': 'left' if 'left' in str(h5_file) else 'right'
                })
        except Exception as e:
            print(f"  ⚠️  {h5_file.name}: {e}")
            continue
    
    images_batch = torch.stack(sample_images)
    print(f"  ✅ Shape: {images_batch.shape}")
    
    return images_batch, metadata


def extract_context_and_latent(model, images, device='cuda'):
    """
    Context vector와 Latent space 추출
    
    Returns:
        context: (N, T, 64, 2048) - VLM 출력
        latent: (N, hidden_size) - LSTM hidden state
        predictions: (N, 2) - Action predictions
    """
    print(f"\n{'='*70}")
    print("Context Vector 및 Latent Space 추출")
    print(f"{'='*70}")
    
    images = images.to(device)
    
    # Hook for latent space
    latent_states = []
    
    def lstm_hook(module, input, output):
        """LSTM forward hook"""
        # output = (output, (h_n, c_n))
        if isinstance(output, tuple) and len(output) == 2:
            h_n, c_n = output[1]
            # h_n shape: (num_layers, batch, hidden_size)
            latent_states.append(h_n[-1].detach().cpu())  # Last layer
    
    # Register hook
    handle = model.model.act_head.rnn.register_forward_hook(lstm_hook)
    
    with torch.no_grad():
        # 1. Context vector 추출
        context = model.model.encode_images(images)
        print(f"  Context shape: {context.shape}")
        
        # 2. Action prediction (이 과정에서 latent 추출됨)
        # Need to prepare full input
        batch_size = images.shape[0]
        
        # Dummy action masks
        action_mask = torch.ones(batch_size, 8, dtype=torch.bool).to(device)
        
        # Forward through action head
        predictions = model.model.act_head(
            context.view(batch_size, -1, context.shape[-1]),  # Flatten temporal
            actions=None,
            action_masks=action_mask
        )
        
        print(f"  Predictions shape: {predictions.shape}")
    
    # Cleanup hook
    handle.remove()
    
    # Get latent
    if latent_states:
        latent = latent_states[0]
        print(f"  Latent shape: {latent.shape}")
    else:
        latent = None
        print(f"  ⚠️  Latent state not captured")
    
    return context.cpu(), latent, predictions.cpu()


def compute_similarity_metrics(context1, context2, name1="Model1", name2="Model2"):
    """
    두 context vector 간 유사도 계산
    """
    print(f"\n{'='*70}")
    print(f"유사도 계산: {name1} vs {name2}")
    print(f"{'='*70}")
    
    # Flatten for comparison
    vec1 = context1.view(-1).numpy()
    vec2 = context2.view(-1).numpy()
    
    # 1. Cosine Similarity
    cos_sim = 1 - cosine(vec1, vec2)
    
    # 2. Euclidean Distance
    euclidean_dist = np.linalg.norm(vec1 - vec2)
    
    # 3. Pearson Correlation
    correlation, p_value = pearsonr(vec1, vec2)
    
    # 4. Mean Squared Error
    mse = np.mean((vec1 - vec2) ** 2)
    
    metrics = {
        'cosine_similarity': float(cos_sim),
        'euclidean_distance': float(euclidean_dist),
        'pearson_correlation': float(correlation),
        'correlation_p_value': float(p_value),
        'mse': float(mse),
        'model1_mean': float(vec1.mean()),
        'model1_std': float(vec1.std()),
        'model2_mean': float(vec2.mean()),
        'model2_std': float(vec2.std())
    }
    
    # Print results
    print(f"\n  📊 유사도 메트릭:")
    print(f"     Cosine Similarity:    {cos_sim:.6f}")
    print(f"     Euclidean Distance:   {euclidean_dist:.6f}")
    print(f"     Pearson Correlation:  {correlation:.6f} (p={p_value:.2e})")
    print(f"     Mean Squared Error:   {mse:.6f}")
    
    print(f"\n  📈 통계:")
    print(f"     {name1}: mean={vec1.mean():.6f}, std={vec1.std():.6f}")
    print(f"     {name2}: mean={vec2.mean():.6f}, std={vec2.std():.6f}")
    
    return metrics


def visualize_comparison(context1, context2, latent1, latent2, 
                        name1="Frozen", name2="LoRA", output_dir="docs/reports/visualizations"):
    """
    비교 시각화
    """
    print(f"\n{'='*70}")
    print("시각화 생성")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Context Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(context1.flatten().numpy(), bins=100, alpha=0.6, 
             label=name1, density=True, color='blue')
    ax1.hist(context2.flatten().numpy(), bins=100, alpha=0.6,
             label=name2, density=True, color='red')
    ax1.set_xlabel('Context Value')
    ax1.set_ylabel('Density')
    ax1.set_title('(A) Context Vector Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Context Scatter (sample mean)
    ax2 = fig.add_subplot(gs[0, 1])
    mean1 = context1.mean(dim=(0, 1, 2)).numpy()  # Average over N, T, tokens
    mean2 = context2.mean(dim=(0, 1, 2)).numpy()
    ax2.scatter(mean1, mean2, alpha=0.3, s=1)
    
    min_val = min(mean1.min(), mean2.min())
    max_val = max(mean1.max(), mean2.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
             linewidth=2, alpha=0.7, label='Perfect Match')
    
    ax2.set_xlabel(f'{name1} Context Mean')
    ax2.set_ylabel(f'{name2} Context Mean')
    ax2.set_title('(B) Per-Feature Context Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Latent Distribution
    if latent1 is not None and latent2 is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(latent1.flatten().numpy(), bins=50, alpha=0.6,
                label=name1, density=True, color='blue')
        ax3.hist(latent2.flatten().numpy(), bins=50, alpha=0.6,
                label=name2, density=True, color='red')
        ax3.set_xlabel('Latent Value')
        ax3.set_ylabel('Density')
        ax3.set_title('(C) Latent Space Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel D: Heatmap comparison (first sample)
    ax4 = fig.add_subplot(gs[1, 0])
    sample1 = context1[0, 0].numpy()  # First sample, first frame
    im = ax4.imshow(sample1, aspect='auto', cmap='viridis')
    ax4.set_title(f'(D) {name1} Context Heatmap')
    ax4.set_xlabel('Features (2048)')
    ax4.set_ylabel('Tokens (64)')
    plt.colorbar(im, ax=ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    sample2 = context2[0, 0].numpy()
    im = ax5.imshow(sample2, aspect='auto', cmap='viridis')
    ax5.set_title(f'(E) {name2} Context Heatmap')
    ax5.set_xlabel('Features (2048)')
    ax5.set_ylabel('Tokens (64)')
    plt.colorbar(im, ax=ax5)
    
    # Panel F: Difference heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    diff = np.abs(sample1 - sample2)
    im = ax6.imshow(diff, aspect='auto', cmap='hot')
    ax6.set_title('(F) Absolute Difference')
    ax6.set_xlabel('Features (2048)')
    ax6.set_ylabel('Tokens (64)')
    plt.colorbar(im, ax=ax6)
    
    plt.suptitle(f'Context Vector Comparison: {name1} vs {name2}',
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / f'context_comparison_{name1}_vs_{name2}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 저장: {output_path}")


def main():
    """
    Main execution
    """
    print("="*70)
    print(" Context Vector & Latent Space 비교 분석")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoints
    checkpoints = {
        'frozen': "RoboVLMs_upstream/runs/mobile_vla_kosmos2_frozen_lora_leftright_20251204/kosmos/mobile_vla_finetune/2025-12-04/mobile_vla_kosmos2_frozen_lora_leftright_20251204/epoch_epoch=08-val_loss=val_loss=0.027.ckpt",
        # 'lora': "path/to/case4_lora_checkpoint.ckpt",  # TODO: Case 4 학습 후
    }
    
    # 1. Load sample images
    images, metadata = load_sample_images(num_samples=50)
    
    # 2. Extract from Frozen model
    model_frozen = load_model(checkpoints['frozen'], device)
    context_frozen, latent_frozen, pred_frozen = extract_context_and_latent(
        model_frozen, images, device
    )
    
    # 3. Extract from LoRA model (if exists)
    if 'lora' in checkpoints and Path(checkpoints['lora']).exists():
        model_lora = load_model(checkpoints['lora'], device)
        context_lora, latent_lora, pred_lora = extract_context_and_latent(
            model_lora, images, device
        )
        
        # 4. Compute similarity
        metrics = compute_similarity_metrics(
            context_frozen, context_lora,
            "Frozen", "LoRA"
        )
        
        # 5. Visualize
        visualize_comparison(
            context_frozen, context_lora,
            latent_frozen, latent_lora,
            "Frozen", "LoRA"
        )
        
        # 6. Save results
        results = {
            'similarity_metrics': metrics,
            'frozen_stats': {
                'context_mean': float(context_frozen.mean()),
                'context_std': float(context_frozen.std()),
                'prediction_mean': float(pred_frozen.mean()),
            },
            'lora_stats': {
                'context_mean': float(context_lora.mean()),
                'context_std': float(context_lora.std()),
                'prediction_mean': float(pred_lora.mean()),
            }
        }
    else:
        print("\n⚠️  LoRA checkpoint not found. Saving Frozen baseline only.")
        
        results = {
            'frozen_stats': {
                'context_mean': float(context_frozen.mean()),
                'context_std': float(context_frozen.std()),
                'context_shape': list(context_frozen.shape),
                'prediction_mean': float(pred_frozen.mean()),
                'prediction_std': float(pred_frozen.std()),
            }
        }
        
        # Save frozen context for later comparison
        np.save('context_frozen_baseline.npy', context_frozen.numpy())
        if latent_frozen is not None:
            np.save('latent_frozen_baseline.npy', latent_frozen.numpy())
        
        print("  ✅ Frozen baseline saved:")
        print("     - context_frozen_baseline.npy")
        print("     - latent_frozen_baseline.npy")
    
    # Save JSON results
    with open('context_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ 완료!")
    print(f"{'='*70}")
    print("\n생성된 파일:")
    print("  - context_comparison_results.json")
    if 'lora' in checkpoints and Path(checkpoints['lora']).exists():
        print("  - docs/reports/visualizations/context_comparison_Frozen_vs_LoRA.png")
    else:
        print("  - context_frozen_baseline.npy")
        print("  - latent_frozen_baseline.npy")


if __name__ == "__main__":
    main()
