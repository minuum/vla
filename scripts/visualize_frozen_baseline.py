#!/usr/bin/env python3
"""
Frozen Baseline 시각화 스크립트
목적: Context vector와 latent space 분석 및 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_frozen_baseline():
    """Load frozen baseline data"""
    print("="*70)
    print("Frozen Baseline 데이터 로드")
    print("="*70)
    
    context = np.load('context_frozen_baseline.npy')
    latent = np.load('latent_frozen_baseline.npy')
    
    with open('context_comparison_results.json', 'r') as f:
        stats = json.load(f)
    
    print(f"  Context shape: {context.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Context mean: {stats['frozen_stats']['context_mean']:.6f}")
    print(f"  Context std: {stats['frozen_stats']['context_std']:.6f}")
    
    return context, latent, stats


def create_comprehensive_visualization(context, latent, stats):
    """Create comprehensive visualization"""
    print("\n" + "="*70)
    print("종합 시각화 생성")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # === Row 1: Context Vector Analysis ===
    
    # Panel A: Context Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    context_flat = context.flatten()
    ax1.hist(context_flat, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(context_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {context_flat.mean():.4f}')
    ax1.axvline(context_flat.mean() + context_flat.std(), color='orange', linestyle=':', linewidth=2, label=f'±1 Std')
    ax1.axvline(context_flat.mean() - context_flat.std(), color='orange', linestyle=':', linewidth=2)
    ax1.set_xlabel('Context Value')
    ax1.set_ylabel('Density')
    ax1.set_title('(A) Context Vector Distribution\nFrozen VLM Baseline', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Per-Sample Mean
    ax2 = fig.add_subplot(gs[0, 1])
    sample_means = context.mean(axis=(1, 2, 3))  # (50,)
    ax2.plot(sample_means, marker='o', linestyle='-', linewidth=2, markersize=6, color='blue')
    ax2.axhline(sample_means.mean(), color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {sample_means.mean():.4f}')
    ax2.fill_between(range(len(sample_means)), 
                      sample_means.mean() - sample_means.std(),
                      sample_means.mean() + sample_means.std(),
                      alpha=0.2, color='blue')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Mean Context Value')
    ax2.set_title('(B) Per-Sample Context Mean\nConsistency Across Samples', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Temporal Evolution
    ax3 = fig.add_subplot(gs[0, 2])
    temporal_means = context.mean(axis=(0, 2, 3))  # (8,) - 8 frames
    ax3.plot(range(8), temporal_means, marker='s', linestyle='-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Frame Index (Time)')
    ax3.set_ylabel('Mean Context Value')
    ax3.set_title('(C) Temporal Evolution\nContext Across 8 Frames', fontweight='bold')
    ax3.set_xticks(range(8))
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Feature Dimension Analysis
    ax4 = fig.add_subplot(gs[0, 3])
    feature_means = context.mean(axis=(0, 1, 2))  # (2048,)
    ax4.plot(feature_means, linewidth=1, color='purple', alpha=0.7)
    ax4.set_xlabel('Feature Dimension')
    ax4.set_ylabel('Mean Value')
    ax4.set_title('(D) Feature Dimension Analysis\n2048 Dimensions', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # === Row 2: Heatmaps ===
    
    # Panel E: Context Heatmap (Sample 0, Frame 0)
    ax5 = fig.add_subplot(gs[1, 0])
    sample_context = context[0, 0]  # (64, 2048)
    im1 = ax5.imshow(sample_context, aspect='auto', cmap='viridis', interpolation='nearest')
    ax5.set_xlabel('Features (2048)')
    ax5.set_ylabel('Tokens (64)')
    ax5.set_title('(E) Context Heatmap\nSample 0, Frame 0', fontweight='bold')
    plt.colorbar(im1, ax=ax5, label='Value')
    
    # Panel F: Context Heatmap (Sample 25, Frame 0) - Right turn
    ax6 = fig.add_subplot(gs[1, 1])
    sample_context_right = context[25, 0]  # (64, 2048)
    im2 = ax6.imshow(sample_context_right, aspect='auto', cmap='viridis', interpolation='nearest')
    ax6.set_xlabel('Features (2048)')
    ax6.set_ylabel('Tokens (64)')
    ax6.set_title('(F) Context Heatmap\nSample 25 (Right), Frame 0', fontweight='bold')
    plt.colorbar(im2, ax=ax6, label='Value')
    
    # Panel G: Difference Heatmap (Left vs Right)
    ax7 = fig.add_subplot(gs[1, 2])
    diff = np.abs(context[0, 0] - context[25, 0])
    im3 = ax7.imshow(diff, aspect='auto', cmap='hot', interpolation='nearest')
    ax7.set_xlabel('Features (2048)')
    ax7.set_ylabel('Tokens (64)')
    ax7.set_title('(G) Absolute Difference\nLeft vs Right Turn', fontweight='bold')
    plt.colorbar(im3, ax=ax7, label='|Difference|')
    
    # Panel H: Latent Space Distribution
    ax8 = fig.add_subplot(gs[1, 3])
    latent_flat = latent.flatten()
    ax8.hist(latent_flat, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax8.axvline(latent_flat.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {latent_flat.mean():.4f}')
    ax8.set_xlabel('Latent Value')
    ax8.set_ylabel('Density')
    ax8.set_title('(H) Latent Space Distribution\nLSTM Hidden State', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # === Row 3: Advanced Analysis ===
    
    # Panel I: Token-wise Variance
    ax9 = fig.add_subplot(gs[2, 0])
    token_variance = context.var(axis=(0, 1, 3))  # (64,)
    ax9.bar(range(64), token_variance, color='teal', alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Token Index')
    ax9.set_ylabel('Variance')
    ax9.set_title('(I) Token-wise Variance\nWhich Tokens Vary Most?', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Panel J: Feature-wise Variance (Top 100)
    ax10 = fig.add_subplot(gs[2, 1])
    feature_variance = context.var(axis=(0, 1, 2))  # (2048,)
    top_100_indices = np.argsort(feature_variance)[-100:]
    ax10.bar(range(100), feature_variance[top_100_indices], color='orange', alpha=0.7, edgecolor='black')
    ax10.set_xlabel('Top 100 Feature Indices')
    ax10.set_ylabel('Variance')
    ax10.set_title('(J) Top 100 High-Variance Features\nMost Informative Dimensions', fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Panel K: Latent Space PCA-like (2D projection)
    ax11 = fig.add_subplot(gs[2, 2])
    # Simple 2D projection using first 2 dimensions
    ax11.scatter(latent[:25, 0], latent[:25, 1], c='blue', label='Left Turn', s=100, alpha=0.6, edgecolors='black')
    ax11.scatter(latent[25:, 0], latent[25:, 1], c='red', label='Right Turn', s=100, alpha=0.6, edgecolors='black')
    ax11.set_xlabel('Latent Dim 0')
    ax11.set_ylabel('Latent Dim 1')
    ax11.set_title('(K) Latent Space Projection\nLeft vs Right Separation', fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Panel L: Statistics Summary
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    summary_text = f"""
    📊 Frozen Baseline Statistics
    
    Context Vector:
      • Shape: {context.shape}
      • Mean: {stats['frozen_stats']['context_mean']:.6f}
      • Std: {stats['frozen_stats']['context_std']:.6f}
      • Min: {context.min():.6f}
      • Max: {context.max():.6f}
    
    Latent Space:
      • Shape: {latent.shape}
      • Mean: {latent.mean():.6f}
      • Std: {latent.std():.6f}
      • Min: {latent.min():.6f}
      • Max: {latent.max():.6f}
    
    Predictions:
      • Mean: {stats['frozen_stats']['prediction_mean']:.6f}
      • Std: {stats['frozen_stats']['prediction_std']:.6f}
    
    Samples:
      • Total: 50 (25 left + 25 right)
      • Frames per sample: 8
      • Tokens per frame: 64
      • Features per token: 2048
    """
    
    ax12.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
              fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Frozen VLM Baseline - Comprehensive Analysis\nMobile-VLA Context Vector & Latent Space',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path('docs/reports/visualizations')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'frozen_baseline_comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 저장: {output_path}")
    
    return output_path


def create_simple_summary_plot(context, latent, stats):
    """Create simple summary plot for presentation"""
    print("\n" + "="*70)
    print("간단한 요약 플롯 생성 (발표용)")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Context Distribution
    ax = axes[0, 0]
    context_flat = context.flatten()
    ax.hist(context_flat, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(context_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {context_flat.mean():.4f}')
    ax.set_xlabel('Context Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Context Vector Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Context Heatmap
    ax = axes[0, 1]
    sample_context = context[0, 0]
    im = ax.imshow(sample_context, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Features (2048)', fontsize=12)
    ax.set_ylabel('Tokens (64)', fontsize=12)
    ax.set_title('Context Heatmap (Sample)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Value')
    
    # Panel 3: Latent Distribution
    ax = axes[1, 0]
    latent_flat = latent.flatten()
    ax.hist(latent_flat, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(latent_flat.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {latent_flat.mean():.4f}')
    ax.set_xlabel('Latent Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Latent Space Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Latent Projection
    ax = axes[1, 1]
    ax.scatter(latent[:25, 0], latent[:25, 1], c='blue', label='Left Turn', s=100, alpha=0.6, edgecolors='black')
    ax.scatter(latent[25:, 0], latent[25:, 1], c='red', label='Right Turn', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Latent Dim 0', fontsize=12)
    ax.set_ylabel('Latent Dim 1', fontsize=12)
    ax.set_title('Latent Space: Left vs Right', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Frozen VLM Baseline - Summary\nContext Vector & Latent Space Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path('docs/reports/visualizations')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'frozen_baseline_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 저장: {output_path}")
    
    return output_path


def main():
    """Main execution"""
    print("="*70)
    print(" Frozen Baseline 시각화")
    print("="*70)
    
    # Load data
    context, latent, stats = load_frozen_baseline()
    
    # Create visualizations
    comprehensive_path = create_comprehensive_visualization(context, latent, stats)
    summary_path = create_simple_summary_plot(context, latent, stats)
    
    print("\n" + "="*70)
    print("✅ 완료!")
    print("="*70)
    print("\n생성된 파일:")
    print(f"  - {comprehensive_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
