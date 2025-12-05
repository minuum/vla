#!/usr/bin/env python3
"""
Frozen Baseline 시각화 스크립트
===============================
Case 3 (Frozen VLM) Context Vector 및 Latent Space 시각화

생성 파일:
- docs/reports/visualizations/frozen_baseline_analysis.png
- docs/reports/visualizations/frozen_context_details.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_baseline_data():
    """Frozen baseline 데이터 로드"""
    print("Loading Frozen baseline data...")
    
    context = np.load('context_frozen_baseline.npy')
    latent = np.load('latent_frozen_baseline.npy')
    
    with open('context_comparison_results.json', 'r') as f:
        stats = json.load(f)
    
    print(f"  Context shape: {context.shape}")
    print(f"  Latent shape: {latent.shape}")
    
    return context, latent, stats


def create_comprehensive_visualization(context, latent, stats, output_path):
    """포괄적인 시각화 생성"""
    print("\nCreating comprehensive visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # === Panel A: Context Distribution ===
    ax1 = fig.add_subplot(gs[0, 0])
    context_flat = context.flatten()
    ax1.hist(context_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(stats['frozen_stats']['context_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean={stats['frozen_stats']['context_mean']:.4f}")
    ax1.set_xlabel('Context Value', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('(A) Context Vector Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Panel B: Context Heatmap (첫 번째 샘플) ===
    ax2 = fig.add_subplot(gs[0, 1])
    sample = context[0, 0]  # First sample, first frame
    im = ax2.imshow(sample, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Features (2048)', fontsize=10)
    ax2.set_ylabel('Tokens (64)', fontsize=10)
    ax2.set_title('(B) Context Heatmap (Sample 0, Frame 0)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    # === Panel C: Token-wise Mean ===
    ax3 = fig.add_subplot(gs[0, 2])
    token_means = context.mean(axis=(0, 1, 3))  # Mean over samples, frames, features
    ax3.plot(token_means, marker='o', markersize=3, linewidth=1, color='darkgreen')
    ax3.set_xlabel('Token Index', fontsize=10)
    ax3.set_ylabel('Mean Value', fontsize=10)
    ax3.set_title('(C) Token-wise Mean Activation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # === Panel D: Frame-wise Evolution ===
    ax4 = fig.add_subplot(gs[0, 3])
    frame_means = context.mean(axis=(0, 2, 3))  # Mean over samples, tokens, features
    ax4.plot(range(8), frame_means, marker='s', markersize=7, linewidth=2, color='purple')
    ax4.set_xlabel('Frame Index (Time)', fontsize=10)
    ax4.set_ylabel('Mean Activation', fontsize=10)
    ax4.set_title('(D) Temporal Evolution (8 frames)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(8))
    
    # === Panel E: Latent Distribution ===
    ax5 = fig.add_subplot(gs[1, 0])
    latent_flat = latent.flatten()
    ax5.hist(latent_flat, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax5.axvline(latent_flat.mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean={latent_flat.mean():.4f}")
    ax5.set_xlabel('Latent Value', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('(E) LSTM Latent Space Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # === Panel F: Latent Heatmap ===
    ax6 = fig.add_subplot(gs[1, 1])
    # Reshape latent for visualization
    latent_vis = latent.T  # (512, 50)
    im = ax6.imshow(latent_vis, aspect='auto', cmap='plasma')
    ax6.set_xlabel('Episodes (50)', fontsize=10)
    ax6.set_ylabel('Hidden Dimensions (512)', fontsize=10)
    ax6.set_title('(F) Latent Space Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6)
    
    # === Panel G: Feature Variance ===
    ax7 = fig.add_subplot(gs[1, 2])
    feature_vars = context.var(axis=(0, 1, 2))  # Var over samples, frames, tokens
    ax7.plot(feature_vars, linewidth=1, color='teal', alpha=0.7)
    ax7.set_xlabel('Feature Index', fontsize=10)
    ax7.set_ylabel('Variance', fontsize=10)
    ax7.set_title('(G) Per-Feature Variance', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    # === Panel H: Correlation Matrix (subset) ===
    ax8 = fig.add_subplot(gs[1, 3])
    # Sample 10 random tokens
    np.random.seed(42)
    sample_tokens = np.random.choice(64, 10, replace=False)
    token_data = context[0, 0, sample_tokens, :]  # (10, 2048)
    corr = np.corrcoef(token_data)
    im = ax8.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax8.set_xlabel('Token Index (sampled)', fontsize=10)
    ax8.set_ylabel('Token Index (sampled)', fontsize=10)
    ax8.set_title('(H) Token Correlation Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax8)
    
    # === Panel I: Statistics Table ===
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Context Mean', f"{stats['frozen_stats']['context_mean']:.6f}"],
        ['Context Std', f"{stats['frozen_stats']['context_std']:.6f}"],
        ['Context Shape', str(stats['frozen_stats']['context_shape'])],
        ['Prediction Mean', f"{stats['frozen_stats']['prediction_mean']:.6f}"],
        ['Prediction Std', f"{stats['frozen_stats']['prediction_std']:.6f}"],
        ['Latent Shape', f"{latent.shape}"],
        ['Latent Mean', f"{latent.mean():.6f}"],
        ['Latent Std', f"{latent.std():.6f}"],
    ]
    
    table = ax9.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('(I) Statistical Summary', fontsize=12, fontweight='bold', 
                  loc='left', pad=20)
    
    # === Panel J: Episode-wise variation ===
    ax10 = fig.add_subplot(gs[2, 2:])
    episode_means = context.mean(axis=(1, 2, 3))  # Mean per episode
    episode_stds = context.std(axis=(1, 2, 3))
    
    ax10.errorbar(range(50), episode_means, yerr=episode_stds, 
                  fmt='o-', linewidth=1, markersize=4, capsize=3, color='navy', alpha=0.7)
    ax10.set_xlabel('Episode Index', fontsize=10)
    ax10.set_ylabel('Mean ± Std', fontsize=10)
    ax10.set_title('(J) Episode-wise Mean Context', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Frozen VLM (Case 3) - Context Vector & Latent Space Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def create_detailed_visualization(context, latent, output_path):
    """세부 분석 시각화"""
    print("\nCreating detailed visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # === Panel A: Multi-frame heatmaps ===
    for i in range(6):
        ax = fig.add_subplot(gs[i//3, i%3])
        frame_idx = i
        sample = context[0, frame_idx]  # First episode, frame i
        im = ax.imshow(sample, aspect='auto', cmap='viridis')
        ax.set_title(f'Frame {frame_idx}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Features (2048)', fontsize=9)
        ax.set_ylabel('Tokens (64)', fontsize=9)
        if i == 5:
            plt.colorbar(im, ax=ax)
    
    plt.suptitle('Frozen VLM - Temporal Context Evolution (Episode 0)', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def main():
    print("="*70)
    print(" Frozen Baseline Visualization")
    print("="*70)
    
    # Load data
    context, latent, stats = load_baseline_data()
    
    # Create output directory
    output_dir = Path("docs/reports/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    create_comprehensive_visualization(
        context, latent, stats,
        output_dir / "frozen_baseline_analysis.png"
    )
    
    create_detailed_visualization(
        context, latent,
        output_dir / "frozen_context_details.png"
    )
    
    print("\n" + "="*70)
    print("✅ 모든 시각화 완료!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - docs/reports/visualizations/frozen_baseline_analysis.png")
    print("  - docs/reports/visualizations/frozen_context_details.png")


if __name__ == "__main__":
    main()
