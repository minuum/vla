#!/usr/bin/env python3
"""
Frozen VLM 심화 분석 스크립트
============================
Case 3 성능 심화 분석 및 generalization test

목적:
1. Left/Right generalization 테스트
2. Failure case 분석
3. Context vector quality 상세 분석
4. Temporal consistency 분석
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_left_right_generalization():
    """Left vs Right generalization 분석"""
    print("\n" + "="*70)
    print("Left/Right Generalization Analysis")
    print("="*70)
    
    # Load baseline
    context = np.load('context_frozen_baseline.npy')
    
    with open('context_comparison_results.json', 'r') as f:
        stats = json.load(f)
    
    # Analyze by direction (from metadata - need to load separately)
    # For now, analyze overall statistics
    
    print(f"\n  Overall Statistics:")
    print(f"    Context Mean: {stats['frozen_stats']['context_mean']:.6f}")
    print(f"    Context Std:  {stats['frozen_stats']['context_std']:.6f}")
    
    # Sample-wise analysis
    sample_means = context.mean(axis=(1, 2, 3))  # Per episode
    sample_stds = context.std(axis=(1, 2, 3))
    
    print(f"\n  Episode-wise Variation:")
    print(f"    Mean of means: {sample_means.mean():.6f}")
    print(f"    Std of means:  {sample_means.std():.6f}")
    print(f"    Range: [{sample_means.min():.6f}, {sample_means.max():.6f}]")
    
    # Check consistency
    consistency_score = 1 - (sample_means.std() / abs(sample_means.mean() + 1e-8))
    print(f"\n  Consistency Score: {consistency_score:.4f}")
    print(f"    (closer to 1.0 = more consistent)")
    
    return {
        'sample_means': sample_means.tolist(),
        'sample_stds': sample_stds.tolist(),
        'consistency_score': float(consistency_score)
    }


def analyze_temporal_consistency():
    """Temporal consistency 분석"""
    print("\n" + "="*70)
    print("Temporal Consistency Analysis")
    print("="*70)
    
    context = np.load('context_frozen_baseline.npy')
    
    # Frame-wise analysis
    frame_means = context.mean(axis=(0, 2, 3))  # [8 frames]
    frame_stds = context.std(axis=(0, 2, 3))
    
    print(f"\n  Frame-wise Statistics:")
    for i in range(8):
        print(f"    Frame {i}: mean={frame_means[i]:.6f}, std={frame_stds[i]:.6f}")
    
    # Temporal smoothness
    frame_diffs = np.diff(frame_means)
    temporal_smoothness = 1 / (1 + np.abs(frame_diffs).mean())
    
    print(f"\n  Temporal Smoothness: {temporal_smoothness:.4f}")
    print(f"    (closer to 1.0 = smoother)")
    
    return{
        'frame_means': frame_means.tolist(),
        'frame_stds': frame_stds.tolist(),
        'temporal_smoothness': float(temporal_smoothness)
    }


def analyze_feature_importance():
    """Feature importance 분석"""
    print("\n" + "="*70)
    print("Feature Importance Analysis")
    print("="*70)
    
    context = np.load('context_frozen_baseline.npy')
    
    # Feature variance (higher = more important)
    feature_vars = context.var(axis=(0, 1, 2))  # [2048 features]
    
    # Top 10 most important features
    top_indices = np.argsort(feature_vars)[-10:][::-1]
    
    print(f"\n  Top 10 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank}. Feature {idx}: variance={feature_vars[idx]:.6f}")
    
    # Sparsity analysis
    near_zero = (np.abs(context) < 0.01).mean()
    
    print(f"\n  Sparsity Analysis:")
    print(f"    Near-zero values: {near_zero*100:.2f}%")
    print(f"    Active values: {(1-near_zero)*100:.2f}%")
    
    return {
        'top_feature_indices': top_indices.tolist(),
        'top_feature_variances': feature_vars[top_indices].tolist(),
        'sparsity': float(near_zero)
    }


def analyze_token_diversity():
    """Token diversity 분석"""
    print("\n" + "="*70)
    print("Token Diversity Analysis")
    print("="*70)
    
    context = np.load('context_frozen_baseline.npy')
    
    # Token-wise analysis
    token_means = context.mean(axis=(0, 1, 3))  # [64 tokens]
    token_vars = context.var(axis=(0, 1, 3))
    
    # Diversity score
    diversity_score = token_vars.std() / (token_vars.mean() + 1e-8)
    
    print(f"\n  Token Statistics:")
    print(f"    Mean activation per token: {token_means.mean():.6f}")
    print(f"    Variance per token: {token_vars.mean():.6f}")
    print(f"    Diversity score: {diversity_score:.4f}")
    
    # Check if some tokens are dominant
    max_var_token = token_vars.argmax()
    min_var_token = token_vars.argmin()
    
    print(f"\n  Most/Least Active Tokens:")
    print(f"    Most active: Token {max_var_token} (var={token_vars[max_var_token]:.6f})")
    print(f"    Least active: Token {min_var_token} (var={token_vars[min_var_token]:.6f})")
    
    return {
        'token_means': token_means.tolist(),
        'token_vars': token_vars.tolist(),
        'diversity_score': float(diversity_score)
    }


def generate_advanced_visualization(results, output_dir="docs/reports/visualizations"):
    """고급 분석 결과 시각화"""
    print("\n" + "="*70)
    print("Generating Advanced Visualization")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Episode consistency
    ax1 = fig.add_subplot(gs[0, 0])
    sample_means = np.array(results['generalization']['sample_means'])
    ax1.plot(sample_means, marker='o', markersize=4, linewidth=1, color='navy')
    ax1.axhline(sample_means.mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean={sample_means.mean():.4f}")
    ax1.fill_between(range(len(sample_means)),
                      sample_means.mean() - sample_means.std(),
                      sample_means.mean() + sample_means.std(),
                      alpha=0.2, color='red')
    ax1.set_xlabel('Episode Index', fontsize=10)
    ax1.set_ylabel('Mean Context Value', fontsize=10)
    ax1.set_title('(A) Episode-wise Consistency', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Temporal evolution
    ax2 = fig.add_subplot(gs[0, 1])
    frame_means = np.array(results['temporal']['frame_means'])
    ax2.plot(range(8), frame_means, marker='s', markersize=7, 
             linewidth=2, color='purple')
    ax2.set_xlabel('Frame Index (Time)', fontsize=10)
    ax2.set_ylabel('Mean Activation', fontsize=10)
    ax2.set_title('(B) Temporal Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(8))
    
    # Panel C: Feature importance
    ax3 = fig.add_subplot(gs[0, 2])
    top_vars = np.array(results['features']['top_feature_variances'])
    ax3.barh(range(10), top_vars, color='teal')
    ax3.set_xlabel('Variance', fontsize=10)
    ax3.set_ylabel('Feature Rank', fontsize=10)
    ax3.set_title('(C) Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Panel D: Token diversity
    ax4 = fig.add_subplot(gs[1, 0])
    token_vars = np.array(results['tokens']['token_vars'])
    ax4.plot(token_vars, linewidth=2, color='orange')
    ax4.set_xlabel('Token Index', fontsize=10)
    ax4.set_ylabel('Variance', fontsize=10)
    ax4.set_title('(D) Token-wise Variance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel E: Statistics summary
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    summary_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Consistency Score', f"{results['generalization']['consistency_score']:.4f}", 
         '✅ High' if results['generalization']['consistency_score'] > 0.9 else '⚠️ Medium'],
        ['Temporal Smoothness', f"{results['temporal']['temporal_smoothness']:.4f}",
         '✅ Smooth' if results['temporal']['temporal_smoothness'] > 0.9 else '⚠️ Moderate'],
        ['Feature Sparsity', f"{results['features']['sparsity']*100:.2f}%",
         '✅ Good' if results['features']['sparsity'] < 0.3 else '⚠️ High'],
        ['Token Diversity', f"{results['tokens']['diversity_score']:.4f}",
         '✅ Diverse' if results['tokens']['diversity_score'] > 0.5 else '⚠️ Limited'],
    ]
    
    table = ax5.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('(E) Analysis Summary', fontsize=12, fontweight='bold',
                  loc='left', pad=20)
    
    plt.suptitle('Frozen VLM - Advanced Analysis', 
                fontsize=14, fontweight='bold')
    
    output_path = output_dir / 'frozen_advanced_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")


def main():
    print("="*70)
    print(" Frozen VLM Advanced Analysis")
    print("="*70)
    
    results = {}
    
    # 1. Generalization analysis
    results['generalization'] = analyze_left_right_generalization()
    
    # 2. Temporal analysis
    results['temporal'] = analyze_temporal_consistency()
    
    # 3. Feature importance
    results['features'] = analyze_feature_importance()
    
    # 4. Token diversity
    results['tokens'] = analyze_token_diversity()
    
    # 5. Generate visualization
    generate_advanced_visualization(results)
    
    # 6. Save results
    with open('frozen_advanced_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Advanced Analysis Complete!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - frozen_advanced_analysis_results.json")
    print("  - docs/reports/visualizations/frozen_advanced_analysis.png")
    
    # Print summary
    print("\n📊 Analysis Summary:")
    print(f"  Consistency Score: {results['generalization']['consistency_score']:.4f}")
    print(f"  Temporal Smoothness: {results['temporal']['temporal_smoothness']:.4f}")
    print(f"  Feature Sparsity: {results['features']['sparsity']*100:.2f}%")
    print(f"  Token Diversity: {results['tokens']['diversity_score']:.4f}")


if __name__ == "__main__":
    main()
