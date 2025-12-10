#!/usr/bin/env python3
"""
Step 2: FT vs NoFT 비교 분석
===========================

placeholder context vectors로 비교 분석 실행
(실제 데이터로 교체 시 같은 분석 사용 가능)
"""

import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup
INPUT_DIR = Path("docs/latent_space_analysis")
OUTPUT_DIR = INPUT_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("FT vs NoFT Context Vector Comparison")
print("="*80)
print()

# ============================================================================
# Load context vectors
# ============================================================================

print("[1/4] Loading context vectors...")

files = {
    'FT-Left': INPUT_DIR / "FT5_left.npy",
    'FT-Right': INPUT_DIR / "FT5_right.npy",
    'NoFT-Left': INPUT_DIR / "noFT_left.npy",
    'NoFT-Right': INPUT_DIR / "noFT_right.npy",
}

contexts = {}
for name, path in files.items():
    if path.exists():
        contexts[name] = np.load(path)
        print(f"  ✅ {name}: {contexts[name].shape}")
    else:
        print(f"  ❌ {name}: Not found")

print()

# ============================================================================
# Helper functions
# ============================================================================

def flatten_contexts(ctx):
    """Flatten [N, 8, 64, 2048] to [N, D]"""
    return ctx.reshape(ctx.shape[0], -1)

def compute_pairwise_similarity(arr1, arr2, metric='cosine'):
    """Compute pairwise similarity between two arrays"""
    arr1_flat = flatten_contexts(arr1)
    arr2_flat = flatten_contexts(arr2)
    
    similarities = []
    for i in range(len(arr1_flat)):
        for j in range(len(arr2_flat)):
            if metric == 'cosine':
                sim = 1 - cosine(arr1_flat[i], arr2_flat[j])
            else:
                sim = -np.linalg.norm(arr1_flat[i] - arr2_flat[j])
            similarities.append(sim)
    
    return np.array(similarities)

def compute_intra_similarity(arr):
    """Compute similarity within same group"""
    arr_flat = flatten_contexts(arr)
    
    similarities = []
    for i in range(len(arr_flat)):
        for j in range(i+1, len(arr_flat)):
            sim = 1 - cosine(arr_flat[i], arr_flat[j])
            similarities.append(sim)
    
    return np.array(similarities)

# ============================================================================
# Analysis 1: Within-group similarity (Intra-class)
# ============================================================================

print("[2/4] Intra-class similarity (same direction)...")
print()

intra_results = {}

# Fine-Tuned
ft_left_intra = compute_intra_similarity(contexts['FT-Left'])
ft_right_intra = compute_intra_similarity(contexts['FT-Right'])

print("Fine-Tuned (FT):")
print(f"  Left-Left:   {ft_left_intra.mean():.4f} ± {ft_left_intra.std():.4f}")
print(f"  Right-Right: {ft_right_intra.mean():.4f} ± {ft_right_intra.std():.4f}")

intra_results['FT'] = {
    'left_left': {'mean': float(ft_left_intra.mean()), 'std': float(ft_left_intra.std())},
    'right_right': {'mean': float(ft_right_intra.mean()), 'std': float(ft_right_intra.std())}
}

# No Fine-Tuning
noFT_left_intra = compute_intra_similarity(contexts['NoFT-Left'])
noFT_right_intra = compute_intra_similarity(contexts['NoFT-Right'])

print()
print("No Fine-Tuning (NoFT):")
print(f"  Left-Left:   {noFT_left_intra.mean():.4f} ± {noFT_left_intra.std():.4f}")
print(f"  Right-Right: {noFT_right_intra.mean():.4f} ± {noFT_right_intra.std():.4f}")

intra_results['NoFT'] = {
    'left_left': {'mean': float(noFT_left_intra.mean()), 'std': float(noFT_left_intra.std())},
    'right_right': {'mean': float(noFT_right_intra.mean()), 'std': float(noFT_right_intra.std())}
}

print()

# ============================================================================
# Analysis 2: Between-group similarity (Inter-class)
# ============================================================================

print("[3/4] Inter-class similarity (different directions)...")
print()

# Fine-Tuned: Left vs Right
ft_left_right = compute_pairwise_similarity(
    contexts['FT-Left'], contexts['FT-Right']
)

print("Fine-Tuned (FT):")
print(f"  Left-Right: {ft_left_right.mean():.4f} ± {ft_left_right.std():.4f}")
print(f"  Separation: {(ft_left_intra.mean() - ft_left_right.mean()):.4f}")

# No Fine-Tuning: Left vs Right
noFT_left_right = compute_pairwise_similarity(
    contexts['NoFT-Left'], contexts['NoFT-Right']
)

print()
print("No Fine-Tuning (NoFT):")
print(f"  Left-Right: {noFT_left_right.mean():.4f} ± {noFT_left_right.std():.4f}")
print(f"  Separation: {(noFT_left_intra.mean() - noFT_left_right.mean()):.4f}")

inter_results = {
    'FT': {
        'left_right': {'mean': float(ft_left_right.mean()), 'std': float(ft_left_right.std())},
        'separation': float(ft_left_intra.mean() - ft_left_right.mean())
    },
    'NoFT': {
        'left_right': {'mean': float(noFT_left_right.mean()), 'std': float(noFT_left_right.std())},
        'separation': float(noFT_left_intra.mean() - noFT_left_right.mean())
    }
}

print()

# ============================================================================
# Analysis 3: Fine-Tuning Effect (Delta)
# ============================================================================

print("[4/4] Fine-Tuning effect (Before vs After)...")
print()

# Same inputs, different models
ft_left_flat = flatten_contexts(contexts['FT-Left'])
noFT_left_flat = flatten_contexts(contexts['NoFT-Left'])

delta_left = []
for i in range(min(len(ft_left_flat), len(noFT_left_flat))):
    sim = 1 - cosine(ft_left_flat[i], noFT_left_flat[i])
    delta_left.append(sim)

delta_left = np.array(delta_left)

print("Same input, different models:")
print(f"  Left delta (NoFT→FT): {delta_left.mean():.4f} ± {delta_left.std():.4f}")

ft_right_flat = flatten_contexts(contexts['FT-Right'])
noFT_right_flat = flatten_contexts(contexts['NoFT-Right'])

delta_right = []
for i in range(min(len(ft_right_flat), len(noFT_right_flat))):
    sim = 1 - cosine(ft_right_flat[i], noFT_right_flat[i])
    delta_right.append(sim)

delta_right = np.array(delta_right)

print(f"  Right delta (NoFT→FT): {delta_right.mean():.4f} ± {delta_right.std():.4f}")

delta_results = {
    'left': {'mean': float(delta_left.mean()), 'std': float(delta_left.std())},
    'right': {'mean': float(delta_right.mean()), 'std': float(delta_right.std())}
}

print()

# ============================================================================
# Summary Report
# ============================================================================

print("="*80)
print("SUMMARY REPORT")
print("="*80)
print()

print("🎯 Key Finding: Direction Separation")
print()
print("Before Fine-Tuning (NoFT):")
print(f"  Same direction:     {noFT_left_intra.mean():.4f}")
print(f"  Different direction: {noFT_left_right.mean():.4f}")
print(f"  Separation:         {inter_results['NoFT']['separation']:.4f}")
if inter_results['NoFT']['separation'] < 0.05:
    print("  → ❌ Poor separation (cannot distinguish directions)")
else:
    print("  → ✅ Some separation")

print()
print("After Fine-Tuning (FT):")
print(f"  Same direction:     {ft_left_intra.mean():.4f}")
print(f"  Different direction: {ft_left_right.mean():.4f}")
print(f"  Separation:         {inter_results['FT']['separation']:.4f}")
if inter_results['FT']['separation'] > 0.10:
    print("  → ✅ Good separation (clear clustering)")
else:
    print("  → ⚠️  Moderate separation")

print()
print("🔄 Fine-Tuning Effect:")
print(f"  Latent space change: {delta_left.mean():.4f}")
if delta_left.mean() < 0.80:
    print("  → ✅ Significant change (LoRA effective)")
else:
    print("  → ⚠️  Small change")

print()

# ============================================================================
# Save Results
# ============================================================================

results = {
    'intra_class': intra_results,
    'inter_class': inter_results,
    'delta': delta_results,
    'summary': {
        'noFT_separation': float(inter_results['NoFT']['separation']),
        'FT_separation': float(inter_results['FT']['separation']),
        'improvement': float(inter_results['FT']['separation'] - inter_results['NoFT']['separation']),
        'latent_change': float(delta_left.mean())
    }
}

with open(OUTPUT_DIR / "comparison_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved: {OUTPUT_DIR / 'comparison_results.json'}")
print()

# ============================================================================
# Quick Visualization
# ============================================================================

print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Intra-class similarity distribution
ax = axes[0, 0]
ax.hist(noFT_left_intra, bins=30, alpha=0.6, label='NoFT Left-Left', density=True)
ax.hist(ft_left_intra, bins=30, alpha=0.6, label='FT Left-Left', density=True)
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Density')
ax.set_title('Intra-class Similarity (Same Direction)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Inter-class similarity distribution
ax = axes[0, 1]
ax.hist(noFT_left_right, bins=30, alpha=0.6, label='NoFT Left-Right', density=True)
ax.hist(ft_left_right, bins=30, alpha=0.6, label='FT Left-Right', density=True)
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Density')
ax.set_title('Inter-class Similarity (Different Directions)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Separation comparison
ax = axes[1, 0]
models = ['NoFT', 'FT']
separations = [inter_results['NoFT']['separation'], inter_results['FT']['separation']]
bars = ax.bar(models, separations, color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Separation (Intra - Inter)')
ax.set_title('Direction Separation: NoFT vs FT')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.05, color='r', linestyle='--', label='Threshold')
ax.legend()

# Add values on bars
for bar, val in zip(bars, separations):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Delta distribution
ax = axes[1, 1]
ax.hist(delta_left, bins=30, alpha=0.7, color='purple', density=True)
ax.axvline(delta_left.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delta_left.mean():.4f}')
ax.set_xlabel('Cosine Similarity (NoFT→FT)')
ax.set_ylabel('Density')
ax.set_title('Fine-Tuning Effect (Latent Space Change)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('FT vs NoFT Context Vector Comparison\n(Placeholder Data)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = OUTPUT_DIR / "comparison_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved: {output_path}")

plt.close()

print()
print("="*80)
print("✅ Analysis complete!")
print("="*80)
print()
print("Next step: python3 scripts/visualize_tsne.py")
