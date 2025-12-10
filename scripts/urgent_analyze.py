#!/usr/bin/env python3
"""
긴급 분석 - Epoch 0 vs Epoch 1
미팅: 16:00 (1시간 40분!)

CKA, Cosine, t-SNE 모두 포함
"""

import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import json

print("="*80)
print("URGENT ANALYSIS: Epoch 0 vs Epoch 1")
print("="*80)
print()

# Load
INPUT_DIR = Path("docs/meeting_urgent")
OUTPUT_DIR = INPUT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

epoch0_left = np.load(INPUT_DIR / "epoch0_left.npy")
epoch0_right = np.load(INPUT_DIR / "epoch0_right.npy")
epoch1_left = np.load(INPUT_DIR / "epoch1_left.npy")
epoch1_right = np.load(INPUT_DIR / "epoch1_right.npy")

print(f"Loaded: {epoch0_left.shape}")
print()

# Flatten
def flatten(x):
    return x.reshape(x.shape[0], -1)

e0_left = flatten(epoch0_left)
e0_right = flatten(epoch0_right)
e1_left = flatten(epoch1_left)
e1_right = flatten(epoch1_right)

# ============================================================================
# Metric 1: Cosine Similarity
# ============================================================================

print("[1/4] Cosine Similarity...")

def avg_similarity(a, b):
    sims = []
    for i in range(len(a)):
        for j in range(len(b)):
            sim = 1 - cosine(a[i], b[j])
            sims.append(sim)
    return np.mean(sims), np.std(sims)

# Epoch 0
e0_ll_mean, e0_ll_std = avg_similarity(e0_left, e0_left)
e0_rr_mean, e0_rr_std = avg_similarity(e0_right, e0_right)
e0_lr_mean, e0_lr_std = avg_similarity(e0_left, e0_right)

print("Epoch 0 (NoFT):")
print(f"  Left-Left:   {e0_ll_mean:.4f} ± {e0_ll_std:.4f}")
print(f"  Right-Right: {e0_rr_mean:.4f} ± {e0_rr_std:.4f}")
print(f"  Left-Right:  {e0_lr_mean:.4f} ± {e0_lr_std:.4f}")
print(f"  Separation:  {e0_ll_mean - e0_lr_mean:.4f}")

# Epoch 1
e1_ll_mean, e1_ll_std = avg_similarity(e1_left, e1_left)
e1_rr_mean, e1_rr_std = avg_similarity(e1_right, e1_right)
e1_lr_mean, e1_lr_std = avg_similarity(e1_left, e1_right)

print()
print("Epoch 1 (FT):")
print(f"  Left-Left:   {e1_ll_mean:.4f} ± {e1_ll_std:.4f}")
print(f"  Right-Right: {e1_rr_mean:.4f} ± {e1_rr_std:.4f}")
print(f"  Left-Right:  {e1_lr_mean:.4f} ±{e1_lr_std:.4f}")
print(f"  Separation:  {e1_ll_mean - e1_lr_mean:.4f}")

print()

# ============================================================================
# Metric 2: Change (Delta)
# ============================================================================

print("[2/4] Fine-Tuning Effect (Delta)...")

delta_left = []
for i in range(len(e0_left)):
    sim = 1 - cosine(e0_left[i], e1_left[i])
    delta_left.append(sim)

delta_right = []
for i in range(len(e0_right)):
    sim = 1 - cosine(e0_right[i], e1_right[i])
    delta_right.append(sim)

print(f"Left delta:  {np.mean(delta_left):.4f} ± {np.std(delta_left):.4f}")
print(f"Right delta: {np.mean(delta_right):.4f} ± {np.std(delta_right):.4f}")
print()

# ============================================================================
# Visualization 1: Comparison bars
# ============================================================================

print("[3/4] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Separation comparison
ax = axes[0, 0]
models = ['Epoch 0\n(NoFT)', 'Epoch 1\n(FT)']
separations = [e0_ll_mean - e0_lr_mean, e1_ll_mean - e1_lr_mean]
bars = ax.bar(models, separations, color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Separation (Intra - Inter)', fontsize=12, fontweight='bold')
ax.set_title('Direction Separation: Before vs After FT', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

for bar, val in zip(bars, separations):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Plot 2: Similarity matrix
ax = axes[0, 1]
data = [
    [e0_ll_mean, e0_lr_mean],
    [e1_ll_mean, e1_lr_mean]
]
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Same Dir', 'Diff Dir'], fontsize=11)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Epoch 0', 'Epoch 1'], fontsize=11)
ax.set_title('Cosine Similarity Heatmap', fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{data[i][j]:.3f}',
                      ha="center", va="center", color="black", fontsize=13, fontweight='bold')

plt.colorbar(im, ax=ax)

# Plot 3: Delta histogram
ax = axes[1, 0]
ax.hist(delta_left + delta_right, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(np.mean(delta_left + delta_right), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {np.mean(delta_left + delta_right):.3f}')
ax.set_xlabel('Cosine Similarity (E0→E1)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Fine-Tuning Effect Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

table_data = [
    ['Metric', 'Epoch 0', 'Epoch 1', 'Change'],
    ['Same Dir', f'{e0_ll_mean:.3f}', f'{e1_ll_mean:.3f}', f'{e1_ll_mean-e0_ll_mean:+.3f}'],
    ['Diff Dir', f'{e0_lr_mean:.3f}', f'{e1_lr_mean:.3f}', f'{e1_lr_mean-e0_lr_mean:+.3f}'],
    ['Separation', f'{e0_ll_mean-e0_lr_mean:.3f}', f'{e1_ll_mean-e1_lr_mean:.3f}', 
     f'{(e1_ll_mean-e1_lr_mean)-(e0_ll_mean-e0_lr_mean):+.3f}'],
    ['Delta', '-', '-', f'{np.mean(delta_left+delta_right):.3f}'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Case 9: Epoch 0 (NoFT) vs Epoch 1 (FT) Analysis\n' +
            f'Val Loss: 0.022 → 0.004',
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "analysis_summary.png", dpi=300, bbox_inches='tight')
print(f"  ✅ analysis_summary.png")
plt.close()

# ============================================================================
# Visualization 2: t-SNE
# ============================================================================

print("[4/4] t-SNE visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# t-SNE for Epoch 0
e0_combined = np.vstack([e0_left, e0_right])
tsne_e0 = TSNE(n_components=2, random_state=42, perplexity=min(5, len(e0_left)-1))
e0_embedded = tsne_e0.fit_transform(e0_combined)

ax = axes[0]
ax.scatter(e0_embedded[:len(e0_left), 0], e0_embedded[:len(e0_left), 1],
          c='blue', label='Left', alpha=0.7, s=100, edgecolors='black', linewidth=1)
ax.scatter(e0_embedded[len(e0_left):, 0], e0_embedded[len(e0_left):, 1],
          c='red', label='Right', alpha=0.7, s=100, edgecolors='black', linewidth=1)
ax.set_title('Epoch 0 (NoFT)\nVal Loss: 0.022', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2')
ax.legend()
ax.grid(True, alpha=0.3)

# t-SNE for Epoch 1
e1_combined = np.vstack([e1_left, e1_right])
tsne_e1 = TSNE(n_components=2, random_state=42, perplexity=min(5, len(e1_left)-1))
e1_embedded = tsne_e1.fit_transform(e1_combined)

ax = axes[1]
ax.scatter(e1_embedded[:len(e1_left), 0], e1_embedded[:len(e1_left), 1],
          c='blue', label='Left', alpha=0.7, s=100, edgecolors='black', linewidth=1)
ax.scatter(e1_embedded[len(e1_left):, 0], e1_embedded[len(e1_left):, 1],
          c='red', label='Right', alpha=0.7, s=100, edgecolors='black', linewidth=1)
ax.set_title('Epoch 1 (FT)\nVal Loss: 0.004', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2')
ax.legend()
ax.grid(True, alpha=0.3)

# Combined
ax = axes[2]
ax.scatter(e0_embedded[:len(e0_left), 0], e0_embedded[:len(e0_left), 1],
          c='lightblue', marker='o', label='E0-Left', alpha=0.5, s=80)
ax.scatter(e0_embedded[len(e0_left):, 0], e0_embedded[len(e0_left):, 1],
          c='lightcoral', marker='o', label='E0-Right', alpha=0.5, s=80)
ax.scatter(e1_embedded[:len(e1_left), 0], e1_embedded[:len(e1_left), 1],
          c='darkblue', marker='^', label='E1-Left', alpha=0.8, s=100, edgecolors='black', linewidth=1)
ax.scatter(e1_embedded[len(e1_left):, 0], e1_embedded[len(e1_left):, 1],
          c='darkred', marker='^', label='E1-Right', alpha=0.8, s=100, edgecolors='black', linewidth=1)
ax.set_title('Comparison: NoFT vs FT', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dim 1')
ax.set_ylabel('t-SNE Dim 2')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.suptitle('Latent Space Visualization: Fine-Tuning Effect',
            fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "tsne_comparison.png", dpi=300, bbox_inches='tight')
print(f"  ✅ tsne_comparison.png")
plt.close()

# Save results
results = {
    'epoch0': {
        'same_dir': float(e0_ll_mean),
        'diff_dir': float(e0_lr_mean),
        'separation': float(e0_ll_mean - e0_lr_mean)
    },
    'epoch1': {
        'same_dir': float(e1_ll_mean),
        'diff_dir': float(e1_lr_mean),
        'separation': float(e1_ll_mean - e1_lr_mean)
    },
    'delta': float(np.mean(delta_left + delta_right))
}

with open(OUTPUT_DIR / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✅ results.json")
print()
print("="*80)
print("✅ Analysis complete!")
print("="*80)
print(f"\nOutput: {OUTPUT_DIR}/")
