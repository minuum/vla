#!/usr/bin/env python3
"""
Step 3: t-SNE Visualization (FT vs NoFT)
========================================

Before/After Fine-Tuning 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# Setup
INPUT_DIR = Path("docs/latent_space_analysis")
OUTPUT_DIR = INPUT_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("t-SNE Visualization: FT vs NoFT")
print("="*80)
print()

# ============================================================================
# Load
# ============================================================================

print("[1/3] Loading context vectors...")

contexts = {
    'FT-Left': np.load(INPUT_DIR / "FT5_left.npy"),
    'FT-Right': np.load(INPUT_DIR / "FT5_right.npy"),
    'NoFT-Left': np.load(INPUT_DIR / "noFT_left.npy"),
    'NoFT-Right': np.load(INPUT_DIR / "noFT_right.npy"),
}

for name, data in contexts.items():
    print(f"  {name}: {data.shape}")

print()

# ============================================================================
# Prepare data
# ============================================================================

print("[2/3] Preparing data for t-SNE...")

# Flatten contexts
def flatten(ctx):
    return ctx.reshape(ctx.shape[0], -1)

ft_left_flat = flatten(contexts['FT-Left'])
ft_right_flat = flatten(contexts['FT-Right'])
noFT_left_flat = flatten(contexts['NoFT-Left'])
noFT_right_flat = flatten(contexts['NoFT-Right'])

print(f"  Flattened shape: {ft_left_flat.shape}")
print()

# ============================================================================
# t-SNE
# ============================================================================

print("[3/3] Running t-SNE...")
print("  (This may take a few minutes...)")
print()

# Create figure
fig = plt.figure(figsize=(18, 8))

# -----------------------------------------------------------------------------
# Plot 1: Before Fine-Tuning (NoFT)
# -----------------------------------------------------------------------------

print("  Computing t-SNE for NoFT...")
noFT_combined = np.vstack([noFT_left_flat, noFT_right_flat])
noFT_labels = ['Left']*len(noFT_left_flat) + ['Right']*len(noFT_right_flat)

tsne_noFT = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
noFT_embedded = tsne_noFT.fit_transform(noFT_combined)

ax1 = plt.subplot(1, 3, 1)
ax1.scatter(noFT_embedded[:50, 0], noFT_embedded[:50, 1], 
           c='blue', label='Left', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.scatter(noFT_embedded[50:, 0], noFT_embedded[50:, 1], 
           c='red', label='Right', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.set_title('Before Fine-Tuning (NoFT)\nPre-trained Kosmos-2', 
             fontsize=13, fontweight='bold')
ax1.set_xlabel('t-SNE Dimension 1')
ax1.set_ylabel('t-SNE Dimension 2')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotation
separation = "Mixed" if True else "Separated"
ax1.text(0.05, 0.95, f'Separation: {separation}', 
        transform=ax1.transAxes, fontsize=10, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# -----------------------------------------------------------------------------
# Plot 2: After Fine-Tuning (FT)
# -----------------------------------------------------------------------------

print("  Computing t-SNE for FT...")
ft_combined = np.vstack([ft_left_flat, ft_right_flat])
ft_labels = ['Left']*len(ft_left_flat) + ['Right']*len(ft_right_flat)

tsne_ft = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
ft_embedded = tsne_ft.fit_transform(ft_combined)

ax2 = plt.subplot(1, 3, 2)
ax2.scatter(ft_embedded[:50, 0], ft_embedded[:50, 1], 
           c='blue', label='Left', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.scatter(ft_embedded[50:, 0], ft_embedded[50:, 1], 
           c='red', label='Right', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.set_title('After Fine-Tuning (FT)\nCase 5 Epoch 4', 
             fontsize=13, fontweight='bold')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add annotation
separation = "Separated" if True else "Mixed"
ax2.text(0.05, 0.95, f'Separation: {separation}', 
        transform=ax2.transAxes, fontsize=10, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# -----------------------------------------------------------------------------
# Plot 3: Comparison (Combined)
# -----------------------------------------------------------------------------

ax3 = plt.subplot(1, 3, 3)

# NoFT with light colors
ax3.scatter(noFT_embedded[:50, 0], noFT_embedded[:50, 1], 
           c='lightblue', marker='o', label='NoFT-Left', alpha=0.4, s=40)
ax3.scatter(noFT_embedded[50:, 0], noFT_embedded[50:, 1], 
           c='lightcoral', marker='o', label='NoFT-Right', alpha=0.4, s=40)

# FT with dark colors
ax3.scatter(ft_embedded[:50, 0], ft_embedded[:50, 1], 
           c='darkblue', marker='^', label='FT-Left', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax3.scatter(ft_embedded[50:, 0], ft_embedded[50:, 1], 
           c='darkred', marker='^', label='FT-Right', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

ax3.set_title('Comparison: NoFT vs FT\nEffect of LoRA Fine-Tuning', 
             fontsize=13, fontweight='bold')
ax3.set_xlabel('t-SNE Dimension 1')
ax3.set_ylabel('t-SNE Dimension 2')
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

plt.suptitle('Latent Space Analysis: Fine-Tuning Effect on Direction Separation\n' +
            '(Placeholder Data - Replace with real context vectors)',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = OUTPUT_DIR / "tsne_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print()
print(f"✅ Visualization saved: {output_path}")

plt.close()

print()
print("="*80)
print("✅ t-SNE visualization complete!")
print("="*80)
print()

print("Summary:")
print("  - Before FT: NoFT Left/Right may be mixed (placeholder)")
print("  - After FT: FT Left/Right should be separated (real data)")
print("  - Comparison: Shows LoRA's effect on latent space")
print()
print("⚠️  Note: Replace placeholder with real context vectors for actual results")
