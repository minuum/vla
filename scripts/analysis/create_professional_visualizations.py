#!/usr/bin/env python3
"""
Simplified Professional Visualization Generator
matplotlib만 사용하여 논문 스타일 시각화 생성
"""

import matplotlib
matplotlib.use('Agg')  # Display 없이 실행

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 출력 디렉토리
OUTPUT_DIR = Path('docs/visualizations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 한글 깨짐 방지
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Professional Visualization Generator")
print("=" * 60)
print()

# ============================================================================
# Data
# ============================================================================

experiment_data = {
    'Case 1': {'val_loss': 0.027, 'strategy': 'Baseline', 'chunk': 10},
    'Case 2': {'val_loss': 0.048, 'strategy': 'Xavier Init', 'chunk': 10},
    'Case 3': {'val_loss': 0.050, 'strategy': 'Aug+Abs', 'chunk': 10},
    'Case 4': {'val_loss': 0.016, 'strategy': 'Right Only', 'chunk': 10},
    'Case 5': {'val_loss': 0.000532, 'strategy': 'No Chunk', 'chunk': 1},
    'Case 8': {'val_loss': 0.004, 'strategy': 'No Chunk+Abs', 'chunk': 1},
}

colors = {
    'Case 1': '#1f77b4', 'Case 2': '#ff7f0e', 'Case 3': '#2ca02c',
    'Case 4': '#d62728', 'Case 5': '#9467bd', 'Case 8': '#8c564b',
}

# Case 5 Epoch 데이터
case5_epochs = [0.013864, 0.002332, 0.001668, 0.001287, 0.000532, 0.000793]

# ============================================================================
# Visualization 1: Bar Chart Comparison
# ============================================================================

print("[1/3] Creating performance comparison bar chart...")

fig, ax = plt.subplots(figsize=(12, 7))

cases = list(experiment_data.keys())
val_losses = [experiment_data[c]['val_loss'] for c in cases]
case_colors = [colors[c] for c in cases]

bars = ax.bar(cases, val_losses, color=case_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Case 5 강조
bars[4].set_alpha(1.0)
bars[4].set_edgecolor('red')
bars[4].set_linewidth(3)

ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
ax.set_xlabel('Experiment Case', fontsize=13, fontweight='bold')
ax.set_title('Fig. 1: Final Validation Loss Comparison\n(Case 5 achieves 30x lower loss)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# 값 레이블
for i, (bar, val) in enumerate(zip(bars, val_losses)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}' if val < 0.01 else f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_loss_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Created: {OUTPUT_DIR / 'fig_loss_comparison.png'}")
plt.close()

# ============================================================================
# Visualization 2: Line Plot - Case 5 Training Progress
# ============================================================================

print("[2/3] Creating Case 5 training progress...")

fig, ax = plt.subplots(figsize=(10, 6))

epochs = list(range(len(case5_epochs)))
ax.plot(epochs, case5_epochs, marker='o', markersize=10,
        color=colors['Case 5'], linewidth=3, label='Case 5: No Chunk')

# Epoch 4 최적점 강조
ax.scatter([4], [0.000532], s=400, marker='*', 
          color='red', edgecolors='black', linewidth=2, zorder=10,
          label='Epoch 4 (Optimal)')

# Epoch 5 과적합 표시
ax.axvspan(5, 5.5, alpha=0.2, color='red', label='Overfitting starts')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Fig. 2: Case 5 Training Progress\n(Best model at Epoch 4)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper right')

# 값 레이블
for i, val in enumerate(case5_epochs):
    ax.annotate(f'{val:.6f}', (i, val), 
                textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_case5_progress.png', dpi=300, bbox_inches='tight')
print(f"✓ Created: {OUTPUT_DIR / 'fig_case5_progress.png'}")
plt.close()

# ============================================================================
# Visualization 3: Strategy Comparison
# ============================================================================

print("[3/3] Creating strategy comparison...")

fig, ax = plt.subplots(figsize=(11, 6))

# Chunking별로 그룹화
no_chunk = {k: v for k, v in experiment_data.items() if v['chunk'] == 1}
chunk = {k: v for k, v in experiment_data.items() if v['chunk'] == 10}

x_no_chunk = list(range(len(no_chunk)))
y_no_chunk = [v['val_loss'] for v in no_chunk.values()]
labels_no_chunk = [f"{k}\n{v['strategy']}" for k, v in no_chunk.items()]

x_chunk = list(range(len(no_chunk), len(no_chunk) + len(chunk)))
y_chunk = [v['val_loss'] for v in chunk.values()]
labels_chunk = [f"{k}\n{v['strategy']}" for k, v in chunk.items()]

# 막대 그래프
bars1 = ax.bar(x_no_chunk, y_no_chunk, width=0.8, 
               color='#9467bd', alpha=0.8, edgecolor='black',
               label='No Chunk (fwd_pred_next_n=1)')
bars2 = ax.bar(x_chunk, y_chunk, width=0.8,
               color='#1f77b4', alpha=0.8, edgecolor='black',
               label='Chunk (fwd_pred_next_n=10)')

# Case 5 강조
bars1[0].set_edgecolor('red')
bars1[0].set_linewidth(3)

ax.set_ylabel('Validation Loss (log scale)', fontsize=12, fontweight='bold')
ax.set_xlabel('Experiment Case', fontsize=12, fontweight='bold')
ax.set_title('Fig. 3: Performance by Action Chunking Strategy\n(No Chunk >> Chunk)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_yscale('log')
ax.set_xticks(x_no_chunk + x_chunk)
ax.set_xticklabels(labels_no_chunk + labels_chunk, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_strategy_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Created: {OUTPUT_DIR / 'fig_strategy_comparison.png'}")
plt.close()

# ============================================================================
# Summary
# ============================================================================

summary = f"""# Professional Visualizations

**Generated**: 2025-12-10 09:16

## Files Created

1. **fig_loss_comparison.png** - Final validation loss comparison (bar chart)
2. **fig_case5_progress.png** - Case 5 training progress (line plot)
3. **fig_strategy_comparison.png** - Chunking strategy comparison

## Key Findings

- **Best**: Case 5 (Val Loss: 0.000532)
- **Strategy**: No Chunk (fwd_pred_next_n=1)
- **Improvement**: 30x better than chunking approaches

All visualizations use:
- English labels only (no Korean)
- Professional color scheme
- High resolution (300 DPI)
- Publication quality

**Output**: `{OUTPUT_DIR}/`
"""

with open(OUTPUT_DIR / 'SUMMARY.md', 'w') as f:
    f.write(summary)

print()
print("=" * 60)
print("✓ All visualizations created successfully!")
print(f"✓ Output directory: {OUTPUT_DIR}")
print("=" * 60)
print()
print("Files created:")
print(f"  - {OUTPUT_DIR / 'fig_loss_comparison.png'}")
print(f"  - {OUTPUT_DIR / 'fig_case5_progress.png'}")
print(f"  - {OUTPUT_DIR / 'fig_strategy_comparison.png'}")
print(f"  - {OUTPUT_DIR / 'SUMMARY.md'}")
