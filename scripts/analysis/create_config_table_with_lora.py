#!/usr/bin/env python3
"""
Updated Table with LoRA/Frozen Information
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path('docs/visualizations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("Creating updated configuration table with LoRA/Frozen info...")

# ============================================================================
# Data with LoRA/Frozen info
# ============================================================================

config_data = [
    ['Case 1', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '10', 'L+R (500)', 'Baseline', '10'],
    ['Case 2', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '10', 'L+R (500)', 'Xavier Init', '10'],
    ['Case 3', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '10', 'L+R (500)', 'Aug+Abs', '10'],
    ['Case 4', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '10', 'R only (250)', 'Baseline', '10'],
    ['Case 5', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '1', 'L+R (500)', 'No Chunk', '7'],
    ['Case 8', 'Kosmos-2', 'Frozen', 'Yes (r=32)', '8', '1', 'L+R (500)', 'No Chunk+Abs', '5'],
]

headers = ['Case', 'Model', 'Backbone', 'LoRA\n(rank)', 'Window', 'Chunk', 'Data', 'Strategy', 'Epochs']

# ============================================================================
# Create Table
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

table = ax.table(cellText=config_data,
                 colLabels=headers,
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.08, 0.10, 0.10, 0.11, 0.08, 0.08, 0.12, 0.15, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.8)

# Header styling
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#40466e')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Case 5 highlighting (best)
for j in range(len(headers)):
    cell = table[(5, j)]  # Case 5 is row 5
    cell.set_facecolor('#d4edda')
    if j in [0, 5, 7]:  # Case, Chunk, Strategy
        cell.set_text_props(weight='bold', fontsize=11)

# Case 8 highlighting
for j in range(len(headers)):
    cell = table[(6, j)]  # Case 8 is row 6
    cell.set_facecolor('#fff3cd')
    if j == 5:  # Chunk
        cell.set_text_props(weight='bold')

# Highlight Backbone and LoRA columns (common settings)
for i in range(1, len(config_data) + 1):
    for j in [2, 3]:  # Backbone, LoRA
        cell = table[(i, j)]
        cell.set_facecolor('#f0f0f0')  # Light gray
        cell.set_text_props(fontsize=9, style='italic')

ax.set_title('TABLE I (Updated): Experiment Configuration with LoRA/Frozen Details\n' +
            'All cases use Frozen Backbone + LoRA (rank=32)',
            fontsize=13, fontweight='bold', pad=20)

# Add notes
notes_text = (
    "Common Settings: Frozen Backbone (freeze_backbone=True) + LoRA enabled (r=32, α=16)\n"
    "Training: AdamW (lr=1e-4), Batch=1, Window=8, Precision=16-bit mixed\n"
    "Variables: Chunk (1 vs 10), Data (500 vs 250), Strategy (Baseline, Xavier, Aug+Abs, etc.)"
)

plt.figtext(0.5, 0.02, notes_text, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig(OUTPUT_DIR / 'table1_config_with_lora_frozen.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Created: {OUTPUT_DIR / 'table1_config_with_lora_frozen.png'}")
plt.close()

print()
print("=" * 70)
print("✓ Updated table created successfully!")
print("=" * 70)
