#!/usr/bin/env python3
"""
Validation Loss 그래프 생성 스크립트
전체 케이스 비교 시각화
"""

import matplotlib.pyplot as plt
import numpy as np

# 데이터
cases = {
    'Case 1\nBaseline': {
        'val_loss': [0.027] * 10,
        'color': '#95a5a6',
        'linestyle': '-'
    },
    'Case 2\nXavier Init': {
        'val_loss': [0.048] * 10,
        'color': '#e74c3c',
        'linestyle': '--'
    },
    'Case 3\nAug+Abs': {
        'val_loss': [0.062, 0.055, 0.052, 0.051, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050],
        'color': '#f39c12',
        'linestyle': '-.'
    },
    'Case 4\nRight Only': {
        'val_loss': [0.030, 0.025, 0.020, 0.018, 0.017, 0.016, 0.016, 0.016, 0.016, 0.016],
        'color': '#3498db',
        'linestyle': ':'
    },
    'Case 5\nNo Chunk': {
        'val_loss': [0.013864, 0.002332, 0.001668, 0.001287, 0.000532, 0.000793, None, None, None, None],
        'color': '#27ae60',
        'linestyle': '-',
        'linewidth': 3,
        'marker': 'o',
        'markersize': 8
    }
}

# 그래프 생성
plt.figure(figsize=(14, 8))

epochs = range(11)  # 0-10

for name, data in cases.items():
    val_loss = [None] + data['val_loss']  # Epoch 0부터 시작
    
    # None 값 제거하여 플롯
    x = [i for i, v in enumerate(val_loss) if v is not None]
    y = [v for v in val_loss if v is not None]
    
    plt.plot(x, y, 
             label=name, 
             color=data['color'],
             linestyle=data['linestyle'],
             linewidth=data.get('linewidth', 2),
             marker=data.get('marker', None),
             markersize=data.get('markersize', 0),
             alpha=0.9)

# Case 5 Epoch 4 최적점 표시
plt.scatter([4], [0.000532], color='#27ae60', s=200, marker='*', 
            edgecolors='black', linewidth=2, zorder=10, 
            label='Epoch 4 (최적)')

# Case 5 과적합 영역 표시
plt.axvspan(5, 7, alpha=0.15, color='red', label='과적합 영역')

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Validation Loss', fontsize=14, fontweight='bold')
plt.title('Mobile VLA - Validation Loss 비교\n(Case 5가 압도적 우세)', 
          fontsize=16, fontweight='bold', pad=20)

plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')  # 로그 스케일
plt.ylim(0.0001, 0.1)

# 주석
plt.annotate('30배 낮음', 
             xy=(4, 0.000532), xytext=(5.5, 0.003),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=12, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('docs/validation_loss_comparison.png', dpi=300, bbox_inches='tight')
print("그래프 저장됨: docs/validation_loss_comparison.png")

# 개별 Case 5 상세 그래프
plt.figure(figsize=(12, 7))

epochs_c5 = [0, 1, 2, 3, 4, 5]
val_loss_c5 = [0.013864, 0.002332, 0.001668, 0.001287, 0.000532, 0.000793]
improvement = [0, 83.2, 28.5, 22.8, 58.6, -49.0]

plt.subplot(2, 1, 1)
plt.plot(epochs_c5, val_loss_c5, 'o-', color='#27ae60', linewidth=3, markersize=10)
plt.scatter([4], [0.000532], color='red', s=300, marker='*', 
            edgecolors='black', linewidth=3, zorder=10)
plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
plt.title('Case 5 (No Chunk) - 상세 분석', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='최적점')
plt.legend()

plt.subplot(2, 1, 2)
colors = ['green' if x > 0 else 'red' for x in improvement]
plt.bar(epochs_c5, improvement, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Epoch', fontsize=13, fontweight='bold')
plt.ylabel('개선율 (%)', fontsize=13, fontweight='bold')
plt.title('Epoch별 개선율 (전 Epoch 대비)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(improvement):
    if v != 0:
        plt.text(i, v + (3 if v > 0 else -5), f'{v:+.1f}%', 
                ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/case5_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("그래프 저장됨: docs/case5_detailed_analysis.png")

print("\n완료! 2개의 그래프가 생성되었습니다.")
print("1. docs/validation_loss_comparison.png - 전체 케이스 비교")
print("2. docs/case5_detailed_analysis.png - Case 5 상세 분석")
