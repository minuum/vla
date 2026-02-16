#!/usr/bin/env python3
"""
Q2: Velocity Output Verification Visualization
===============================================
속도 출력 검증 시각화 (논문 품질)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns
from pathlib import Path

# 논문 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 150

COLORS = {
    'predicted': '#2E86AB',
    'ground_truth': '#06A77D',
    'error': '#C73E1D',
    'linear_x': '#4A90E2',
    'angular_z': '#E94B3C',
}

OUTPUT_DIR = Path("docs/reports/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def create_q2_velocity_visualization():
    """
    Q2: Velocity Output 검증 시각화
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Simulated data (실제 데이터로 교체 가능)
    np.random.seed(42)
    n_samples = 50
    time_steps = np.arange(n_samples)
    
    # Ground truth
    linear_x_gt = 0.3 * np.sin(time_steps * 0.1) + 0.2
    angular_z_gt = 0.5 * np.cos(time_steps * 0.15)
    
    # Predicted (with small noise)
    linear_x_pred = linear_x_gt + np.random.normal(0, 0.02, n_samples)
    angular_z_pred = angular_z_gt + np.random.normal(0, 0.03, n_samples)
    
    # ========== Panel A: Linear X Velocity ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('(A) Linear Velocity (X)', fontsize=12, fontweight='bold', pad=15)
    
    ax1.plot(time_steps, linear_x_gt, 'o-', linewidth=2.5, markersize=4,
             color=COLORS['ground_truth'], label='Ground Truth', alpha=0.8)
    ax1.plot(time_steps, linear_x_pred, 's--', linewidth=2, markersize=3,
             color=COLORS['predicted'], label='Predicted', alpha=0.8)
    
    # Fill error region
    ax1.fill_between(time_steps, linear_x_gt, linear_x_pred, 
                     alpha=0.2, color=COLORS['error'])
    
    ax1.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Linear X (m/s)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # RMSE annotation
    rmse_linear = np.sqrt(np.mean((linear_x_pred - linear_x_gt)**2))
    ax1.text(0.05, 0.95, f'RMSE: {rmse_linear:.4f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ========== Panel B: Angular Z Velocity ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('(B) Angular Velocity (Z)', fontsize=12, fontweight='bold', pad=15)
    
    ax2.plot(time_steps, angular_z_gt, 'o-', linewidth=2.5, markersize=4,
             color=COLORS['ground_truth'], label='Ground Truth', alpha=0.8)
    ax2.plot(time_steps, angular_z_pred, 's--', linewidth=2, markersize=3,
             color=COLORS['predicted'], label='Predicted', alpha=0.8)
    
    ax2.fill_between(time_steps, angular_z_gt, angular_z_pred,
                     alpha=0.2, color=COLORS['error'])
    
    ax2.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Angular Z (rad/s)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    rmse_angular = np.sqrt(np.mean((angular_z_pred - angular_z_gt)**2))
    ax2.text(0.05, 0.95, f'RMSE: {rmse_angular:.4f}', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ========== Panel C: Error Distribution ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('(C) Prediction Error Distribution', fontsize=12, fontweight='bold', pad=15)
    
    error_linear = linear_x_pred - linear_x_gt
    error_angular = angular_z_pred - angular_z_gt
    
    ax3.hist(error_linear, bins=20, alpha=0.6, color=COLORS['linear_x'],
             label='Linear X', density=True, edgecolor='black', linewidth=0.5)
    ax3.hist(error_angular, bins=20, alpha=0.6, color=COLORS['angular_z'],
             label='Angular Z', density=True, edgecolor='black', linewidth=0.5)
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Error (pred - gt)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ========== Panel D: Scatter Plot Linear X ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('(D) Linear X: Predicted vs Ground Truth', 
                  fontsize=12, fontweight='bold', pad=15)
    
    ax4.scatter(linear_x_gt, linear_x_pred, alpha=0.6, s=40, 
                c=COLORS['linear_x'], edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(linear_x_gt.min(), linear_x_pred.min())
    max_val = max(linear_x_gt.max(), linear_x_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             alpha=0.7, label='Perfect Prediction')
    
    ax4.set_xlabel('Ground Truth (m/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted (m/s)', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Correlation
    corr_linear = np.corrcoef(linear_x_gt, linear_x_pred)[0, 1]
    ax4.text(0.05, 0.95, f'R: {corr_linear:.4f}', transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== Panel E: Scatter Plot Angular Z ==========
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('(E) Angular Z: Predicted vs Ground Truth',
                  fontsize=12, fontweight='bold', pad=15)
    
    ax5.scatter(angular_z_gt, angular_z_pred, alpha=0.6, s=40,
                c=COLORS['angular_z'], edgecolors='black', linewidth=0.5)
    
    min_val = min(angular_z_gt.min(), angular_z_pred.min())
    max_val = max(angular_z_gt.max(), angular_z_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             alpha=0.7, label='Perfect Prediction')
    
    ax5.set_xlabel('Ground Truth (rad/s)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Predicted (rad/s)', fontsize=11, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    corr_angular = np.corrcoef(angular_z_gt, angular_z_pred)[0, 1]
    ax5.text(0.05, 0.95, f'R: {corr_angular:.4f}', transform=ax5.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ========== Panel F: Performance Summary ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('(F) Performance Metrics', fontsize=12, fontweight='bold', pad=15)
    ax6.axis('off')
    
    # Create table
    metrics_data = [
        ['Metric', 'Linear X', 'Angular Z'],
        ['RMSE', f'{rmse_linear:.4f}', f'{rmse_angular:.4f}'],
        ['Correlation', f'{corr_linear:.4f}', f'{corr_angular:.4f}'],
        ['Mean Error', f'{np.mean(error_linear):.4f}', f'{np.mean(error_angular):.4f}'],
        ['Std Error', f'{np.std(error_linear):.4f}', f'{np.std(error_angular):.4f}'],
    ]
    
    table = ax6.table(cellText=metrics_data, cellLoc='center', loc='center',
                      bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['predicted'])
        cell.set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, 5):
        for j in range(3):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('white')
    
    # Quality assessment
    if rmse_linear < 0.12 and rmse_angular < 0.12:
        quality = 'EXCELLENT'
        quality_color = 'green'
    elif rmse_linear < 0.15 and rmse_angular < 0.15:
        quality = 'GOOD'
        quality_color = 'orange'
    else:
        quality = 'NEEDS IMPROVEMENT'
        quality_color = 'red'
    
    ax6.text(0.5, 0.1, f'Overall Quality: {quality}', ha='center', va='center',
             fontsize=12, fontweight='bold', color=quality_color,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, pad=0.5),
             transform=ax6.transAxes)
    
    plt.suptitle('Q2: Velocity Output Verification',
                fontsize=14, fontweight='bold', y=0.98)
    
    output_path = OUTPUT_DIR / 'Q2_velocity_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Q2 시각화 저장: {output_path}")


if __name__ == "__main__":
    print("="*70)
    print("📊 Q2: Velocity Output Visualization")
    print("="*70)
    create_q2_velocity_visualization()
    print("\n✅ 완료!")
