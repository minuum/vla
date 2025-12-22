#!/usr/bin/env python3
"""
Left vs Right Chunk10 비교 시각화
완료된 2개 실험 결과 비교
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_left_right_comparison():
    """Left vs Right Chunk10 비교 시각화"""
    
    print("🎨 Left vs Right Chunk10 비교 시각화 시작...")
    
    # CSV 경로
    base_path = Path('runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18')
    
    left_csv = base_path / 'mobile_vla_left_chunk10_20251218/mobile_vla_left_chunk10_20251218/version_1/metrics.csv'
    right_csv = base_path / 'mobile_vla_right_chunk10_20251218/mobile_vla_right_chunk10_20251218/version_1/metrics.csv'
    
    # 데이터 로딩
    print("📊 Loading CSV data...")
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)
    
    # Val loss 추출
    left_val = left_df[left_df['val_loss'].notna()].groupby('epoch').last().reset_index()
    right_val = right_df[right_df['val_loss'].notna()].groupby('epoch').last().reset_index()
    
    print(f"   Left: {len(left_val)} epochs")
    print(f"   Right: {len(right_val)} epochs")
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Left vs Right Navigation (Chunk10 Comparison)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Val Loss 비교
    ax = axes[0, 0]
    ax.plot(left_val['epoch'], left_val['val_loss'], 
            marker='o', label='Left Navigation', linewidth=2, markersize=6, color='blue')
    ax.plot(right_val['epoch'], right_val['val_loss'], 
            marker='s', label='Right Navigation', linewidth=2, markersize=6, color='orange')
    
    # Best epoch 표시
    left_best_idx = left_val['val_loss'].idxmin()
    right_best_idx = right_val['val_loss'].idxmin()
    
    ax.scatter(left_val.loc[left_best_idx, 'epoch'], 
               left_val.loc[left_best_idx, 'val_loss'],
               s=200, color='blue', marker='*', zorder=5, 
               label=f'Left Best (Epoch {int(left_val.loc[left_best_idx, "epoch"])})')
    ax.scatter(right_val.loc[right_best_idx, 'epoch'], 
               right_val.loc[right_best_idx, 'val_loss'],
               s=200, color='orange', marker='*', zorder=5,
               label=f'Right Best (Epoch {int(right_val.loc[right_best_idx, "epoch"])})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Val RMSE 비교
    ax = axes[0, 1]
    ax.plot(left_val['epoch'], left_val['val_rmse_velocity_act'], 
            marker='o', label='Left Navigation', linewidth=2, markersize=6, color='blue')
    ax.plot(right_val['epoch'], right_val['val_rmse_velocity_act'], 
            marker='s', label='Right Navigation', linewidth=2, markersize=6, color='orange')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (m/s)')
    ax.set_title('Validation RMSE Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Best Model 비교 (Bar Chart)
    ax = axes[1, 0]
    
    left_best_loss = left_val.loc[left_best_idx, 'val_loss']
    right_best_loss = right_val.loc[right_best_idx, 'val_loss']
    
    bars = ax.bar(['Left\nNavigation', 'Right\nNavigation'], 
                   [left_best_loss, right_best_loss],
                   color=['blue', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
    
    # 값 표시
    for bar, val in zip(bars, [left_best_loss, right_best_loss]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 성능 차이 표시
    diff_pct = ((right_best_loss - left_best_loss) / right_best_loss) * 100
    ax.text(0.5, max(left_best_loss, right_best_loss) * 0.8,
            f'Left is {diff_pct:.1f}% better',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Best Validation Loss (MSE)')
    ax.set_title('Best Model Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. 학습 속도 비교 (첫 5 Epoch)
    ax = axes[1, 1]
    
    epochs = range(5)
    left_progress = [left_val.loc[i, 'val_loss'] if i < len(left_val) else None for i in epochs]
    right_progress = [right_val.loc[i, 'val_loss'] if i < len(right_val) else None for i in epochs]
    
    ax.plot(epochs, left_progress, marker='o', label='Left', linewidth=2, color='blue')
    ax.plot(epochs, right_progress, marker='s', label='Right', linewidth=2, color='orange')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Learning Speed (First 5 Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = Path('docs/left_vs_right_chunk10_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    
    # Summary 출력
    print("\n" + "="*60)
    print("📊 Comparison Summary")
    print("="*60)
    
    print(f"\n🔵 Left Navigation (Chunk10)")
    print(f"  Best Epoch: {int(left_val.loc[left_best_idx, 'epoch'])}")
    print(f"  Best Val Loss: {left_best_loss:.4f}")
    print(f"  Best Val RMSE: {left_val.loc[left_best_idx, 'val_rmse_velocity_act']:.4f} m/s")
    
    print(f"\n🟠 Right Navigation (Chunk10)")
    print(f"  Best Epoch: {int(right_val.loc[right_best_idx, 'epoch'])}")
    print(f"  Best Val Loss: {right_best_loss:.4f}")
    print(f"  Best Val RMSE: {right_val.loc[right_best_idx, 'val_rmse_velocity_act']:.4f} m/s")
    
    print(f"\n🎯 Performance Comparison:")
    print(f"  Val Loss: Left is {diff_pct:.1f}% better than Right")
    
    rmse_diff = ((right_val.loc[right_best_idx, 'val_rmse_velocity_act'] - 
                  left_val.loc[left_best_idx, 'val_rmse_velocity_act']) / 
                 right_val.loc[right_best_idx, 'val_rmse_velocity_act']) * 100
    print(f"  RMSE: Left is {rmse_diff:.1f}% better than Right")
    
    print("="*60)

if __name__ == "__main__":
    print("🚀 Left vs Right Chunk10 Comparison Visualization")
    print("="*60 + "\n")
    
    plot_left_right_comparison()
    
    print("\n✅ Visualization complete!")
