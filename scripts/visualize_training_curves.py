#!/usr/bin/env python3
"""
학습 곡선 시각화 스크립트
Chunk5와 Chunk10의 학습 추이를 비교
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(csv_path):
    """CSV 메트릭 파일 로딩"""
    df = pd.read_csv(csv_path)
    
    # Epoch별로 aggregation (중복 제거)
    # val_loss가 있는 행만 추출 (epoch 완료 시점)
    val_df = df[df['val_loss'].notna()].copy()
    
    # Epoch별 최종 값만 추출
    result = val_df.groupby('epoch').last().reset_index()
    
    return result

def plot_training_curves():
    """학습 곡선 시각화"""
    
    # 경로 설정
    base_path = Path('/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17')
    
    chunk5_csv = base_path / 'mobile_vla_chunk5_20251217/mobile_vla_chunk5_20251217/version_1/metrics.csv'
    chunk10_csv = base_path / 'mobile_vla_chunk10_20251217/mobile_vla_chunk10_20251217/version_1/metrics.csv'
    
    # CSV 로딩
    print("📊 Loading metrics...")
    chunk5_df = load_metrics(chunk5_csv)
    chunk10_df = load_metrics(chunk10_csv)
    
    print(f"   Chunk5: {len(chunk5_df)} epochs")
    print(f"   Chunk10: {len(chunk10_df)} epochs")
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mobile VLA Training Comparison: Chunk5 vs Chunk10', fontsize=16, fontweight='bold')
    
    # 1. Train Loss
    ax = axes[0, 0]
    ax.plot(chunk5_df['epoch'], chunk5_df['train_loss'], 
            marker='o', label='Chunk5 (fwd_pred_n=5)', linewidth=2, markersize=6)
    ax.plot(chunk10_df['epoch'], chunk10_df['train_loss'], 
            marker='s', label='Chunk10 (fwd_pred_n=10)', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (MSE)')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Val Loss
    ax = axes[0, 1]
    ax.plot(chunk5_df['epoch'], chunk5_df['val_loss'], 
            marker='o', label='Chunk5', linewidth=2, markersize=6, color='blue')
    ax.plot(chunk10_df['epoch'], chunk10_df['val_loss'], 
            marker='s', label='Chunk10', linewidth=2, markersize=6, color='orange')
    
    # Best epoch 표시
    chunk5_best_idx = chunk5_df['val_loss'].idxmin()
    chunk10_best_idx = chunk10_df['val_loss'].idxmin()
    
    ax.scatter(chunk5_df.loc[chunk5_best_idx, 'epoch'], 
               chunk5_df.loc[chunk5_best_idx, 'val_loss'],
               s=200, color='blue', marker='*', zorder=5, 
               label=f'Chunk5 Best (Epoch {int(chunk5_df.loc[chunk5_best_idx, "epoch"])})')
    ax.scatter(chunk10_df.loc[chunk10_best_idx, 'epoch'], 
               chunk10_df.loc[chunk10_best_idx, 'val_loss'],
               s=200, color='orange', marker='*', zorder=5,
               label=f'Chunk10 Best (Epoch {int(chunk10_df.loc[chunk10_best_idx, "epoch"])})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. RMSE Comparison
    ax = axes[1, 0]
    ax.plot(chunk5_df['epoch'], chunk5_df['train_rmse_velocity_act'], 
            marker='o', label='Chunk5 Train', linewidth=2, markersize=5, linestyle='--', alpha=0.7)
    ax.plot(chunk5_df['epoch'], chunk5_df['val_rmse_velocity_act'], 
            marker='o', label='Chunk5 Val', linewidth=2, markersize=6, color='blue')
    ax.plot(chunk10_df['epoch'], chunk10_df['train_rmse_velocity_act'], 
            marker='s', label='Chunk10 Train', linewidth=2, markersize=5, linestyle='--', alpha=0.7)
    ax.plot(chunk10_df['epoch'], chunk10_df['val_rmse_velocity_act'], 
            marker='s', label='Chunk10 Val', linewidth=2, markersize=6, color='orange')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (m/s)')
    ax.set_title('RMSE Comparison (Train vs Val)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Best Model Val Loss Bar Chart
    ax = axes[1, 1]
    
    chunk5_best_loss = chunk5_df.loc[chunk5_best_idx, 'val_loss']
    chunk10_best_loss = chunk10_df.loc[chunk10_best_idx, 'val_loss']
    
    bars = ax.bar(['Chunk5\n(Best)', 'Chunk10\n(Best)'], 
                   [chunk5_best_loss, chunk10_best_loss],
                   color=['blue', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, [chunk5_best_loss, chunk10_best_loss])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 개선율 표시
    improvement = ((chunk10_best_loss - chunk5_best_loss) / chunk10_best_loss) * 100
    ax.text(0.5, max(chunk5_best_loss, chunk10_best_loss) * 0.8,
            f'Chunk5 is {improvement:.1f}% better',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Best Validation Loss (MSE)')
    ax.set_title('Best Model Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = Path('/home/billy/25-1kp/vla/docs/training_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    
    # Summary 출력
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    
    print(f"\nChunk5 (Action Chunking: 5 steps)")
    print(f"  Best Epoch: {int(chunk5_df.loc[chunk5_best_idx, 'epoch'])}")
    print(f"  Best Val Loss: {chunk5_best_loss:.4f}")
    print(f"  Best Val RMSE: {chunk5_df.loc[chunk5_best_idx, 'val_rmse_velocity_act']:.4f} m/s")
    print(f"  Final Train Loss: {chunk5_df.iloc[-1]['train_loss']:.4f}")
    
    print(f"\nChunk10 (Action Chunking: 10 steps)")
    print(f"  Best Epoch: {int(chunk10_df.loc[chunk10_best_idx, 'epoch'])}")
    print(f"  Best Val Loss: {chunk10_best_loss:.4f}")
    print(f"  Best Val RMSE: {chunk10_df.loc[chunk10_best_idx, 'val_rmse_velocity_act']:.4f} m/s")
    print(f"  Final Train Loss: {chunk10_df.iloc[-1]['train_loss']:.4f}")
    
    print(f"\n🎯 Performance Gain:")
    print(f"  Val Loss: {improvement:.1f}% improvement (Chunk5 better)")
    
    rmse_improvement = ((chunk10_df.loc[chunk10_best_idx, 'val_rmse_velocity_act'] - 
                        chunk5_df.loc[chunk5_best_idx, 'val_rmse_velocity_act']) / 
                       chunk10_df.loc[chunk10_best_idx, 'val_rmse_velocity_act']) * 100
    print(f"  Val RMSE: {rmse_improvement:.1f}% improvement (Chunk5 better)")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    print("🚀 Mobile VLA Training Curves Visualization")
    print("="*60 + "\n")
    
    plot_training_curves()
    
    print("\n✅ Visualization complete!")
