import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def plot_resource_comparison():
    # 데이터 (RoboVLMs vs Mobile VLA FP32 vs Mobile VLA INT8)
    models = ['RoboVLMs\n(7B)', 'Mobile VLA\n(FP32)', 'Mobile VLA\n(INT8)']
    gpu_mem = [14.0, 6.3, 1.8]  # GB
    latency = [15.0, 15.0, 0.495]  # Seconds (Log scale for visualization?)
    
    # 1. GPU Memory Comparison
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    bars = ax1.bar(models, gpu_mem, color=['gray', '#4c72b0', '#55a868'], width=0.6, alpha=0.9)
    
    ax1.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax1.set_title('GPU Memory Usage Comparison', fontsize=14, pad=15)
    ax1.set_ylim(0, 16)
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height} GB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # 절감율 표시
    # RoboVLMs -> Mobile INT8
    reduction = (1 - 1.8/14.0) * 100
    ax1.annotate(f'-{reduction:.0f}%', 
                xy=(2, 1.8), xytext=(1.5, 8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
                fontsize=11, color='#55a868', fontweight='bold')

    plt.tight_layout()
    plt.savefig("docs/figures/gpu_memory_comparison.png", dpi=300)
    print("✅ GPU Memory chart saved.")
    
    # 2. Inference Speed Comparison (Log scale)
    fig, ax2 = plt.subplots(figsize=(8, 6))
    
    # Latency Bar Chart
    bars2 = ax2.bar(models, latency, color=['gray', '#4c72b0', '#c44e52'], width=0.6, alpha=0.9)
    
    ax2.set_ylabel('Inference Latency (seconds)', fontsize=12)
    ax2.set_title('Inference Speed Comparison (Lower is Better)', fontsize=14, pad=15)
    
    # Log scale 적용 (차이가 너무 커서)
    ax2.set_yscale('log')
    ax2.set_ylim(0.1, 100)
    
    # 값 표시
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{height} s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speed up 표시
    # FP32 -> INT8
    speedup = 15.0 / 0.495
    ax2.annotate(f'{speedup:.0f}x Faster', 
                xy=(2, 0.495), xytext=(1.2, 0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2', color='black'),
                fontsize=11, color='#c44e52', fontweight='bold')

    plt.tight_layout()
    plt.savefig("docs/figures/inference_speed_comparison.png", dpi=300)
    print("✅ Inference Speed chart saved.")

if __name__ == "__main__":
    plot_resource_comparison()
