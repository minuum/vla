import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_training_curves():
    # 로그 파일 경로 (실제 경로로 수정 필요)
    # 2025-12-17 실행 로그 추정
    chunk5_log = "/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/mobile_vla_chunk5_20251217/version_1/metrics.csv"
    chunk10_log = "/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/mobile_vla_chunk10_20251217/version_1/metrics.csv"
    
    # 경로 존재 확인
    if not Path(chunk5_log).exists() or not Path(chunk10_log).exists():
        print(f"❌ Log files not found. Please check paths.")
        # 임시 데이터 생성 (시각화 테스트용) -> 실제 데이터 확인 후 제거
        return

    # 데이터 로드
    df5 = pd.read_csv(chunk5_log)
    df10 = pd.read_csv(chunk10_log)
    
    # Train/Val 분리 및 정리
    # metrics.csv는 step마다 기록되므로 epoch 단위로 aggregation 필요할 수 있음
    # 또는 'epoch' 컬럼 기준으로 평균
    
    def process_df(df):
        # val_loss가 있는 행과 train_loss가 있는 행이 다를 수 있음
        # epoch 기준으로 grouping
        train = df.dropna(subset=['train_loss']).groupby('epoch')['train_loss'].mean()
        val = df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].mean()
        return train, val

    train5, val5 = process_df(df5)
    train10, val10 = process_df(df10)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train5.index, train5.values, label='Chunk5 Train', linestyle='--', color='blue', alpha=0.6)
    ax.plot(val5.index, val5.values, label='Chunk5 Val', linestyle='-', color='blue', linewidth=2, marker='o')
    
    ax.plot(train10.index, train10.values, label='Chunk10 Train', linestyle='--', color='red', alpha=0.6)
    ax.plot(val10.index, val10.values, label='Chunk10 Val', linestyle='-', color='red', linewidth=2, marker='s')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Curves: Chunk5 vs Chunk10', fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Best Point Annotation
    min5 = val5.min()
    epoch5 = val5.idxmin()
    ax.annotate(f'Best: {min5:.3f}', xy=(epoch5, min5), xytext=(epoch5, min5+0.05),
                arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')

    min10 = val10.min()
    epoch10 = val10.idxmin()
    ax.annotate(f'Best: {min10:.3f}', xy=(epoch10, min10), xytext=(epoch10, min10+0.05),
                arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    # 저장
    output_path = "docs/figures/training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {output_path}")

if __name__ == "__main__":
    plot_training_curves()
