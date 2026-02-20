import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

def plot_training_metrics(log_base_dir, output_dir):
    """
    TensorBoard 로그에서 train_loss와 val_loss를 추출하여 그래프를 생성합니다.
    """
    sns.set_theme(style="whitegrid")
    versions = sorted(glob.glob(os.path.join(log_base_dir, "version_*")))
    
    train_steps, train_values = [], []
    val_steps, val_values = [], []
    
    for v in versions:
        event_files = glob.glob(os.path.join(v, "events.out.tfevents.*"))
        if not event_files: continue
        
        ea = EventAccumulator(max(event_files, key=os.path.getmtime))
        ea.Reload()
        tags = ea.Tags()['scalars']
        
        if 'train_loss' in tags:
            for ev in ea.Scalars('train_loss'):
                train_steps.append(ev.step)
                train_values.append(ev.value)
        
        if 'val_loss' in tags:
            for ev in ea.Scalars('val_loss'):
                val_steps.append(ev.step)
                val_values.append(ev.value)

    if not train_values and not val_values:
        print("❌ 시각화할 데이터를 찾을 수 없습니다.")
        return

    # 1. Train vs Val Loss Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_values, label='Train Loss', alpha=0.6, color='blue')
    plt.plot(val_steps, val_values, 'o-', label='Val Loss', markersize=8, color='red', linewidth=2)
    
    # 과적합 발생 지점 표시 (Val Loss 최저점)
    if val_values:
        min_val_idx = val_values.index(min(val_values))
        plt.annotate(f'Best (Overfitting Start)\nStep: {val_steps[min_val_idx]}',
                     xy=(val_steps[min_val_idx], val_values[min_val_idx]),
                     xytext=(val_steps[min_val_idx], val_values[min_val_idx] + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center', fontsize=10, fontweight='bold')

    plt.title('V2 Classification Training: Train vs Val Loss', fontsize=15)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log') # Loss 차이가 클 수 있으므로 log scale 권장
    plt.legend()
    
    output_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(output_path)
    print(f"✅ 그래프 저장 완료: {output_path}")

    # 2. Accuracy Plot (있을 경우)
    plt.figure(figsize=(12, 6))
    # Accuracy 추출 및 플로팅 로직 추가 가능... (생략)
    
    plt.close('all')

if __name__ == "__main__":
    LOG_DIR = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v2_classification/kosmos/mobile_vla_v2_classification/2026-02-17/v2-classification-9cls/v2-classification-9cls/"
    OUT_DIR = "/home/billy/25-1kp/vla/docs/plots/v2_classification"
    plot_training_metrics(LOG_DIR, OUT_DIR)
