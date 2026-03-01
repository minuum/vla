import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def analyze_tb_logs():
    # 최신 로그 파일 찾기
    log_dir = "RoboVLMs_upstream/runs/v2_classification/kosmos/mobile_vla_v2_classification/2026-02-17/v2-classification-9cls/v2-classification-9cls/version_5"
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print("❌ 로그 파일을 찾을 수 없습니다.")
        return

    latest_file = max(event_files, key=os.path.getmtime)
    print(f"📊 분석 중인 로그 파일: {latest_file}")

    # 데이터 로드
    ea = EventAccumulator(latest_file)
    ea.Reload()

    # 사용 가능한 태그 확인
    tags = ea.Tags()['scalars']
    
    metrics = {}
    target_tags = ['train_loss', 'train_acc_velocity_act', 'train_rmse_velocity_act']
    
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Step':<6} | {'Value':<10}")
    print("-" * 50)
    
    for tag in target_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            last_event = events[-1]
            metrics[tag] = last_event.value
            print(f"{tag:<25} | {last_event.step:<6} | {last_event.value:.4f}")

    print("="*50)
    
    # 종합 분석
    acc = metrics.get('train_acc_velocity_act', 0)
    loss = metrics.get('train_loss', 0)
    
    if acc > 0.9 and loss < 0.1:
        print("\n🔥 결론: 모델이 학습 데이터를 완벽하게 암기(Convergence)했습니다.")
        print("💡 다음 단계: Validation 데이터셋에서도 이 정도 정확도가 나오는지 확인이 필요합니다.")
    elif acc > 0.5:
        print("\n📈 결론: 모델이 방향성을 잡고 학습 중입니다.")
    else:
        print("\n⏳ 결론: 아직 학습 초기 단계입니다.")

if __name__ == "__main__":
    analyze_tb_logs()
