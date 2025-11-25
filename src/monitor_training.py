#!/usr/bin/env python3
"""
Mobile VLA 학습 모니터링 스크립트
실시간으로 학습 로그를 파싱하여 명확한 메트릭 표시
"""

import time
import re
from pathlib import Path

def parse_epoch_line(line):
    """Epoch 진행 상황 파싱"""
    # Epoch 2:  98%|█████████▊| 44/45 [00:16<00:00,  2.67it/s, v_num=8_19, train_loss=0.0813, train_loss_arm_act=0.0813, ...
    
    # Epoch 번호
    epoch_match = re.search(r'Epoch (\d+):', line)
    if not epoch_match:
        return None
    
    epoch = int(epoch_match.group(1))
    
    # Progress
    progress_match = re.search(r'(\d+)%', line)
    progress = int(progress_match.group(1)) if progress_match else 0
    
    # Train loss (mobile 2D velocity)
    train_loss_match = re.search(r'train_loss_arm_act=([\d.]+)', line)
    train_loss = float(train_loss_match.group(1)) if train_loss_match else None
    
    # Val loss (mobile 2D velocity)
    val_loss_match = re.search(r'val_loss_arm_act=([\d.]+)', line)
    val_loss = float(val_loss_match.group(1)) if val_loss_match else None
    
    # Gripper accuracy (더미, 참고용)
    acc_match = re.search(r'train_acc_gripper_act=([\d.]+)', line)
    acc = float(acc_match.group(1)) if acc_match else None
    
    return {
        'epoch': epoch,
        'progress': progress,
        'train_loss_2d': train_loss,
        'val_loss_2d': val_loss,
        'dummy_acc': acc
    }

def monitor_log(log_file, interval=5):
    """로그 파일을 모니터링하며 결과 출력"""
    
    print("=" * 80)
    print("🚀 Mobile VLA LoRA 학습 모니터링")
    print("=" * 80)
    print()
    print("📊 메트릭 설명:")
    print("  - train_loss_2d: Mobile Robot 2D 속도 [linear_x, linear_y] 학습 loss")
    print("  - val_loss_2d:   검증 데이터에서의 2D 속도 예측 loss")
    print("  - dummy_acc:     그리퍼 정확도 (더미, 무시 가능)")
    print()
    print("-" * 80)
    print()
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    last_position = 0
    last_epoch_data = {}
    
    try:
        while True:
            with open(log_path, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
                for line in new_lines:
                    data = parse_epoch_line(line)
                    if data and data['progress'] == 100:
                        # Epoch 완료
                        epoch = data['epoch']
                        
                        if data['val_loss_2d'] is not None:
                            # Validation 포함된 최종 결과
                            print(f"✅ Epoch {epoch:2d} 완료:")
                            print(f"   Train Loss (2D Velocity): {data['train_loss_2d']:.4f}")
                            print(f"   Val Loss (2D Velocity):   {data['val_loss_2d']:.4f}")
                            
                            # 개선율 계산
                            if epoch > 0 and epoch-1 in last_epoch_data:
                                prev = last_epoch_data[epoch-1]
                                if prev['val_loss_2d'] is not None:
                                    improvement = (prev['val_loss_2d'] - data['val_loss_2d']) / prev['val_loss_2d'] * 100
                                    print(f"   개선율: {improvement:+.2f}%")
                            
                            print()
                            last_epoch_data[epoch] = data
                    
                    # 완료 메시지 감지
                    if '`Trainer.fit` stopped' in line:
                        print("=" * 80)
                        print("🎉 학습 완료!")
                        print("=" * 80)
                        print()
                        
                        # 최종 요약
                        if last_epoch_data:
                            epochs = sorted(last_epoch_data.keys())
                            print("📈 학습 결과 요약:")
                            print()
                            print("  Epoch | Train Loss | Val Loss | 개선율")
                            print("  ------|------------|----------|--------")
                            
                            for i, epoch in enumerate(epochs):
                                d = last_epoch_data[epoch]
                                if d['val_loss_2d'] is not None:
                                    if i == 0:
                                        print(f"  {epoch:5d} | {d['train_loss_2d']:10.4f} | {d['val_loss_2d']:8.4f} | -")
                                    else:
                                        prev = last_epoch_data[epochs[i-1]]
                                        if prev['val_loss_2d'] is not None:
                                            imp = (prev['val_loss_2d'] - d['val_loss_2d']) / prev['val_loss_2d'] * 100
                                            print(f"  {epoch:5d} | {d['train_loss_2d']:10.4f} | {d['val_loss_2d']:8.4f} | {imp:+6.2f}%")
                            
                            # 전체 개선율
                            if len(epochs) > 1:
                                first_val = last_epoch_data[epochs[0]]['val_loss_2d']
                                last_val = last_epoch_data[epochs[-1]]['val_loss_2d']
                                if first_val and last_val:
                                    total_imp = (first_val - last_val) / first_val * 100
                                    print()
                                    print(f"  💡 총 개선율: {total_imp:.2f}%")
                        
                        return
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  모니터링 중단됨")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = '/home/billy/25-1kp/vla/lora_training_20epochs_20251112.log'
    
    monitor_log(log_file)

