#!/bin/bash

# EXP-16, 17 Window 최적화 학습 스크립트
# Window 6, 8을 순차적으로 학습하여 최적 Window Size 탐색

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_train_window_opt.log"
}

wait_for_process() {
    PROCESS_NAME=$1
    while pgrep -f "$PROCESS_NAME" > /dev/null; do
        log_message "⏳ Waiting for $PROCESS_NAME to finish..."
        sleep 30
    done
    log_message "✅ $PROCESS_NAME finished."
}

train_experiment() {
    EXP_NAME=$1
    CONFIG_FILE=$2
    
    log_message "========================================="
    log_message "🚀 Training $EXP_NAME"
    log_message "Config: $CONFIG_FILE"
    log_message "========================================="
    
    # 학습 시작
    CUDA_VISIBLE_DEVICES=0 python3 RoboVLMs/main.py \
        --config "$CONFIG_FILE" \
        > "$LOG_DIR/training_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    TRAIN_PID=$!
    log_message "📊 Training PID: $TRAIN_PID"
    
    # 학습 완료 대기
    wait $TRAIN_PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        log_message "✅ $EXP_NAME training completed successfully!"
    else
        log_message "❌ $EXP_NAME training failed with exit code $EXIT_CODE"
    fi
    
    log_message ""
    sleep 10
}

# 메인 시퀀스 시작
log_message "🤖 Window Optimization Training Sequence Started"
log_message "📂 Experiments: EXP-16 (Window 6), EXP-17 (Window 8)"
log_message ""

# 기존 학습 프로세스 대기
wait_for_process "main.py"

# EXP-16: Window 6 + Chunk 1
train_experiment "exp16_win6_k1" "Mobile_VLA/configs/mobile_vla_exp16_win6_k1.json"

# EXP-17: Window 8 + Chunk 1 (CALVIN-aligned)
train_experiment "exp17_win8_k1" "Mobile_VLA/configs/mobile_vla_exp17_win8_k1.json"

# 완료
log_message "========================================="
log_message "🎉 All Window Optimization Experiments Completed!"
log_message "========================================="
log_message "📁 Logs saved in: $LOG_DIR"
log_message ""
log_message "Next steps:"
log_message "1. Check training logs for any errors"
log_message "2. Run inference tests on both models"
log_message "3. Compare with EXP-05 (Window 12, k=1): 89.72%"
log_message "4. Select optimal Window size for final model"
