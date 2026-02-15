#!/bin/bash

# Configuration
LOG_DIR="logs"
CONFIG_DIR="Mobile_VLA/configs"
CURRENT_PID_FILE="$LOG_DIR/train_pid"

mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_train_sequence.log"
}

wait_for_current_process() {
    if [ -f "$CURRENT_PID_FILE" ]; then
        PID=$(cat "$CURRENT_PID_FILE")
        if ps -p $PID > /dev/null; then
            log_message "⏳ Waiting for current training (PID: $PID) to finish..."
            while ps -p $PID > /dev/null; do
                sleep 60
            done
            log_message "✅ Previous training (PID: $PID) completed."
        else
            log_message "⚠️ PID file exists but process $PID is not running."
        fi
    else
        log_message "ℹ️ No active training PID file found. Starting sequence immediately."
    fi
}

run_experiment() {
    EXP_NAME=$1
    CONFIG_FILE=$2
    LOG_FILE="$LOG_DIR/training_${EXP_NAME}.log"

    log_message "🚀 Starting Experiment: $EXP_NAME"
    log_message "   Config: $CONFIG_FILE"
    
    # Run training
    export CUDA_VISIBLE_DEVICES=0
    python3 RoboVLMs/main.py "$CONFIG_DIR/$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    
    echo $NEW_PID > "$CURRENT_PID_FILE"
    log_message "   PID: $NEW_PID"
    
    # Wait for this experiment to finish
    wait $NEW_PID
    
    if [ $? -eq 0 ]; then
        log_message "✅ Experiment $EXP_NAME finished successfully."
    else
        log_message "❌ Experiment $EXP_NAME failed! Check $LOG_FILE."
        # Optional: exit on failure? For now, continue to next.
    fi
    
    # Tiny cooldown
    sleep 30
}

# --- Main Sequence ---

log_message "🤖 Auto Training Scheduler Started"

# 1. Wait for EXP-05 (k=1) to finish
# (Assuming EXP-05 is currently running and its PID is in logs/train_pid)
wait_for_current_process

# 2. Run EXP-06: Unified Reg + Resampler
run_experiment "unified_reg_win12_k6_resampler" "mobile_vla_unified_reg_win12_k6_resampler.json"

# 3. Run EXP-07: Unified Reg + INT8 (QLoRA)
run_experiment "unified_reg_win12_k6_int8" "mobile_vla_unified_reg_win12_k6_int8.json"

# 4. Run EXP-08: Unified Classification (Baseline)
run_experiment "unified_class_win12_k6" "mobile_vla_unified_class_win12_k6.json"

log_message "🎉 All scheduled experiments completed!"
