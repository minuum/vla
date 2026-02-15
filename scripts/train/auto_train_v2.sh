#!/bin/bash

# Configuration
LOG_DIR="logs"
CONFIG_DIR="Mobile_VLA/configs"
CURRENT_PID_FILE="$LOG_DIR/train_pid"

mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_train_v2.log"
}

run_experiment() {
    EXP_NAME=$1
    CONFIG_FILE=$2
    LOG_FILE="$LOG_DIR/training_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

    log_message "🚀 Starting Experiment: $EXP_NAME"
    log_message "   Config: $CONFIG_FILE"
    
    # Run training
    export CUDA_VISIBLE_DEVICES=0
    python3 RoboVLMs/main.py "$CONFIG_DIR/$CONFIG_FILE" > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log_message "✅ Experiment $EXP_NAME finished successfully."
    else
        log_message "❌ Experiment $EXP_NAME failed! Check $LOG_FILE."
    fi
    
    # Tiny cooldown
    sleep 30
}

# --- Main Sequence ---

log_message "🤖 Auto Training Scheduler V2 Started"

# 1. EXP-09: Latent 128
run_experiment "exp09_latent128" "mobile_vla_exp09_latent128.json"

# 2. EXP-10: Window 16
run_experiment "exp10_win16" "mobile_vla_exp10_win16.json"

# 3. EXP-11: Discrete
run_experiment "exp11_resampler_discrete" "mobile_vla_exp11_resampler_discrete.json"

log_message "🎉 All scheduled experiments completed!"
