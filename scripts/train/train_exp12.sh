#!/bin/bash
# EXP-12 Training Script (Hybrid Masterpiece)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/train_exp12.log"
}

EXP_NAME="exp12_win6_k1_resampler"
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_exp12_win6_k1_resampler.json"

log_message "========================================="
log_message "🚀 Starting Hybrid Experiment: $EXP_NAME"
log_message "📈 Goal: Higher than 90% PM/DA via Window 6 + Resampler"
log_message "========================================="

export CUDA_VISIBLE_DEVICES=0

python3 RoboVLMs/main.py "$CONFIG_FILE" > "$LOG_DIR/training_${EXP_NAME}.log" 2>&1

if [ $? -eq 0 ]; then
    log_message "✅ $EXP_NAME training completed successfully!"
else
    log_message "❌ $EXP_NAME training failed. Check logs/training_${EXP_NAME}.log"
fi
