#!/bin/bash
# English Instruction 재학습
set -e
cd /home/billy/25-1kp/vla

echo "=========================================="
echo "Mobile VLA English Instruction Training"  
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_english_chunk5_${TIMESTAMP}.log"
mkdir -p logs

CONFIG_FILE="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"

echo "Starting English instruction training..."
echo "Log: $LOG_FILE"

nohup python3 RoboVLMs_upstream/main.py "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Started (PID: $PID)"
echo "Monitor: tail -f $LOG_FILE"
