#!/bin/bash
# Left navigation only - Chunk5 학습

LOG_FILE="logs/train_left_chunk5_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting Left Chunk5 Training..."
echo "Log: $LOG_FILE"

python3 RoboVLMs_upstream/main.py \
    Mobile_VLA/configs/mobile_vla_left_chunk5_20251218.json \
    2>&1 | tee "$LOG_FILE"

echo "✅ Training completed. Check $LOG_FILE"
