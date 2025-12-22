#!/bin/bash
# Left navigation only - Chunk10 학습

LOG_FILE="logs/train_left_chunk10_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting Left Chunk10 Training..."
echo "Log: $LOG_FILE"

python3 RoboVLMs_upstream/main.py \
    Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json \
    2>&1 | tee "$LOG_FILE"

echo "✅ Training completed. Check $LOG_FILE"
