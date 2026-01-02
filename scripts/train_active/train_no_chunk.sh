#!/bin/bash
# Case 6: No Chunking Training
# Action Chunking 없이 매 스텝 1개의 액션만 예측

set -e

# 프로젝트 루트로 이동
cd /home/soda/25-1kp/vla

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_no_chunk_${TIMESTAMP}.log"
mkdir -p logs

echo "🚀 Starting Case 6: No Chunking"
echo "Log: $LOG_FILE"

python3 RoboVLMs_upstream/main.py \
    "Mobile_VLA/configs/mobile_vla_no_chunk_20251209.json" \
    > "$LOG_FILE" 2>&1
