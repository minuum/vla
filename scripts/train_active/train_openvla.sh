#!/bin/bash
# Case 5: OpenVLA Style Training
# OpenVLA/RT-2 논문 기반 설정 (LR 2e-5, Epoch 27)

set -e

# 프로젝트 루트로 이동
cd /home/billy/25-1kp/vla

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_openvla_${TIMESTAMP}.log"
mkdir -p logs

echo "🚀 Starting Case 5: OpenVLA Style"
echo "Log: $LOG_FILE"

# max_epochs는 config에 있지만 커맨드라인에서 덮어쓸 수도 있음 (이미 config에 27로 설정됨)
python3 RoboVLMs_upstream/main.py \
    "Mobile_VLA/configs/mobile_vla_openvla_style_20251209.json" \
    > "$LOG_FILE" 2>&1
