#!/bin/bash
# Basket Navigation Left Only 학습 스크립트
# 작성일: 2026-02-01

set -e

cd /home/billy/25-1kp/vla

echo "=================================================="
echo "🚀 Basket Navigation Left Only 학습 시작"
echo "=================================================="

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_basket_left_only_${TIMESTAMP}.log"
mkdir -p logs

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_basket_left_only.json"

echo "📁 로그 파일: $LOG_FILE"
echo "🚀 config: $CONFIG_FILE"

nohup python3 RoboVLMs_upstream/main.py \
    "$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ 학습 프로세스 시작됨 (PID: $PID)"
echo "모니터링: tail -f $LOG_FILE"
