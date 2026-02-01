#!/bin/bash
# Frozen VLM 학습 스크립트 (LoRA 비활성화)
# 작성일: 2025-12-16
# 목적: Action Head만 학습하여 LoRA 효과 증명

set -e

cd /home/billy/25-1kp/vla

echo "=================================================="
echo "🚀 Frozen VLM 학습 (Action Head만)"
echo "=================================================="
echo ""
echo "학습 내용:"
echo "  - VLM: Frozen (Pre-trained Kosmos-2)"
echo "  - LoRA: 비활성화"
echo "  - 학습: Action Head (LSTM) only"
echo "  - 목적: LoRA Fine-tuning 효과 비교"
echo ""

# 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_frozen_vlm_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_frozen_vlm_20251216.json"

echo "🚀 학습 시작..."
echo "  Config: $CONFIG_FILE"
echo " Start time: $(date)"
echo ""

# 학습 실행 (nohup으로 백그라운드 실행)
nohup python3 RoboVLMs_upstream/main.py \
    "$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ 학습 프로세스 시작됨 (PID: $PID)"
echo ""
echo "모니터링 명령어:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi"
echo ""
echo "프로세스 확인:"
echo "  ps aux | grep main.py"
echo ""
echo "예상 소요 시간: ~30분 (4 epochs)"
