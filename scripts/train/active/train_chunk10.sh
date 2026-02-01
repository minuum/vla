#!/bin/bash
# Action Chunking (chunk=10) 학습 스크립트
# 작성일: 2025-12-17
# 목적: fwd_pred_next_n=10 action chunking 효과 검증

set -e

cd /home/billy/25-1kp/vla

echo "=================================================="
echo "🚀 Action Chunking 학습 (chunk=10)"
echo "=================================================="
echo ""
echo "학습 내용:"
echo "  - VLM: Frozen (Pre-trained Kosmos-2)"
echo "  - LoRA: 비활성화"
echo "  - Action Chunking: 10 steps"
echo "  - 목적: Action chunking 효과 검증"
echo ""

# 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_chunk10_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_chunk10_20251217.json"

echo "🚀 학습 시작..."
echo "  Config: $CONFIG_FILE"
echo "  Start time: $(date)"
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
echo "예상 소요 시간: ~30-60분 (10 epochs)"
