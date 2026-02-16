#!/bin/bash
# Model_LEFT 학습 스크립트
# 작성일: 2026-01-11
# 목적: Instruction-Specific Model - LEFT navigation only

set -e

cd /home/billy/25-1kp/vla

echo "=================================================="
echo "🚀 Model_LEFT 학습 (LEFT Navigation Only)"
echo "=================================================="
echo ""
echo "학습 내용:"
echo "  - VLM: Frozen (Pre-trained Kosmos-2)"
echo "  - Data: LEFT episodes only (~250 episodes)"
echo "  - Task: Single-task LEFT navigation"
echo "  - Strategy: Instruction-specific model"
echo ""

# 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_left_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_left_only.json"

# 데이터 확인
echo "📊 데이터 확인..."
LEFT_COUNT=$(ls -1 ROS_action/mobile_vla_dataset/episode_202512*left*.h5 2>/dev/null | wc -l)
echo "  LEFT episodes: $LEFT_COUNT"
echo ""

if [ "$LEFT_COUNT" -lt 100 ]; then
    echo "❌ 에러: LEFT episodes가 부족합니다 (최소 100개 필요)"
    exit 1
fi

echo "🚀 학습 시작..."
echo "  Config: $CONFIG_FILE"
echo "  Episodes: $LEFT_COUNT"
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
echo "  watch -n 1 nvidia-smi"
echo ""
echo "프로세스 확인:"
echo "  ps aux | grep main.py"
echo ""
echo "예상 소요 시간: ~2-3시간 (20 epochs)"
echo ""
echo "🎯 목표: LEFT navigation policy 학습"
echo "  - Train loss < 0.10"
echo "  - Val loss < 0.15"
echo ""
