#!/bin/bash
# Basket Navigation 학습 스크립트 (Chunk=5)
# 작성일: 2026-01-29

set -e

cd /home/billy/25-1kp/vla

echo "=================================================="
echo "🚀 Basket Navigation 학습 (Chunk=5)"
echo "=================================================="
echo ""
echo "학습 내용:"
echo "  - VLM: Frozen (Pre-trained Kosmos-2)"
echo "  - Action Head: Trainable"
echo "  - Dataset: ROS_action/basket_dataset"
echo "  - Instruction Source: dataset_index.csv"
echo "  - Action Chunking: 5 steps"
echo ""

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_basket_chunk5_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_chunk5_basket.json"

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
echo "프로세스 관리:"
echo "  ps aux | grep main.py"
echo "  kill $PID"  # 종료 명령어 안내
echo ""
echo "예상 소요 시간: ~30-60분 (10 epochs)"
