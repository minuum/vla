#!/bin/bash
# 방향 제거 버전 학습 스크립트
# 작성일: 2025-12-09
# 목적: linear_y의 절대값만 학습하여 크기만 예측하도록 함
#       방향은 언어 명령에서 추출하여 결정

set -e

# 프로젝트 루트로 이동
cd /home/soda/25-1kp/vla


echo "=================================================="
echo "🚀 방향 제거 버전 학습 (abs_action)"
echo "=================================================="
echo ""
echo "학습 내용:"
echo "  - linear_y: 크기만 학습 (절대값)"
echo "  - 방향: 추론 시 언어에서 'left'/'right' 추출"
echo ""

# 환경 설정
cd /home/soda/25-1kp/vla
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_abs_action_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_kosmos2_abs_action_20251209.json"

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
