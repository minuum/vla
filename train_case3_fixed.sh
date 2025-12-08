#!/bin/bash
# 재학습 스크립트: action_token Xavier 초기화 수정 후 재학습
# 작성일: 2025-12-09
# 목적: action_token zeros → Xavier 초기화로 수정 후 재학습
#       언어 조건부 학습이 제대로 되도록 개선

set -e

echo "=================================================="
echo "🚀 Case 3 재학습 (action_token 초기화 수정됨)"
echo "=================================================="
echo ""
echo "수정 내용:"
echo "  - base_backbone.py: action_token zeros → Xavier 초기화"
echo "  - 목적: VLM에서 언어 정보가 action_token에 전달되도록 개선"
echo ""

# 환경 설정
cd /home/billy/25-1kp/vla
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_case3_fixed_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# config 파일 경로
CONFIG_FILE="Mobile_VLA/configs/mobile_vla_kosmos2_fixed_20251209.json"

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
