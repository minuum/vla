#!/bin/bash
# Mirroring Augmentation + abs_action 학습 스크립트
# 작성일: 2025-12-09
# 목적: 데이터 증강(Mirroring) 효과 검증
#       (기존 abs_action 전략에 데이터 2배 증강 추가)

set -e

# 프로젝트 루트로 이동
cd /home/soda/25-1kp/vla


echo "=================================================="
echo "🚀 Mirroring Augmentation 학습 (Case 4)"
echo "=================================================="
echo ""
echo "학습 설정:"
echo "  - Config: mobile_vla_kosmos2_aug_abs_20251209.json"
echo "  - 증강: Mirroring (Image Flip, Action Invert, Text Swap)"
echo "  - 전략: abs_action (크기 학습 + 언어 방향 추출)"
echo ""

# 환경 설정
cd /home/soda/25-1kp/vla
export CUDA_VISIBLE_DEVICES=0

# 로그 파일
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_aug_abs_${TIMESTAMP}.log"
mkdir -p logs

echo "📁 로그 파일: $LOG_FILE"
echo ""

# 학습 실행
nohup python3 RoboVLMs_upstream/main.py \
    "Mobile_VLA/configs/mobile_vla_kosmos2_aug_abs_20251209.json" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ 학습 프로세스 시작됨 (PID: $PID)"
echo ""
echo "모니터링 명령어:"
echo "  tail -f $LOG_FILE"
