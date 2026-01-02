#!/bin/bash
# 모든 실험 순차 실행 스크립트 (Resumed)
# 작성일: 2025-12-09
# Status: Case 4 (aug_abs) 완료됨. Case 5, 6 진행.

set -e

# 프로젝트 루트로 이동
cd /home/soda/25-1kp/vla

mkdir -p logs

run_experiment() {
    NAME=$1
    CONFIG=$2
    DESC=$3
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/train_${NAME}_${TIMESTAMP}.log"

    echo ""
    echo "=================================================="
    echo "🚀 실험 시작: $DESC"
    echo "   Config: $CONFIG"
    echo "   Log: $LOG_FILE"
    echo "=================================================="
    
    START_TIME=$(date +%s)
    
    # python 명령어 직접 실행
    python3 RoboVLMs_upstream/main.py "$CONFIG" > "$LOG_FILE" 2>&1
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "✅ 실험 완료 (소요시간: ${DURATION}초)"
    echo ""
}

# 1. Case 4: Augmented -> 완료됨 (Skip)
echo "⏭️  Case 4 (aug_abs)는 이미 완료되어 건너뜁니다."

# 2. Case 5: OpenVLA Style
run_experiment "openvla" \
    "Mobile_VLA/configs/mobile_vla_openvla_style_20251209.json" \
    "Case 5: OpenVLA Style (Low LR, 27 Epochs)"

# 3. Case 6: No Chunking
run_experiment "no_chunk" \
    "Mobile_VLA/configs/mobile_vla_no_chunk_20251209.json" \
    "Case 6: No Action Chunking"

echo "🎉 남은 실험 시퀀스 완료!"
