#!/bin/bash
# Case 9 학습 시작 스크립트
# No Chunk + Aug + Abs Action

echo "=========================================="
echo "Case 9: No Chunk + Aug + Abs 학습 시작"
echo "=========================================="

# Config 확인
CONFIG="Mobile_VLA/configs/mobile_vla_no_chunk_aug_abs_20251210.json"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config 파일이 없습니다: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
echo "예상 학습 시간: 5-6시간 (4-5 epochs)"
echo "예상 Val Loss: ~0.0008"
echo ""

# 학습 시작
LOG_FILE="logs/train_no_chunk_aug_abs_$(date +%Y%m%d_%H%M%S).log"
echo "로그 파일: $LOG_FILE"
echo ""

nohup python3 RoboVLMs_upstream/main.py \
    $CONFIG \
    > $LOG_FILE 2>&1 &

PID=$!
echo "학습 시작됨 (PID: $PID)"
echo ""
echo "모니터링:"
echo "  tail -f $LOG_FILE"
echo ""
echo "중단:"
echo "  kill $PID"
