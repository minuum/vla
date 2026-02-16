#!/bin/bash
# Case 8 학습 시작 스크립트
# No Chunk + Abs Action

echo "=========================================="
echo "Case 8: No Chunk + Abs Action 학습 시작"
echo "=========================================="

# Config 확인
CONFIG="Mobile_VLA/configs/mobile_vla_no_chunk_abs_20251210.json"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config 파일이 없습니다: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
echo "예상 학습 시간: 4-5시간 (4 epochs)"
echo "예상 Val Loss: ~0.001"
echo ""

# 학습 시작
LOG_FILE="logs/train_no_chunk_abs_$(date +%Y%m%d_%H%M%S).log"
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
