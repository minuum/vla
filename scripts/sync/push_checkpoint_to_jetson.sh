#!/bin/bash
# Billy 서버에서 Jetson으로 체크포인트 전송
# 사용법: bash scripts/sync/push_checkpoint_to_jetson.sh [checkpoint_name]

set -e

# 설정
JETSON_HOST="${JETSON_HOST:-soda@100.99.189.94}"  # Tailscale IP
JETSON_PATH="${JETSON_PATH:-~/vla}"
LOCAL_PATH="/home/soda/25-1kp/vla"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 Billy → Jetson: Checkpoint 전송"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. 체크포인트 선택
if [ -n "$1" ]; then
    CHECKPOINT="$1"
else
    echo "사용 가능한 체크포인트:"
    find runs -name "*.ckpt" -type f 2>/dev/null | head -10
    echo ""
    read -p "체크포인트 경로 입력: " CHECKPOINT
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 파일을 찾을 수 없습니다: $CHECKPOINT"
    exit 1
fi

SIZE=$(du -h "$CHECKPOINT" | cut -f1)
echo "📦 전송할 파일: $CHECKPOINT ($SIZE)"
echo "🎯 목적지: ${JETSON_HOST}:${JETSON_PATH}/ROS_action/last.ckpt"
echo ""

# 2. 확인
read -p "전송하시겠습니까? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "취소됨"
    exit 0
fi

# 3. rsync로 전송 (재개 가능, 압축)
echo ""
echo "📡 전송 시작..."
rsync -avz --progress \
    "$CHECKPOINT" \
    "${JETSON_HOST}:${JETSON_PATH}/ROS_action/last.ckpt"

echo ""
echo "✅ 전송 완료!"
echo ""
echo "Jetson에서 확인:"
echo "  ssh ${JETSON_HOST}"
echo "  ls -lh ~/vla/ROS_action/last.ckpt"
