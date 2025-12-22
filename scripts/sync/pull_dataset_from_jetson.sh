#!/bin/bash
# Jetson에서 Billy 서버로 데이터셋 전송
# 사용법: bash scripts/sync/pull_dataset_from_jetson.sh [pattern]

set -e

# 설정
JETSON_HOST="${JETSON_HOST:-soda@100.99.189.94}"
JETSON_DATASET_PATH="${JETSON_DATASET_PATH:-~/vla/ROS_action/mobile_vla_dataset}"
LOCAL_PATH="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Jetson → Billy: 데이터셋 전송"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Jetson의 파일 목록 확인
echo "Jetson의 데이터셋 확인 중..."
ssh "$JETSON_HOST" "ls -lh ${JETSON_DATASET_PATH}/*.h5 2>/dev/null | wc -l" || echo "0"

echo ""
echo "최근 파일 (10개):"
ssh "$JETSON_HOST" "ls -lth ${JETSON_DATASET_PATH}/*.h5 2>/dev/null | head -10" || echo "파일 없음"

echo ""
echo "전송 옵션:"
echo "  1) 전체 전송 (모든 .h5 파일)"
echo "  2) 특정 날짜만 (예: episode_20251216*.h5)"
echo "  3) 특정 수만큼 (최근 N개)"
echo ""

read -p "선택 (1-3): " OPTION

case $OPTION in
    1)
        PATTERN="*.h5"
        ;;
    2)
        read -p "날짜 패턴 (예: 20251216): " DATE
        PATTERN="episode_${DATE}*.h5"
        ;;
    3)
        read -p "개수: " COUNT
        PATTERN="recent_${COUNT}"
        ;;
    *)
        echo "잘못된 선택"
        exit 1
        ;;
esac

# 2. 전송
mkdir -p "$LOCAL_PATH"

echo ""
echo "📡 전송 시작..."

if [ "$OPTION" = "3" ]; then
    # 최근 N개만
    ssh "$JETSON_HOST" "ls -t ${JETSON_DATASET_PATH}/*.h5 | head -${COUNT}" | while read file; do
        rsync -avz --progress \
            "${JETSON_HOST}:$file" \
            "${LOCAL_PATH}/"
    done
else
    # 패턴으로 전송
    rsync -avz --progress \
        "${JETSON_HOST}:${JETSON_DATASET_PATH}/${PATTERN}" \
        "${LOCAL_PATH}/"
fi

echo ""
echo "✅ 전송 완료!"
echo ""
echo "전송된 파일 확인:"
ls -lh "${LOCAL_PATH}"/*.h5 | tail -10
