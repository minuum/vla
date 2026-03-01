#!/bin/bash
# 학습 완료 모니터링 및 자동 추론 테스트 스크립트

echo "🔍 학습 완료 대기 중..."
echo "현재 시각: $(date +%H:%M:%S)"
echo ""

# 프로세스 확인
PID=1546813
if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ 학습 프로세스가 이미 종료되었습니다!"
else
    echo "⏳ 학습 진행 중... (PID: $PID)"
    
    # 현재 상태 표시
    CURRENT_EPOCH=$(tail -20 logs/train_no_chunk_20251209_160112.log | grep "Epoch" | tail -1)
    echo "$CURRENT_EPOCH"
    echo ""
    
    # 완료 대기
    echo "학습 완료까지 대기합니다..."
    echo "(Ctrl+C로 중단 가능)"
    
    while ps -p $PID > /dev/null 2>&1; do
        sleep 30
        # 30초마다 진행 상황 출력
        PROGRESS=$(tail -5 logs/train_no_chunk_20251209_160112.log | grep "Epoch" | tail -1)
        if [ ! -z "$PROGRESS" ]; then
            echo "[$(date +%H:%M:%S)] $PROGRESS"
        fi
    done
    
    echo ""
    echo "✅ 학습 완료!"
fi

echo "완료 시각: $(date +%H:%M:%S)"
echo ""

# 체크포인트 확인
echo "📦 체크포인트 확인 중..."
CKPT_DIR="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-09/mobile_vla_no_chunk_20251209"
LAST_CKPT=$(find $CKPT_DIR -name "last.ckpt" -type f 2>/dev/null | head -1)

if [ -z "$LAST_CKPT" ]; then
    echo "❌ last.ckpt를 찾을 수 없습니다."
    echo "사용 가능한 체크포인트:"
    find $CKPT_DIR -name "*.ckpt" -type f 2>/dev/null
    exit 1
fi

echo "✅ 체크포인트 발견: $LAST_CKPT"
echo ""

# 추론 테스트 실행
echo "🚀 추론 테스트 시작..."
export PATH="/home/billy/.local/bin:$PATH"
export POETRY_PYTHON=/home/billy/.cache/pypoetry/virtualenvs/robovlms-ASlHafON-py3.10/bin/python
export VLA_CHECKPOINT_PATH="$(pwd)/$LAST_CKPT"

echo "체크포인트: $VLA_CHECKPOINT_PATH"
echo ""

cd /home/billy/25-1kp/vla
$POETRY_PYTHON test_inference_stepbystep.py

echo ""
echo "✅ 완료!"
