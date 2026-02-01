#!/bin/bash
# 4개 실험 순차 자동 실행 스크립트
# 순서: Left Chunk10 -> Right Chunk10 -> Left Chunk5 -> Right Chunk5

set -e

cd /home/billy/25-1kp/vla

MAIN_LOG="logs/auto_train_sequence_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 4개 실험 순차 자동 실행 시작" | tee -a "$MAIN_LOG"
echo "순서: Left10 -> Right10 -> Left5 -> Right5" | tee -a "$MAIN_LOG"
echo "시작 시간: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 함수: 학습 완료 대기
wait_for_training() {
    local log_file=$1
    local exp_name=$2
    
    echo "⏳ $exp_name 학습 완료 대기 중..." | tee -a "$MAIN_LOG"
    
    # 로그 파일이 생성될 때까지 대기
    while [ ! -f "$log_file" ]; do
        sleep 5
    done
    
    # "max_epochs" reached 또는 완료 메시지가 나올 때까지 대기
    while true; do
        if grep -q "max_epochs.*reached\|Trainer.fit.*stopped\|Training completed" "$log_file" 2>/dev/null; then
            echo "✅ $exp_name 학습 완료!" | tee -a "$MAIN_LOG"
            echo "   완료 시간: $(date)" | tee -a "$MAIN_LOG"
            break
        fi
        sleep 10
    done
    
    # 추가 대기 (cleanup 시간)
    sleep 5
}

# ============================================
# 1. Left Chunk10 (현재 실행 중이므로 대기만)
# ============================================
echo "1️⃣  Left Chunk10 (이미 실행 중)" | tee -a "$MAIN_LOG"
CURRENT_LOG=$(ls -t logs/train_left_chunk10_*.log 2>/dev/null | head -1)

if [ -n "$CURRENT_LOG" ]; then
    echo "   로그: $CURRENT_LOG" | tee -a "$MAIN_LOG"
    wait_for_training "$CURRENT_LOG" "Left Chunk10"
else
    echo "   로그 파일을 찾을 수 없습니다. 새로 시작합니다." | tee -a "$MAIN_LOG"
    bash scripts/train_active/train_left_chunk10.sh &
    sleep 10
    CURRENT_LOG=$(ls -t logs/train_left_chunk10_*.log 2>/dev/null | head -1)
    wait_for_training "$CURRENT_LOG" "Left Chunk10"
fi

echo "" | tee -a "$MAIN_LOG"

# ============================================
# 2. Right Chunk10
# ============================================
echo "2️⃣  Right Chunk10 시작" | tee -a "$MAIN_LOG"
echo "   시작 시간: $(date)" | tee -a "$MAIN_LOG"

bash scripts/train_active/train_right_chunk10.sh &
sleep 10

RIGHT10_LOG=$(ls -t logs/train_right_chunk10_*.log 2>/dev/null | head -1)
echo "   로그: $RIGHT10_LOG" | tee -a "$MAIN_LOG"
wait_for_training "$RIGHT10_LOG" "Right Chunk10"

echo "" | tee -a "$MAIN_LOG"

# ============================================
# 3. Left Chunk5
# ============================================
echo "3️⃣  Left Chunk5 시작" | tee -a "$MAIN_LOG"
echo "   시작 시간: $(date)" | tee -a "$MAIN_LOG"

bash scripts/train_active/train_left_chunk5.sh &
sleep 10

LEFT5_LOG=$(ls -t logs/train_left_chunk5_*.log 2>/dev/null | head -1)
echo "   로그: $LEFT5_LOG" | tee -a "$MAIN_LOG"
wait_for_training "$LEFT5_LOG" "Left Chunk5"

echo "" | tee -a "$MAIN_LOG"

# ============================================
# 4. Right Chunk5
# ============================================
echo "4️⃣  Right Chunk5 시작 (마지막)" | tee -a "$MAIN_LOG"
echo "   시작 시간: $(date)" | tee -a "$MAIN_LOG"

bash scripts/train_active/train_right_chunk5.sh &
sleep 10

RIGHT5_LOG=$(ls -t logs/train_right_chunk5_*.log 2>/dev/null | head -1)
echo "   로그: $RIGHT5_LOG" | tee -a "$MAIN_LOG"
wait_for_training "$RIGHT5_LOG" "Right Chunk5"

echo "" | tee -a "$MAIN_LOG"

# ============================================
# 완료
# ============================================
echo "=" | tee -a "$MAIN_LOG"
echo "=" | tee -a "$MAIN_LOG"
echo "🎉 모든 학습 완료!" | tee -a "$MAIN_LOG"
echo "=" | tee -a "$MAIN_LOG"
echo "=" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "완료 시간: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "로그 파일:" | tee -a "$MAIN_LOG"
echo "  1. Left Chunk10:  $CURRENT_LOG" | tee -a "$MAIN_LOG"
echo "  2. Right Chunk10: $RIGHT10_LOG" | tee -a "$MAIN_LOG"
echo "  3. Left Chunk5:   $LEFT5_LOG" | tee -a "$MAIN_LOG"
echo "  4. Right Chunk5:  $RIGHT5_LOG" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "체크포인트 디렉토리:" | tee -a "$MAIN_LOG"
echo "  runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Best model 찾기
echo "Best Models:" | tee -a "$MAIN_LOG"
for exp in mobile_vla_left_chunk10_20251218 mobile_vla_right_chunk10_20251218 mobile_vla_left_chunk5_20251218 mobile_vla_right_chunk5_20251218; do
    best_ckpt=$(find "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/$exp" -name "epoch*.ckpt" 2>/dev/null | grep -v "last.ckpt" | sort | head -1)
    if [ -n "$best_ckpt" ]; then
        echo "  $exp:" | tee -a "$MAIN_LOG"
        echo "    $best_ckpt" | tee -a "$MAIN_LOG"
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "다음 단계: 결과 분석 및 시각화" | tee -a "$MAIN_LOG"
echo "  python3 scripts/visualize_training_curves_all.py" | tee -a "$MAIN_LOG"
