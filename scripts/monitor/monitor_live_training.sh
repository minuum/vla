#!/bin/bash
# 현재 실행 중인 학습 프로세스 실시간 모니터링

# 자동으로 최신 메트릭 파일 찾기
BASE_DIR="RoboVLMs_upstream/runs/mobile_vla_lora_20251114/kosmos/mobile_vla_finetune/2025-11-20/mobile_vla_lora_20251114"
METRICS_FILE=$(find "$BASE_DIR" -name "metrics.csv" -type f -mmin -30 2>/dev/null | head -1)
CHECKPOINT_DIR="$BASE_DIR"

# 메트릭 파일이 없으면 기본 경로 사용
if [ -z "$METRICS_FILE" ]; then
    METRICS_FILE="$BASE_DIR/mobile_vla_lora_20251114/version_55/metrics.csv"
fi

clear
echo "========================================="
echo "🚀 실시간 학습 모니터링"
echo "========================================="
echo ""

while true; do
    # 현재 시간
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    # 1. 프로세스 상태
    PROC_COUNT=$(ps aux | grep -E "python.*main.py.*mobile_vla_20251114_lora.json" | grep -v grep | wc -l)
    if [ "$PROC_COUNT" -gt 0 ]; then
        echo "✅ 학습 프로세스: $PROC_COUNT개 실행 중"
        ps aux | grep -E "python.*main.py.*mobile_vla_20251114_lora.json" | grep -v grep | head -1 | awk '{printf "   PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
    else
        echo "❌ 학습 프로세스가 실행 중이 아닙니다."
    fi
    echo ""
    
    # 2. GPU 사용률
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
        echo "🔥 GPU 상태:"
        echo "$GPU_INFO" | awk -F', ' '{printf "   사용률: %s%%, 메모리: %s/%s MB, 온도: %s°C\n", $1, $2, $3, $4}'
    fi
    echo ""
    
    # 3. 최신 메트릭
    if [ -f "$METRICS_FILE" ]; then
        echo "📊 최신 학습 메트릭:"
        # validation이 포함된 최신 행 찾기
        LATEST_METRIC=$(grep -E "^[0-9]+," "$METRICS_FILE" | tail -1)
        if [ -n "$LATEST_METRIC" ]; then
            echo "$LATEST_METRIC" | awk -F',' '{
                epoch = $1
                lr = $2
                step = $3
                train_loss = $4
                train_loss_arm = $6
                val_loss = $9
                val_loss_arm = $11
                printf "   Step: %s, Epoch: %s, LR: %s\n", step, epoch, lr
                if (train_loss != "") printf "   Train Loss: %s (arm_act: %s)\n", train_loss, train_loss_arm
                if (val_loss != "") printf "   Val Loss: %s (arm_act: %s)\n", val_loss, val_loss_arm
            }'
        else
            echo "   ⚠️  메트릭 데이터가 아직 없습니다."
        fi
    else
        echo "⚠️  메트릭 파일을 찾을 수 없습니다: $METRICS_FILE"
    fi
    echo ""
    
    # 4. 최신 체크포인트
    if [ -d "$CHECKPOINT_DIR" ]; then
        LATEST_CKPT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_NAME=$(basename "$LATEST_CKPT")
            CKPT_SIZE=$(du -h "$LATEST_CKPT" 2>/dev/null | cut -f1)
            CKPT_TIME=$(stat -c %y "$LATEST_CKPT" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
            echo "💾 최신 체크포인트:"
            echo "   $CKPT_NAME ($CKPT_SIZE, 수정: $CKPT_TIME)"
        fi
    fi
    echo ""
    
    # 5. 최근 5개 메트릭 히스토리 (validation 포함된 것만)
    if [ -f "$METRICS_FILE" ]; then
        echo "📈 최근 메트릭 히스토리 (validation 포함, 최근 5개):"
        grep -E "^[0-9]+," "$METRICS_FILE" | tail -5 | awk -F',' '{
            epoch = $1
            step = $3
            train_loss = $4
            train_loss_arm = $6
            val_loss = $9
            val_loss_arm = $11
            if (step != "") {
                printf "   Epoch %s, Step %s: train=%s", epoch, step, train_loss
                if (train_loss_arm != "") printf " (arm=%s)", train_loss_arm
                if (val_loss != "") printf ", val=%s", val_loss
                if (val_loss_arm != "") printf " (arm=%s)", val_loss_arm
                printf "\n"
            }
        }'
    fi
    echo ""
    
    echo "========================================="
    echo "5초 후 업데이트... (Ctrl+C로 종료)"
    echo ""
    
    sleep 5
    clear
    echo "========================================="
    echo "🚀 실시간 학습 모니터링"
    echo "========================================="
    echo ""
done

