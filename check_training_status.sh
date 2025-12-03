#!/bin/bash
# 학습 진행 상태 확인 스크립트

echo "========================================="
echo "🔍 학습 진행 상태 확인"
echo "========================================="
echo ""

# 1. 프로세스 확인
echo "1️⃣  Python 학습 프로세스 확인:"
echo "----------------------------------------"
PROCESSES=$(ps aux | grep -E "python.*main.py|python.*train" | grep -v grep)
if [ -z "$PROCESSES" ]; then
    echo "   ❌ 실행 중인 학습 프로세스가 없습니다."
else
    echo "   ✅ 학습 프로세스 실행 중:"
    echo "$PROCESSES" | while read line; do
        echo "   $line"
    done
fi
echo ""

# 2. GPU 사용률 확인
echo "2️⃣  GPU 사용률 확인:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s (%s): %s%% 사용, 메모리: %s/%s MB\n", $1, $2, $3, $4, $5}'
else
    echo "   ⚠️  nvidia-smi를 사용할 수 없습니다."
    echo "   (Jetson 환경에서는 torch.cuda.is_available()로 확인)"
fi
echo ""

# 3. 체크포인트 디렉토리 확인
echo "3️⃣  체크포인트 디렉토리 확인:"
echo "----------------------------------------"
CHECKPOINT_DIRS=(
    "RoboVLMs_upstream/runs"
    "runs/mobile_vla_lora_20251114"
)

for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   📁 $dir:"
        # 날짜별 디렉토리 찾기
        LATEST_DIR=$(find "$dir" -type d -name "*mobile_vla_lora_20251114*" -o -type d -name "20*" | head -1)
        if [ -n "$LATEST_DIR" ]; then
            echo "      최신 디렉토리: $LATEST_DIR"
            
            # 체크포인트 파일 확인
            CHECKPOINTS=$(find "$LATEST_DIR" -name "*.ckpt" -o -name "checkpoint-*" 2>/dev/null | head -5)
            if [ -n "$CHECKPOINTS" ]; then
                echo "      ✅ 체크포인트 파일 발견:"
                echo "$CHECKPOINTS" | while read ckpt; do
                    SIZE=$(du -h "$ckpt" 2>/dev/null | cut -f1)
                    MTIME=$(stat -c %y "$ckpt" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
                    echo "         - $(basename $ckpt) ($SIZE, 수정: $MTIME)"
                done
            else
                echo "      ⚠️  체크포인트 파일이 아직 없습니다."
            fi
        fi
    fi
done
echo ""

# 4. 로그 파일 확인
echo "4️⃣  로그 파일 확인:"
echo "----------------------------------------"
LOG_PATHS=(
    "RoboVLMs_upstream/runs"
    "runs/mobile_vla_lora_20251114"
    "."
)

FOUND_LOGS=0
for base_path in "${LOG_PATHS[@]}"; do
    if [ -d "$base_path" ]; then
        # find를 사용하여 로그 파일 찾기
        while IFS= read -r log; do
            if [ -f "$log" ]; then
                FOUND_LOGS=1
                SIZE=$(du -h "$log" 2>/dev/null | cut -f1)
                MTIME=$(stat -c %y "$log" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
                echo "   📄 $log ($SIZE, 수정: $MTIME)"
                
                # 최근 로그 내용 확인
                echo "      최근 로그 (마지막 3줄):"
                tail -3 "$log" 2>/dev/null | sed 's/^/         /'
            fi
        done < <(find "$base_path" -name "*.log" -type f 2>/dev/null | head -10)
    fi
done

if [ $FOUND_LOGS -eq 0 ]; then
    echo "   ⚠️  로그 파일을 찾을 수 없습니다."
fi
echo ""

# 5. 최근 학습 진행 상황 (로그에서)
echo "5️⃣  최근 학습 진행 상황:"
echo "----------------------------------------"
for base_path in "${LOG_PATHS[@]}"; do
    if [ -d "$base_path" ]; then
        while IFS= read -r log; do
            if [ -f "$log" ]; then
                EPOCH_INFO=$(tail -100 "$log" 2>/dev/null | grep -E "Epoch [0-9]+:" | tail -1)
                if [ -n "$EPOCH_INFO" ]; then
                    echo "   📊 $log:"
                    echo "      $EPOCH_INFO" | sed 's/^/         /'
                fi
            fi
        done < <(find "$base_path" -name "*.log" -type f 2>/dev/null | head -10)
    fi
done
echo ""

echo "========================================="
echo "💡 추가 명령어:"
echo "========================================="
echo ""
echo "   실시간 로그 확인:"
echo "   tail -f <로그파일경로>"
echo ""
echo "   모니터링 스크립트 실행:"
echo "   python monitor_training.py <로그파일경로>"
echo ""
echo "   프로세스 상세 확인:"
echo "   ps aux | grep python | grep main.py"
echo ""

