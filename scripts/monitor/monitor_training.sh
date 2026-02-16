#!/bin/bash
# Mobile-VLA 학습 모니터링 - 셸 스크립트 버전
# 메뉴 선택 방식

EXPERIMENT="${1:-mobile_vla_kosmos2_frozen_lora_leftright_20251204}"
RUN_DIR="RoboVLMs_upstream/runs/${EXPERIMENT}"
LOG_FILE=$(ls -t case3_kosmos2_leftright_*.txt 2>/dev/null | head -1)

show_menu() {
    clear
    echo "=========================================="
    echo "📊 Mobile-VLA 학습 모니터링 도구"
    echo "=========================================="
    echo "실험: ${EXPERIMENT}"
    echo "로그: ${LOG_FILE:-없음}"
    echo ""
    echo "선택 가능한 옵션:"
    echo "  [1] 📊 모니터링 요약 보기"
    echo "  [2] 📈 Tensorboard 실행"
    echo "  [3] 📜 실시간 로그 (tail -f)"
    echo "  [4] 🔄 새로고침"
    echo "  [0] 종료"
    echo "=========================================="
}

show_summary() {
    clear
    echo "=========================================="
    echo "📊 모니터링 요약"
    echo "=========================================="
    
    # 프로세스 확인
    echo ""
    echo "[1] 프로세스 상태"
    echo "------------------------------------------"
    if ps aux | grep "python.*main.py.*${EXPERIMENT}" | grep -v grep > /dev/null; then
        PID=$(ps aux | grep "python.*main.py.*${EXPERIMENT}" | grep -v grep | awk '{print $2}' | head -1)
        echo "  ✅ 실행 중 (PID: $PID)"
    else
        echo "  ❌ 실행 안 됨"
    fi
    
    # 최근 로그
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "[2] 최근 로그 (15 lines)"
        echo "------------------------------------------"
        tail -15 "$LOG_FILE" | grep -E "Epoch|Loss|Error|training|validation" || tail -15 "$LOG_FILE"
    fi
    
    # Checkpoint
    echo ""
    echo "[3] Checkpoint 현황"
    echo "------------------------------------------"
    if [ -d "$RUN_DIR" ]; then
        CKPT_COUNT=$(find "$RUN_DIR" -name "*.ckpt" 2>/dev/null | wc -l)
        echo "  저장된 checkpoint: ${CKPT_COUNT}개"
        if [ $CKPT_COUNT -gt 0 ]; then
            echo "  최근 checkpoint:"
            find "$RUN_DIR" -name "*.ckpt" 2>/dev/null | xargs ls -lth | head -3 | awk '{print "    - " $9 " (" $5 ")"}'
        fi
    else
        echo "  Run 디렉토리 없음"
    fi
    
    echo ""
    echo "Press Enter to continue..."
    read
}

run_tensorboard() {
    clear
    echo "=========================================="
    echo "📈 Tensorboard 실행"
    echo "=========================================="
    
    if [ ! -d "$RUN_DIR" ]; then
        echo "  ❌ Run 디렉토리 없음: $RUN_DIR"
        echo ""
        echo "Press Enter to continue..."
        read
        return
    fi
    
    echo "  Starting Tensorboard..."
    echo "  URL: http://localhost:6006"
    echo ""
    echo "  종료: Ctrl+C"
    echo ""
    
    tensorboard --logdir "$RUN_DIR"
}

tail_log() {
    if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
        echo ""
        echo "  ❌ 로그 파일 없음"
        echo ""
        echo "Press Enter to continue..."
        read
        return
    fi
    
    clear
    echo "=========================================="
    echo "📜 실시간 로그: $LOG_FILE"
    echo "=========================================="
    echo "  종료: Ctrl+C"
    echo ""
    
    tail -f "$LOG_FILE"
}

# 메인 루프
while true; do
    show_menu
    echo -n "선택 (0-4): "
    read choice
    
    case $choice in
        0)
            echo ""
            echo "종료합니다."
            exit 0
            ;;
        1)
            show_summary
            ;;
        2)
            run_tensorboard
            ;;
        3)
            tail_log
            ;;
        4)
            continue
            ;;
        *)
            echo ""
            echo "잘못된 선택입니다."
            sleep 1
            ;;
    esac
done
