#!/bin/bash
# Case 3 학습 모니터링 스크립트

EXPERIMENT="mobile_vla_kosmos2_frozen_lora_leftright_20251204"
LOG_FILE=$(ls -t case3_kosmos2_leftright_*.txt 2>/dev/null | head -1)
RUN_DIR="RoboVLMs_upstream/runs/${EXPERIMENT}"

echo "=========================================="
echo "📊 Case 3 학습 모니터링"
echo "=========================================="
echo ""
echo "실험: ${EXPERIMENT}"
echo "로그: ${LOG_FILE:-없음}"
echo ""

# 1. 프로세스 확인
echo "[1] 프로세스 상태"
echo "------------------------------------------"
if ps aux | grep "python.*main.py.*${EXPERIMENT}" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "python.*main.py.*${EXPERIMENT}" | grep -v grep | awk '{print $2}' | head -1)
    echo "  ✅ 실행 중 (PID: $PID)"
else
    echo "  ❌ 실행 안 됨"
fi
echo ""

# 2. 로그 확인
if [ -f "$LOG_FILE" ]; then
    echo "[2] 최근 로그 (20 lines)"
    echo "------------------------------------------"
    tail -20 "$LOG_FILE" | grep -E "Epoch|Loss|Error|training|validation" || tail -20 "$LOG_FILE"
    echo ""
else
    echo "[2] 로그 파일 없음"
    echo ""
fi

# 3. Checkpoint 확인
echo "[3] Checkpoint 현황"
echo "------------------------------------------"
if [ -d "$RUN_DIR" ]; then
    CKPT_COUNT=$(find "$RUN_DIR" -name "*.ckpt" 2>/dev/null | wc -l)
    echo "  저장된 checkpoint: ${CKPT_COUNT}개"
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "  최근 checkpoint:"
        find "$RUN_DIR" -name "*.ckpt" 2>/dev/null | xargs ls -lth | head -3 | awk '{print "    " $9 " (" $5 ")"}'
    fi
else
    echo "  Run 디렉토리 없음"
fi
echo ""

# 4. Tensorboard 로그
echo "[4] Tensorboard 이벤트"
echo "------------------------------------------"
TB_EVENTS=$(find "$RUN_DIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
echo "  이벤트 파일: ${TB_EVENTS}개"
if [ $TB_EVENTS -gt 0 ]; then
    echo "  최근 이벤트:"
    find "$RUN_DIR" -name "events.out.tfevents.*" 2>/dev/null | xargs ls -lth | head -2 | awk '{print "    " $9}'
fi
echo ""

# 5. 실시간 로그 옵션
echo "=========================================="
echo "실시간 모니터링:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Tensorboard (선택):"
echo "  tensorboard --logdir ${RUN_DIR}"
echo "=========================================="
