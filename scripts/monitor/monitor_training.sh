#!/bin/bash
# 학습 진행 상황 모니터링 스크립트

LOG_FILE="logs/train_english_chunk5_20260107_075340.log"
CKPT_DIR="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2026-01-07/mobile_vla_chunk5_20251217"

echo "=========================================="
echo "영어 Instruction 재학습 모니터링"
echo "=========================================="
echo ""

# 학습 프로세스 확인
echo "=== 학습 프로세스 ==="
PROC_COUNT=$(ps aux | grep "main.py.*chunk5" | grep -v grep | wc -l)
if [ "$PROC_COUNT" -gt 0 ]; then
    echo "✅ 학습 진행 중 (프로세스 수: $PROC_COUNT)"
    ps aux | grep "main.py.*chunk5" | grep -v grep | head -3 | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Time:", $10}'
else
    echo "❌ 학습 프로세스 없음 (완료 또는 중단)"
fi
echo ""

# 최신 로그 확인
echo "=== 최신 로그 (최근 5줄) ==="
tail -5 "$LOG_FILE" 2>/dev/null | grep -E "Epoch|val_loss" || echo "로그 파일 없음"
echo ""

# 체크포인트 확인
echo "=== 체크포인트 현황 ==="
if [ -d "$CKPT_DIR" ]; then
    echo "생성된 체크포인트:"
    ls -lht "$CKPT_DIR"/*.ckpt 2>/dev/null | grep -E "epoch_epoch" | head -5 | awk '{print "  ", $9}' | sed 's/.*epoch_/epoch_/' | sed 's/.ckpt//'
    
    echo ""
    echo "Best Checkpoint (val_loss 기준):"
    ls -lh "$CKPT_DIR"/epoch_*.ckpt 2>/dev/null | \
        grep -v "last" | \
        sed 's/.*epoch_//' | \
        sed 's/.ckpt//' | \
        sort -t= -k3 -n | \
        head -1 | \
        awk '{print "  ✨ epoch_" $0}'
else
    echo "  체크포인트 디렉토리 없음"
fi
echo ""

# GPU 사용률
echo "=== GPU 사용률 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "GPU 정보 없음"
echo ""

echo "=========================================="
echo "모니터링 명령어:"
echo "  실시간 로그: tail -f $LOG_FILE"
echo "  체크포인트: ls -lht $CKPT_DIR/*.ckpt | head -10"
echo "=========================================="
