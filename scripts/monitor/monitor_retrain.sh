#!/bin/bash
# 재학습 모니터링 스크립트
# 작성일: 2025-12-09

echo "=================================================="
echo "📊 재학습 모니터링 (action_token Fix)"
echo "=================================================="

LOG_FILE=$(ls -t logs/train_case3_fixed_*.log | head -1)
echo "로그 파일: $LOG_FILE"
echo ""

# 프로세스 확인
echo "=== 프로세스 상태 ==="
ps aux | grep "[m]ain.py" | head -3 || echo "❌ 프로세스가 실행 중이 아닙니다"
echo ""

# GPU 상태
echo "=== GPU 사용량 ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# 최근 학습 상태
echo "=== 최근 학습 로그 ==="
tail -5 "$LOG_FILE" 2>/dev/null | grep -E "Epoch|train_loss|val_loss" || tail -5 "$LOG_FILE"
echo ""

# Epoch 진행 상황
echo "=== Epoch 진행 상황 ==="
grep "Epoch" "$LOG_FILE" 2>/dev/null | tail -3

# 체크포인트 확인
echo ""
echo "=== 체크포인트 ==="
ls -lh runs/mobile_vla_kosmos2_fixed_20251209/*/mobile_vla_finetune/*/*.ckpt 2>/dev/null | tail -3 || echo "아직 체크포인트 없음"
