#!/bin/bash
# EXP-V3-07: LEFT+RIGHT 양방향 데이터 통합 + L/R 균형 class_weight + lr 1e-5
# 목적: 실물 로봇 양방향 일반화 - exp06 실물 주행 결과 기반 개선
#
# 변경점 vs exp06:
#   - 데이터: LEFT(278) + RIGHT(250) = 전체 528 에피소드 (filter_keyword=null)
#   - class_weight[L]=8.0, class_weight[R]=8.0 (L/R 대칭 균형)
#   - class_weight[FR]=3.0 (exp06의 3.5에서 완화)
#   - learning_rate: 3e-5 → 1e-5 (더 안정적인 수렴)
#   - history_dropout_prob: 0.3 → 0.2 (소폭 완화)
#   - max_epochs: 15 → 20 (더 많은 데이터에 맞춰 증가)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="Mobile_VLA/configs/mobile_vla_v3_exp07_lora.json"
LOG_FILE="logs/train_v3_exp07_lora.log"
PID_FILE="train_v3_exp07.pid"

mkdir -p logs

echo "======================================"
echo " V3-EXP07 LoRA 학습 시작"
echo " Config: $CONFIG"
echo " Log:    $LOG_FILE"
echo " 주요 설정:"
echo "   - 데이터: LEFT+RIGHT 전체 528 에피소드 (양방향 통합)"
echo "   - class_weight[L]=8.0, class_weight[R]=8.0 (대칭 균형)"
echo "   - History Dropout: 0.2 (exp06 0.3에서 완화)"
echo "   - LoRA rank: 32, alpha: 64"
echo "   - learning_rate: 1e-5 (안정 수렴)"
echo "   - max_epochs: 20"
echo "   - 실물 주행 결과 기반: Snake 궤적 능력 유지 + 양방향 일반화"
echo "======================================"

# 이전 exp 학습이 돌고 있으면 확인
for EXP in 05 06; do
    if [ -f "train_v3_exp${EXP}.pid" ]; then
        PREV_PID=$(cat "train_v3_exp${EXP}.pid")
        if kill -0 $PREV_PID 2>/dev/null; then
            echo "[WARNING] EXP${EXP} 학습이 아직 실행 중입니다 (PID: $PREV_PID)"
            echo "[WARNING] GPU 메모리 부족 가능성이 있습니다."
            echo "[WARNING] 계속하려면 Enter, 중단하려면 Ctrl+C"
            read -r
        fi
    fi
done

cd "$SCRIPT_DIR"

cd RoboVLMs_upstream
nohup python3 -u main.py \
    "../Mobile_VLA/configs/mobile_vla_v3_exp07_lora.json" \
    2>&1 | tee "../logs/train_v3_exp07_lora.log" &
TRAIN_PID=$!
cd ..

echo $TRAIN_PID > "$PID_FILE"
echo "[EXP07] 학습 시작됨. PID: $TRAIN_PID"
echo "[EXP07] 로그 확인: tail -f $LOG_FILE"
echo "[EXP07] 학습 모니터링: watch -n 30 'tail -n 5 $LOG_FILE'"
