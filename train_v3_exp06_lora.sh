#!/bin/bash
# EXP-V3-06: LEFT 에피소드만 사용 + History Dropout + LoRA rank 32
# 목적: 스텝 시퀀스 암기 방지, 이미지-언어 바인딩 강화

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="Mobile_VLA/configs/mobile_vla_v3_exp06_lora.json"
LOG_FILE="logs/train_v3_exp06_lora.log"
PID_FILE="train_v3_exp06.pid"

mkdir -p logs

echo "=============================="
echo " V3-EXP06 LoRA 학습 시작"
echo " Config: $CONFIG"
echo " Log:    $LOG_FILE"
echo " 주요 설정:"
echo "   - 데이터: LEFT 에피소드 278개만 (filter_keyword='left')"
echo "   - History Dropout: 0.3 (과거 프레임 30% 랜덤 마스킹)"
echo "   - LoRA rank: 32 (exp05 대비 2배, 언어 바인딩 강화)"
echo "   - FR class_weight: 3.5 (페널티 강화)"
echo "   - L  class_weight: 15.0 (강력 강화)"
echo "   - F  class_weight: 0.08 (최대 억제)"
echo "   - max_epochs: 15"
echo "=============================="

# exp05 학습이 아직 돌고 있으면 확인
if [ -f "train_v3_exp05.pid" ]; then
    EXP05_PID=$(cat train_v3_exp05.pid)
    if kill -0 $EXP05_PID 2>/dev/null; then
        echo "[WARNING] EXP05 학습이 아직 실행 중입니다 (PID: $EXP05_PID)"
        echo "[WARNING] GPU 메모리 부족 가능성이 있습니다."
        echo "[WARNING] 계속하려면 Enter, 중단하려면 Ctrl+C"
        read -r
    fi
fi

cd "$SCRIPT_DIR"

cd RoboVLMs_upstream
nohup python3 -u main.py \
    "../Mobile_VLA/configs/mobile_vla_v3_exp06_lora.json" \
    2>&1 | tee "../logs/train_v3_exp06_lora.log" &
cd ..

echo $! > "$PID_FILE"
echo "[EXP06] 학습 시작됨. PID: $(cat $PID_FILE)"
echo "[EXP06] 로그 확인: tail -f $LOG_FILE"
echo "[EXP06] 학습 모니터링: watch -n 30 'tail -n 5 $LOG_FILE'"
