#!/bin/bash
# EXP-V3-08: Center-Goal Instruction Experiment
# EXP-07 세팅 완전 유지, instruction만 center-goal 형식으로 변경
# 실행 방식: EXP-07과 동일하게 train.py를 __main__으로 직접 실행

PROJECT_ROOT="/home/billy/25-1kp/vla"
CONFIG="$PROJECT_ROOT/configs/mobile_vla_v3_exp08_center_goal.json"
LOG="$PROJECT_ROOT/logs/train_v3_exp08_center_goal.log"

mkdir -p "$PROJECT_ROOT/logs"

echo "======================================"
echo " EXP-V3-08 center-goal 학습 시작"
echo " Config: $CONFIG"
echo " Log:    $LOG"
echo " 변경사항: instruction = 'Navigate toward basket until centered in frame'"
echo "======================================"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/RoboVLMs"

# EXP-07과 동일한 방식: train.py를 __main__으로 직접 실행
# → train.py 내부에서 chdir(RoboVLMs) 후 'from main import main; main()' 수행
nohup python3 "$PROJECT_ROOT/robovlm_nav/train.py" "$CONFIG" \
    > "$LOG" 2>&1 &

EXP08_PID=$!
echo "[EXP-08] 학습 시작됨. PID: $EXP08_PID"
echo "[EXP-08] 로그 확인: tail -f $LOG"
echo "[EXP-08] 모니터링: watch -n 30 'tail -n 5 $LOG'"
echo $EXP08_PID > "$PROJECT_ROOT/logs/train_v3_exp08.pid"
