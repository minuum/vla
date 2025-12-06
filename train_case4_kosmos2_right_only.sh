#!/bin/bash
# RIGHT ONLY 학습 스크립트
# 박스를 오른쪽으로 피해가는 에피소드만 학습

echo "=== Mobile VLA Right Only Training ==="
echo "Date: $(date)"
echo "Config: mobile_vla_kosmos2_right_only_20251207.json"
echo ""

cd /home/billy/25-1kp/vla/RoboVLMs_upstream

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:/home/billy/25-1kp/vla/RoboVLMs_upstream"
export CUDA_VISIBLE_DEVICES=0

# Right 파일 개수 확인
RIGHT_FILES=$(ls /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/*right*.h5 2>/dev/null | wc -l)
echo "Right episode files: $RIGHT_FILES"
echo ""

# 학습 실행
python robovlms/train/train.py \
    --cfg ../Mobile_VLA/configs/mobile_vla_kosmos2_right_only_20251207.json \
    2>&1 | tee ../logs/train_right_only_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=== Training completed ==="
