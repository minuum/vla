#!/bin/bash
# Mobile VLA LoRA Fine-tuning Script for 20251106 Episodes
# 참조: https://github.com/Robot-VLAs/RoboVLMs

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Mobile VLA LoRA Fine-tuning (20251106)${NC}"
echo -e "${GREEN}========================================${NC}"

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/home/billy/25-1kp/vla"
cd $PROJECT_ROOT

# Python 경로 설정
export PYTHONPATH="${PROJECT_ROOT}/Mobile_VLA/src:${PYTHONPATH}"

# CUDA 설정
export CUDA_VISIBLE_DEVICES=0

# Config 파일 경로
CONFIG_PATH="${PROJECT_ROOT}/Mobile_VLA/configs/finetune_mobile_vla_lora_20251106.json"

echo -e "${YELLOW}📄 Config: ${CONFIG_PATH}${NC}"
echo -e "${YELLOW}🔧 Device: CUDA${NC}"
echo ""

# CUDA 사용 가능 여부 확인
echo -e "${YELLOW}🔍 CUDA 확인 중...${NC}"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 데이터셋 확인
echo -e "${YELLOW}📊 데이터셋 확인 중...${NC}"
python3 -c "
import glob
episodes = glob.glob('${PROJECT_ROOT}/ROS_action/mobile_vla_dataset/episode_20251106_*.h5')
print(f'20251106 에피소드: {len(episodes)}개')
for ep in episodes[:5]:
    print(f'  - {ep.split(\"/\")[-1]}')
if len(episodes) > 5:
    print(f'  ... 외 {len(episodes)-5}개')
"
echo ""

# LoRA Fine-tuning 시작
echo -e "${GREEN}🚀 LoRA Fine-tuning 시작...${NC}"
echo ""

# 시작 시간 기록
START_TIME=$(date +%s)

# Fine-tuning 실행
python3 ${PROJECT_ROOT}/Mobile_VLA/src/training/finetune_lora_20251106.py \
    --config ${CONFIG_PATH} \
    --device cuda

# 종료 시간 기록
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))
ELAPSED_SEC=$((ELAPSED_TIME % 60))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ LoRA Fine-tuning 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}⏱️  총 소요 시간: ${ELAPSED_MIN}분 ${ELAPSED_SEC}초${NC}"
echo ""

# 결과 확인
echo -e "${YELLOW}📊 학습 결과 확인:${NC}"
ls -lh ${PROJECT_ROOT}/Mobile_VLA/runs/mobile_vla_lora/checkpoints/
echo ""
echo -e "${YELLOW}📈 로그 확인:${NC}"
ls -lh ${PROJECT_ROOT}/Mobile_VLA/runs/mobile_vla_lora/logs/
echo ""

echo -e "${GREEN}🎉 모든 작업 완료!${NC}"

