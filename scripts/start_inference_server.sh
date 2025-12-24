#!/bin/bash
# Mobile VLA 추론 서버 시작 스크립트
# Billy 서버에서 실행 (A5000 GPU)

set -e

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 프로젝트 디렉토리
PROJECT_DIR="/home/soda/vla"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}  Mobile VLA 추론 서버 시작${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"

# 체크포인트 경로 확인
LATEST_CKPT="$PROJECT_DIR/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
LEGACY_CKPT="$PROJECT_DIR/ROS_action/last.ckpt"

# 사용할 체크포인트 선택 (최신 우선)
if [ -f "$LATEST_CKPT" ]; then
    CHECKPOINT_PATH="$LATEST_CKPT"
    echo -e "${GREEN}✓ 최신 체크포인트 발견${NC}"
    echo -e "  경로: ${YELLOW}$CHECKPOINT_PATH${NC}"
elif [ -f "$LEGACY_CKPT" ]; then
    CHECKPOINT_PATH="$LEGACY_CKPT"
    echo -e "${YELLOW}! 레거시 체크포인트 사용${NC}"
    echo -e "  경로: ${YELLOW}$CHECKPOINT_PATH${NC}"
else
    echo -e "${RED}✗ 체크포인트 파일을 찾을 수 없습니다${NC}"
    echo -e "  확인 경로 1: $LATEST_CKPT"
    echo -e "  확인 경로 2: $LEGACY_CKPT"
    exit 1
fi

# GPU 확인
echo ""
echo -e "${BLUE}GPU 상태 확인...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi를 찾을 수 없습니다${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Python 환경 확인
echo -e "${BLUE}Python 환경 확인...${NC}"
python3 --version
echo ""

# API 키 확인/생성
if [ -z "$VLA_API_KEY" ]; then
    echo -e "${YELLOW}! VLA_API_KEY 환경변수가 설정되지 않았습니다${NC}"
    echo -e "${YELLOW}! API 서버가 자동으로 키를 생성합니다${NC}"
else
    echo -e "${GREEN}✓ VLA_API_KEY 설정됨${NC}"
fi
echo ""

# 환경변수 설정
export VLA_CHECKPOINT_PATH="$CHECKPOINT_PATH"

# API 서버 시작
cd "$PROJECT_DIR"

echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  API 서버 시작 중...${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "  💡 Tip: 터미널을 나눠서 로그를 확인하세요:"
echo -e "  ${YELLOW}tail -f $LOG_DIR/api_server.log${NC}"
echo ""

# API 서버 실행
python3 api_server.py

# If we get here, server stopped
echo ""
echo -e "${YELLOW}API 서버가 종료되었습니다${NC}"
