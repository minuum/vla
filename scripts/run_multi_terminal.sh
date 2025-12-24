#!/bin/bash
# 터미널을 나눠서 추론 서버와 모니터링을 동시 실행
# tmux를 사용하여 멀티 터미널 환경 구축

set -e

PROJECT_DIR="/home/soda/vla"

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}  Mobile VLA 멀티 터미널 실행${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# tmux 설치 확인
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ tmux가 설치되지 않았습니다${NC}"
    echo -e "다음 명령어로 설치: ${YELLOW}sudo apt install tmux${NC}"
    exit 1
fi

SESSION_NAME="vla_inference"

# 기존 세션 종료
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo -e "${GREEN}✓ tmux 세션 생성: $SESSION_NAME${NC}"
echo ""

# 새 세션 생성 및 창 분할
tmux new-session -d -s "$SESSION_NAME" -n "VLA"

# 창을 수평 분할 (위: 서버, 아래: 로그)
tmux split-window -v -t "$SESSION_NAME:0"

# 위쪽 패널 크기 조정 (70%)
tmux resize-pane -t "$SESSION_NAME:0.0" -y 25

# 위쪽 패널: API 서버 실행
tmux send-keys -t "$SESSION_NAME:0.0" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '════════ API 서버 시작 ════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "bash scripts/start_inference_server.sh" C-m

# 서버 시작 대기
sleep 2

# 아래쪽 패널: 로그 모니터링
tmux send-keys -t "$SESSION_NAME:0.1" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '════════ 로그 모니터링 ════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "bash scripts/monitor_inference_server.sh" C-m

echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  tmux 세션 실행 중${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "세션 접속: ${YELLOW}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo -e "패널 이동:"
echo -e "  - ${YELLOW}Ctrl+b, 위/아래 화살표${NC}"
echo -e "  - ${YELLOW}Ctrl+b, o${NC} (다음 패널)"
echo ""
echo -e "세션 종료:"
echo -e "  - ${YELLOW}tmux kill-session -t $SESSION_NAME${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"

# 세션에 자동 접속
tmux attach -t "$SESSION_NAME"
