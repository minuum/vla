#!/bin/bash
# 추론 서버 로그 모니터링 스크립트

PROJECT_DIR="/home/soda/vla"
LOG_FILE="$PROJECT_DIR/logs/api_server.log"

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}  추론 서버 로그 모니터링${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# 로그 파일 확인
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}로그 파일이 아직 생성되지 않았습니다${NC}"
    echo -e "서버가 시작되면 로그가 나타납니다..."
    echo ""
    
    # 로그 파일이 생성될 때까지 대기
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
fi

echo -e "${GREEN}로그 모니터링 시작...${NC}"
echo -e "${YELLOW}Ctrl+C로 종료${NC}"
echo ""
echo -e "${BLUE}────────────────────────────────────────${NC}"

# 실시간 로그 표시
tail -f "$LOG_FILE" | while read line; do
    # 에러 강조
    if echo "$line" | grep -i "error\|fail\|exception" > /dev/null; then
        echo -e "\033[0;31m$line\033[0m"  # Red
    # 경고 강조
    elif echo "$line" | grep -i "warning\|warn" > /dev/null; then
        echo -e "\033[1;33m$line\033[0m"  # Yellow
    # 성공 강조
    elif echo "$line" | grep -i "success\|✓\|✅" > /dev/null; then
        echo -e "\033[0;32m$line\033[0m"  # Green
    # 일반 로그
    else
        echo "$line"
    fi
done
