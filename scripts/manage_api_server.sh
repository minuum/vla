#!/bin/bash
# API 서버 관리 스크립트
# Usage: ./manage_api_server.sh [start|stop|restart|status|logs]

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="/home/billy/25-1kp/vla"
LOG_FILE="$PROJECT_DIR/logs/api_server.log"
PID_FILE="$PROJECT_DIR/logs/api_server.pid"

# 프로세스 확인
check_process() {
    # PID 파일 존재하고 프로세스 실행 중인지 확인
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"
            return 1  # Not running
        fi
    fi
    
    # PID 파일이 없으면 프로세스 이름으로 검색
    if pgrep -f "python3 api_server.py" > /dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# 서버 시작
start_server() {
    echo -e "${BLUE}Starting API server...${NC}"
    
    if check_process; then
        echo -e "${YELLOW}API server is already running${NC}"
        return 1
    fi
    
    cd "$PROJECT_DIR"
    mkdir -p logs
    
    # 백그라운드 실행
    nohup python3 api_server.py > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # 시작 대기
    sleep 2
    
    if check_process; then
        echo -e "${GREEN}✓ API server started (PID: $PID)${NC}"
        echo -e "Log file: ${YELLOW}$LOG_FILE${NC}"
        
        # Health check
        sleep 1
        if curl -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}✓ Server is healthy${NC}"
        else
            echo -e "${YELLOW}⚠ Server started but health check failed${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to start API server${NC}"
        echo -e "Check logs: tail -f $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 서버 중지
stop_server() {
    echo -e "${BLUE}Stopping API server...${NC}"
    
    if ! check_process; then
        echo -e "${YELLOW}API server is not running${NC}"
        return 1
    fi
    
    # PID로 종료 시도
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill "$PID" 2>/dev/null
        rm -f "$PID_FILE"
    fi
    
    # 프로세스 이름으로 종료
    pkill -f "python3 api_server.py"
    
    # 종료 확인
    sleep 1
    if ! check_process; then
        echo -e "${GREEN}✓ API server stopped${NC}"
    else
        echo -e "${RED}✗ Failed to stop API server${NC}"
        echo -e "Try: ${YELLOW}pkill -9 -f api_server.py${NC}"
        return 1
    fi
}

# 서버 재시작
restart_server() {
    echo -e "${BLUE}Restarting API server...${NC}"
    stop_server
    sleep 1
    start_server
}

# 서버 상태 확인
status_server() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}API Server Status${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if check_process; then
        PID=$(pgrep -f "python3 api_server.py")
        echo -e "Status: ${GREEN}Running ✓${NC}"
        echo -e "PID: ${YELLOW}$PID${NC}"
        
        # CPU/Memory 사용량
        ps -p "$PID" -o %cpu,%mem,cmd 2>/dev/null | tail -n 1
        
        # Health check
        echo ""
        echo "Health Check:"
        curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Failed"
        
    else
        echo -e "Status: ${RED}Not Running ✗${NC}"
    fi
    
    echo ""
    echo -e "Log file: ${YELLOW}$LOG_FILE${NC}"
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE"
    fi
}

# 로그 보기
view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        return 1
    fi
    
    echo -e "${BLUE}API Server Logs (tail -f)${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
    echo ""
    tail -f "$LOG_FILE"
}

# Main
case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        view_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the API server"
        echo "  stop    - Stop the API server"
        echo "  restart - Restart the API server"
        echo "  status  - Show server status (default)"
        echo "  logs    - View server logs (tail -f)"
        exit 1
        ;;
esac
