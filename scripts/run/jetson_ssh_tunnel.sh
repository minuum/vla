#!/bin/bash
# SSH 터널링 스크립트 - Jetson에서 Billy 서버로 안전한 연결
# Billy 서버는 Port 10022에서 SSH를 listening하고 있습니다!

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔒 SSH 터널 설정 스크립트${NC}"
echo "========================================"

# 1. 환경 변수 확인
if [ -f "./secrets.sh" ]; then
    source ./secrets.sh
    echo -e "${GREEN}✅${NC} secrets.sh 로드 완료"
else
    echo -e "${RED}❌ secrets.sh 파일이 없습니다!${NC}"
    echo "다음 내용으로 secrets.sh를 생성하세요:"
    echo ""
    echo "export BILLY_IP='100.86.152.29'"
    echo "export VLA_API_KEY='your-secret-key-here'"
    echo ""
    exit 1
fi

# 2. 필수 환경 변수 확인
if [ -z "${BILLY_IP:-}" ]; then
    echo -e "${RED}❌ BILLY_IP가 설정되지 않았습니다!${NC}"
    exit 1
fi

if [ -z "${VLA_API_KEY:-}" ]; then
    echo -e "${YELLOW}⚠️  VLA_API_KEY가 설정되지 않았습니다.${NC}"
    echo "   API 인증이 필요한 경우 secrets.sh에 추가하세요."
fi

# 3. Tailscale 연결 확인
echo ""
echo "Tailscale 연결 확인 중..."
if ! command -v tailscale &> /dev/null; then
    echo -e "${RED}❌ Tailscale이 설치되지 않았습니다!${NC}"
    exit 1
fi

if ! tailscale status | grep -q "${BILLY_IP}"; then
    echo -e "${RED}❌ Billy 서버(${BILLY_IP})에 연결할 수 없습니다!${NC}"
    echo "Tailscale 상태:"
    tailscale status
    exit 1
fi

echo -e "${GREEN}✅${NC} Tailscale 연결 확인됨: ${BILLY_IP}"

# 4. 기존 SSH 터널 확인 및 종료
echo ""
echo "기존 SSH 터널 확인 중..."
EXISTING_TUNNEL=$(ps aux | grep "ssh -N" | grep "8000:localhost:8000" | grep -v grep || true)

if [ -n "$EXISTING_TUNNEL" ]; then
    echo -e "${YELLOW}⚠️  기존 SSH 터널이 실행 중입니다:${NC}"
    echo "$EXISTING_TUNNEL"
    read -p "종료하고 새로 시작하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TUNNEL_PID=$(echo "$EXISTING_TUNNEL" | awk '{print $2}')
        kill "$TUNNEL_PID" 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✅${NC} 기존 터널 종료됨"
    else
        echo "기존 터널을 유지합니다."
        exit 0
    fi
fi

# 5. SSH 연결 테스트 (Port 10022!)
echo ""
echo "SSH 연결 테스트 중 (Port 10022)..."
if ssh -p 10022 -o ConnectTimeout=5 -o BatchMode=yes billy@${BILLY_IP} "echo 'SSH 연결 성공'" 2>/dev/null; then
    echo -e "${GREEN}✅${NC} SSH Key 인증 성공!"
    SSH_AUTH="key"
else
    echo -e "${YELLOW}⚠️  SSH Key 인증 실패. 비밀번호가 필요합니다.${NC}"
    SSH_AUTH="password"
fi

# 6. SSH 터널 생성
echo ""
echo "SSH 터널 생성 중..."
echo "Command: ssh -N -f -L 8000:localhost:8000 billy@${BILLY_IP} -p 10022"

if [ "$SSH_AUTH" = "key" ]; then
    # SSH Key 인증
    if ssh -N -f -L 8000:localhost:8000 billy@${BILLY_IP} -p 10022; then
        echo -e "${GREEN}✅${NC} SSH 터널 생성 성공!"
    else
        echo -e "${RED}❌ SSH 터널 생성 실패!${NC}"
        exit 1
    fi
else
    # 비밀번호 인증
    echo -e "${YELLOW}비밀번호를 입력하세요:${NC}"
    if ssh -N -f -L 8000:localhost:8000 billy@${BILLY_IP} -p 10022; then
        echo -e "${GREEN}✅${NC} SSH 터널 생성 성공!"
    else
        echo -e "${RED}❌ SSH 터널 생성 실패!${NC}"
        exit 1
    fi
fi

# 7. 터널 동작 확인
sleep 2
if ps aux | grep "ssh -N" | grep "8000:localhost:8000" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✅${NC} SSH 터널이 정상 동작 중입니다!"
    ps aux | grep "ssh -N" | grep "8000:localhost:8000" | grep -v grep | awk '{print "PID:", $2, "CMD:", $11, $12, $13, $14}'
else
    echo -e "${RED}❌ SSH 터널이 시작되지 않았습니다!${NC}"
    exit 1
fi

# 8. API 서버 테스트
echo ""
echo "API 서버 Health Check 중..."
export VLA_API_SERVER="http://localhost:8000"

if curl -s -H "X-API-Key: ${VLA_API_KEY}" http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} API 서버 응답 성공!"
    echo "Health Check 결과:"
    curl -s -H "X-API-Key: ${VLA_API_KEY}" http://localhost:8000/health | python3 -m json.tool 2>/dev/null || \
    curl -s -H "X-API-Key: ${VLA_API_KEY}" http://localhost:8000/health
else
    echo -e "${YELLOW}⚠️  API 서버 응답 없음 (Billy 서버에서 API 서버가 실행 중인지 확인하세요)${NC}"
fi

# 9. 환경 변수 안내
echo ""
echo "========================================"
echo -e "${GREEN}🚀 SSH 터널 준비 완료!${NC}"
echo ""
echo "다음 환경 변수를 사용하세요:"
echo ""
echo "export VLA_API_SERVER=\"http://localhost:8000\""
echo "export VLA_API_KEY=\"${VLA_API_KEY}\""
echo ""
echo "ROS2 실행:"
echo "ros2 run mobile_vla_package api_client_node"
echo ""
echo "터널 종료:"
echo "ps aux | grep 'ssh -N' | grep '8000:localhost:8000' | awk '{print \$2}' | xargs kill"
echo "========================================"
