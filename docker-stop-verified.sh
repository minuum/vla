#!/bin/bash

# =============================================================================
# 🛑 Mobile VLA Docker 중지 스크립트 - 검증된 VLA 환경 기반
# =============================================================================

set -e

# 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🛑 Mobile VLA Docker 컨테이너 중지 중...${NC}"

# 컨테이너 중지
docker-compose -f docker-compose.mobile-vla.yml down

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Mobile VLA 컨테이너 중지 완료!${NC}"
    
    # X11 권한 복원
    echo -e "${BLUE}🖥️  X11 권한 복원 중...${NC}"
    xhost -local:docker
    
    echo
    echo -e "${BLUE}📋 상태:${NC}"
    echo "   🔍 실행 중인 컨테이너: $(docker ps | grep mobile_vla | wc -l)개"
    echo "   💾 데이터는 docker_volumes/ 폴더에 보존됨"
    echo
    echo -e "${BLUE}🚀 재시작하려면:${NC}"
    echo "   ./docker-run-verified.sh"
    
else
    echo -e "${RED}❌ 컨테이너 중지 실패!${NC}"
    echo -e "${YELLOW}🔍 실행 중인 컨테이너 확인:${NC}"
    docker ps | grep mobile_vla || echo "Mobile VLA 컨테이너가 실행되지 않고 있습니다."
    exit 1
fi