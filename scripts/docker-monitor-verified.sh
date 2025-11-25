#!/bin/bash

# =============================================================================
# 📊 Mobile VLA 모니터링 스크립트 - 검증된 VLA 환경 기반
# =============================================================================

set -e

# 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}📊 Mobile VLA 모니터링 서비스 시작 중...${NC}"

# 모니터링 서비스 시작
docker-compose -f docker-compose.mobile-vla.yml --profile monitoring up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 모니터링 서비스 시작 완료!${NC}"
    echo
    echo -e "${PURPLE}📊 모니터링 대시보드:${NC}"
    echo "   🔍 실시간 로그: docker logs -f mobile_vla_monitoring"
    echo "   📈 시스템 상태: 30초마다 자동 업데이트"
    echo "   🎯 모니터링 항목:"
    echo "     - GPU 사용률 및 온도"
    echo "     - 메모리 사용량"
    echo "     - 디스크 사용량"
    echo "     - ROS2 노드 상태"
    echo "     - 카메라 토픽 상태"
    echo
    echo -e "${BLUE}📋 유용한 명령어:${NC}"
    echo "   docker logs -f mobile_vla_monitoring     # 모니터링 로그 실시간 확인"
    echo "   docker stats mobile_vla_verified         # 메인 컨테이너 리소스 확인"
    echo "   docker exec -it mobile_vla_verified nvidia-smi  # GPU 상태 직접 확인"
    echo
    echo -e "${YELLOW}🛑 모니터링 중지:${NC}"
    echo "   docker-compose -f docker-compose.mobile-vla.yml --profile monitoring down"
    
else
    echo -e "${RED}❌ 모니터링 서비스 시작 실패!${NC}"
    exit 1
fi