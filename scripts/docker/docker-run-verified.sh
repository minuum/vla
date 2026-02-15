#!/bin/bash

# =============================================================================
# 🚀 Mobile VLA Docker 실행 스크립트 - 검증된 VLA 환경 기반
# =============================================================================

set -e

# 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Mobile VLA Docker 컨테이너 시작 (검증된 VLA 환경)${NC}"

# X11 권한 설정 (GUI 지원)
echo -e "${BLUE}🖥️  X11 권한 설정 중...${NC}"
xhost +local:docker

# 필요한 디렉토리 생성
echo -e "${BLUE}📁 볼륨 디렉토리 확인 중...${NC}"
mkdir -p docker_volumes/cache
mkdir -p docker_volumes/dataset
mkdir -p docker_volumes/logs

# 컨테이너 시작
echo -e "${BLUE}🐳 Docker 컨테이너 시작 중...${NC}"
docker-compose -f docker-compose.mobile-vla.yml up -d mobile-vla

# 시작 확인
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✅ Mobile VLA 컨테이너 시작 완료!${NC}"
    echo -e "${BLUE}📋 유용한 명령어들:${NC}"
    echo
    echo "🔧 기본 명령어:"
    echo "   docker exec -it mobile_vla_verified bash                    # 컨테이너 접속"
    echo "   docker exec -it mobile_vla_verified /usr/local/bin/healthcheck.sh  # 헬스체크"
    echo
    echo "🔍 테스트 명령어:"
    echo "   docker exec -it mobile_vla_verified torch_cuda_test         # 기존 VLA CUDA 테스트"
    echo "   docker exec -it mobile_vla_verified cuda-test               # 간단 CUDA 테스트"
    echo
    echo "🤖 Mobile VLA 명령어:"
    echo "   docker exec -it mobile_vla_verified vla-build               # ROS2 워크스페이스 빌드"
    echo "   docker exec -it mobile_vla_verified vla-source              # ROS2 환경 소싱"
    echo "   docker exec -it mobile_vla_verified vla-camera              # CSI 카메라 시작"
    echo "   docker exec -it mobile_vla_verified vla-collect             # 데이터 수집 시작"
    echo
    echo "📊 모니터링:"
    echo "   docker logs -f mobile_vla_verified                          # 컨테이너 로그"
    echo "   docker stats mobile_vla_verified                            # 리소스 사용량"
    echo "   ./docker-monitor-verified.sh                                # 전용 모니터링 서비스"
    echo
    echo "🛑 중지:"
    echo "   ./docker-stop-verified.sh                                   # 컨테이너 중지"
    echo "   docker-compose -f docker-compose.mobile-vla.yml down        # 직접 중지"
    echo
    
    # 컨테이너 상태 확인
    echo -e "${BLUE}🔍 컨테이너 상태:${NC}"
    docker ps | grep mobile_vla_verified
    
    echo
    echo -e "${YELLOW}💡 팁: 컨테이너가 완전히 준비되려면 1-2분 정도 기다려주세요.${NC}"
    echo -e "${BLUE}    헬스체크로 상태를 확인할 수 있습니다.${NC}"
    
else
    echo -e "${RED}❌ 컨테이너 시작 실패!${NC}"
    echo -e "${YELLOW}🔍 로그 확인:${NC}"
    docker-compose -f docker-compose.mobile-vla.yml logs mobile-vla
    exit 1
fi