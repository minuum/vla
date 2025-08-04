#!/bin/bash
echo "🚀 Mobile VLA Jetson Docker 컨테이너 시작 중..."

# X11 권한 설정 (GUI 지원)
xhost +local:docker

# 컨테이너 시작
docker-compose -f docker-compose.jetson.yml up -d mobile-vla

echo "✅ 컨테이너 시작 완료!"
echo "📋 유용한 명령어:"
echo "   docker exec -it mobile_vla_jetson bash           # 컨테이너 접속"
echo "   docker exec -it mobile_vla_jetson vla-camera     # CSI 카메라 시작"
echo "   docker exec -it mobile_vla_jetson vla-collect    # 데이터 수집 시작"
echo "   docker-compose -f docker-compose.jetson.yml logs # 로그 확인"
echo "   docker-compose -f docker-compose.jetson.yml down # 컨테이너 중지"
