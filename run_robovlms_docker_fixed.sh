#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 - FIXED
# tail 명령어 문제 해결

set -e

echo "🚀 RoboVLMs Docker 환경 시작 (FIXED)"
echo "🔧 문제 해결: tail 명령어 실행 방식 변경"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_robovlms_final 2>/dev/null || true
docker rm mobile_vla_robovlms_final 2>/dev/null || true

# 이미지 빌드 (기존 이미지 사용)
echo "🔨 Docker 이미지 빌드 중..."
docker build -f Dockerfile.mobile-vla -t mobile_vla:robovlms-final .

# 컨테이너 실행 (FIXED - sleep 명령어 사용)
echo "🚀 컨테이너 실행 중 (FIXED)..."
docker run -d \
    --name mobile_vla_robovlms_final \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    mobile_vla:robovlms-final \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep mobile_vla_robovlms_final

if docker ps | grep -q mobile_vla_robovlms_final; then
    echo "✅ 컨테이너가 성공적으로 실행 중입니다!"
    echo ""
    echo "🚀 컨테이너에 접속합니다..."
    echo ""
    
    # 컨테이너에 접속
    docker exec -it mobile_vla_robovlms_final bash
else
    echo "❌ 컨테이너 실행 실패"
    echo "📋 오류 로그:"
    docker logs mobile_vla_robovlms_final
fi
