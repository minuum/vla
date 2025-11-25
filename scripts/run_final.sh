#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 (JetPack 6 + l4t-ml 기반 최종 버전)

set -e

# 최종 이미지 및 컨테이너 이름 설정
IMAGE_NAME="mobile_vla:robovlms-jp6-final"
CONTAINER_NAME="mobile_vla_robovlms_jp6_final"
DOCKERFILE="Dockerfile.jp6.final"

echo "🚀 RoboVLMs Docker 환경 시작 (JetPack 6, l4t-ml 기반)"
echo "🧹 기존 컨테이너 정리 중..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🔨 Docker 이미지 빌드 중 ($DOCKERFILE 사용)..."
docker build -f $DOCKERFILE -t $IMAGE_NAME .

echo "🚀 컨테이너 실행 중..."
docker run -d \
    --name $CONTAINER_NAME \
    --runtime nvidia \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    $IMAGE_NAME \
    tail -f /dev/null

echo "✅ 컨테이너가 백그라운드에서 실행 중입니다."
echo "🚀 컨테이너에 바로 접속합니다..."
# ... (도움말 메시지) ...

docker exec -it $CONTAINER_NAME bash