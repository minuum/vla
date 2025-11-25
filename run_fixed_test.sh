#!/bin/bash

# 🚀 Mobile VLA Fixed Test Docker 실행 스크립트
# 문제가 되는 부분들을 수정한 버전으로 테스트

set -e

echo "🚀 Mobile VLA Fixed Test Docker 환경 시작"
echo "🔧 수정된 버전: ROS2 Humble + 최신 Transformers"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_fixed_test 2>/dev/null || true
docker rm mobile_vla_fixed_test 2>/dev/null || true

# 이미지 빌드 (수정된 버전)
echo "🔨 Docker 이미지 빌드 중..."
docker build -f Dockerfile.mobile-vla-fixed -t mobile_vla:fixed-test .

# 컨테이너 실행 (Jetson 최적화 설정)
echo "🚀 컨테이너 실행 중..."
docker run -d \
    --name mobile_vla_fixed_test \
    --network bridge \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    mobile_vla:fixed-test \
    tail -f /dev/null

echo "✅ 컨테이너가 백그라운드에서 실행 중입니다."
echo ""
echo "🚀 컨테이너에 바로 접속합니다..."
echo ""

# 컨테이너에 바로 접속
docker exec -it mobile_vla_fixed_test bash
