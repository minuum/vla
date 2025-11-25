#!/bin/bash

# 🚀 Mobile VLA Simple Test Docker 실행 스크립트
# 기본 기능만 포함하여 안정성 테스트

set -e

echo "🚀 Mobile VLA Simple Test Docker 환경 시작"
echo "🔧 기본 PyTorch + CUDA 환경만 포함"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_simple_test 2>/dev/null || true
docker rm mobile_vla_simple_test 2>/dev/null || true

# 이미지 빌드 (간단한 버전)
echo "🔨 Docker 이미지 빌드 중..."
docker build -f docker/Dockerfile.mobile-vla-simple -t mobile_vla:simple-test .

# 컨테이너 실행 (최소한의 설정)
echo "🚀 컨테이너 실행 중..."
docker run -d \
    --name mobile_vla_simple_test \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    mobile_vla:simple-test \
    tail -f /dev/null

echo "✅ 컨테이너가 백그라운드에서 실행 중입니다."
echo ""
echo "🚀 컨테이너에 바로 접속합니다..."
echo ""

# 컨테이너에 바로 접속
docker exec -it mobile_vla_simple_test bash
