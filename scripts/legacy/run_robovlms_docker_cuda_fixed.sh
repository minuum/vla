#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 - CUDA FIXED
# CUDA 버전 문제 해결

set -e

echo "🚀 RoboVLMs Docker 환경 시작 (CUDA FIXED)"
echo "🔧 문제 해결: CUDA 환경 변수 및 런타임 설정"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_robovlms_final 2>/dev/null || true
docker rm mobile_vla_robovlms_final 2>/dev/null || true

# NVIDIA 런타임 확인
echo "🔍 NVIDIA 런타임 확인:"
docker info | grep nvidia || echo "⚠️  NVIDIA 런타임이 설정되지 않았습니다"

# 이미지 빌드 (기존 이미지 사용)
echo "🔨 Docker 이미지 빌드 중..."
docker build -f docker/Dockerfile.mobile-vla -t mobile_vla:robovlms-final .

# 컨테이너 실행 (CUDA FIXED)
echo "🚀 컨테이너 실행 중 (CUDA FIXED)..."
docker run -d \
    --name mobile_vla_robovlms_final \
    --gpus all \
    --runtime=nvidia \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -e CUDA_VISIBLE_DEVICES=0 \
    mobile_vla:robovlms-final \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep mobile_vla_robovlms_final

if docker ps | grep -q mobile_vla_robovlms_final; then
    echo "✅ 컨테이너가 성공적으로 실행 중입니다!"
    echo ""
    echo "🔍 CUDA 테스트 실행 중..."
    
    # CUDA 테스트
    echo "📊 GPU 정보:"
    docker exec mobile_vla_robovlms_final nvidia-smi || echo "⚠️  nvidia-smi 실행 실패"
    
    echo "📊 PyTorch CUDA 테스트:"
    docker exec mobile_vla_robovlms_final torch_cuda_test || echo "⚠️  torch_cuda_test 실행 실패"
    
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
