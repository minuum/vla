#!/bin/bash

# 🚀 CUDA True로 작동했던 버전 실행 스크립트
# 과거 기록에서 확인된 mobile_vla:pytorch-2.3.0-cuda 사용

set -e

echo "🚀 CUDA True로 작동했던 버전 실행"
echo "🔧 과거 기록에서 확인된 mobile_vla:pytorch-2.3.0-cuda 사용"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_cuda_working 2>/dev/null || true
docker rm mobile_vla_cuda_working 2>/dev/null || true

# 사용 가능한 이미지 확인
echo "📋 사용 가능한 이미지 확인:"
docker images | grep pytorch || echo "⚠️  pytorch 이미지가 없습니다"

# CUDA True로 작동했던 설정으로 컨테이너 실행
echo "🚀 CUDA True로 작동했던 설정으로 컨테이너 실행 중..."
docker run -d \
    --name mobile_vla_cuda_working \
    --gpus all \
    -v /usr/local/cuda:/usr/local/cuda \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
    -v /dev/bus/usb:/dev/bus/usb \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep mobile_vla_cuda_working

if docker ps | grep -q mobile_vla_cuda_working; then
    echo "✅ 컨테이너가 성공적으로 실행 중입니다!"
    echo ""
    echo "🔍 CUDA 테스트 실행 중..."
    
    # CUDA 테스트
    echo "📊 GPU 정보:"
    docker exec mobile_vla_cuda_working nvidia-smi || echo "⚠️  nvidia-smi 실행 실패"
    
    echo "📊 PyTorch CUDA 테스트:"
    docker exec mobile_vla_cuda_working python3 -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'CUDA 디바이스 수: {torch.cuda.device_count()}')
    print(f'디바이스 이름: {torch.cuda.get_device_name(0)}')
    print('✅ CUDA True로 작동 중!')
else:
    print('❌ CUDA를 사용할 수 없습니다')
" || echo "⚠️  PyTorch CUDA 테스트 실행 실패"
    
    echo ""
    echo "🚀 컨테이너에 접속합니다..."
    echo ""
    
    # 컨테이너에 접속
    docker exec -it mobile_vla_cuda_working bash
else
    echo "❌ 컨테이너 실행 실패"
    echo "📋 오류 로그:"
    docker logs mobile_vla_cuda_working
fi
