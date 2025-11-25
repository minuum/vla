#!/bin/bash

# 🚀 ROS 설정이 포함된 도커 이미지 빌드 스크립트
# 컨테이너 간 ROS2 통신을 위한 완전한 환경 구성

set -e

echo "🚀 ROS 설정이 포함된 도커 이미지 빌드"
echo "======================================"
echo ""

# 기존 이미지 정리
echo "🧹 기존 이미지 정리 중..."
docker rmi mobile_vla:ros 2>/dev/null || echo "기존 이미지가 없습니다."

# ROS 설정이 포함된 이미지 빌드
echo "🔨 ROS 설정이 포함된 이미지 빌드 중..."
echo "⚠️ 빌드 시간이 오래 걸릴 수 있습니다 (10-15분)"
echo ""

docker build -f Dockerfile.mobile-vla-ros -t mobile_vla:ros .

if [ $? -eq 0 ]; then
    echo "✅ ROS 설정이 포함된 이미지 빌드 완료!"
    echo ""
    
    # 이미지 정보 확인
    echo "📋 이미지 정보:"
    docker images | grep mobile_vla
    echo ""
    
    # 이미지 상세 정보
    echo "📋 이미지 상세 정보:"
    docker inspect mobile_vla:ros --format='{{.Config.Env}}' | tr ' ' '\n' | grep -E "(ROS|CUDA|PYTHON)"
    echo ""
    
    # 포트 정보
    echo "📋 노출된 포트:"
    docker inspect mobile_vla:ros --format='{{.Config.ExposedPorts}}'
    echo ""
    
else
    echo "❌ 이미지 빌드 실패"
    exit 1
fi

# 빌드된 이미지 테스트
echo "🧪 빌드된 이미지 테스트 중..."
echo ""

# 테스트 컨테이너 실행
echo "📦 테스트 컨테이너 실행 중..."
docker run --rm -it \
    --name ros_test_container \
    --gpus all \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    mobile_vla:ros \
    bash -c "
        echo '🔍 ROS2 환경 테스트 중...'
        source /opt/ros/setup_ros.sh
        echo '✅ ROS2 환경 설정 완료'
        echo ''
        echo '🔍 ROS2 버전 확인:'
        ros2 --help | head -5
        echo ''
        echo '🔍 ROS2 패키지 목록 (상위 10개):'
        ros2 pkg list | head -10
        echo ''
        echo '🔍 CUDA 상태 확인:'
        python3 -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')\" 2>/dev/null || echo 'PyTorch CUDA 테스트 건너뜀'
        echo ''
        echo '🔍 GPU 정보:'
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        echo ''
        echo '✅ 이미지 테스트 완료!'
    "

echo ""
echo "🎯 ROS 설정이 포함된 도커 이미지 빌드 및 테스트 완료!"
echo ""

# 사용 방법 안내
echo "📋 사용 방법:"
echo "1. 단일 컨테이너 실행:"
echo "   docker run -it --gpus all mobile_vla:ros bash"
echo ""
echo "2. 컨테이너 간 통신 테스트:"
echo "   ./test_ros2_communication.sh"
echo ""
echo "3. ROS2 워크스페이스 설정:"
echo "   cd /workspace/ros2_ws"
echo "   colcon build"
echo "   source install/setup.bash"
echo ""

echo "💡 이제 ROS2 환경이 완전히 설정된 도커 이미지가 준비되었습니다!"
echo "💡 컨테이너 간 ROS2 통신이 가능합니다!"
