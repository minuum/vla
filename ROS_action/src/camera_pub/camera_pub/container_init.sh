#!/bin/bash

# =============================================================================
# 🐳 Docker Container ROS2 Environment Auto-Setup
# 도커 컨테이너 실행 시 자동으로 ROS2 환경을 설정하는 스크립트
# =============================================================================

echo "==============================================================================="
echo "🐳 Docker Container ROS2 Environment Auto-Setup"
echo "==============================================================================="

# 1. ROS2 Humble 환경 설정
if [ -f "/opt/ros/humble/setup.bash" ]; then
    echo "✅ ROS2 Humble 환경 설정 중..."
    source /opt/ros/humble/setup.bash
else
    echo "❌ ROS2 Humble이 설치되지 않았습니다."
    echo "📋 ROS2 Humble 설치 중..."
    apt update && apt install -y ros-humble-desktop
    source /opt/ros/humble/setup.bash
fi

# 2. 환경 변수 설정
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo "✅ ROS_DOMAIN_ID=42 설정 완료"
echo "✅ RMW_IMPLEMENTATION=rmw_fastrtps_cpp 설정 완료"

# 3. ROS2 워크스페이스 확인 및 설정
if [ -f "/workspace/vla/ROS_action/install/local_setup.bash" ]; then
    echo "✅ ROS2 워크스페이스 발견, 환경 설정 중..."
    cd /workspace/vla/ROS_action/install && source local_setup.bash
    cd /workspace/vla
elif [ -f "/workspace/vla/ROS_action/install/setup.bash" ]; then
    echo "✅ ROS2 워크스페이스 발견, 환경 설정 중..."
    cd /workspace/vla/ROS_action/install && source setup.bash
    cd /workspace/vla
else
    echo "⚠️  ROS2 워크스페이스가 없습니다. 빌드가 필요합니다."
    echo "📋 ROS 워크스페이스 빌드 중..."
    cd /workspace/vla/ROS_action
    colcon build
    source install/local_setup.bash
    cd /workspace/vla
fi

# 4. 필요한 패키지 설치 확인
echo "📋 필요한 패키지 설치 확인 중..."
pip3 install opencv-python numpy

# 5. 환경 변수 확인
echo ""
echo "🔍 환경 변수 확인:"
echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "   ROS_DISTRO: $ROS_DISTRO"

# 6. ROS2 상태 확인
echo ""
echo "🔍 ROS2 상태 확인:"
if command -v ros2 &> /dev/null; then
    echo "✅ ROS2 명령어 사용 가능"
    echo "📋 사용 가능한 패키지 수: $(ros2 pkg list | wc -l)"
else
    echo "❌ ROS2 명령어를 찾을 수 없습니다."
fi

echo ""
echo "🎉 Docker Container ROS2 Environment Auto-Setup 완료!"
echo "📋 사용 가능한 명령어:"
echo "   ros2 pkg list          : 패키지 목록"
echo "   ros2 node list         : 노드 목록"
echo "   ros2 topic list        : 토픽 목록"
echo "   ros2 run <pkg> <node>  : 노드 실행"
echo "   container-run          : 컨테이너 내부 실행 메뉴"
echo ""

# 7. 컨테이너 실행 메뉴 표시
echo "🚀 Mobile VLA + PyTorch 2.3.0 Docker 환경"
echo "📋 사용 가능한 명령어:"
echo "   cuda-test              : PyTorch/CUDA 상태 확인"
echo "   mobile-vla-test        : Mobile VLA 카메라 테스트"
echo "   torch_cuda_test        : 상세 PyTorch CUDA 테스트"
echo "   run-mobile-vla         : Mobile VLA 시스템 실행"
echo "   container-run          : 컨테이너 내부 실행 메뉴"
echo ""

