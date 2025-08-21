#!/bin/bash

# =============================================================================
# 🚀 ROS2 Host Environment Setup Script
# 호스트에서 ROS2 환경을 자동으로 설정하는 스크립트
# =============================================================================

echo "==============================================================================="
echo "🚀 ROS2 Host Environment Setup"
echo "==============================================================================="

# 1. ROS2 Humble 설치 확인
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo "❌ ROS2 Humble이 설치되지 않았습니다."
    echo "📋 설치 명령어:"
    echo "   sudo apt update"
    echo "   sudo apt install ros-humble-desktop"
    exit 1
fi

# 2. ROS2 환경 설정
echo "✅ ROS2 Humble 환경 설정 중..."
source /opt/ros/humble/setup.bash

# 3. ROS_DOMAIN_ID 설정
export ROS_DOMAIN_ID=42
echo "✅ ROS_DOMAIN_ID=42 설정 완료"

# 4. RMW 구현체 설정
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
echo "✅ RMW_IMPLEMENTATION=rmw_fastrtps_cpp 설정 완료"

# 5. ROS2 워크스페이스 확인 및 설정
if [ -f "/home/soda/vla/ROS_action/install/setup.bash" ]; then
    echo "✅ ROS2 워크스페이스 발견, 환경 설정 중..."
    source /home/soda/vla/ROS_action/install/setup.bash
else
    echo "⚠️  ROS2 워크스페이스가 없습니다. 빌드가 필요합니다."
    echo "📋 빌드 명령어:"
    echo "   cd /home/soda/vla/ROS_action"
    echo "   colcon build"
fi

# 6. 환경 변수 확인
echo ""
echo "🔍 환경 변수 확인:"
echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "   ROS_DISTRO: $ROS_DISTRO"

# 7. ROS2 상태 확인
echo ""
echo "🔍 ROS2 상태 확인:"
if command -v ros2 &> /dev/null; then
    echo "✅ ROS2 명령어 사용 가능"
    echo "📋 사용 가능한 패키지 수: $(ros2 pkg list | wc -l)"
else
    echo "❌ ROS2 명령어를 찾을 수 없습니다."
fi

echo ""
echo "🎉 ROS2 Host Environment Setup 완료!"
echo "📋 사용 가능한 명령어:"
echo "   ros2 pkg list          : 패키지 목록"
echo "   ros2 node list         : 노드 목록"
echo "   ros2 topic list        : 토픽 목록"
echo "   ros2 run <pkg> <node>  : 노드 실행"
echo ""
