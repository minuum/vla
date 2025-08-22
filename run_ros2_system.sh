#!/bin/bash

# =============================================================================
# 🚀 ROS2 System Runner Script
# 호스트에서 ROS2 시스템을 실행하는 통합 스크립트
# =============================================================================

echo "==============================================================================="
echo "🚀 ROS2 System Runner"
echo "==============================================================================="

# 1. ROS2 환경 설정
echo "📋 ROS2 환경 설정 중..."
source ./setup_ros2_host.sh

# 2. Docker 컨테이너 상태 확인
echo ""
echo "🔍 Docker 컨테이너 상태 확인 중..."
if docker ps | grep -q "infallible_elion"; then
    echo "✅ 기존 컨테이너 발견: infallible_elion"
    echo "📋 컨테이너에 접속하려면: docker exec -it infallible_elion bash"
else
    echo "⚠️  실행 중인 컨테이너가 없습니다."
    echo "📋 컨테이너를 시작하려면: ./run_mobile_vla_system.sh"
fi

# 3. ROS2 노드 실행 메뉴
echo ""
echo "🎮 ROS2 노드 실행 메뉴:"
echo "   1. 로봇 제어 노드 실행"
echo "   2. 카메라 노드 실행"
echo "   3. VLA 추론 노드 실행"
echo "   4. 데이터 수집 노드 실행"
echo "   5. 시스템 상태 확인"
echo "   6. 종료"
echo ""

read -p "선택하세요 (1-6): " choice

case $choice in
    1)
        echo "🚀 로봇 제어 노드 실행 중..."
        ros2 run mobile_vla_package simple_robot_mover
        ;;
    2)
        echo "📷 카메라 노드 실행 중..."
        ros2 run camera_pub camera_publisher_continuous.py
        ;;
    3)
        echo "🧠 VLA 추론 노드 실행 중..."
        ros2 run mobile_vla_package vla_inference_node.py
        ;;
    4)
        echo "📊 데이터 수집 노드 실행 중..."
        ros2 run mobile_vla_package mobile_vla_data_collector.py
        ;;
    5)
        echo "🔍 시스템 상태 확인 중..."
        echo "📋 실행 중인 노드:"
        ros2 node list
        echo ""
        echo "📋 활성 토픽:"
        ros2 topic list
        echo ""
        echo "📋 패키지 목록:"
        ros2 pkg list | grep -E "(mobile_vla|camera_pub)"
        ;;
    6)
        echo "👋 종료합니다."
        exit 0
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac
