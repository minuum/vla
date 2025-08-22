#!/bin/bash

# =============================================================================
# 🐳 ROS2 Docker System Runner Script
# Docker 컨테이너에서 ROS2 시스템을 실행하는 통합 스크립트
# =============================================================================

echo "==============================================================================="
echo "🐳 ROS2 Docker System Runner"
echo "==============================================================================="

# 1. ROS2 환경 설정
echo "📋 ROS2 환경 설정 중..."
source ./setup_ros2_docker.sh

# 2. ROS2 노드 실행 메뉴
echo ""
echo "🎮 ROS2 노드 실행 메뉴:"
echo "   1. 로봇 제어 노드 실행"
echo "   2. 카메라 노드 실행"
echo "   3. VLA 추론 노드 실행"
echo "   4. 데이터 수집 노드 실행"
echo "   5. 시스템 상태 확인"
echo "   6. 토픽 모니터링"
echo "   7. 종료"
echo ""

read -p "선택하세요 (1-7): " choice

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
        echo ""
        echo "📋 환경 변수:"
        echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
        echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
        echo "   ROS_LOCALHOST_ONLY: $ROS_LOCALHOST_ONLY"
        ;;
    6)
        echo "📡 토픽 모니터링 중..."
        echo "📋 사용 가능한 토픽:"
        ros2 topic list
        echo ""
        read -p "모니터링할 토픽을 입력하세요 (예: /cmd_vel): " topic_name
        if [ -n "$topic_name" ]; then
            echo "📡 $topic_name 토픽 모니터링 중... (Ctrl+C로 종료)"
            ros2 topic echo "$topic_name"
        fi
        ;;
    7)
        echo "👋 종료합니다."
        exit 0
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac
