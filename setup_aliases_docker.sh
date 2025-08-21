#!/bin/bash

# =============================================================================
# 🐳 ROS2 Docker Aliases Setup Script
# Docker 컨테이너에서 간단한 명령어로 ROS2 노드를 실행할 수 있도록 별칭 설정
# =============================================================================

echo "==============================================================================="
echo "🐳 ROS2 Docker Aliases Setup"
echo "==============================================================================="

# 1. ROS2 환경 설정
source ./setup_ros2_docker.sh

# 2. 별칭 함수들 정의
echo "📋 ROS2 노드 실행 별칭 설정 중..."

# 로봇 제어 노드
run_robot_control() {
    echo "🚀 로봇 제어 노드 실행 중..."
    ros2 run mobile_vla_package simple_robot_mover
}

# 카메라 서버 노드
run_camera_server() {
    echo "📷 카메라 서버 노드 실행 중..."
    ros2 run camera_pub camera_publisher_continuous.py
}

# VLA 데이터 수집 노드
run_vla_collector() {
    echo "📊 VLA 데이터 수집 노드 실행 중..."
    ros2 run mobile_vla_package vla_collector
}

# VLA 추론 노드
run_vla_inference() {
    echo "🧠 VLA 추론 노드 실행 중..."
    ros2 run mobile_vla_package vla_inference_node
}

# 시스템 상태 확인
check_system() {
    echo "🔍 시스템 상태 확인 중..."
    echo "📋 실행 중인 노드:"
    ros2 node list
    echo ""
    echo "📋 활성 토픽:"
    ros2 topic list
    echo ""
    echo "📋 환경 변수:"
    echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
    echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
    echo "   ROS_LOCALHOST_ONLY: $ROS_LOCALHOST_ONLY"
}

# 토픽 모니터링
monitor_topic() {
    local topic_name=${1:-"/cmd_vel"}
    echo "📡 $topic_name 토픽 모니터링 중... (Ctrl+C로 종료)"
    ros2 topic echo "$topic_name"
}

# 컨테이너 상태 확인
check_container() {
    echo "🔍 컨테이너 상태 확인 중..."
    echo "📋 호스트명: $(hostname)"
    echo "📋 IP 주소: $(hostname -I)"
    echo "📋 네트워크 모드: $(if [ "$(hostname)" = "$(hostname -I | awk '{print $1}')" ]; then echo "호스트 네트워크"; else echo "브리지 네트워크"; fi)"
    echo "📋 현재 디렉토리: $(pwd)"
    echo "📋 ROS2 워크스페이스: $(if [ -f "/workspace/vla/ROS_action/install/setup.bash" ]; then echo "설정됨"; else echo "설정되지 않음"; fi)"
}

# 3. 함수들을 현재 셸에 export
export -f run_robot_control
export -f run_camera_server
export -f run_vla_collector
export -f run_vla_inference
export -f check_system
export -f monitor_topic
export -f check_container

# 4. .bashrc에 별칭 추가 (선택사항)
echo ""
echo "📋 사용 가능한 명령어:"
echo "   run_robot_control    : 로봇 제어 노드 실행"
echo "   run_camera_server    : 카메라 서버 노드 실행"
echo "   run_vla_collector    : VLA 데이터 수집 노드 실행"
echo "   run_vla_inference    : VLA 추론 노드 실행"
echo "   check_system         : 시스템 상태 확인"
echo "   check_container      : 컨테이너 상태 확인"
echo "   monitor_topic <topic>: 토픽 모니터링"
echo ""
echo "🎉 ROS2 Docker Aliases Setup 완료!"
echo "📋 이제 간단한 명령어로 노드를 실행할 수 있습니다!"
echo ""
