#!/bin/bash

# 🚀 ROS 환경 빌드 및 시스템 실행 명령어 모음
# 핵심 명령어들을 쉽게 실행할 수 있도록 정리

echo "🚀 ROS 환경 빌드 및 시스템 실행 명령어"
echo "======================================"
echo ""

# 1. ROS 환경 설정
setup_ros_env() {
    echo "📋 1. ROS 환경 설정"
    echo "source /opt/ros/humble/setup.bash"
    echo "export ROS_DOMAIN_ID=42"
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    echo ""
}

# 2. 워크스페이스 빌드
build_workspace() {
    echo "📋 2. 워크스페이스 빌드"
    echo "cd ./ROS_action"
    echo "rosdep install --from-paths src --ignore-src -r -y"
    echo "colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference"
    echo "source install/setup.bash"
    echo ""
}

# 3. 핵심 노드 실행 명령어
run_core_nodes() {
    echo "📋 3. 핵심 노드 실행 명령어"
    echo ""
    echo "📷 카메라 서비스:"
    echo "  # CSI 카메라 서비스"
    echo "  ros2 run camera_pub camera_service_server"
    echo "  # USB 카메라 서비스"
    echo "  ros2 run camera_pub usb_camera_service_server"
    echo ""
    echo "🧠 VLA 추론 (SOTA):"
    echo "  # 메인 추론 노드 (MAE 0.212)"
    echo "  ros2 run mobile_vla_package robovlms_inference"
    echo "  # 시스템 컨트롤러"
    echo "  ros2 run mobile_vla_package robovlms_controller"
    echo "  # 성능 모니터링"
    echo "  ros2 run mobile_vla_package robovlms_monitor"
    echo ""
    echo "🤖 로봇 제어:"
    echo "  # WASD 키보드 + VLA 통합 제어"
    echo "  ros2 run robot_control robot_control_node"
    echo ""
    echo "📊 데이터 수집:"
    echo "  # 실시간 데이터 수집"
    echo "  ros2 run mobile_vla_package mobile_vla_data_collector"
    echo ""
    echo "🚀 전체 시스템:"
    echo "  # SOTA 모델 전체 시스템"
    echo "  ros2 launch mobile_vla_package robovlms_system.launch.py"
    echo ""
}

# 4. 시스템 상태 확인 명령어
check_system_status() {
    echo "📋 4. 시스템 상태 확인 명령어"
    echo ""
    echo "🔍 ROS 상태:"
    echo "  # 패키지 목록 확인"
    echo "  ros2 pkg list | grep -E '(camera|mobile_vla|robot|vla)'"
    echo "  # 노드 목록 확인"
    echo "  ros2 node list"
    echo "  # 토픽 목록 확인"
    echo "  ros2 topic list"
    echo "  # 서비스 목록 확인"
    echo "  ros2 service list"
    echo ""
    echo "🔍 시스템 상태:"
    echo "  # GPU 상태 확인"
    echo "  nvidia-smi"
    echo "  # 시스템 리소스 확인"
    echo "  htop"
    echo "  # ROS 토픽 모니터링"
    echo "  ros2 topic echo /cmd_vel"
    echo "  ros2 topic echo /mobile_vla/inference_result"
    echo ""
}

# 5. 테스트 명령어
test_commands() {
    echo "📋 5. 테스트 명령어"
    echo ""
    echo "🧪 카메라 테스트:"
    echo "  # 카메라 서비스 테스트"
    echo "  ros2 service call /get_image_service camera_interfaces/srv/GetImage"
    echo "  # 카메라 토픽 확인"
    echo "  ros2 topic echo /camera/image_raw"
    echo ""
    echo "🧪 VLA 추론 테스트:"
    echo "  # 추론 결과 확인"
    echo "  ros2 topic echo /mobile_vla/inference_result"
    echo "  # 추론 상태 확인"
    echo "  ros2 topic echo /mobile_vla/status"
    echo ""
    echo "🧪 로봇 제어 테스트:"
    echo "  # 제어 명령 확인"
    echo "  ros2 topic echo /cmd_vel"
    echo "  # 제어 상태 확인"
    echo "  ros2 topic echo /robot_control/status"
    echo ""
}

# 6. 발표용 시연 명령어
demo_commands() {
    echo "📋 6. 발표용 시연 명령어"
    echo ""
    echo "🎯 1단계: 시스템 시작 (30초)"
    echo "  # ROS 환경 설정"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  export ROS_DOMAIN_ID=42"
    echo "  cd ./ROS_action && source install/setup.bash"
    echo ""
    echo "🎯 2단계: 핵심 노드 실행 (1분)"
    echo "  # 터미널 1: 카메라 서비스"
    echo "  ros2 run camera_pub usb_camera_service_server"
    echo "  # 터미널 2: VLA 추론"
    echo "  ros2 run mobile_vla_package robovlms_inference"
    echo "  # 터미널 3: 로봇 제어"
    echo "  ros2 run robot_control robot_control_node"
    echo ""
    echo "🎯 3단계: 실제 테스트 (1분)"
    echo "  # WASD 키보드 제어 테스트"
    echo "  # VLA 추론 결과 확인"
    echo "  # 성능 모니터링"
    echo ""
    echo "🎯 4단계: 성능 확인 (30초)"
    echo "  # GPU 사용률 확인"
    echo "  nvidia-smi"
    echo "  # 추론 속도 확인"
    echo "  ros2 topic echo /mobile_vla/inference_result"
    echo ""
}

# 7. 문제 해결 명령어
troubleshoot_commands() {
    echo "📋 7. 문제 해결 명령어"
    echo ""
    echo "🔧 빌드 문제:"
    echo "  # 의존성 재설치"
    echo "  rosdep install --from-paths src --ignore-src -r -y"
    echo "  # 캐시 클리어 후 재빌드"
    echo "  rm -rf build/ install/ log/"
    echo "  colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference"
    echo ""
    echo "🔧 실행 문제:"
    echo "  # 노드 강제 종료"
    echo "  pkill -f 'ros2 run'"
    echo "  # 서비스 재시작"
    echo "  ros2 service call /reset_camera_service std_srvs/srv/Empty"
    echo ""
    echo "🔧 네트워크 문제:"
    echo "  # ROS 도메인 확인"
    echo "  echo \$ROS_DOMAIN_ID"
    echo "  # 네트워크 인터페이스 확인"
    echo "  ip addr show"
    echo ""
}

# 메인 함수
main() {
    setup_ros_env
    build_workspace
    run_core_nodes
    check_system_status
    test_commands
    demo_commands
    troubleshoot_commands
    
    echo "🎯 핵심 명령어 요약:"
    echo "======================================"
    echo "1. ROS 환경: source /opt/ros/humble/setup.bash"
    echo "2. 빌드: cd ROS_action && colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference"
    echo "3. 환경: source install/setup.bash"
    echo "4. 카메라: ros2 run camera_pub usb_camera_service_server"
    echo "5. 추론: ros2 run mobile_vla_package robovlms_inference"
    echo "6. 제어: ros2 run robot_control robot_control_node"
    echo "7. 전체: ros2 launch mobile_vla_package robovlms_system.launch.py"
    echo ""
    echo "💡 팁: 각 명령어는 별도 터미널에서 실행하세요!"
}

# 스크립트 실행
main
