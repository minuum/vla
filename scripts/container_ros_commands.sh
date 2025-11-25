#!/bin/bash

# 🚀 도커 컨테이너 내부 ROS 명령어 모음
# 컨테이너 환경에 최적화된 명령어들

echo "🚀 도커 컨테이너 내부 ROS 명령어"
echo "================================"
echo ""

# 1. 컨테이너 환경 확인
check_container_env() {
    echo "📋 1. 컨테이너 환경 확인"
    echo "🔍 현재 디렉토리: $(pwd)"
    echo "🔍 컨테이너 ID: $(hostname)"
    echo "🔍 CUDA 상태:"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
    echo "🔍 GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
}

# 2. ROS 환경 설정 (컨테이너용)
setup_ros_env_container() {
    echo "📋 2. ROS 환경 설정 (컨테이너용)"
    echo "source /opt/ros/humble/setup.bash"
    echo "export ROS_DOMAIN_ID=42"
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    echo "export ROS_LOCALHOST_ONLY=1"
    echo ""
}

# 3. 워크스페이스 빌드 (컨테이너용)
build_workspace_container() {
    echo "📋 3. 워크스페이스 빌드 (컨테이너용)"
    echo "cd /workspace/vla/ROS_action"
    echo "rosdep install --from-paths src --ignore-src -r -y"
    echo "colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference"
    echo "source install/setup.bash"
    echo ""
}

# 4. 핵심 노드 실행 명령어 (컨테이너용)
run_core_nodes_container() {
    echo "📋 4. 핵심 노드 실행 명령어 (컨테이너용)"
    echo ""
    echo "📷 카메라 서비스:"
    echo "  # USB 카메라 서비스 (컨테이너 권장)"
    echo "  ros2 run camera_pub usb_camera_service_server"
    echo "  # CSI 카메라 서비스 (Jetson 전용)"
    echo "  ros2 run camera_pub camera_service_server"
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

# 5. 컨테이너 내부 테스트 명령어
test_commands_container() {
    echo "📋 5. 컨테이너 내부 테스트 명령어"
    echo ""
    echo "🧪 시스템 상태:"
    echo "  # ROS 상태 확인"
    echo "  ros2 node list"
    echo "  ros2 topic list"
    echo "  ros2 service list"
    echo "  # GPU 상태 확인"
    echo "  nvidia-smi"
    echo "  # 시스템 리소스 확인"
    echo "  htop"
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

# 6. 컨테이너 내부 발표용 시연
demo_commands_container() {
    echo "📋 6. 컨테이너 내부 발표용 시연"
    echo ""
    echo "🎯 1단계: 환경 설정 (30초)"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  export ROS_DOMAIN_ID=42"
    echo "  cd /workspace/vla/ROS_action"
    echo "  source install/setup.bash"
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

# 7. 컨테이너 문제 해결
troubleshoot_container() {
    echo "📋 7. 컨테이너 문제 해결"
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
    echo "  # 컨테이너 네트워크 확인"
    echo "  ip addr show"
    echo ""
    echo "🔧 권한 문제:"
    echo "  # 카메라 장치 권한 확인"
    echo "  ls -la /dev/video*"
    echo "  # USB 장치 권한 확인"
    echo "  ls -la /dev/bus/usb/"
    echo ""
}

# 메인 함수
main() {
    check_container_env
    setup_ros_env_container
    build_workspace_container
    run_core_nodes_container
    test_commands_container
    demo_commands_container
    troubleshoot_container
    
    echo "🎯 컨테이너 핵심 명령어 요약:"
    echo "======================================"
    echo "1. 환경: source /opt/ros/humble/setup.bash"
    echo "2. 빌드: cd /workspace/vla/ROS_action && colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference"
    echo "3. 환경: source install/setup.bash"
    echo "4. 카메라: ros2 run camera_pub usb_camera_service_server"
    echo "5. 추론: ros2 run mobile_vla_package robovlms_inference"
    echo "6. 제어: ros2 run robot_control robot_control_node"
    echo "7. 전체: ros2 launch mobile_vla_package robovlms_system.launch.py"
    echo ""
    echo "💡 팁: 컨테이너 내부에서는 ROS_LOCALHOST_ONLY=1 설정이 권장됩니다!"
}

# 스크립트 실행
main
