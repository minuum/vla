#!/bin/bash

# 🚀 핵심 시스템 바로 실행 스크립트
# ROS 환경 설정 후 핵심 노드들을 바로 실행

set -e

echo "🚀 핵심 시스템 바로 실행"
echo "========================"
echo ""

# 1. ROS 환경 설정
echo "📋 1. ROS 환경 설정 중..."
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 2. 워크스페이스 환경 설정
echo "📋 2. 워크스페이스 환경 설정 중..."
cd ./ROS_action
source install/setup.bash
echo "✅ 환경 설정 완료"
echo ""

# 3. 시스템 상태 확인
echo "📋 3. 시스템 상태 확인 중..."
echo "🔍 사용 가능한 패키지:"
ros2 pkg list | grep -E "(camera|mobile_vla|robot|vla)" || echo "⚠️ 패키지 목록 확인 실패"
echo ""

# 4. 핵심 노드 실행
echo "📋 4. 핵심 노드 실행 중..."
echo ""

echo "📷 카메라 서비스 시작 중..."
echo "터미널 1에서 실행: ros2 run camera_pub usb_camera_service_server"
echo ""

echo "🧠 VLA 추론 노드 시작 중..."
echo "터미널 2에서 실행: ros2 run mobile_vla_package robovlms_inference"
echo ""

echo "🤖 로봇 제어 노드 시작 중..."
echo "터미널 3에서 실행: ros2 run robot_control robot_control_node"
echo ""

echo "📊 시스템 모니터링 시작 중..."
echo "터미널 4에서 실행: ros2 run mobile_vla_package robovlms_monitor"
echo ""

# 5. 실행 안내
echo "📋 5. 실행 안내:"
echo "🎯 각 노드를 별도 터미널에서 실행하세요:"
echo ""
echo "터미널 1 (카메라):"
echo "  cd /workspace/vla/ROS_action"
echo "  source install/setup.bash"
echo "  ros2 run camera_pub usb_camera_service_server"
echo ""
echo "터미널 2 (추론):"
echo "  cd /workspace/vla/ROS_action"
echo "  source install/setup.bash"
echo "  ros2 run mobile_vla_package robovlms_inference"
echo ""
echo "터미널 3 (제어):"
echo "  cd /workspace/vla/ROS_action"
echo "  source install/setup.bash"
echo "  ros2 run robot_control robot_control_node"
echo ""
echo "터미널 4 (모니터링):"
echo "  cd /workspace/vla/ROS_action"
echo "  source install/setup.bash"
echo "  ros2 run mobile_vla_package robovlms_monitor"
echo ""

# 6. 테스트 명령어
echo "📋 6. 테스트 명령어:"
echo "🧪 시스템 상태 확인:"
echo "  ros2 node list"
echo "  ros2 topic list"
echo "  ros2 service list"
echo ""
echo "🧪 카메라 테스트:"
echo "  ros2 service call /get_image_service camera_interfaces/srv/GetImage"
echo ""
echo "🧪 추론 테스트:"
echo "  ros2 topic echo /mobile_vla/inference_result"
echo ""
echo "🧪 제어 테스트:"
echo "  ros2 topic echo /cmd_vel"
echo ""

# 7. 전체 시스템 실행 (선택사항)
echo "📋 7. 전체 시스템 실행 (선택사항):"
echo "🚀 모든 노드를 한 번에 실행하려면:"
echo "  ros2 launch mobile_vla_package robovlms_system.launch.py"
echo ""

echo "🎉 핵심 시스템 실행 준비 완료!"
echo "💡 각 터미널에서 위의 명령어들을 실행하세요."
echo ""
