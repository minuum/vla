#!/bin/bash

# =============================================================================
# 🚀 Install ROS2 Aliases to Host .bashrc
# 호스트의 .bashrc에 ROS2 별칭을 자동으로 추가하는 스크립트
# =============================================================================

echo "==============================================================================="
echo "🚀 Install ROS2 Aliases to Host .bashrc"
echo "==============================================================================="

# 1. .bashrc 백업
echo "📋 .bashrc 백업 중..."
cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)

# 2. ROS2 별칭 섹션 추가
echo "📋 ROS2 별칭 섹션 추가 중..."

# 별칭 섹션이 이미 있는지 확인
if ! grep -q "# ROS2 Mobile VLA Aliases" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# =============================================================================" >> ~/.bashrc
    echo "# ROS2 Mobile VLA Aliases" >> ~/.bashrc
    echo "# =============================================================================" >> ~/.bashrc
    echo "" >> ~/.bashrc
    
    # ROS2 환경 설정 함수
    echo "# ROS2 환경 설정 함수" >> ~/.bashrc
    echo "setup_ros2_env() {" >> ~/.bashrc
    echo "    export ROS_DOMAIN_ID=42" >> ~/.bashrc
    echo "    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc
    echo "    source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "    if [ -f \"/home/soda/vla/ROS_action/install/setup.bash\" ]; then" >> ~/.bashrc
    echo "        source /home/soda/vla/ROS_action/install/setup.bash" >> ~/.bashrc
    echo "    fi" >> ~/.bashrc
    echo "}" >> ~/.bashrc
    echo "" >> ~/.bashrc
    
    # ROS2 노드 실행 함수들
    echo "# ROS2 노드 실행 함수들" >> ~/.bashrc
    echo "run_robot_control() { setup_ros2_env; echo \"🚀 로봇 제어 노드 실행 중...\"; ros2 run mobile_vla_package simple_robot_mover; }" >> ~/.bashrc
    echo "run_camera_server() { setup_ros2_env; echo \"📷 카메라 서버 노드 실행 중...\"; ros2 run camera_pub camera_publisher_continuous.py; }" >> ~/.bashrc
    echo "run_vla_collector() { setup_ros2_env; echo \"📊 VLA 데이터 수집 노드 실행 중...\"; ros2 run mobile_vla_package vla_collector; }" >> ~/.bashrc
    echo "run_vla_inference() { setup_ros2_env; echo \"🧠 VLA 추론 노드 실행 중...\"; ros2 run mobile_vla_package vla_inference_node; }" >> ~/.bashrc
    echo "" >> ~/.bashrc
    
    # 시스템 확인 함수들
    echo "# 시스템 확인 함수들" >> ~/.bashrc
    echo "check_system() { setup_ros2_env; echo \"🔍 시스템 상태 확인 중...\"; ros2 node list; echo \"\"; ros2 topic list; }" >> ~/.bashrc
    echo "monitor_topic() { setup_ros2_env; local topic=\${1:-\"/cmd_vel\"}; echo \"📡 \$topic 토픽 모니터링 중...\"; ros2 topic echo \"\$topic\"; }" >> ~/.bashrc
    echo "" >> ~/.bashrc
    
    # 도움말 함수
    echo "# 도움말 함수" >> ~/.bashrc
    echo "ros2_help() {" >> ~/.bashrc
    echo "    echo \"📋 사용 가능한 ROS2 명령어:\"" >> ~/.bashrc
    echo "    echo \"   run_robot_control    : 로봇 제어 노드 실행\"" >> ~/.bashrc
    echo "    echo \"   run_camera_server    : 카메라 서버 노드 실행\"" >> ~/.bashrc
    echo "    echo \"   run_vla_collector    : VLA 데이터 수집 노드 실행\"" >> ~/.bashrc
    echo "    echo \"   run_vla_inference    : VLA 추론 노드 실행\"" >> ~/.bashrc
    echo "    echo \"   check_system         : 시스템 상태 확인\"" >> ~/.bashrc
    echo "    echo \"   monitor_topic <topic>: 토픽 모니터링\"" >> ~/.bashrc
    echo "    echo \"   ros2_help            : 이 도움말 표시\"" >> ~/.bashrc
    echo "}" >> ~/.bashrc
    echo "" >> ~/.bashrc
    
    echo "✅ ROS2 별칭이 .bashrc에 추가되었습니다!"
else
    echo "⚠️  ROS2 별칭이 이미 .bashrc에 존재합니다."
fi

# 3. 완료 메시지
echo ""
echo "🎉 ROS2 Aliases Installation 완료!"
echo "📋 새 터미널을 열거나 다음 명령어를 실행하세요:"
echo "   source ~/.bashrc"
echo ""
echo "📋 사용 가능한 명령어:"
echo "   run_robot_control    : 로봇 제어 노드 실행"
echo "   run_camera_server    : 카메라 서버 노드 실행"
echo "   run_vla_collector    : VLA 데이터 수집 노드 실행"
echo "   run_vla_inference    : VLA 추론 노드 실행"
echo "   check_system         : 시스템 상태 확인"
echo "   monitor_topic <topic>: 토픽 모니터링"
echo "   ros2_help            : 도움말 표시"
echo ""
