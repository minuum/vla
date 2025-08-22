#!/bin/bash

# =============================================================================
# 🚀 Fix Zsh and Install ROS2 Aliases
# zsh 에러를 해결하고 ROS2 별칭을 추가하는 스크립트
# =============================================================================

echo "==============================================================================="
echo "🚀 Fix Zsh and Install ROS2 Aliases"
echo "==============================================================================="

# 1. .zshrc 백업
echo "📋 .zshrc 백업 중..."
cp ~/.zshrc ~/.zshrc.backup.$(date +%Y%m%d_%H%M%S)

# 2. ROS2 별칭 섹션 추가
echo "📋 ROS2 별칭 섹션 추가 중..."

# 별칭 섹션이 이미 있는지 확인
if ! grep -q "# ROS2 Mobile VLA Aliases" ~/.zshrc; then
    echo "" >> ~/.zshrc
    echo "# =============================================================================" >> ~/.zshrc
    echo "# ROS2 Mobile VLA Aliases" >> ~/.zshrc
    echo "# =============================================================================" >> ~/.zshrc
    echo "" >> ~/.zshrc
    
    # ROS2 환경 설정 함수
    echo "# ROS2 환경 설정 함수" >> ~/.zshrc
    echo "setup_ros2_env() {" >> ~/.zshrc
    echo "    export ROS_DOMAIN_ID=42" >> ~/.zshrc
    echo "    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.zshrc
    echo "    source /opt/ros/humble/setup.bash" >> ~/.zshrc
    echo "    if [ -f \"/home/soda/vla/ROS_action/install/setup.bash\" ]; then" >> ~/.zshrc
    echo "        source /home/soda/vla/ROS_action/install/setup.bash" >> ~/.zshrc
    echo "    else" >> ~/.zshrc
    echo "        echo \"⚠️  ROS2 워크스페이스가 빌드되지 않았습니다. cd /home/soda/vla/ROS_action && colcon build 실행하세요.\"" >> ~/.zshrc
    echo "        return 1" >> ~/.zshrc
    echo "    fi" >> ~/.zshrc
    echo "}" >> ~/.zshrc
    echo "" >> ~/.zshrc
    
    # ROS2 노드 실행 함수들
    echo "# ROS2 노드 실행 함수들" >> ~/.zshrc
    echo "run_robot_control() { setup_ros2_env; echo \"🚀 로봇 제어 노드 실행 중...\"; ros2 run mobile_vla_package simple_robot_mover }" >> ~/.zshrc
    echo "run_camera_server() { setup_ros2_env; echo \"📷 카메라 서버 노드 실행 중...\"; ros2 run camera_pub camera_publisher_continuous.py }" >> ~/.zshrc
    echo "run_vla_collector() { setup_ros2_env; echo \"📊 VLA 데이터 수집 노드 실행 중...\"; ros2 run mobile_vla_package vla_collector }" >> ~/.zshrc
    echo "run_vla_inference() { setup_ros2_env; echo \"🧠 VLA 추론 노드 실행 중...\"; ros2 run mobile_vla_package vla_inference_node }" >> ~/.zshrc
    echo "" >> ~/.zshrc
    
    # 시스템 확인 함수들
    echo "# 시스템 확인 함수들" >> ~/.zshrc
    echo "check_system() { setup_ros2_env; echo \"🔍 시스템 상태 확인 중...\"; ros2 node list; echo \"\"; ros2 topic list }" >> ~/.zshrc
    echo "monitor_topic() { setup_ros2_env; local topic=\${1:-\"/cmd_vel\"}; echo \"📡 \$topic 토픽 모니터링 중...\"; ros2 topic echo \"\$topic\" }" >> ~/.zshrc
    echo "" >> ~/.zshrc
    
    # 도움말 함수
    echo "# 도움말 함수" >> ~/.zshrc
    echo "ros2_help() {" >> ~/.zshrc
    echo "    echo \"📋 사용 가능한 ROS2 명령어:\"" >> ~/.zshrc
    echo "    echo \"   run_robot_control    : 로봇 제어 노드 실행\"" >> ~/.zshrc
    echo "    echo \"   run_camera_server    : 카메라 서버 노드 실행\"" >> ~/.zshrc
    echo "    echo \"   run_vla_collector    : VLA 데이터 수집 노드 실행\"" >> ~/.zshrc
    echo "    echo \"   run_vla_inference    : VLA 추론 노드 실행\"" >> ~/.zshrc
    echo "    echo \"   check_system         : 시스템 상태 확인\"" >> ~/.zshrc
    echo "    echo \"   monitor_topic <topic>: 토픽 모니터링\"" >> ~/.zshrc
    echo "    echo \"   ros2_help            : 이 도움말 표시\"" >> ~/.zshrc
    echo "}" >> ~/.zshrc
    echo "" >> ~/.zshrc
    
    echo "✅ ROS2 별칭이 .zshrc에 추가되었습니다!"
else
    echo "⚠️  ROS2 별칭이 이미 .zshrc에 존재합니다."
fi

# 3. 완료 메시지
echo ""
echo "🎉 Zsh Fix and ROS2 Aliases Installation 완료!"
echo "📋 새 터미널을 열거나 다음 명령어를 실행하세요:"
echo "   source ~/.zshrc"
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
