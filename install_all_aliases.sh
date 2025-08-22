#!/bin/bash

# =============================================================================
# 🚀 Install All ROS2 Aliases
# 호스트와 Docker 컨테이너 모두에 ROS2 별칭을 설치하는 통합 스크립트
# =============================================================================

echo "==============================================================================="
echo "🚀 Install All ROS2 Aliases"
echo "==============================================================================="

# 1. 호스트에 별칭 설치
echo "📋 호스트에 ROS2 별칭 설치 중..."
./install_aliases_host.sh

# 2. Docker 컨테이너 상태 확인
echo ""
echo "🔍 Docker 컨테이너 상태 확인 중..."
if docker ps | grep -q "infallible_elion"; then
    echo "✅ 실행 중인 컨테이너 발견: infallible_elion"
    
    # 3. Docker 컨테이너에 스크립트 복사
    echo "📋 Docker 컨테이너에 스크립트 복사 중..."
    docker cp install_aliases_docker.sh infallible_elion:/workspace/vla/
    docker cp setup_aliases_docker.sh infallible_elion:/workspace/vla/
    
    # 4. Docker 컨테이너에서 별칭 설치
    echo "📋 Docker 컨테이너에 ROS2 별칭 설치 중..."
    docker exec infallible_elion bash -c "cd /workspace/vla && chmod +x install_aliases_docker.sh && ./install_aliases_docker.sh"
    
    echo "✅ Docker 컨테이너 별칭 설치 완료!"
else
    echo "⚠️  실행 중인 컨테이너가 없습니다."
    echo "📋 컨테이너를 먼저 시작하세요:"
    echo "   ./run_mobile_vla_system.sh"
    echo "📋 그 후 다시 이 스크립트를 실행하세요."
fi

# 5. 완료 메시지
echo ""
echo "🎉 All ROS2 Aliases Installation 완료!"
echo ""
echo "📋 사용 방법:"
echo ""
echo "🔹 호스트에서:"
echo "   source ~/.bashrc"
echo "   run_robot_control"
echo "   run_camera_server"
echo "   run_vla_collector"
echo "   run_vla_inference"
echo "   check_system"
echo "   monitor_topic /cmd_vel"
echo "   ros2_help"
echo ""
echo "🔹 Docker 컨테이너에서:"
echo "   docker exec -it infallible_elion bash"
echo "   source ~/.bashrc"
echo "   run_robot_control"
echo "   run_camera_server"
echo "   run_vla_collector"
echo "   run_vla_inference"
echo "   check_system"
echo "   check_container"
echo "   monitor_topic /cmd_vel"
echo "   ros2_help"
echo ""
echo "🎯 이제 간단한 명령어로 ROS2 노드를 실행할 수 있습니다!"
