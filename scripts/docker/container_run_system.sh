#!/bin/bash

# 🚀 Container 내부 Mobile VLA System 실행 스크립트
# 컨테이너 내부에서 ROS2 노드들을 실행하는 메뉴 시스템

echo "🚀 Container 내부 Mobile VLA System 실행..."
echo "✅ Container 내부 Mobile VLA System 스크립트 로드 완료"
echo ""

# ROS 환경 설정
setup_ros_env() {
    echo "ℹ️  ROS 워크스페이스 설정 중..."
    source /opt/ros/setup_ros.sh
    
    # ROS 워크스페이스 디렉토리로 이동
    cd /workspace/ros2_ws
    
    # src 디렉토리 확인
    if [ ! -d "src" ]; then
        echo "⚠️  src 디렉토리가 없습니다. 생성 중..."
        mkdir -p src
    fi
    
    # ROS_action 디렉토리 확인
    if [ ! -d "src/ROS_action" ]; then
        echo "⚠️  ROS_action 디렉토리가 없습니다. 생성 중..."
        mkdir -p src/ROS_action
        echo "# ROS_action 패키지" > src/ROS_action/package.xml
    fi
    
    echo "✅ ROS 워크스페이스 설정 완료"
}

# ROS 워크스페이스 빌드
build_workspace() {
    echo "ℹ️  ROS 워크스페이스 빌드 중..."
    echo "ℹ️  의존성 설치 중..."
    
    # rosdep 설치 (src 디렉토리가 있는 경우에만)
    if [ -d "src" ] && [ "$(ls -A src)" ]; then
        rosdep install --from-paths src --ignore-src -r -y || echo "⚠️  rosdep 설치 실패 (계속 진행)"
    else
        echo "⚠️  src 디렉토리가 비어있습니다. 빌드를 건너뜁니다."
        return
    fi
    
    echo "ℹ️  패키지 빌드 중..."
    colcon build --packages-select ROS_action || echo "⚠️  빌드 실패 (계속 진행)"
    
    echo "ℹ️  환경 설정 중..."
    source install/setup.bash
    
    echo "✅ ROS 워크스페이스 빌드 완료"
}

# 카메라 노드 실행
run_camera_node() {
    echo "ℹ️  카메라 노드만 실행"
    setup_ros_env
    
    # 빌드 확인
    if [ ! -f "install/setup.bash" ]; then
        echo "⚠️  ROS 워크스페이스 빌드가 필요합니다. 빌드를 진행하시겠습니까? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            build_workspace
        else
            echo "❌ 빌드가 필요합니다."
            return
        fi
    fi
    
    echo "🚀 카메라 노드 실행 중..."
    # 실제 카메라 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run camera_pub usb_camera_service_server
    # 임시로 테스트 토픽 발행
    ros2 topic pub /camera/image_raw std_msgs/msg/String 'data: Camera Image Data' --rate 1
}

# 추론 노드 실행
run_inference_node() {
    echo "ℹ️  추론 노드만 실행"
    setup_ros_env
    
    # 빌드 확인
    if [ ! -f "install/setup.bash" ]; then
        echo "⚠️  ROS 워크스페이스 빌드가 필요합니다. 빌드를 진행하시겠습니까? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            build_workspace
        else
            echo "❌ 빌드가 필요합니다."
            return
        fi
    fi
    
    echo "🚀 추론 노드 실행 중..."
    # 실제 추론 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run mobile_vla_package robovlms_inference
    # 임시로 테스트 토픽 구독
    ros2 topic echo /camera/image_raw
}

# 전체 시스템 실행
run_full_system() {
    echo "ℹ️  전체 시스템 실행"
    setup_ros_env
    
    # 빌드 확인
    if [ ! -f "install/setup.bash" ]; then
        echo "⚠️  ROS 워크스페이스 빌드가 필요합니다. 빌드를 진행하시겠습니까? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            build_workspace
        else
            echo "❌ 빌드가 필요합니다."
            return
        fi
    fi
    
    echo "🚀 전체 시스템 실행 중..."
    # 실제 전체 시스템 실행 (ROS_action 코드가 있다면)
    # ros2 launch mobile_vla_package launch_mobile_vla.launch.py
    # 임시로 여러 노드 실행
    ros2 topic pub /camera/image_raw std_msgs/msg/String 'data: Camera Image Data' --rate 1 &
    sleep 2
    ros2 topic echo /camera/image_raw &
    sleep 2
    ros2 topic pub /inference/result std_msgs/msg/String 'data: Inference Result' --rate 0.5
}

# 데이터 수집 모드
run_data_collection() {
    echo "ℹ️  데이터 수집 모드"
    setup_ros_env
    
    echo "🚀 데이터 수집 모드 실행 중..."
    # 실제 데이터 수집 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run mobile_vla_package mobile_vla_data_collector
    # 임시로 데이터 수집 시뮬레이션
    echo "📊 데이터 수집 중..."
    for i in {1..10}; do
        echo "데이터 포인트 $i 수집 중..."
        sleep 1
    done
    echo "✅ 데이터 수집 완료"
}

# 시스템 상태 확인
check_system_status() {
    echo "ℹ️  시스템 상태 확인"
    setup_ros_env
    
    echo "📋 시스템 상태:"
    echo "ROS2 버전:"
    ros2 --help | head -3
    echo ""
    echo "ROS2 노드 목록:"
    ros2 node list
    echo ""
    echo "ROS2 토픽 목록:"
    ros2 topic list
    echo ""
    echo "ROS2 서비스 목록:"
    ros2 service list
    echo ""
    echo "워크스페이스 상태:"
    ls -la /workspace/ros2_ws/
    echo ""
    echo "src 디렉토리 상태:"
    ls -la /workspace/ros2_ws/src/ 2>/dev/null || echo "src 디렉토리가 없습니다."
}

# 메인 메뉴
show_menu() {
    echo "🎯 Container 내부 Mobile VLA 실행 메뉴"
    echo "====================================="
    echo "1. 카메라 노드만 실행"
    echo "2. 추론 노드만 실행"
    echo "3. 전체 시스템 실행"
    echo "4. 데이터 수집 모드"
    echo "5. ROS 워크스페이스 빌드"
    echo "6. 시스템 상태 확인"
    echo "7. 종료"
    echo ""
    echo "선택하세요 (1-7): "
}

# 메인 루프
main() {
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            1)
                run_camera_node
                ;;
            2)
                run_inference_node
                ;;
            3)
                run_full_system
                ;;
            4)
                run_data_collection
                ;;
            5)
                setup_ros_env
                build_workspace
                ;;
            6)
                check_system_status
                ;;
            7)
                echo "👋 종료합니다."
                exit 0
                ;;
            *)
                echo "❌ 잘못된 선택입니다. 1-7 중에서 선택하세요."
                ;;
        esac
        
        echo ""
        echo "계속하려면 Enter를 누르세요..."
        read -r
        clear
    done
}

# 스크립트 실행
main
