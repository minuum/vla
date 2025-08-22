#!/bin/bash

# =============================================================================
# 🚀 Enhanced Container Run Menu System
# 도커 컨테이너 내부에서 사용할 완전한 실행 메뉴 시스템
# =============================================================================

# 환경 설정 함수
setup_ros2_env() {
    echo "ℹ️  ROS2 워크스페이스 설정 중..."
    
    # ROS2 환경 설정
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi
    
    # 환경 변수 설정
    export ROS_DOMAIN_ID=42
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    
    # 워크스페이스 설정
    if [ -f "/workspace/vla/ROS_action/install/local_setup.bash" ]; then
        cd /workspace/vla/ROS_action/install && source local_setup.bash
        cd /workspace/vla
    elif [ -f "/workspace/vla/ROS_action/install/setup.bash" ]; then
        cd /workspace/vla/ROS_action/install && source setup.bash
        cd /workspace/vla
    else
        echo "⚠️  ROS 워크스페이스가 없습니다. 빌드가 필요합니다."
        return 1
    fi
    
    echo "✅ ROS2 워크스페이스 환경 설정 완료"
    return 0
}

# 카메라 노드만 실행
run_camera_only() {
    echo "ℹ️  카메라 노드만 실행"
    if setup_ros2_env; then
        echo "ℹ️  카메라 테스트를 위해 간단한 토픽을 발행합니다..."
        timeout 10 ros2 run camera_pub usb_camera_service_server &
        sleep 2
        echo "📸 카메라 서비스 서버가 백그라운드에서 실행 중입니다."
        echo "📋 서비스 확인: ros2 service list | grep usb"
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# 추론 노드만 실행
run_inference_only() {
    echo "ℹ️  추론 노드만 실행"
    if setup_ros2_env; then
        echo "🧠 VLA 추론 노드 실행 중..."
        timeout 10 ros2 run mobile_vla_package vla_inference_node &
        sleep 2
        echo "📊 추론 노드가 백그라운드에서 실행 중입니다."
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# 전체 시스템 실행
run_full_system() {
    echo "ℹ️  전체 시스템 실행"
    if setup_ros2_env; then
        echo "🚀 전체 Mobile VLA 시스템 실행 중..."
        
        # 카메라 서비스 서버 시작
        echo "📷 카메라 서비스 서버 시작..."
        ros2 run camera_pub usb_camera_service_server &
        CAMERA_PID=$!
        sleep 2
        
        # VLA 추론 노드 시작
        echo "🧠 VLA 추론 노드 시작..."
        ros2 run mobile_vla_package vla_inference_node &
        INFERENCE_PID=$!
        sleep 2
        
        # 데이터 수집 노드 시작
        echo "📊 데이터 수집 노드 시작..."
        ros2 run mobile_vla_package vla_collector &
        COLLECTOR_PID=$!
        sleep 2
        
        echo "✅ 전체 시스템이 백그라운드에서 실행 중입니다."
        echo "📋 실행 중인 노드:"
        ros2 node list
        echo ""
        echo "📋 활성 토픽:"
        ros2 topic list
        echo ""
        echo "📋 활성 서비스:"
        ros2 service list
        echo ""
        echo "🛑 시스템 중지하려면: pkill -f 'ros2 run'"
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# 데이터 수집 모드
run_data_collection() {
    echo "ℹ️  데이터 수집 모드"
    if setup_ros2_env; then
        echo "📊 데이터 수집 모드 실행 중..."
        echo "📋 수집할 데이터:"
        echo "   - 카메라 이미지"
        echo "   - 로봇 상태"
        echo "   - 액션 데이터"
        
        # 데이터 수집 노드 실행
        ros2 run mobile_vla_package vla_collector &
        COLLECTOR_PID=$!
        
        echo "✅ 데이터 수집 노드가 실행 중입니다."
        echo "📁 데이터 저장 위치: /workspace/vla/mobile_vla_dataset"
        echo "🛑 중지하려면: kill $COLLECTOR_PID"
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# ROS 워크스페이스 빌드
build_workspace() {
    echo "ℹ️  ROS 워크스페이스 빌드 중..."
    cd /workspace/vla/ROS_action
    
    echo "ℹ️  의존성 설치 중..."
    rosdep install --from-paths src --ignore-src -r -y
    
    echo "ℹ️  ROS 패키지 빌드 중..."
    colcon build
    
    if [ $? -eq 0 ]; then
        echo "✅ ROS 워크스페이스 빌드 완료!"
        echo "📋 빌드된 패키지:"
        ls -la install/
    else
        echo "❌ ROS 워크스페이스 빌드 실패"
    fi
    
    cd /workspace/vla
}

# 시스템 상태 확인
check_system_status() {
    echo "ℹ️  시스템 상태 확인"
    if setup_ros2_env; then
        echo "📋 실행 중인 노드:"
        ros2 node list
        echo ""
        echo "📋 활성 토픽:"
        ros2 topic list
        echo ""
        echo "📋 활성 서비스:"
        ros2 service list
        echo ""
        echo "📋 환경 변수:"
        echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
        echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
        echo "   ROS_DISTRO: $ROS_DISTRO"
        echo ""
        echo "📋 사용 가능한 패키지:"
        ros2 pkg list | grep -E "(camera|mobile_vla)" | head -10
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# 카메라 테스트
test_camera() {
    echo "ℹ️  카메라 테스트"
    if setup_ros2_env; then
        echo "📷 카메라 연결 테스트 중..."
        
        # USB 카메라 서비스 서버 테스트
        echo "🔍 USB 카메라 서비스 서버 테스트..."
        timeout 5 ros2 run camera_pub usb_camera_service_server &
        CAMERA_PID=$!
        sleep 3
        
        # 서비스 확인
        echo "📋 서비스 목록:"
        ros2 service list | grep usb
        
        # 노드 확인
        echo "📋 실행 중인 노드:"
        ros2 node list
        
        # 프로세스 정리
        kill $CAMERA_PID 2>/dev/null
        echo "✅ 카메라 테스트 완료"
    else
        echo "❌ ROS2 환경 설정 실패"
    fi
}

# 네트워크 테스트
test_network() {
    echo "ℹ️  네트워크 테스트"
    echo "📡 ROS2 네트워크 연결 테스트 중..."
    
    # ROS_DOMAIN_ID 확인
    echo "🔍 ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
    
    # 토픽 발행 테스트
    echo "📤 테스트 토픽 발행 중..."
    timeout 3 ros2 topic pub /test_topic std_msgs/msg/String "data: 'Hello ROS2'" &
    sleep 1
    
    # 토픽 확인
    echo "📋 활성 토픽:"
    ros2 topic list
    
    echo "✅ 네트워크 테스트 완료"
}

# 메인 메뉴
show_menu() {
    echo "🚀 Container 내부 Mobile VLA System 실행..."
    echo "✅ Container 내부 Mobile VLA System 스크립트 로드 완료"
    echo ""
    echo "🎯 Container 내부 Mobile VLA 실행 메뉴"
    echo "====================================="
    echo "1. 카메라 노드만 실행"
    echo "2. 추론 노드만 실행"
    echo "3. 전체 시스템 실행"
    echo "4. 데이터 수집 모드"
    echo "5. ROS 워크스페이스 빌드"
    echo "6. 시스템 상태 확인"
    echo "7. 카메라 테스트"
    echo "8. 네트워크 테스트"
    echo "9. 모든 노드 중지"
    echo "0. 종료"
    echo ""
}

# 모든 노드 중지
stop_all_nodes() {
    echo "🛑 모든 ROS2 노드 중지 중..."
    pkill -f "ros2 run"
    pkill -f "camera_publisher"
    pkill -f "vla_collector"
    pkill -f "vla_inference"
    echo "✅ 모든 노드가 중지되었습니다."
}

# 메인 실행 함수
main() {
    while true; do
        show_menu
        read -p "선택하세요 (0-9): " choice
        
        case $choice in
            1)
                run_camera_only
                ;;
            2)
                run_inference_only
                ;;
            3)
                run_full_system
                ;;
            4)
                run_data_collection
                ;;
            5)
                build_workspace
                ;;
            6)
                check_system_status
                ;;
            7)
                test_camera
                ;;
            8)
                test_network
                ;;
            9)
                stop_all_nodes
                ;;
            0)
                echo "👋 Container Run 메뉴를 종료합니다."
                exit 0
                ;;
            *)
                echo "❌ 잘못된 선택입니다. 0-9 중에서 선택하세요."
                ;;
        esac
        
        echo ""
        read -p "계속하려면 Enter를 누르세요..."
        clear
    done
}

# 스크립트 실행
main

