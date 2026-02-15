#!/bin/bash

# 🚀 Container 내부용 Mobile VLA 실행 스크립트
# 컨테이너 내부에서 ROS 시스템을 쉽게 실행할 수 있도록 함

set -e

echo "🚀 Container 내부 Mobile VLA System 실행..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수들
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ROS 워크스페이스 설정
setup_ros_workspace() {
    print_info "ROS 워크스페이스 설정 중..."
    
    # ROS2 환경 소스 및 RMW 구현체 설정
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DISTRO=humble
    export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
    
    # ROS 환경 변수 설정
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
    
    if [ ! -d "/workspace/vla/vla/ROS_action" ]; then
        print_error "ROS 워크스페이스가 마운트되지 않았습니다!"
        print_info "다음 명령어로 컨테이너를 실행하세요:"
        echo "docker run --rm --gpus all \\"
        echo "  -v /usr/local/cuda:/usr/local/cuda:ro \\"
        echo "  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \\"
        echo "  -v \$(pwd):/workspace/vla \\"
        echo "  --network host --privileged -it \\"
        echo "  mobile_vla:pytorch-2.3.0-cuda bash"
        return 1
    fi
    
    cd /workspace/vla/vla/ROS_action
    
    # ROS 환경 설정
    if [ -f "install/setup.bash" ]; then
        source install/setup.bash
        print_success "ROS 워크스페이스 환경 설정 완료"
    else
        print_warning "ROS 워크스페이스 빌드가 필요합니다. 빌드를 진행하시겠습니까? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            build_ros_workspace
        else
            print_info "ROS2 시스템 패키지만 사용합니다."
        fi
    fi
}

# ROS 워크스페이스 빌드
build_ros_workspace() {
    print_info "ROS 워크스페이스 빌드 중..."
    
    # ROS2 환경 소스 및 RMW 구현체 설정
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DISTRO=humble
    export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
    
    # ROS 환경 변수 설정
    export ROS_DOMAIN_ID=42
    export ROS_LOCALHOST_ONLY=0
    export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
    
    cd /workspace/vla/vla/ROS_action
    
    # 기존 빌드 캐시 정리 (완전 삭제)
    print_info "기존 빌드 캐시 정리 중..."
    rm -rf build/ install/ log/
    find . -name "CMakeCache.txt" -delete
    find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # 의존성 설치 (오류 무시)
    print_info "의존성 설치 중..."
    rosdep install --from-paths src --ignore-src -r -y || true
    
    # 빌드 (오류 무시하고 계속, 병렬 빌드 비활성화)
    print_info "ROS 패키지 빌드 중..."
    colcon build --symlink-install --continue-on-error --parallel-workers 1
    
    # 환경 설정
    source install/setup.bash
    print_success "ROS 워크스페이스 빌드 완료"
}

# 메뉴 표시
show_menu() {
    echo ""
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
    read -p "선택하세요 (1-7): " choice
}

# 카메라 노드 실행
run_camera_only() {
    print_info "카메라 노드만 실행"
    
    # ROS2 환경 설정
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=42
    
    # ROS2 환경 확인
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2가 설치되지 않았습니다!"
        return 1
    fi
    
    print_success "ROS2 환경 설정 완료"
    
    # 카메라 노드 실행 (Python 스크립트 직접 실행)
    print_info "카메라 테스트 스크립트를 실행합니다..."
    cd /workspace/vla
    
    # 백그라운드에서 실행
    python3 kosmos_camera_test.py &
    CAMERA_PID=$!
    print_success "카메라 노드가 시작되었습니다 (PID: $CAMERA_PID)"
    
    # 3초 대기 후 상태 확인
    sleep 3
    if kill -0 $CAMERA_PID 2>/dev/null; then
        print_success "✅ 카메라 노드가 정상적으로 실행 중입니다"
        print_info "노드를 종료하려면: kill $CAMERA_PID"
    else
        print_error "❌ 카메라 노드 실행에 실패했습니다"
    fi
}

# 추론 노드 실행
run_inference_only() {
    print_info "추론 노드만 실행"
    setup_ros_workspace
    
    # ROS2 환경 확인
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2가 설치되지 않았습니다!"
        return 1
    fi
    
    # 추론 노드 직접 실행
    print_info "추론 노드를 직접 실행합니다..."
    cd /workspace/vla/vla/ROS_action
    source install/setup.bash
    
    # 추론 노드 실행
    ros2 run mobile_vla_package vla_inference_node &
    INFERENCE_PID=$!
    print_success "추론 노드가 시작되었습니다 (PID: $INFERENCE_PID)"
    
    # 5초 대기 후 상태 확인
    sleep 5
    if kill -0 $INFERENCE_PID 2>/dev/null; then
        print_success "추론 노드가 정상적으로 실행 중입니다"
    else
        print_error "추론 노드 실행에 실패했습니다"
    fi
}

# 전체 시스템 실행
run_full_system() {
    print_info "전체 시스템 실행"
    setup_ros_workspace
    
    # ROS2 환경 확인
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2가 설치되지 않았습니다!"
        return 1
    fi
    
    # 전체 시스템 노드들 직접 실행
    print_info "전체 시스템 노드들을 직접 실행합니다..."
    cd /workspace/vla/vla/ROS_action
    source install/setup.bash
    
    # 카메라 노드 실행
    print_info "카메라 노드 시작 중..."
    ros2 run camera_pub camera_publisher_continuous &
    CAMERA_PID=$!
    
    # 3초 대기
    sleep 3
    
    # 추론 노드 실행
    print_info "추론 노드 시작 중..."
    ros2 run mobile_vla_package vla_inference_node &
    INFERENCE_PID=$!
    
    # 3초 대기
    sleep 3
    
    # 로봇 제어 노드 실행
    print_info "로봇 제어 노드 시작 중..."
    ros2 run mobile_vla_package robot_control_node &
    CONTROL_PID=$!
    
    print_success "전체 시스템이 시작되었습니다:"
    print_success "  - 카메라 노드 (PID: $CAMERA_PID)"
    print_success "  - 추론 노드 (PID: $INFERENCE_PID)"
    print_success "  - 로봇 제어 노드 (PID: $CONTROL_PID)"
    
    # 5초 대기 후 상태 확인
    sleep 5
    print_info "시스템 상태 확인 중..."
    ros2 node list
    ros2 topic list
}

# 데이터 수집 모드
run_data_collection() {
    print_info "데이터 수집 모드"
    setup_ros_workspace
    
    # ROS2 환경 확인
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2가 설치되지 않았습니다!"
        return 1
    fi
    
    # 데이터 수집 노드 직접 실행
    print_info "데이터 수집 노드를 직접 실행합니다..."
    cd /workspace/vla/vla/ROS_action
    source install/setup.bash
    
    # 데이터 수집 노드 실행
    ros2 run mobile_vla_package mobile_vla_data_collector &
    COLLECTOR_PID=$!
    print_success "데이터 수집 노드가 시작되었습니다 (PID: $COLLECTOR_PID)"
    
    # 5초 대기 후 상태 확인
    sleep 5
    if kill -0 $COLLECTOR_PID 2>/dev/null; then
        print_success "데이터 수집 노드가 정상적으로 실행 중입니다"
    else
        print_error "데이터 수집 노드 실행에 실패했습니다"
    fi
}
# 시스템 상태 확인
check_system_status() {
    print_info "시스템 상태 확인 중..."
    
    echo ""
    echo "🔍 PyTorch & CUDA 상태:"
    python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
print(f'  Device Count: {torch.cuda.device_count()}')
"
    
    echo ""
    echo "🔍 ROS2 환경:"
    if command -v ros2 &> /dev/null; then
        echo "  ROS2 설치: ✅ 완료"
        source /opt/ros/humble/setup.bash
        echo "  ROS2 버전: $(ros2 --help | head -1)"
        echo "  ROS2 도메인 ID: $ROS_DOMAIN_ID"
    else
        echo "  ROS2 설치: ❌ 없음"
    fi
    
    if [ -d "/workspace/vla/vla/ROS_action" ]; then
        echo "  ROS 워크스페이스: ✅ 존재"
        if [ -f "/workspace/vla/vla/ROS_action/install/setup.bash" ]; then
            echo "  ROS 워크스페이스 빌드: ✅ 완료"
        else
            echo "  ROS 워크스페이스 빌드: ❌ 필요"
        fi
    else
        echo "  ROS 워크스페이스: ❌ 없음"
    fi
    
    echo ""
    echo "🔍 필수 패키지:"
    python3 -c "
try:
    import transformers
    print(f'  Transformers: ✅ {transformers.__version__}')
except ImportError:
    print('  Transformers: ❌ 없음')

try:
    import cv2
    print(f'  OpenCV: ✅ {cv2.__version__}')
except ImportError:
    print('  OpenCV: ❌ 없음')

try:
    import h5py
    print(f'  HDF5: ✅ {h5py.__version__}')
except ImportError:
    print('  HDF5: ❌ 없음')
"
    
    echo ""
    echo "🔍 ROS2 노드 및 토픽:"
    if command -v ros2 &> /dev/null; then
        source /opt/ros/humble/setup.bash
        echo "  활성 노드:"
        ros2 node list 2>/dev/null || echo "    (노드 없음)"
        echo "  활성 토픽:"
        ros2 topic list 2>/dev/null || echo "    (토픽 없음)"
    fi
}

# 메인 루프
main() {
    print_success "Container 내부 Mobile VLA System 스크립트 로드 완료"
    
    # ROS2 환경 설정 강화
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DISTRO=humble
    export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
    
    while true; do
        show_menu
        
        case $choice in
            1) run_camera_only ;;
            2) run_inference_only ;;
            3) run_full_system ;;
            4) run_data_collection ;;
            5) build_ros_workspace ;;
            6) check_system_status ;;
            7) 
                print_success "시스템 종료"
                exit 0
                ;;
            *) 
                print_error "잘못된 선택"
                ;;
        esac
        
        echo ""
        read -p "계속하려면 Enter를 누르세요..."
    done
}

# 스크립트 시작
main
