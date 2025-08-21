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
    
    if [ ! -d "/workspace/vla/ROS_action" ]; then
        print_error "ROS 워크스페이스가 마운트되지 않았습니다!"
        print_info "다음 명령어로 컨테이너를 실행하세요:"
        echo "docker run --rm --gpus all \\"
        echo "  -v /usr/local/cuda:/usr/local/cuda:ro \\"
        echo "  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \\"
        echo "  -v \$(pwd)/ROS_action:/workspace/vla/ROS_action \\"
        echo "  -v \$(pwd)/mobile_vla_dataset:/workspace/vla/mobile_vla_dataset \\"
        echo "  --network host --privileged -it \\"
        echo "  mobile_vla:pytorch-2.3.0-ros2-complete bash"
        return 1
    fi
    
    cd /workspace/vla/ROS_action
    
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
    
    cd /workspace/vla/ROS_action
    
    # 의존성 설치
    print_info "의존성 설치 중..."
    rosdep install --from-paths src --ignore-src -r -y
    
    # 빌드
    print_info "ROS 패키지 빌드 중..."
    colcon build --symlink-install
    
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
    setup_ros_workspace
    
    # ROS2 환경 확인
    if ! command -v ros2 &> /dev/null; then
        print_error "ROS2가 설치되지 않았습니다!"
        return 1
    fi
    
    # 워크스페이스에 launch 파일이 있는지 확인
    if [ -f "/workspace/vla/ROS_action/launch/launch_mobile_vla.launch.py" ]; then
        cd /workspace/vla/ROS_action
        ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=false control_node:=false data_collector:=false
    else
        print_warning "launch 파일이 없습니다. 기본 ROS2 노드들을 실행합니다."
        print_info "카메라 테스트를 위해 간단한 토픽을 발행합니다..."
        ros2 topic pub /test_camera std_msgs/String "data: 'Camera test message'" --rate 1
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
    
    # 워크스페이스에 launch 파일이 있는지 확인
    if [ -f "/workspace/vla/ROS_action/launch/launch_mobile_vla.launch.py" ]; then
        cd /workspace/vla/ROS_action
        ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=false data_collector:=false
    else
        print_warning "launch 파일이 없습니다. 기본 ROS2 노드들을 실행합니다."
        print_info "추론 테스트를 위해 간단한 토픽을 구독합니다..."
        ros2 topic echo /vla_inference_result
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
    
    # 워크스페이스에 launch 파일이 있는지 확인
    if [ -f "/workspace/vla/ROS_action/launch/launch_mobile_vla.launch.py" ]; then
        cd /workspace/vla/ROS_action
        ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=true data_collector:=false
    else
        print_warning "launch 파일이 없습니다. 기본 ROS2 노드들을 실행합니다."
        print_info "전체 시스템 테스트를 위해 여러 토픽을 모니터링합니다..."
        ros2 topic list
        ros2 node list
    fi
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
    
    # 워크스페이스에 launch 파일이 있는지 확인
    if [ -f "/workspace/vla/ROS_action/launch/launch_mobile_vla.launch.py" ]; then
        cd /workspace/vla/ROS_action
        ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=false control_node:=false data_collector:=true
    else
        print_warning "launch 파일이 없습니다. 기본 ROS2 노드들을 실행합니다."
        print_info "데이터 수집 테스트를 위해 간단한 토픽을 발행합니다..."
        ros2 topic pub /data_collection std_msgs/String "data: 'Data collection test message'" --rate 0.5
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
        echo "  ROS2 버전: $(ros2 --version | head -1)"
        echo "  ROS2 도메인 ID: $ROS_DOMAIN_ID"
    else
        echo "  ROS2 설치: ❌ 없음"
    fi
    
    if [ -d "/workspace/vla/ROS_action" ]; then
        echo "  ROS 워크스페이스: ✅ 존재"
        if [ -f "/workspace/vla/ROS_action/install/setup.bash" ]; then
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
