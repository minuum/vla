#!/bin/bash

# 🚀 Mobile VLA System 실행 스크립트
# PyTorch 2.3.0 + ROS2 + VLA 추론 시스템

set -e

echo "🚀 Mobile VLA System 시작..."

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

# Docker 컨테이너 실행 함수
run_container() {
    local mode=$1
    local container_name="mobile_vla_${mode}"
    
    print_info "Docker 컨테이너 실행 중... (모드: $mode)"
    
    case $mode in
        "main")
                    docker run --rm --gpus all \
            -v /usr/local/cuda:/usr/local/cuda:ro \
            -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
            -v $(pwd)/ROS_action:/workspace/vla/ROS_action \
            -v $(pwd)/mobile_vla_dataset:/workspace/vla/mobile_vla_dataset \
            --network host \
            --privileged \
            -it \
            --name $container_name \
            mobile_vla:pytorch-2.3.0-ros2-complete \
            bash
            ;;
        "dev")
            docker run --rm --gpus all \
                -v /usr/local/cuda:/usr/local/cuda:ro \
                -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
                -v $(pwd)/ROS_action:/workspace/vla/ROS_action \
                -v $(pwd)/mobile_vla_dataset:/workspace/vla/mobile_vla_dataset \
                -v $(pwd)/scripts:/workspace/vla/scripts \
                -v $(pwd)/logs:/workspace/vla/logs \
                --network host \
                --privileged \
                -it \
                --name $container_name \
                mobile_vla:pytorch-2.3.0-ros2-complete \
                bash
            ;;
        "test")
            docker run --rm --gpus all \
                -v /usr/local/cuda:/usr/local/cuda:ro \
                -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
                -v $(pwd)/ROS_action:/workspace/vla/ROS_action \
                -v $(pwd)/mobile_vla_dataset:/workspace/vla/mobile_vla_dataset \
                -v $(pwd)/tests:/workspace/vla/tests \
                --network host \
                --privileged \
                -it \
                --name $container_name \
                mobile_vla:pytorch-2.3.0-ros2-complete \
                bash
            ;;
        *)
            print_error "알 수 없는 모드: $mode"
            exit 1
            ;;
    esac
}

# ROS 시스템 실행 함수
run_ros_system() {
    local mode=$1
    
    print_info "ROS2 시스템 실행 중... (모드: $mode)"
    
    case $mode in
        "camera_only")
            print_info "카메라 노드만 실행"
            cd /workspace/vla/ROS_action
            source install/setup.bash
            ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=false control_node:=false data_collector:=false
            ;;
        "inference_only")
            print_info "추론 노드만 실행"
            cd /workspace/vla/ROS_action
            source install/setup.bash
            ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=false data_collector:=false
            ;;
        "full_system")
            print_info "전체 시스템 실행"
            cd /workspace/vla/ROS_action
            source install/setup.bash
            ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=true data_collector:=false
            ;;
        "data_collection")
            print_info "데이터 수집 모드"
            cd /workspace/vla/ROS_action
            source install/setup.bash
            ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=false control_node:=false data_collector:=true
            ;;
        *)
            print_error "알 수 없는 ROS 모드: $mode"
            exit 1
            ;;
    esac
}

# 메인 메뉴
show_menu() {
    echo ""
    echo "🎯 Mobile VLA System 실행 메뉴"
    echo "================================"
    echo "1. Docker 컨테이너 실행"
    echo "2. ROS2 시스템 실행"
    echo "3. 전체 시스템 (Docker + ROS)"
    echo "4. 테스트 실행"
    echo "5. 종료"
    echo ""
    read -p "선택하세요 (1-5): " choice
}

# Docker 메뉴
show_docker_menu() {
    echo ""
    echo "🐳 Docker 컨테이너 모드"
    echo "======================"
    echo "1. 메인 모드 (실행용)"
    echo "2. 개발 모드 (디버깅용)"
    echo "3. 테스트 모드"
    echo "4. 뒤로가기"
    echo ""
    read -p "선택하세요 (1-4): " docker_choice
    
    case $docker_choice in
        1) run_container "main" ;;
        2) run_container "dev" ;;
        3) run_container "test" ;;
        4) return ;;
        *) print_error "잘못된 선택" ;;
    esac
}

# ROS 메뉴
show_ros_menu() {
    echo ""
    echo "🤖 ROS2 시스템 모드"
    echo "=================="
    echo "1. 카메라만 실행"
    echo "2. 추론만 실행"
    echo "3. 전체 시스템"
    echo "4. 데이터 수집"
    echo "5. 뒤로가기"
    echo ""
    read -p "선택하세요 (1-5): " ros_choice
    
    case $ros_choice in
        1) run_ros_system "camera_only" ;;
        2) run_ros_system "inference_only" ;;
        3) run_ros_system "full_system" ;;
        4) run_ros_system "data_collection" ;;
        5) return ;;
        *) print_error "잘못된 선택" ;;
    esac
}

# 전체 시스템 실행
run_full_system() {
    print_info "전체 시스템 실행 중..."
    
    # Docker Compose로 실행
    if command -v docker-compose &> /dev/null; then
        print_info "Docker Compose로 전체 시스템 실행"
        docker-compose up -d mobile-vla
        docker-compose exec mobile-vla bash -c "
            cd /workspace/vla/ROS_action
            source install/setup.bash
            ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=true data_collector:=false
        "
    else
        print_warning "Docker Compose 없음 - 수동 실행"
        run_container "main"
    fi
}

# 테스트 실행
run_tests() {
    print_info "시스템 테스트 실행 중..."
    
    # PyTorch 테스트
    print_info "PyTorch 2.3.0 테스트..."
    docker run --rm --gpus all \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
        mobile_vla:pytorch-2.3.0-ros2-complete \
        python3 -c "
import torch
print(f'🚀 PyTorch: {torch.__version__}')
print(f'🔥 CUDA Available: {torch.cuda.is_available()}')
print(f'💪 Device Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = torch.mm(x, y)
    print(f'✅ GPU 연산 성공: {z.shape}')
"
    
    # ROS 노드 테스트
    print_info "ROS 노드 테스트..."
    cd ROS_action
    source install/setup.bash
    ros2 node list
}

# 메인 루프
main() {
    while true; do
        show_menu
        
        case $choice in
            1) show_docker_menu ;;
            2) show_ros_menu ;;
            3) run_full_system ;;
            4) run_tests ;;
            5) 
                print_success "시스템 종료"
                exit 0
                ;;
            *) 
                print_error "잘못된 선택"
                ;;
        esac
    done
}

# 스크립트 시작
print_success "Mobile VLA System 스크립트 로드 완료"
main
