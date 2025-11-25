#!/bin/bash
# =============================================================================
# 🚀 Mobile VLA 시스템 실행 스크립트
# PyTorch 2.3.0 + ROS2 + VLA 추론 시스템
# =============================================================================

set -e  # 오류 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 헤더 출력
echo "=============================================================================="
echo "🚀 Mobile VLA 시스템 실행 스크립트"
echo "=============================================================================="
echo "📋 시스템 구성:"
echo "   • PyTorch 2.3.0 (CUDA 가속)"
echo "   • ROS2 Humble"
echo "   • VLA 추론 노드"
echo "   • 로봇 제어 노드"
echo "   • 카메라 서비스 노드"
echo "=============================================================================="

# 1. 환경 확인
log_info "🔍 시스템 환경 확인 중..."

# Docker 확인
if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다."
    exit 1
fi

# NVIDIA Container Toolkit 확인
if ! docker info | grep -q "nvidia"; then
    log_warning "NVIDIA Container Toolkit이 설정되지 않았습니다."
    log_warning "GPU 가속이 제한될 수 있습니다."
fi

# X11 권한 확인
if [ -z "$DISPLAY" ]; then
    log_warning "DISPLAY 환경 변수가 설정되지 않았습니다."
    log_warning "GUI 기능이 제한될 수 있습니다."
else
    log_success "X11 디스플레이 설정 확인됨: $DISPLAY"
fi

# 2. 이미지 빌드 확인
log_info "🔍 Docker 이미지 확인 중..."

if ! docker images | grep -q "mobile_vla:pytorch-2.3.0-cuda"; then
    log_warning "Mobile VLA 이미지가 없습니다. 빌드를 시작합니다..."
    
    # Dockerfile 확인
    if [ ! -f "docker/Dockerfile.mobile-vla" ]; then
        log_error "docker/Dockerfile.mobile-vla이 없습니다."
        exit 1
    fi
    
    # 이미지 빌드
    log_info "🔨 Docker 이미지 빌드 중... (시간이 오래 걸릴 수 있습니다)"
    docker build -t mobile_vla:pytorch-2.3.0-cuda -f docker/Dockerfile.mobile-vla .
    
    if [ $? -eq 0 ]; then
        log_success "Docker 이미지 빌드 완료"
    else
        log_error "Docker 이미지 빌드 실패"
        exit 1
    fi
else
    log_success "Mobile VLA 이미지 확인됨"
fi

# 3. 시스템 실행
log_info "🚀 Mobile VLA 시스템 시작 중..."

# 기존 컨테이너 정리
log_info "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_main 2>/dev/null || true
docker rm mobile_vla_main 2>/dev/null || true

# X11 권한 설정
log_info "🖥️ X11 권한 설정 중..."
xhost +local:docker 2>/dev/null || log_warning "X11 권한 설정 실패"

# 컨테이너 실행
log_info "🐳 Docker 컨테이너 실행 중..."

docker run -d \
    --name mobile_vla_main \
    --runtime=nvidia \
    --network=host \
    --privileged \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTHORITY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
    -v $(pwd)/vla:/workspace/vla \
    -v $(pwd)/ROS_action:/workspace/vla/ROS_action \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/video0:/dev/video0:rw \
    -v /dev/ttyUSB0:/dev/ttyUSB0:rw \
    -v /dev/ttyUSB1:/dev/ttyUSB1:rw \
    -v /dev/input:/dev/input:ro \
    -p 8888:8888 \
    -p 6006:6006 \
    mobile_vla:pytorch-2.3.0-cuda

if [ $? -eq 0 ]; then
    log_success "컨테이너 실행 성공"
else
    log_error "컨테이너 실행 실패"
    exit 1
fi

# 4. 시스템 초기화 대기
log_info "⏳ 시스템 초기화 대기 중... (30초)"
sleep 30

# 5. 시스템 상태 확인
log_info "🔍 시스템 상태 확인 중..."

# 컨테이너 상태 확인
if docker ps | grep -q "mobile_vla_main"; then
    log_success "컨테이너 실행 중"
else
    log_error "컨테이너가 실행되지 않았습니다."
    docker logs mobile_vla_main
    exit 1
fi

# 헬스체크 실행
log_info "🏥 헬스체크 실행 중..."
docker exec mobile_vla_main /usr/local/bin/healthcheck.sh

if [ $? -eq 0 ]; then
    log_success "시스템 헬스체크 통과"
else
    log_warning "시스템 헬스체크 실패 (일부 기능 제한될 수 있음)"
fi

# 6. ROS2 환경 설정 및 노드 실행
log_info "🤖 ROS2 노드 실행 준비 중..."

# ROS2 환경 설정
docker exec mobile_vla_main bash -c "
    source /opt/ros/humble/setup.bash
    source /workspace/vla/ROS_action/install/setup.bash
    echo '✅ ROS2 환경 설정 완료'
"

# 7. 사용법 안내
echo ""
echo "=============================================================================="
echo "🎉 Mobile VLA 시스템 실행 완료!"
echo "=============================================================================="
echo "📋 사용 가능한 명령어:"
echo ""
echo "🔍 시스템 상태 확인:"
echo "   docker logs mobile_vla_main"
echo "   docker exec mobile_vla_main nvidia-smi"
echo ""
echo "🤖 ROS2 노드 실행:"
echo "   docker exec -it mobile_vla_main bash"
echo "   # 컨테이너 내에서:"
echo "   source /opt/ros/humble/setup.bash"
echo "   source /workspace/vla/ROS_action/install/setup.bash"
echo "   ros2 run camera_pub camera_publisher_continuous"
echo "   ros2 run vla_inference vla_inference_node"
echo "   ros2 run robot_control robot_control_node"
echo ""
echo "🎮 제어 모드:"
echo "   M: 수동 모드 (WASD)"
echo "   V: VLA 자동 모드"
echo "   H: 하이브리드 모드"
echo "   F/G: 속도 조절"
echo ""
echo "🛑 시스템 종료:"
echo "   docker stop mobile_vla_main"
echo "   docker rm mobile_vla_main"
echo ""
echo "📊 모니터링:"
echo "   docker stats mobile_vla_main"
echo "   ros2 topic list"
echo "   ros2 topic echo /vla_inference_result"
echo "=============================================================================="

# 8. 자동 실행 옵션
read -p "🚀 자동으로 모든 노드를 실행하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "🤖 모든 노드 자동 실행 중..."
    
    docker exec -d mobile_vla_main bash -c "
        source /opt/ros/humble/setup.bash
        source /workspace/vla/ROS_action/install/setup.bash
        python3 /workspace/vla/launch_mobile_vla_system.py
    "
    
    log_success "자동 실행 시작됨"
    log_info "로그 확인: docker logs -f mobile_vla_main"
else
    log_info "수동 실행 모드 - 위의 명령어를 사용하여 노드를 실행하세요."
fi

echo ""
log_success "Mobile VLA 시스템이 성공적으로 시작되었습니다! 🚀"
