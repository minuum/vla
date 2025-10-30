#!/bin/bash

# =============================================================================
# 🚀 Jetson Orin NX Mobile VLA Docker 환경 설정 스크립트
# JetPack 6.0 (L4T R36.4) 최적화 버전
# =============================================================================

set -e

# 색상 코드 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

# 헤더 출력
clear
log_header "==============================================================================="
log_header "🚀 Jetson Orin NX Mobile VLA Docker 환경 설정"
log_header "   JetPack 6.0 (L4T R36.4) 최적화 버전"
log_header "==============================================================================="
echo

# 1️⃣ 시스템 정보 확인
log_info "시스템 정보 확인 중..."
echo "🖥️  호스트 OS: $(lsb_release -d | cut -f2)"
echo "🔧 커널: $(uname -r)"
echo "🏗️  아키텍처: $(uname -m)"

# Jetson 정보 확인
if [ -f "/etc/nv_tegra_release" ]; then
    echo "📟 Jetson: $(cat /etc/nv_tegra_release)"
else
    log_warning "Jetson 시스템이 아닐 수 있습니다"
fi

# CUDA 버전 확인
if command -v nvcc &> /dev/null; then
    echo "🎯 CUDA: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
else
    log_warning "CUDA가 설치되지 않았습니다"
fi

echo

# 2️⃣ Docker 및 NVIDIA Container Runtime 확인
log_info "Docker 환경 확인 중..."

if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다. 먼저 Docker를 설치해주세요."
    exit 1
fi

echo "🐳 Docker: $(docker --version)"

# NVIDIA Container Runtime 확인
if docker info | grep -q nvidia; then
    log_success "NVIDIA Container Runtime 감지됨"
else
    log_warning "NVIDIA Container Runtime이 설정되지 않았을 수 있습니다"
fi

# Docker 권한 확인
if groups | grep -q docker; then
    log_success "Docker 그룹 권한 있음"
else
    log_warning "Docker 그룹 권한이 없습니다. 다음 명령어로 추가하세요:"
    echo "  sudo usermod -aG docker $USER"
    echo "  그리고 로그아웃 후 다시 로그인하세요."
fi

echo

# 3️⃣ 필요한 디렉토리 생성
log_info "Docker 볼륨 디렉토리 생성 중..."

mkdir -p docker_volumes/cache
mkdir -p docker_volumes/dataset  
mkdir -p docker_volumes/logs

log_success "볼륨 디렉토리 생성 완료"
echo

# 4️⃣ 환경 변수 설정 파일 생성
log_info ".env 파일 생성 중..."

cat > .env << EOF
# Mobile VLA Docker 환경 변수
COMPOSE_PROJECT_NAME=mobile-vla
COMPOSE_FILE=docker-compose.jetson.yml

# ROS2 설정
ROS_DOMAIN_ID=42

# NVIDIA 설정
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# 디스플레이 설정 (GUI 지원)
DISPLAY=${DISPLAY:-:0}

# 데이터 경로
MOBILE_VLA_DATA_DIR=./mobile_vla_dataset
MOBILE_VLA_CACHE_DIR=./docker_volumes/cache
MOBILE_VLA_LOG_DIR=./docker_volumes/logs
EOF

log_success ".env 파일 생성 완료"
echo

# 5️⃣ 빠른 실행 스크립트들 생성
log_info "편의 스크립트 생성 중..."

# Docker 빌드 스크립트
cat > docker-build.sh << 'EOF'
#!/bin/bash
echo "🔨 Mobile VLA Jetson Docker 이미지 빌드 중..."
docker-compose -f docker-compose.jetson.yml build --no-cache mobile-vla
echo "✅ 빌드 완료!"
EOF

# Docker 실행 스크립트
cat > docker-run.sh << 'EOF'
#!/bin/bash
echo "🚀 Mobile VLA Jetson Docker 컨테이너 시작 중..."

# X11 권한 설정 (GUI 지원)
xhost +local:docker

# 컨테이너 시작
docker-compose -f docker-compose.jetson.yml up -d mobile-vla

echo "✅ 컨테이너 시작 완료!"
echo "📋 유용한 명령어:"
echo "   docker exec -it mobile_vla_jetson bash           # 컨테이너 접속"
echo "   docker exec -it mobile_vla_jetson vla-camera     # CSI 카메라 시작"
echo "   docker exec -it mobile_vla_jetson vla-collect    # 데이터 수집 시작"
echo "   docker-compose -f docker-compose.jetson.yml logs # 로그 확인"
echo "   docker-compose -f docker-compose.jetson.yml down # 컨테이너 중지"
EOF

# Docker 중지 스크립트
cat > docker-stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Mobile VLA Jetson Docker 컨테이너 중지 중..."
docker-compose -f docker-compose.jetson.yml down
echo "✅ 컨테이너 중지 완료!"
EOF

# 모니터링 시작 스크립트
cat > docker-monitor.sh << 'EOF'
#!/bin/bash
echo "📊 Mobile VLA 모니터링 서비스 시작 중..."
docker-compose -f docker-compose.jetson.yml --profile monitoring up -d
echo "✅ 모니터링 서비스 시작 완료!"
echo "📊 모니터링 로그 확인: docker logs -f mobile_vla_monitoring"
EOF

# 실행 권한 부여
chmod +x docker-build.sh docker-run.sh docker-stop.sh docker-monitor.sh

log_success "편의 스크립트 생성 완료"
echo

# 6️⃣ CSI 카메라 권한 확인
log_info "CSI 카메라 권한 확인 중..."

if [ -c "/dev/video0" ]; then
    log_success "/dev/video0 디바이스 존재함"
    ls -la /dev/video* | head -3
else
    log_warning "/dev/video0 디바이스가 없습니다"
fi

# nvargus-daemon 상태 확인
if systemctl is-active --quiet nvargus-daemon; then
    log_success "nvargus-daemon 실행 중"
else
    log_warning "nvargus-daemon이 실행되지 않고 있습니다"
fi

echo

# 7️⃣ 테스트 명령어 제공
log_info "테스트 명령어 생성 중..."

cat > test-docker-gpu.sh << 'EOF'
#!/bin/bash
echo "🧪 Docker GPU 지원 테스트..."
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/l4t-base:r36.4.0 \
  python3 -c "
import platform
print(f'🖥️  Platform: {platform.platform()}')
print(f'🏗️  Architecture: {platform.machine()}')

try:
    import torch
    print(f'🔥 PyTorch: {torch.__version__}')
    print(f'🎯 CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'📟 CUDA Device: {torch.cuda.get_device_name(0)}')
        print(f'💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except ImportError:
    print('⚠️  PyTorch not available in base image')

print('✅ Docker GPU 테스트 완료!')
"
EOF

chmod +x test-docker-gpu.sh

log_success "테스트 스크립트 생성 완료"
echo

# 8️⃣ 설정 완료 메시지
log_header "==============================================================================="
log_success "🎉 Mobile VLA Jetson Docker 환경 설정 완료!"
log_header "==============================================================================="
echo

echo "📋 다음 단계:"
echo "   1️⃣  Docker 이미지 빌드:     ./docker-build.sh"
echo "   2️⃣  GPU 테스트:             ./test-docker-gpu.sh"
echo "   3️⃣  컨테이너 시작:          ./docker-run.sh"
echo "   4️⃣  컨테이너 접속:          docker exec -it mobile_vla_jetson bash"
echo "   5️⃣  CSI 카메라 테스트:       docker exec -it mobile_vla_jetson vla-camera"
echo "   6️⃣  Mobile VLA 데이터 수집:  docker exec -it mobile_vla_jetson vla-collect"
echo

log_info "문제가 발생하면 다음을 확인하세요:"
echo "   - Docker 그룹 권한: groups | grep docker"
echo "   - NVIDIA Runtime: docker info | grep nvidia"
echo "   - CSI 카메라: ls -la /dev/video*"
echo "   - nvargus-daemon: systemctl status nvargus-daemon"
echo

log_header "Happy Mobile VLA Development! 🚀"