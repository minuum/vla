#!/bin/bash

# Mobile VLA RoboVLMs Docker Manager
# 통합된 Docker 관리 스크립트

set -e

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

# 도움말
show_help() {
    echo "Mobile VLA RoboVLMs Docker Manager"
    echo ""
    echo "사용법: $0 [명령어]"
    echo ""
    echo "명령어:"
    echo "  build     - Docker 이미지 빌드"
    echo "  run       - Docker 컨테이너 실행"
    echo "  stop      - Docker 컨테이너 정지"
    echo "  status    - 컨테이너 상태 확인"
    echo "  logs      - 로그 확인"
    echo "  shell     - 컨테이너 쉘 접속"
    echo "  clean     - 정리 (컨테이너, 이미지 삭제)"
    echo "  help      - 이 도움말 표시"
}

# Docker 이미지 빌드
build_image() {
    log_info "Docker 이미지 빌드 시작..."
    
    if [ ! -f "docker/Dockerfile.mobile-vla" ]; then
        log_error "docker/Dockerfile.mobile-vla 파일을 찾을 수 없습니다."
        exit 1
    fi
    
    docker build -t mobile_vla:robovlms -f docker/Dockerfile.mobile-vla .
    
    if [ $? -eq 0 ]; then
        log_success "Docker 이미지 빌드 완료!"
    else
        log_error "Docker 이미지 빌드 실패!"
        exit 1
    fi
}

# Docker 컨테이너 실행
run_container() {
    log_info "Docker 컨테이너 실행..."
    
    # 기존 컨테이너 확인
    if docker ps -q -f name=mobile_vla_container | grep -q .; then
        log_warning "이미 실행 중인 컨테이너가 있습니다."
        read -p "기존 컨테이너를 중지하고 새로 실행하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            stop_container
        else
            log_info "실행을 취소했습니다."
            return
        fi
    fi
    
    # 컨테이너 실행
    docker run -d \
        --name mobile_vla_container \
        --runtime nvidia \
        --network host \
        --privileged \
        -v /dev/video0:/dev/video0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd):/workspace/vla \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_VISIBLE_DEVICES=all \
        mobile_vla:robovlms
    
    if [ $? -eq 0 ]; then
        log_success "컨테이너가 성공적으로 실행되었습니다!"
        log_info "컨테이너 ID: $(docker ps -q -f name=mobile_vla_container)"
    else
        log_error "컨테이너 실행 실패!"
        exit 1
    fi
}

# Docker 컨테이너 정지
stop_container() {
    log_info "Docker 컨테이너 정지..."
    
    if docker ps -q -f name=mobile_vla_container | grep -q .; then
        docker stop mobile_vla_container
        log_success "컨테이너가 정지되었습니다."
    else
        log_warning "실행 중인 컨테이너가 없습니다."
    fi
}

# 컨테이너 상태 확인
check_status() {
    log_info "컨테이너 상태 확인..."
    
    if docker ps -a -f name=mobile_vla_container | grep -q mobile_vla_container; then
        docker ps -a -f name=mobile_vla_container
    else
        log_warning "mobile_vla_container가 존재하지 않습니다."
    fi
}

# 로그 확인
show_logs() {
    log_info "컨테이너 로그 확인..."
    
    if docker ps -q -f name=mobile_vla_container | grep -q .; then
        docker logs mobile_vla_container
    else
        log_error "실행 중인 컨테이너가 없습니다."
    fi
}

# 컨테이너 쉘 접속
access_shell() {
    log_info "컨테이너 쉘 접속..."
    
    if docker ps -q -f name=mobile_vla_container | grep -q .; then
        docker exec -it mobile_vla_container /bin/bash
    else
        log_error "실행 중인 컨테이너가 없습니다."
    fi
}

# 정리
cleanup() {
    log_info "Docker 정리 시작..."
    
    # 컨테이너 정지 및 삭제
    if docker ps -a -q -f name=mobile_vla_container | grep -q .; then
        docker stop mobile_vla_container 2>/dev/null || true
        docker rm mobile_vla_container 2>/dev/null || true
        log_success "컨테이너 삭제 완료"
    fi
    
    # 이미지 삭제
    if docker images mobile_vla:robovlms | grep -q mobile_vla; then
        docker rmi mobile_vla:robovlms
        log_success "이미지 삭제 완료"
    fi
    
    log_success "정리 완료!"
}

# 메인 로직
case "${1:-help}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    shell)
        access_shell
        ;;
    clean)
        cleanup
        ;;
    help|*)
        show_help
        ;;
esac
