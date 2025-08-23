#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 (SOTA 모델)
# 🏆 Kosmos2 + CLIP 하이브리드 (MAE 0.212) 최고 성능 모델 사용

set -e

echo "🚀 RoboVLMs Docker 환경 시작 (SOTA 모델)"
echo "🏆 최고 성능 모델: Kosmos2 + CLIP 하이브리드 (MAE 0.212)"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_robovlms_final 2>/dev/null || true
docker rm mobile_vla_robovlms_final 2>/dev/null || true

# 이미지 빌드 (최신 SOTA 모델 포함)
echo "🔨 Docker 이미지 빌드 중..."
docker build -f Dockerfile.mobile-vla -t mobile_vla:robovlms-final .

# 컨테이너 실행 (Jetson 최적화 설정)
echo "🚀 컨테이너 실행 중..."
docker run -d \
    --name mobile_vla_robovlms_final \
    --network bridge \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    mobile_vla:robovlms-final \
    /bin/bash

echo "✅ 컨테이너가 백그라운드에서 실행 중입니다."
echo ""
echo "🎯 다음 명령어로 컨테이너에 접속하세요:"
echo "   docker exec -it mobile_vla_robovlms_final bash"
echo ""
echo "🏆 SOTA 모델 테스트 명령어:"
echo "   robovlms-system    # 전체 시스템 실행"
echo "   robovlms-inference # 추론 노드만 실행"
echo "   robovlms-test      # 모델 테스트"
echo ""
echo "🚀 최적화 옵션:"
echo "   --ros-args -p optimization_mode:=test    # 테스트 모드 (메모리 최소화)"
echo "   --ros-args -p optimization_mode:=auto    # 자동 최적화 (권장)"
echo "   --ros-args -p optimization_mode:=fp16    # FP16 양자화"
echo ""
echo "📊 모델 정보:"
echo "   🏆 Kosmos2 + CLIP 하이브리드 (MAE 0.212) - 최고 성능"
echo "   🥈 순수 Kosmos2 (MAE 0.222) - 2위 성능"
echo "   ⚡ 예상 성능: 765.7 FPS (FP16)"
echo ""
echo "🎮 키보드 제어:"
echo "   WASD: 로봇 이동"
echo "   Enter: AI 추론 토글"
echo "   R/T: 속도 조절"
echo "   P: 상태 확인"
echo "   H: 도움말"
