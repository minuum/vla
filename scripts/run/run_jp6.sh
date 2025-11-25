#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 (JetPack 6 호스트용)
# 🏆 Kosmos2 + CLIP 하이브리드 (MAE 0.212) 최고 성능 모델 사용

set -e

# JP6용 새 이미지 및 컨테이너 이름 설정
IMAGE_NAME="mobile_vla:robovlms-jp6"
CONTAINER_NAME="mobile_vla_robovlms_jp6"
DOCKERFILE="Dockerfile.jp6"

echo "🚀 RoboVLMs Docker 환경 시작 (JetPack 6 호스트)"
echo "🏆 최고 성능 모델: Kosmos2 + CLIP 하이브리드 (MAE 0.212)"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# 이미지 빌드 (JetPack 6 호환 Dockerfile 사용)
echo "🔨 Docker 이미지 빌드 중 ($DOCKERFILE 사용)..."
docker build -f $DOCKERFILE -t $IMAGE_NAME .

# 컨테이너 실행 (JetPack 6 최적화 설정)
echo "🚀 컨테이너 실행 중..."
docker run -d \
    --name $CONTAINER_NAME \
    --network bridge \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    $IMAGE_NAME \
    tail -f /dev/null

echo "✅ 컨테이너가 백그라운드에서 실행 중입니다."
echo ""
echo "🚀 컨테이너에 바로 접속합니다..."
echo ""
# ... (이하 도움말 메시지는 동일) ...
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
echo ""

# 컨테이너에 바로 접속
docker exec -it $CONTAINER_NAME bash