#!/bin/bash
# rsync 테스트 스크립트

echo "=========================================="
echo "rsync 문제 해결 스크립트"
echo "=========================================="
echo ""

REMOTE_IP="192.168.101.101"
REMOTE_USER="soda"
REMOTE_PATH="/home/soda/vla/ROS_action/mobile_vla_dataset"
LOCAL_PATH="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"

echo "1️⃣ SSH 연결 테스트..."
ssh ${REMOTE_USER}@${REMOTE_IP} "echo 'SSH 연결 성공!'"
echo ""

echo "2️⃣ 원격 파일 개수 확인..."
ssh ${REMOTE_USER}@${REMOTE_IP} "ls ${REMOTE_PATH}/*.h5 2>/dev/null | wc -l"
echo ""

echo "3️⃣ 원격 파일 목록 일부 확인..."
ssh ${REMOTE_USER}@${REMOTE_IP} "ls -lh ${REMOTE_PATH}/*.h5 | head -3"
echo ""

echo "4️⃣ 작은 범위로 rsync 테스트 (최신 파일 5개만)..."
rsync -avz --progress \
  ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}/episode_20251119_*.h5 \
  ${LOCAL_PATH}/ \
  2>&1 | head -30
echo ""

echo "5️⃣ 전체 동기화 (위 테스트 성공 후 실행)..."
echo "rsync -avz --progress \\"
echo "  ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}/*.h5 \\"
echo "  ${LOCAL_PATH}/"
echo ""

