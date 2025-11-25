#!/bin/bash
# 서버 간 데이터 동기화 스크립트
# 사용법: ./QUICK_SYNC_COMMANDS.sh

echo "=========================================="
echo "서버 간 데이터 동기화 가이드"
echo "=========================================="
echo ""

# 로봇 서버 IP 확인 (로봇 서버에서 실행)
echo "1️⃣ 로봇 서버(apexs)에서 IP 주소 확인:"
echo "   hostname -I"
echo "   또는"
echo "   ip addr show | grep 'inet ' | grep -v '127.0.0.1'"
echo ""

# 방법 1: IP 주소 사용
echo "2️⃣ 방법 1: IP 주소 사용 (로컬 서버 billy에서 실행)"
echo "   rsync -avz --progress \\"
echo "     soda@로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \\"
echo "     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/"
echo ""

# 방법 2: SSH Config 설정
echo "3️⃣ 방법 2: SSH Config 설정 (로컬 서버 billy에서 실행)"
echo "   # ~/.ssh/config 파일 편집"
echo "   cat >> ~/.ssh/config << EOF"
echo "   Host apexs"
echo "       HostName 로봇서버IP주소"
echo "       User soda"
echo "       Port 22"
echo "   EOF"
echo ""

# 방법 3: 역방향 전송
echo "4️⃣ 방법 3: 역방향 전송 (로봇 서버 apexs에서 실행)"
echo "   rsync -avz --progress \\"
echo "     /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \\"
echo "     billy@로컬서버IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/"
echo ""

# 방법 4: 공유 스토리지
echo "5️⃣ 방법 4: 공유 스토리지 사용 (두 서버 모두 접근 가능한 경로)"
echo "   # 예: NFS, CIFS 등"
echo ""

echo "=========================================="
echo "현재 로봇 서버 정보:"
echo "=========================================="
echo "호스트명: apexs"
echo "사용자: soda"
echo "데이터 경로: /home/soda/vla/ROS_action/mobile_vla_dataset"
echo "파일 개수: 468개"
echo "총 크기: 약 12GB"
echo ""

echo "로컬 서버 정보:"
echo "사용자: billy"
echo "데이터 경로: /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"
echo "현재 파일: 237개"
echo "필요 파일: 231개 추가"
echo ""

echo "⚠️  주의: 로봇 서버 IP 주소를 먼저 확인하세요!"

