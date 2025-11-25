#!/bin/bash
# GPU 서버에서 실행할 동기화 명령어

echo "=========================================="
echo "GPU 서버에서 로봇 서버 데이터 동기화"
echo "=========================================="
echo ""

echo "1️⃣ GPU 서버에서 로봇 서버 접속 테스트:"
echo "   ssh soda@10.109.0.187 'echo 연결성공'"
echo ""

echo "2️⃣ 로봇 서버 파일 개수 확인:"
echo "   ssh soda@10.109.0.187 'ls /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l'"
echo ""

echo "3️⃣ GPU 서버로 파일 가져오기:"
echo "   mkdir -p /tmp/vla_sync"
echo "   rsync -avz --progress \\"
echo "     soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \\"
echo "     /tmp/vla_sync/"
echo ""

echo "4️⃣ 로컬 컴퓨터로 다운로드 (로컬 컴퓨터에서 실행):"
echo "   rsync -avz --progress \\"
echo "     GPU서버사용자@223.194.115.11:/tmp/vla_sync/*.h5 \\"
echo "     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/"
echo ""

