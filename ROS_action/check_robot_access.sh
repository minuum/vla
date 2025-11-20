#!/bin/bash
# 로봇 서버 접속 확인 스크립트

echo "=========================================="
echo "로봇 서버 접속 방법 확인"
echo "=========================================="
echo ""

echo "확인된 로봇 서버 IP:"
echo "  1. 10.109.0.187 (시도했으나 Connection timed out)"
echo "  2. 192.168.101.101 (로컬 네트워크, 확인 필요)"
echo ""

echo "테스트할 명령어:"
echo ""
echo "1. 다른 IP 시도:"
echo "   ssh soda@192.168.101.101 'echo 연결성공'"
echo ""
echo "2. 다른 포트 시도:"
echo "   ssh -p 2222 soda@10.109.0.187 'echo 연결성공'"
echo "   ssh -p 2200 soda@10.109.0.187 'echo 연결성공'"
echo ""
echo "3. GPU 서버에서 접속 테스트:"
echo "   # GPU 서버(223.194.115.11)에 접속 후"
echo "   ssh soda@10.109.0.187 'echo 연결성공'"
echo "   ssh soda@192.168.101.101 'echo 연결성공'"
echo ""
echo "4. VPN 연결 확인:"
echo "   # 내부 네트워크 접속을 위해 VPN 필요할 수 있음"
echo ""

