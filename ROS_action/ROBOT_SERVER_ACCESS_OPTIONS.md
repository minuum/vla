# 로봇 서버 접속 방법 옵션

## 현재 상황

- 로컬 컴퓨터(billy) → 로봇 서버(10.109.0.187): **Connection timed out**
- 로컬 컴퓨터(billy) → GPU 서버(223.194.115.11): 접속 가능
- 로봇 서버 확인된 IP: 192.168.101.101 (로컬 네트워크)

## 가능한 해결 방법

### 방법 1: 다른 IP 주소 시도

로봇 서버에 여러 IP가 있을 수 있습니다:

```bash
# 로컬 컴퓨터에서 시도
ssh soda@192.168.101.101 "echo '연결 성공'"

# 또는 다른 IP 확인 필요
```

### 방법 2: 다른 포트 사용

SSH가 기본 포트(22)가 아닐 수 있습니다:

```bash
# 포트 2222 시도
ssh -p 2222 soda@10.109.0.187 "echo '연결 성공'"

# 또는 다른 포트들
ssh -p 2200 soda@10.109.0.187 "echo '연결 성공'"
ssh -p 8022 soda@10.109.0.187 "echo '연결 성공'"
```

### 방법 3: VPN 연결 필요

내부 네트워크 접속을 위해 VPN이 필요할 수 있습니다:

```bash
# VPN 연결 후 시도
# VPN 연결 방법은 네트워크 관리자에게 문의
```

### 방법 4: GPU 서버 경유 (ProxyJump)

GPU 서버를 점프 호스트로 사용:

```bash
# 로컬 컴퓨터에서 실행
# ~/.ssh/config 설정
cat >> ~/.ssh/config << EOF
Host gpu-server
    HostName 223.194.115.11
    User GPU서버사용자

Host robot-server
    HostName 10.109.0.187
    User soda
    ProxyJump gpu-server
    # 또는 다른 포트
    # Port 2222
EOF

# 그 후 사용
ssh robot-server "echo '연결 성공'"
```

### 방법 5: GPU 서버에서 직접 접속

GPU 서버가 로봇 서버와 같은 네트워크에 있을 수 있습니다:

```bash
# GPU 서버(223.194.115.11)에 SSH 접속 후
ssh soda@10.109.0.187 "echo '연결 성공'"

# 또는 다른 IP
ssh soda@192.168.101.101 "echo '연결 성공'"
```

### 방법 6: 로봇 서버에서 역방향 전송

로봇 서버에서 GPU 서버나 로컬 컴퓨터로 직접 전송:

```bash
# 로봇 서버에서 실행 (직접 접속 가능한 경우)
rsync -avz --progress \
  /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  GPU서버사용자@223.194.115.11:/tmp/vla_sync/
```

## 확인 사항

### 1. 로봇 서버의 실제 접속 정보 확인

로봇 서버에 직접 물리적으로 접근 가능한 경우:

```bash
# 로봇 서버에서 실행
hostname -I
ip addr show
cat /etc/ssh/sshd_config | grep Port
```

### 2. 네트워크 관리자에게 문의

- 로봇 서버 접속 방법
- VPN 연결 필요 여부
- 특별한 포트나 경로
- 내부 네트워크 접근 방법

### 3. GPU 서버에서 접속 테스트

```bash
# GPU 서버(223.194.115.11)에 접속 후
ping 10.109.0.187
ping 192.168.101.101

# SSH 접속 테스트
ssh -v soda@10.109.0.187 "echo 'test'"
ssh -v soda@192.168.101.101 "echo 'test'"
```

## 임시 해결책

### 옵션 A: USB/외장하드 사용

로봇 서버에 직접 접근 가능한 경우:

```bash
# 로봇 서버에서 USB에 복사
cp /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 /media/usb/

# 로컬 컴퓨터에서 USB에서 복사
cp /media/usb/*.h5 /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 옵션 B: Git LFS 대기

11일 후 Git LFS 대역폭 리셋 대기:

```bash
# 11일 후 로컬 컴퓨터에서
git pull
git lfs pull
```

### 옵션 C: 부분 다운로드

로컬에 있는 237개 파일로 먼저 작업 진행

## 다음 단계

1. **로봇 서버 접속 정보 확인**
   - IP 주소 (10.109.0.187 vs 192.168.101.101)
   - SSH 포트
   - VPN 필요 여부

2. **GPU 서버에서 접속 테스트**
   - GPU 서버가 로봇 서버와 같은 네트워크인지 확인

3. **네트워크 관리자 문의**
   - 로봇 서버 접속 방법
   - 내부 네트워크 접근 권한

