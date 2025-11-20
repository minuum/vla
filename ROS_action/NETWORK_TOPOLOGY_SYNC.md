# 네트워크 토폴로지 기반 데이터 동기화

## 네트워크 구조

```
로컬 컴퓨터 (billy)
    ↓ (직접 접속 불가)
로봇 서버 (10.109.101.187) - 468개 H5 파일
    ↓ (직접 접속 가능)
GPU 서버 (223.194.115.11)
    ↓ (직접 접속 가능)
로컬 컴퓨터 (billy)
```

## 문제점

- 로컬 컴퓨터에서 로봇 서버로 직접 SSH 접속 불가
- GPU 서버는 로컬 컴퓨터와 로봇 서버 모두 접속 가능

## 해결 방법

### 방법 1: GPU 서버 경유 (2단계 전송) ⭐ 추천

#### 1단계: 로봇 서버 → GPU 서버

```bash
# 로봇 서버(10.109.0.187)에서 실행
rsync -avz --progress \
  /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  사용자명@223.194.115.11:/tmp/vla_sync/
```

#### 2단계: GPU 서버 → 로컬 컴퓨터

```bash
# 로컬 컴퓨터(billy)에서 실행
rsync -avz --progress \
  사용자명@223.194.115.11:/tmp/vla_sync/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 2: GPU 서버에서 원격 실행 (1단계)

```bash
# GPU 서버(223.194.115.11)에서 실행
# 로봇 서버에서 파일을 가져와서 로컬 컴퓨터로 전송
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/

# 그 다음 로컬 컴퓨터로 전송
rsync -avz --progress \
  /tmp/vla_sync/*.h5 \
  billy@로컬컴퓨터IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 3: 로컬 컴퓨터에서 GPU 서버 경유 (SSH ProxyJump)

```bash
# 로컬 컴퓨터(billy)에서 실행
# GPU 서버를 점프 호스트로 사용
rsync -avz --progress -e "ssh -J 사용자명@223.194.115.11" \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 4: GPU 서버에서 스크립트 실행

GPU 서버에서 한 번에 처리:

```bash
# GPU 서버(223.194.115.11)에서 실행
#!/bin/bash
# 1. 로봇 서버에서 파일 가져오기
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/

# 2. 로컬 컴퓨터로 전송
rsync -avz --progress \
  /tmp/vla_sync/*.h5 \
  billy@로컬컴퓨터IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/

# 3. 임시 파일 정리
rm -rf /tmp/vla_sync
```

## 단계별 실행 가이드

### 방법 1 실행 (2단계 전송)

#### Step 1: 로봇 서버 → GPU 서버

```bash
# 로봇 서버(10.109.0.187)에서 실행
# GPU 서버 접속 정보 필요
rsync -avz --progress \
  /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  GPU서버사용자@223.194.115.11:/tmp/vla_sync/
```

#### Step 2: GPU 서버 → 로컬 컴퓨터

```bash
# 로컬 컴퓨터(billy)에서 실행
rsync -avz --progress \
  GPU서버사용자@223.194.115.11:/tmp/vla_sync/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 3 실행 (ProxyJump)

```bash
# 로컬 컴퓨터(billy)에서 실행
# ~/.ssh/config 파일 설정
cat >> ~/.ssh/config << EOF
Host gpu-server
    HostName 223.194.115.11
    User GPU서버사용자

Host robot-server
    HostName 10.109.0.187
    User soda
    ProxyJump gpu-server
EOF

# 그 후 사용
rsync -avz --progress \
  robot-server:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 필요한 정보

1. **GPU 서버 사용자명**: ?
2. **로컬 컴퓨터 IP 주소**: ?
3. **GPU 서버에서 로봇 서버 접속 가능 여부**: 확인 필요

## 확인 명령어

### GPU 서버에서 로봇 서버 접속 테스트

```bash
# GPU 서버(223.194.115.11)에서 실행
ssh soda@10.109.0.187 "echo '연결 성공'"
```

### 로컬 컴퓨터에서 GPU 서버 접속 테스트

```bash
# 로컬 컴퓨터(billy)에서 실행
ssh GPU서버사용자@223.194.115.11 "echo '연결 성공'"
```

## 추천 방법

**방법 1 (2단계 전송)**을 추천합니다:
- 가장 안정적
- 각 단계별로 확인 가능
- 문제 발생 시 재시도 용이

