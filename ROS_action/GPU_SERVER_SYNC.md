# GPU 서버에서 로봇 서버 데이터 동기화

## 네트워크 구조

```
로컬 컴퓨터 (billy)
    ↓ SSH 접속
GPU 서버 (223.194.115.11)
    ↓ SSH 접속 가능
로봇 서버 (10.109.0.187) - 468개 H5 파일
```

## 해결 방법: GPU 서버에서 직접 동기화

### 방법 1: GPU 서버에서 로봇 서버로 직접 접속 (가장 간단) ⭐ 추천

```bash
# GPU 서버(223.194.115.11)에 SSH 접속 후 실행
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/

# 그 다음 로컬 컴퓨터로 다운로드
# (로컬 컴퓨터에서 실행)
rsync -avz --progress \
  GPU서버사용자@223.194.115.11:/tmp/vla_sync/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 2: GPU 서버에서 한 번에 처리

```bash
# GPU 서버(223.194.115.11)에 SSH 접속 후 실행
# 로봇 서버 → GPU 서버 → 로컬 컴퓨터 (한 번에)

# 1단계: 로봇 서버에서 GPU 서버로
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/

# 2단계: GPU 서버에서 로컬 컴퓨터로 (로컬 컴퓨터 IP 필요)
rsync -avz --progress \
  /tmp/vla_sync/*.h5 \
  billy@로컬컴퓨터IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 3: GPU 서버에서 스크립트로 자동화

```bash
# GPU 서버(223.194.115.11)에 SSH 접속 후 실행
cat > /tmp/sync_vla_data.sh << 'SCRIPT'
#!/bin/bash
SOURCE="soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset"
TEMP_DIR="/tmp/vla_sync"
DEST="billy@로컬컴퓨터IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"

echo "1단계: 로봇 서버 → GPU 서버"
mkdir -p $TEMP_DIR
rsync -avz --progress $SOURCE/*.h5 $TEMP_DIR/

echo "2단계: GPU 서버 → 로컬 컴퓨터"
rsync -avz --progress $TEMP_DIR/*.h5 $DEST/

echo "정리 중..."
rm -rf $TEMP_DIR
echo "완료!"
SCRIPT

chmod +x /tmp/sync_vla_data.sh
/tmp/sync_vla_data.sh
```

## 단계별 실행 가이드

### Step 1: GPU 서버 접속

```bash
# 로컬 컴퓨터에서 실행
ssh GPU서버사용자@223.194.115.11
```

### Step 2: 로봇 서버 접속 테스트

```bash
# GPU 서버에서 실행
ssh soda@10.109.0.187 "echo '연결 성공'"
```

### Step 3: 로봇 서버 파일 확인

```bash
# GPU 서버에서 실행
ssh soda@10.109.0.187 "ls /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l"
```

### Step 4: GPU 서버로 파일 가져오기

```bash
# GPU 서버에서 실행
mkdir -p /tmp/vla_sync
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/
```

### Step 5: 로컬 컴퓨터로 다운로드

```bash
# 로컬 컴퓨터(billy)에서 실행
rsync -avz --progress \
  GPU서버사용자@223.194.115.11:/tmp/vla_sync/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 간단한 원라이너 (GPU 서버에서)

```bash
# GPU 서버(223.194.115.11)에 SSH 접속 후 실행
rsync -avz --progress \
  soda@10.109.0.187:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /tmp/vla_sync/ && \
echo "로봇 서버 → GPU 서버 완료! 이제 로컬 컴퓨터에서 다운로드하세요."
```

## 확인 명령어

### GPU 서버에서 로봇 서버 접속 확인

```bash
# GPU 서버에서 실행
ssh -v soda@10.109.0.187 "echo 'test'"
```

### 파일 개수 확인

```bash
# GPU 서버에서 실행
ssh soda@10.109.0.187 "ls /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l"
```

### 전송된 파일 확인

```bash
# GPU 서버에서 실행
ls -lh /tmp/vla_sync/*.h5 | wc -l
```

## 추천 방법

**방법 1 (GPU 서버에서 직접 접속)**을 추천합니다:
- 가장 간단하고 직접적
- 중간 단계 최소화
- 문제 발생 시 디버깅 용이

## 주의사항

1. **디스크 공간 확인**
   ```bash
   # GPU 서버에서 실행
   df -h /tmp
   ```

2. **임시 디렉토리 정리**
   ```bash
   # 전송 완료 후
   rm -rf /tmp/vla_sync
   ```

