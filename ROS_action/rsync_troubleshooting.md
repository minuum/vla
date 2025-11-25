# rsync 문제 해결 가이드

## 현재 상황
rsync 명령 실행 후 아무 출력이 없는 경우

## 가능한 원인

1. **SSH 연결 대기 중** (비밀번호 입력 필요)
2. **원격 파일 목록 읽는 중** (468개 파일이 많아서 시간 소요)
3. **네트워크 지연**
4. **SSH 키 인증 대기**

## 해결 방법

### 방법 1: SSH 연결 먼저 확인

```bash
# SSH 연결 테스트 (비밀번호 입력)
ssh soda@192.168.101.101

# 연결 성공 후 나가기
exit
```

### 방법 2: 파일 목록 먼저 확인

```bash
# 원격 서버의 파일 개수 확인
ssh soda@192.168.101.101 "ls /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l"

# 파일 목록 일부 확인
ssh soda@192.168.101.101 "ls -lh /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | head -5"
```

### 방법 3: rsync with verbose + SSH 옵션

```bash
# 더 상세한 출력 + SSH 옵션 추가
rsync -avz --progress -e "ssh -v" \
  soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 4: 작은 범위로 테스트

```bash
# 먼저 1개 파일만 테스트
rsync -avz --progress \
  soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/episode_20251119_*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/ \
  | head -20
```

### 방법 5: 백그라운드 실행 + 로그 저장

```bash
# 백그라운드 실행 + 로그 파일 저장
rsync -avz --progress \
  soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/ \
  > rsync_log.txt 2>&1 &

# 진행 상황 확인
tail -f rsync_log.txt
```

### 방법 6: 타임아웃 설정

```bash
# SSH 타임아웃 설정
rsync -avz --progress -e "ssh -o ConnectTimeout=10" \
  soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 단계별 확인

### 1단계: SSH 연결 확인
```bash
ssh -v soda@192.168.101.101 "echo 'test'"
```

### 2단계: 네트워크 연결 확인
```bash
ping -c 3 192.168.101.101
```

### 3단계: 포트 확인
```bash
nc -zv 192.168.101.101 22
```

### 4단계: 작은 파일로 테스트
```bash
# 1개 파일만 전송 테스트
rsync -avz --progress \
  soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/episode_20251119_080007_1box_hori_right_core_medium.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 권장 실행 순서

1. **SSH 연결 먼저 확인**
   ```bash
   ssh soda@192.168.101.101
   ```

2. **파일 개수 확인**
   ```bash
   ssh soda@192.168.101.101 "ls /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l"
   ```

3. **작은 범위로 테스트**
   ```bash
   rsync -avz --progress \
     soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/episode_20251119_*.h5 \
     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
   ```

4. **전체 동기화 (성공 후)**
   ```bash
   rsync -avz --progress \
     soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
   ```

