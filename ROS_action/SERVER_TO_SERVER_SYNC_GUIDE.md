# 서버 간 데이터 동기화 가이드

## 현재 상황

- **로봇 서버**: `/home/soda/vla/ROS_action/mobile_vla_dataset` (468개 H5 파일)
- **로컬 서버**: `/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset` (237개 H5 파일)
- **차이**: 231개 파일 (약 3.3 GB)
- **Git LFS 대역폭**: 10GB/10GB 초과 (11일 후 리셋)

## 권장 방법: rsync (추천)

### 방법 1: rsync (IP 주소 사용) ⭐ 추천

호스트명 해석 실패 시 IP 주소를 사용하세요:

```bash
# 로컬 서버(billy)에서 실행
# 로봇 서버 IP 주소 확인 필요 (예: 192.168.1.100)
rsync -avz --progress \
  soda@로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/

# 또는 전체 디렉토리 동기화 (JSON 파일 포함)
rsync -avz --progress \
  soda@로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset/ \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 1-1: SSH Config 설정 (호스트명 등록)

`~/.ssh/config` 파일에 호스트 등록:

```bash
# 로컬 서버(billy)에서 실행
cat >> ~/.ssh/config << EOF
Host apexs
    HostName 로봇서버IP주소
    User soda
    Port 22
EOF

# 그 후 호스트명으로 접속 가능
rsync -avz --progress \
  apexs:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 2: 역방향 전송 (로봇 서버에서 실행) ⭐ 대안

로컬 서버에서 접속이 안 될 경우, 로봇 서버에서 역방향으로 전송:

```bash
# 로봇 서버(apexs)에서 실행
rsync -avz --progress \
  /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  billy@로컬서버IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

**옵션 설명:**
- `-a`: 아카이브 모드 (권한, 타임스탬프 유지)
- `-v`: 상세 출력
- `-z`: 압축 전송 (네트워크 효율성)
- `--progress`: 진행률 표시

### 방법 2: rsync (공유 스토리지 경유)

만약 두 서버가 공유 스토리지(NFS, CIFS 등)를 사용한다면:

```bash
# 공유 스토리지 경로 사용
rsync -avz --progress \
  /shared/storage/vla/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 3: scp (간단하지만 느림)

```bash
# 로컬 서버에서 실행
scp -r soda@로봇서버주소:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 단계별 실행 가이드

### 1단계: 로봇 서버 IP 주소 확인

```bash
# 로봇 서버(apexs)에서 실행하여 IP 확인
hostname -I
# 또는
ip addr show | grep "inet " | grep -v "127.0.0.1"
```

### 2단계: SSH 연결 테스트 (IP 주소 사용)

```bash
# 로컬 서버(billy)에서 실행
ssh soda@로봇서버IP "echo '연결 성공'"
```

### 3단계: 디렉토리 확인

```bash
# 로컬 서버에서 실행
ssh soda@로봇서버IP "ls -lh /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l"
```

### 4단계: 드라이런 (실제 복사 전 테스트)

```bash
# 로컬 서버에서 실행
rsync -avz --dry-run --progress \
  soda@로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 5단계: 실제 동기화

```bash
# 로컬 서버에서 실행
rsync -avz --progress \
  soda@로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 대안: 역방향 전송 (로봇 서버에서 실행)

로컬 서버에서 접속이 안 될 경우:

```bash
# 로봇 서버(apexs)에서 실행
# 1. 로컬 서버 IP 확인 필요
rsync -avz --progress \
  /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  billy@로컬서버IP:/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 대안 방법

### 방법 A: 중간 스토리지 사용

1. 로봇 서버 → 외부 스토리지 (USB, 외장하드 등)
2. 외부 스토리지 → 로컬 서버

### 방법 B: tar + 압축 전송

```bash
# 로봇 서버에서 실행
cd /home/soda/vla/ROS_action/mobile_vla_dataset
tar -czf - *.h5 | ssh billy@로컬서버주소 "cd /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset && tar -xzf -"
```

### 방법 C: Git LFS 대기 (11일 후)

- 11일 후 Git LFS 대역폭 리셋
- 그때 `git pull` 또는 `git lfs pull` 사용

## 주의사항

1. **디스크 공간 확인**
   ```bash
   # 로컬 서버에서 실행
   df -h /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset
   ```

2. **기존 파일 보호**
   - rsync는 기본적으로 기존 파일을 덮어쓰지 않음 (타임스탬프 비교)
   - 강제 덮어쓰기: `--ignore-existing` 제거

3. **네트워크 대역폭**
   - 대용량 파일 전송 시 네트워크 속도 확인
   - `--bwlimit=10000` (10MB/s 제한) 옵션 사용 가능

4. **중단 시 재개**
   - rsync는 중단 후 재실행 시 이어서 진행 가능
   - 부분 전송된 파일은 자동으로 재전송

## 예상 소요 시간

- 파일 크기: 약 3.3 GB (231개 파일)
- 네트워크 속도에 따라:
  - 100 Mbps: 약 4-5분
  - 1 Gbps: 약 30초
  - 10 Mbps: 약 45분

## 검증

동기화 후 확인:

```bash
# 로컬 서버에서 실행
ls -lh /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l
# 예상: 468개

# 파일 크기 비교
du -sh /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

## 추천 방법

**rsync (방법 1)**를 추천합니다:
- 안정적이고 신뢰성 높음
- 중단 후 재개 가능
- 네트워크 효율적 (압축 전송)
- 진행률 표시 가능

