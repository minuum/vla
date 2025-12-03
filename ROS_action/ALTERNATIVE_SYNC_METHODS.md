# SSH/rsync 대안 데이터 전송 방법

## 현재 상황
- SSH/rsync 접속 불가
- 로봇 서버: 468개 H5 파일 (약 12GB)
- 로컬 컴퓨터: 237개 H5 파일

## 대안 방법

### 방법 1: Python HTTP 서버 (가장 간단) ⭐ 추천

#### 로봇 서버에서 실행

```bash
# 로봇 서버(10.109.0.187 또는 192.168.101.101)에서 실행
cd /home/soda/vla/ROS_action/mobile_vla_dataset
python3 -m http.server 8000

# 또는 특정 IP만 허용
python3 -m http.server 8000 --bind 0.0.0.0
```

#### 로컬 컴퓨터에서 다운로드

```bash
# 로컬 컴퓨터(billy)에서 실행
# wget으로 다운로드
wget -r -np -nH --cut-dirs=3 \
  http://로봇서버IP:8000/*.h5 \
  -P /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/

# 또는 curl
curl -O http://로봇서버IP:8000/episode_*.h5
```

### 방법 2: FTP/SFTP 서버

#### 로봇 서버에서 FTP 서버 실행

```bash
# vsftpd 설치 (필요시)
sudo apt install vsftpd

# 또는 Python FTP 서버
pip install pyftpdlib
python3 -m pyftpdlib -p 2121 -w
```

#### 로컬 컴퓨터에서 접속

```bash
# FTP 클라이언트 사용
ftp 로봇서버IP
# 또는
sftp 로봇서버IP
```

### 방법 3: NFS/CIFS 공유 스토리지

#### 로봇 서버에서 NFS 공유

```bash
# /etc/exports 파일 편집
echo "/home/soda/vla/ROS_action/mobile_vla_dataset *(ro,sync)" | sudo tee -a /etc/exports
sudo exportfs -ra
```

#### 로컬 컴퓨터에서 마운트

```bash
sudo mount -t nfs 로봇서버IP:/home/soda/vla/ROS_action/mobile_vla_dataset /mnt/robot_data
cp /mnt/robot_data/*.h5 /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 4: 클라우드 스토리지 (Google Drive, Dropbox 등)

#### 로봇 서버에서 업로드

```bash
# rclone 설치 및 설정
rclone copy /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
  gdrive:vla_dataset/
```

#### 로컬 컴퓨터에서 다운로드

```bash
rclone copy gdrive:vla_dataset/ \
  /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 5: tar + netcat (직접 연결)

#### 로봇 서버에서 실행

```bash
# 로봇 서버에서 tar + netcat로 전송
cd /home/soda/vla/ROS_action/mobile_vla_dataset
tar -czf - *.h5 | nc -l -p 12345
```

#### 로컬 컴퓨터에서 수신

```bash
# 로컬 컴퓨터에서 수신
nc 로봇서버IP 12345 | tar -xzf - -C /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 6: USB/외장하드 (물리적 전송)

```bash
# 로봇 서버에 직접 접근 가능한 경우
cp /home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 /media/usb/

# 로컬 컴퓨터에서
cp /media/usb/*.h5 /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 7: Git LFS 대기 (11일 후)

```bash
# 11일 후 Git LFS 대역폭 리셋되면
git pull
git lfs pull
```

### 방법 8: SCP (SSH 기반이지만 별도 시도)

```bash
# SCP 직접 시도
scp -r soda@192.168.101.101:/home/soda/vla/ROS_action/mobile_vla_dataset/*.h5 \
     /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/
```

### 방법 9: 웹 기반 파일 관리자

#### 로봇 서버에서 실행

```bash
# FileBrowser 설치
docker run -d \
  -v /home/soda/vla/ROS_action/mobile_vla_dataset:/srv \
  -p 8080:80 \
  filebrowser/filebrowser
```

#### 로컬 컴퓨터에서 접속

브라우저에서 `http://로봇서버IP:8080` 접속하여 다운로드

### 방법 10: 기존 Git 저장소 활용

이미 Git LFS로 추적 중이므로:

```bash
# 로컬 컴퓨터에서
git fetch origin
git checkout origin/feature/bsp-analysis-clean -- ROS_action/mobile_vla_dataset/*.h5
```

## 추천 방법 순위

1. **Python HTTP 서버** ⭐⭐⭐
   - 가장 간단하고 빠름
   - 추가 설치 불필요
   - 방화벽만 열리면 됨

2. **클라우드 스토리지** ⭐⭐
   - 안정적
   - 중간 저장소 역할
   - 다운로드 속도는 느릴 수 있음

3. **USB/외장하드** ⭐⭐
   - 가장 빠름
   - 물리적 접근 필요

4. **Git LFS 대기** ⭐
   - 추가 작업 없음
   - 11일 대기 필요

## Python HTTP 서버 상세 가이드

### 로봇 서버에서 실행

```bash
# 1. 로봇 서버에 접속 (직접 또는 다른 방법)
cd /home/soda/vla/ROS_action/mobile_vla_dataset

# 2. HTTP 서버 시작
python3 -m http.server 8000

# 또는 특정 IP만 허용하고 백그라운드 실행
nohup python3 -m http.server 8000 --bind 0.0.0.0 > /tmp/http_server.log 2>&1 &
```

### 로컬 컴퓨터에서 다운로드

```bash
# 방법 1: wget 사용
cd /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset
wget -r -np -nH --cut-dirs=3 \
  http://로봇서버IP:8000/*.h5

# 방법 2: curl 사용
for file in $(curl -s http://로봇서버IP:8000/ | grep -o 'episode_[^"]*\.h5'); do
  curl -O http://로봇서버IP:8000/$file
done

# 방법 3: Python 스크립트
python3 << 'EOF'
import urllib.request
import re
import os

url = "http://로봇서버IP:8000/"
response = urllib.request.urlopen(url)
html = response.read().decode('utf-8')
files = re.findall(r'href="(episode_.*?\.h5)"', html)

os.chdir("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
for file in files:
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(url + file, file)
EOF
```

## 보안 주의사항

HTTP 서버는 인증이 없으므로:
- 임시로만 사용
- 사용 후 즉시 종료
- 방화벽으로 접근 제한

```bash
# 서버 종료
pkill -f "python3 -m http.server"
```

