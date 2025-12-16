# Billy 서버 API 서버 시작 가이드

## 서버 정보
- **IP**: 223.194.115.11
- **포트**: 8000
- **프로토콜**: HTTP

---

## Billy 서버에서 실행할 명령어

### 1. 디렉토리 이동
```bash
cd /home/billy/25-1kp/vla
```

### 2. 방화벽 포트 열기 (최초 1회)
```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

### 3. API 서버 실행

**옵션 A: 포그라운드 실행 (테스트용)**
```bash
python3 api_server.py
```

**옵션 B: 백그라운드 실행 (권장)**
```bash
# logs 디렉토리 생성
mkdir -p logs

# 백그라운드 실행
nohup python3 api_server.py > logs/api_server.log 2>&1 &

# 프로세스 확인
ps aux | grep api_server

# 로그 확인
tail -f logs/api_server.log
```

### 4. 실행 확인

**로컬에서 테스트**:
```bash
curl http://localhost:8000/health
```

**예상 응답**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_memory": {
    "allocated_gb": 3.2,
    "device_name": "NVIDIA RTX A5000"
  }
}
```

### 5. 외부 접속 테스트

**자신의 IP 확인**:
```bash
hostname -I
curl ifconfig.me
```

**Jetson에서 테스트** (Billy 서버 실행 후):
```bash
curl http://223.194.115.11:8000/health
```

---

## 트러블슈팅

### 모델 로딩 실패
```bash
# 체크포인트 확인
ls -lh ROS_action/last.ckpt
ls -lh .vlms/kosmos-2-patch14-224/

# 로그 확인
tail -100 logs/api_server.log
```

### 포트 이미 사용 중
```bash
# 기존 프로세스 확인
lsof -i :8000

# 프로세스 종료
pkill -f api_server.py
```

### GPU 메모리 부족
```bash
# GPU 상태 확인
nvidia-smi

# 다른 프로세스 종료 필요시
```

---

## 서버 종료

```bash
# 백그라운드 프로세스 종료
pkill -f api_server.py

# 확인
ps aux | grep api_server
```

---

## Jetson에서 대기 중

Jetson에서는 이미 다음 설정이 완료되었습니다:
```bash
export VLA_API_SERVER="http://223.194.115.11:8000"
```

Billy 서버에서 API 서버를 실행하면 Jetson에서 바로 사용 가능합니다!

---

**작성**: 2025-12-16 23:05  
**상태**: Billy 서버에서 API 서버 실행 대기 중
