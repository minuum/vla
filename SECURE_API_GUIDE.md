# Billy 서버 API 서버 시작 가이드 (보안 강화 버전)

**날짜**: 2025-12-16  
**버전**: v2.0 (API Key 인증 추가)

---

## 🔐 보안 기능

### 1. API Key 인증
- 모든 `/predict`와 `/test` 엔드포인트는 API Key 필수
- `/health`와 `/`는 인증 불필요 (모니터링용)

### 2. Tailscale VPN 권장
- 외부 인터넷 노출 대신 Tailscale 사용 권장
- IP: `100.99.189.94` (Tailscale 내부 IP)
- 안전한 p2p 암호화 통신

---

## 📡 서버 정보

### 네트워크 옵션

**옵션 A: Tailscale VPN (권장 - 가장 안전)** ⭐
- IP: `100.99.189.94`
- 포트: 8000
- 장점: 암호화, 인증, 방화벽 불필요
- Jetson도 Tailscale 설치 필요

**옵션 B: 외부 IP (간단하지만 보안 주의)**
- IP: `223.194.115.11`
- 포트: 8000
- 장점: 설정 간단
- 단점: 외부 노출, 방화벽 설정 필요

**옵션 C: Local (같은 서버)**
- IP: `localhost` 또는 `127.0.0.1`
- 포트: 8000
- 사용: 테스트용

---

## 🚀 Billy 서버에서 실행

### 1. 디렉토리 이동
```bash
cd /home/billy/25-1kp/vla
```

### 2. API Key 생성 및 설정

**첫 실행 시 자동 생성됩니다. 출력된 API Key를 저장하세요!**

```bash
# API Key 직접 설정 (권장)
export VLA_API_KEY="your-secret-api-key-here-make-it-long-and-random"

# 또는 자동 생성도 가능 (서버 시작 시)
```

**영구 저장**:
```bash
echo 'export VLA_API_KEY="your-secret-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 방화벽 설정

**Tailscale 사용 시 (권장)**:
```bash
# 방화벽 설정 불필요! Tailscale이 자동 처리
```

**외부 IP 사용 시**:
```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

### 4. 서버 실행

**옵션 A: 포그라운드 (테스트용)**
```bash
python3 api_server.py
# 또는
python3 Mobile_VLA/inference_server.py

# 출력에서 API Key 확인!
# 🔑 API Key: xxxxxxxxxxxxxxxxxx
```

**옵션 B: 백그라운드 (운영용)** ⭐
```bash
# logs 디렉토리 생성
mkdir -p logs

# 백그라운드 실행
nohup python3 api_server.py > logs/api_server.log 2>&1 &

# 프로세스 확인
ps aux | grep api_server

# API Key 확인
grep "API Key:" logs/api_server.log

# 로그 확인
tail -f logs/api_server.log
```

### 5. 서버 확인

```bash
# Health check (인증 불필요)
curl http://localhost:8000/health

# 응답:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cuda",
#   "gpu_memory": {
#     "allocated_gb": 3.2,
#     "device_name": "NVIDIA RTX A5000"
#   }
# }
```

---

## 🤖 Jetson 서버 설정

### 1. Tailscale 설치 (권장)

```bash
# Tailscale 설치
curl -fsSL https://tailscale.com/install.sh | sh

# Tailscale 시작
sudo tailscale up

# Billy 서버 ping 테스트
ping 100.99.189.94
```

### 2. 환경 변수 설정

**Tailscale 사용 (권장)**:
```bash
export VLA_API_SERVER="http://100.99.189.94:8000"
export VLA_API_KEY="your-api-key-from-billy-server"
```

**외부 IP 사용**:
```bash
export VLA_API_SERVER="http://223.194.115.11:8000"
export VLA_API_KEY="your-api-key-from-billy-server"
```

**영구 저장**:
```bash
echo 'export VLA_API_SERVER="http://100.99.189.94:8000"' >> ~/.bashrc
echo 'export VLA_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 네트워크 테스트

```bash
# Billy 서버로 파일 복사 (한 번만)
scp billy@100.99.189.94:/home/billy/25-1kp/vla/scripts/test_network.sh ~/

# 네트워크 테스트 실행
bash test_network.sh 100.99.189.94

# 또는 외부 IP
bash test_network.sh 223.194.115.11
```

### 4. API 테스트

```bash
# Health check
curl http://100.99.189.94:8000/health

# Test endpoint (API Key 필요)
curl -H "X-API-Key: your-api-key" http://100.99.189.94:8000/test

# Python 클라이언트 테스트
python3 vla_api_client.py --test
```

---

## 🧪 API 사용 예시

### Health Check (인증 불필요)
```bash
curl http://100.99.189.94:8000/health
```

### Test Endpoint (API Key 필요)
```bash
curl -H "X-API-Key: your-api-key" \
     http://100.99.189.94:8000/test
```

### Predict (API Key 필요)
```bash
curl -X POST http://100.99.189.94:8000/predict \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
  }'
```

### Python 클라이언트
```python
import os
os.environ["VLA_API_SERVER"] = "http://100.99.189.94:8000"
os.environ["VLA_API_KEY"] = "your-api-key"

from vla_api_client import VLAClient

client = VLAClient()
action, latency = client.predict(image, instruction)
```

---

## 🔧 트러블슈팅

### API Key 인증 실패
```
❌ 403 Forbidden: Invalid API Key
```

**해결**:
1. Billy 서버 로그에서 정확한 API Key 확인
   ```bash
   grep "API Key:" logs/api_server.log
   ```
2. Jetson에서 환경 변수 확인
   ```bash
   echo $VLA_API_KEY
   ```
3. 일치하는지 확인

### 연결 실패 (Connection refused)

**Tailscale 사용 시**:
```bash
# 1. Tailscale 연결 확인
tailscale status

# 2. Billy 서버 ping
ping 100.99.189.94

# 3. Health check
curl http://100.99.189.94:8000/health
```

**외부 IP 사용 시**:
```bash
# 1. Billy 서버에서 API 서버 실행 확인
ps aux | grep api_server

# 2. 방화벽 확인
sudo ufw status | grep 8000

# 3. Jetson에서 ping
ping 223.194.115.11
```

### 모델 로딩 실패
```bash
# 체크포인트 확인
ls -lh runs/mobile_vla_no_chunk_20251209/checkpoints/

# 환경 변수로 경로 지정
export VLA_CHECKPOINT_PATH="runs/.../last.ckpt"
python3 api_server.py
```

---

## 📊 모니터링

### Billy 서버
```bash
# API 서버 로그
tail -f logs/api_server.log

# GPU 사용량
nvidia-smi

# Control Center
python3 scripts/control_center.py
```

### Jetson 서버
```bash
# Latency 측정
for i in {1..10}; do
  curl -w "Latency: %{time_total}s\n" \
       -o /dev/null -s \
       http://100.99.189.94:8000/health
done
```

---

## 🛑 서버 종료

```bash
# 백그라운드 프로세스 종료
pkill -f api_server.py

# 확인
ps aux | grep api_server
```

---

## ✅ 검증 체크리스트

### Billy 서버
- [ ] API Key 설정 확인 (`echo $VLA_API_KEY`)
- [ ] 서버 실행 (`python3 api_server.py`)
- [ ] Health check 성공 (`curl http://localhost:8000/health`)
- [ ] API Key 저장 (안전한 곳에)

### Jetson 서버
- [ ] Tailscale 설치 및 연결
- [ ] 환경 변수 설정 (`VLA_API_SERVER`, `VLA_API_KEY`)
- [ ] Ping 성공 (`ping 100.99.189.94`)
- [ ] Health check 성공
- [ ] API 테스트 성공 (API Key 포함)

---

## 🔐 보안 권장사항

1. **Tailscale 사용** (가장 안전)
   - 암호화된 VPN
   - 자동 인증
   - 방화벽 불필요

2. **API Key 관리**
   - 긴 랜덤 문자열 사용 (32자 이상)
   - 환경 변수로만 저장
   - Git에 커밋하지 말 것
   - 정기적으로 갱신

3. **외부 IP 사용 시**
   - 방화벽으로 특정 IP만 허용
   - HTTPS 고려 (추후)
   - Rate limiting 고려 (추후)

---

**작성**: 2025-12-16 23:25  
**버전**: v2.0 (API Key 인증 + Tailscale 권장)
