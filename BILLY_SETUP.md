# Billy 서버 설정 가이드

## 📅 Billy 모델 서버 (A5000)

Billy 서버에서 이 파일의 내용대로 설정하세요.

## 1️⃣ 프로젝트 클론

```bash
cd ~
git clone <repository-url> vla
cd vla
```

## 2️⃣ Aliases 설정

```bash
# .zshrc에 추가
echo 'source ~/vla/.vla_aliases' >> ~/.zshrc
source ~/.zshrc

# 확인
vla-env
# 출력:
# Server Name: Billy Model Server
# Role: model
```

## 3️⃣ Tailscale 설정

```bash
# Tailscale 설치 (미설치 시)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Tailscale IP 확인
tailscale ip -4
# 예: 100.XXX.XXX.XXX

# 이 IP를 Jetson 서버에 알려주세요!
```

## 4️⃣ 환경 변수 설정

`~/.zshrc` 또는 `.vla_aliases`를 열어서 Billy의 Tailscale IP를 설정:

```bash
# ~/.zshrc 맨 아래에 추가
export BILLY_TAILSCALE_IP="100.XXX.XXX.XXX"  # 위에서 확인한 IP
```

## 5️⃣ Python 환경 설정

```bash
cd ~/vla

# 필요한 패키지 설치
pip3 install -r requirements.txt

# RoboVLMs 설치
pip3 install -e RoboVLMs/

# 확인
python3 -c "import robovlms; print('✓ robovlms installed')"
```

## 6️⃣ 모델 체크포인트 확인

```bash
# Fine-tuned 모델 확인
ls -lh runs/mobile_vla_no_chunk_20251209/checkpoints/

# Base 모델 확인
ls -lh .vlms/kosmos-2-patch14-224/
```

## 7️⃣ 방화벽 설정

```bash
# 포트 8000 개방
sudo ufw allow 8000/tcp
sudo ufw status
```

## 8️⃣ API 서버 시작

```bash
# 환경 설정
vla-model-env

# API 서버 시작
vla-start

# 상태 확인
vla-status

# 로그에서 API Key 확인
vla-logs | grep "API Key"
# 출력 예시:
# 생성된 API Key: jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc
```

## 9️⃣ API Key를 Jetson에 전달

위에서 확인한 API Key를 Jetson 서버 담당자에게 전달:

```
Billy Server Information:
- Tailscale IP: 100.XXX.XXX.XXX
- API Server: http://100.XXX.XXX.XXX:8000
- API Key: jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc
```

## 🔟 테스트

```bash
# Health check
vla-health

# Test endpoint
vla-test

# GPU 상태
vla-gpu
```

## ✅ 체크리스트

- [ ] Git repository 클론
- [ ] `.vla_aliases` 소싱
- [ ] Tailscale 설치 및 IP 확인
- [ ] `BILLY_TAILSCALE_IP` 환경 변수 설정
- [ ] Python 패키지 설치
- [ ] RoboVLMs 설치
- [ ] 모델 체크포인트 확인
- [ ] 방화벽 포트 8000 개방
- [ ] API 서버 시작
- [ ] API Key 확인 및 Jetson에 전달
- [ ] API 테스트 성공

## 🚀 일상적인 사용

```bash
# 서버 시작
source ~/.zshrc
vla-model-env
vla-start

# 상태 확인
vla-status

# 로그 모니터링
vla-logs

# Jetson 연결 확인
vla-network-check

# 서버 종료 (필요시)
vla-stop
```

## 📝 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `vla-model-env` | 모델 서버 환경 설정 |
| `vla-start` | API 서버 시작 |
| `vla-stop` | API 서버 중지 |
| `vla-restart` | API 서버 재시작 |
| `vla-status` | API 서버 상태 |
| `vla-logs` | API 서버 로그 |
| `vla-test` | API 전체 테스트 |
| `vla-health` | Health check |
| `vla-network-check` | Jetson 연결 확인 |
| `vla-env` | 환경 변수 확인 |
| `vla-gpu` | GPU 상태 (nvidia-smi) |

## 🔍 트러블슈팅

### API 서버가 시작되지 않을 때

```bash
# 로그 확인
vla-logs

# 포트 충돌 확인
lsof -i :8000

# 기존 프로세스 종료
vla-stop
pkill -f api_server.py

# 재시작
vla-start
```

### 모델 로딩 실패

```bash
# robovlms 설치 확인
python3 -c "import robovlms"

# 재설치
pip3 install -e RoboVLMs/

# 체크포인트 확인
ls -lh runs/mobile_vla_no_chunk_20251209/checkpoints/
```

### Jetson에서 연결 안 됨

```bash
# 방화벽 확인
sudo ufw status

# 포트 개방
sudo ufw allow 8000/tcp

# Tailscale 상태
tailscale status

# API 서버 상태
vla-status
```

## 📡 Jetson과의 통신 예시

```python
# Jetson에서 이렇게 호출합니다:
import requests
import base64

response = requests.post(
    "http://100.XXX.XXX.XXX:8000/predict",
    headers={
        "X-API-Key": "your-api-key"
    },
    json={
        "image": "base64_encoded_image_data",
        "instruction": "Navigate to the left box"
    }
)

print(response.json())
# {'action': [1.15, 0.319], 'latency_ms': 123.45, ...}
```

---

**서버**: Billy (A5000)  
**역할**: VLA Model Inference  
**문서**: `docs/VLA_MULTI_SERVER_SETUP.md`
