# VLA API Server - Quick Start Guide

빠르게 API 서버를 시작하고 테스트하는 방법입니다.

## 🚀 빠른 시작

### 1. Aliases 설정 (최초 1회만)

```bash
# .zshrc에 추가
echo 'source /home/soda/vla/.vla_aliases' >> ~/.zshrc
source ~/.zshrc

# 또는 현재 세션에만 적용
source /home/soda/vla/.vla_aliases
```

### 2. 서버 시작

```bash
vla-start
```

### 3. 테스트

```bash
# 빠른 health check
vla-curl-health

# 전체 테스트
vla-test
```

## 📋 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `vla-help` | 모든 명령어 보기 |
| `vla-start` | 서버 시작 |
| `vla-stop` | 서버 중지 |
| `vla-restart` | 서버 재시작 |
| `vla-status` | 서버 상태 확인 |
| `vla-logs` | 로그 보기 (실시간) |
| `vla-test` | API 전체 테스트 |
| `vla-health` | Health check만 |
| `vla-curl-health` | Health check (빠름) |
| `vla-ps` | 프로세스 확인 |

## 🔧 환경 변수

서버가 시작되면 자동으로 API Key가 생성됩니다. 
생성된 API Key는 로그에서 확인할 수 있습니다:

```bash
vla-logs  # 또는
cat logs/api_server.log | grep "API Key"
```

현재 설정 확인:
```bash
vla-env
```

## 📊 서버 상태 확인

```bash
# 상세 상태
vla-status

# 간단한 확인
vla-ps

# 실시간 모니터링
watch -n 1 vla-status
```

## 🧪 API 테스트 방법

### 1. Health Check (인증 불필요)
```bash
curl http://localhost:8000/health
```

### 2. Test Endpoint (API Key 필요)
```bash
curl -X GET http://localhost:8000/test \
  -H "X-API-Key: YOUR_API_KEY"
```

### 3. Predict Endpoint (API Key 필요)
Python 스크립트 예시:
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 이미지 준비
img = Image.new('RGB', (1280, 720), color=(0, 128, 255))
buffer = BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# API 호출
response = requests.post(
    "http://localhost:8000/predict",
    headers={"X-API-Key": "YOUR_API_KEY"},
    json={
        "image": img_base64,
        "instruction": "Navigate to the left box"
    }
)

print(response.json())
```

## 🔍 문제 해결

### 서버가 시작되지 않을 때
```bash
# 로그 확인
vla-logs

# 또는
cat logs/api_server.log

# 기존 프로세스 확인 및 종료
vla-ps
pkill -9 -f api_server.py

# 재시작
vla-start
```

### 포트가 이미 사용 중
```bash
# 포트 8000 사용 확인
lsof -i :8000
netstat -tuln | grep 8000

# 해당 프로세스 종료 후 재시작
```

### API Key 오류
```bash
# 환경 변수 확인
vla-env

# 로그에서 생성된 API Key 확인
cat logs/api_server.log | grep "API Key"

# 환경 변수 설정
export VLA_API_KEY="your-api-key-here"
```

## 📖 자세한 문서

- `scripts/README.md` - 스크립트 상세 설명
- `BILLY_SERVER_START_GUIDE.md` - 서버 설정 가이드

## 💡 유용한 팁

```bash
# 서버 자동 재시작 (cron)
*/5 * * * * pgrep -f api_server.py || /home/soda/vla/scripts/manage_api_server.sh start

# 원격 서버 테스트
export VLA_API_SERVER="http://223.194.115.11:8000"
vla-test

# GPU 모니터링 (Jetson)
vla-gpu
```

## 🎯 일반적인 워크플로우

```bash
# 1. 프로젝트로 이동
vla-cd

# 2. 서버 시작
vla-start

# 3. 상태 확인
vla-status

# 4. 테스트
vla-test

# 5. 작업 완료 후 (선택사항)
vla-stop
```

---

**작성**: 2025-12-17  
**버전**: 1.0
