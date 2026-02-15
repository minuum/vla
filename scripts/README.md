# VLA API Server - Scripts & Aliases

API 서버를 쉽게 관리하고 테스트할 수 있는 스크립트와 alias 모음입니다.

## 📦 설치 (Setup)

### 1. Aliases 활성화

`~/.zshrc` 또는 `~/.bashrc`에 다음 라인을 추가:

```bash
source /home/soda/vla/.vla_aliases
```

그리고 shell을 재시작하거나:

```bash
source ~/.zshrc  # or source ~/.bashrc
```

## 🚀 사용법

### 서버 관리

```bash
# 서버 시작
vla-start

# 서버 중지
vla-stop

# 서버 재시작
vla-restart

# 서버 상태 확인
vla-status

# 로그 보기 (실시간)
vla-logs
```

### API 테스트

```bash
# 전체 테스트 (health + test + predict)
vla-test

# Health check만
vla-health

# Test endpoint만
vla-test-endpoint

# Predict endpoint만 (모델이 로드되어야 함)
vla-predict
```

### 빠른 명령어

```bash
# Health check (curl)
vla-curl-health

# Test endpoint (curl)
vla-curl-test

# 프로세스 확인
vla-ps

# 프로젝트 디렉토리로 이동
vla-cd

# GPU 상태 보기 (Jetson)
vla-gpu

# 환경 변수 보기
vla-env

# 도움말
vla-help
```

## 📝 스크립트 직접 사용

Alias 없이 스크립트를 직접 실행할 수도 있습니다:

### 서버 관리 스크립트

```bash
# 사용법
./scripts/manage_api_server.sh {start|stop|restart|status|logs}

# 예시
./scripts/manage_api_server.sh start
./scripts/manage_api_server.sh status
```

### 테스트 스크립트

```bash
# 사용법
./scripts/test_api_server.sh [health|test|predict|all]

# 예시
./scripts/test_api_server.sh all
./scripts/test_api_server.sh health
```

## 🔧 환경 변수

다음 환경 변수를 설정할 수 있습니다:

```bash
# API 서버 URL (기본값: http://localhost:8000)
export VLA_API_SERVER="http://localhost:8000"

# API Key
export VLA_API_KEY="your-api-key-here"

# 프로젝트 디렉토리 (기본값: /home/soda/vla)
export VLA_PROJECT_DIR="/home/soda/vla"
```

## 📂 파일 구조

```
/home/soda/vla/
├── scripts/
│   ├── manage_api_server.sh   # 서버 관리 스크립트
│   └── test_api_server.sh     # API 테스트 스크립트
├── .vla_aliases               # Shell aliases 정의
├── api_server.py              # API 서버 코드
└── logs/
    ├── api_server.log         # 서버 로그
    └── api_server.pid         # 프로세스 ID
```

## 🎯 일반적인 워크플로우

### 1. 서버 시작하기

```bash
vla-start
```

### 2. 상태 확인

```bash
vla-status
```

### 3. API 테스트

```bash
# 빠른 health check
vla-curl-health

# 전체 테스트
vla-test
```

### 4. 문제 발생 시

```bash
# 로그 확인
vla-logs

# 서버 재시작
vla-restart
```

### 5. 서버 종료

```bash
vla-stop
```

## 🔍 트러블슈팅

### 서버가 시작되지 않을 때

```bash
# 로그 확인
cat logs/api_server.log

# 프로세스 확인
vla-ps

# 강제 종료 후 재시작
pkill -9 -f api_server.py
vla-start
```

### 포트가 이미 사용 중일 때

```bash
# 포트 8000 사용 중인 프로세스 확인
lsof -i :8000

# 또는
netstat -tuln | grep 8000
```

### API Key 오류

```bash
# 현재 설정 확인
vla-env

# 새 API Key 설정
export VLA_API_KEY="new-api-key-here"
```

## 📊 예시 출력

### vla-status

```
========================================
API Server Status
========================================
Status: Running ✓
PID: 12345
 0.5 2.1 python3 api_server.py

Health Check:
{
  "status": "healthy",
  "model_loaded": false,
  "device": "cuda",
  "gpu_memory": {
    "allocated_gb": 0.0,
    "reserved_gb": 0.0,
    "device_name": "Orin"
  }
}

Log file: /home/soda/vla/logs/api_server.log
Last 5 lines:
...
```

### vla-test

```
========================================
VLA API Server Test Script
========================================
API URL: http://localhost:8000
API Key: jFLQzbwEch8_S2lpioP6sC...

[1/3] Health Check
Request: GET http://localhost:8000/health
✓ Status: 200
{
  "status": "healthy",
  ...
}

[2/3] Test Endpoint
Request: GET http://localhost:8000/test
✓ Status: 200
{
  "message": "Test endpoint - using dummy data",
  ...
}

[3/3] Predict Endpoint
Creating sample image (1280x720)...
Request: POST http://localhost:8000/predict
✓ Status: 200
{
  "action": [1.15, 0.319],
  ...
}

========================================
All tests passed! ✓
========================================
```

## 💡 Tips

1. **자동 시작**: 서버를 부팅 시 자동으로 시작하려면 systemd service를 설정하세요.
2. **모니터링**: `watch -n 1 vla-status` 명령으로 실시간 모니터링 가능
3. **원격 테스트**: `VLA_API_SERVER` 환경 변수를 변경하여 원격 서버 테스트 가능

## 🔗 관련 문서

- `BILLY_SERVER_START_GUIDE.md` - 서버 설정 가이드
- `api_server.py` - API 서버 코드
