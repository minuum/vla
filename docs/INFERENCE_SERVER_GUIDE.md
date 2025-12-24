# Mobile VLA 추론 서버 실행 가이드

## 📅 작성: 2025-12-24

## 🎯 목표
최신 체크포인트 기준으로 추론 서버를 작동시키고, 터미널을 나눠서 서버와 로그를 동시에 모니터링합니다.

## 📦 최근 변경사항 (Billy 서버 기준)

### 최신 커밋 (2025-12-17 ~ 2025-12-24)
- `83477421`: 체크포인트 검사 유틸리티 및 gitignore 업데이트
- `c1b41264`: feature/inference-integration 브랜치 병합
- `5fe2cb27`: 프로젝트 전체 상황 종합 README 업데이트
- `206c8640`: 교수님 브리핑에 정량적 메트릭 추가 (방향 정확도 100%, 안정성 96%)

### 사용 가능한 체크포인트

#### 1. 최신 체크포인트 (권장 ✅)
```bash
체크포인트: runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt
크기: 6.4GB
날짜: 2025-12-17
성능: val_loss=0.067 (최고 성능)
Chunk: 5
설명: Chunk5 Epoch 6 모델 - 검증 손실이 가장 낮음
```

#### 2. 레거시 체크포인트
```bash
체크포인트: ROS_action/last.ckpt
크기: 6.9GB
날짜: 2025-12-09
설명: 이전 학습 체크포인트 (호환성을 위해 유지)
```

## 🚀 실행 방법

### Option 1: 멀티 터미널 자동 실행 (tmux 사용, 권장 ✅)

**한 번에 서버 + 로그 모니터링 시작:**

```bash
cd /home/soda/vla
bash scripts/run_multi_terminal.sh
```

**화면 구성:**
```
┌─────────────────────────────────────┐
│  추론 서버 (상단 70%)                │
│  - API 서버 실행                     │
│  - 모델 로딩 상태                    │
├─────────────────────────────────────┤
│  로그 모니터링 (하단 30%)            │
│  - 실시간 로그 스트리밍              │
│  - 에러/경고 색상 강조               │
└─────────────────────────────────────┘
```

**tmux 단축키:**
- `Ctrl+b, 위/아래 화살표`: 패널 이동
- `Ctrl+b, o`: 다음 패널로 이동
- `Ctrl+b, d`: 세션 detach (백그라운드 실행)
- `tmux attach -t vla_inference`: 세션 재접속

**세션 종료:**
```bash
tmux kill-session -t vla_inference
```

### Option 2: 수동 실행 (두 개의 터미널 창 사용)

#### 터미널 1: 추론 서버 시작
```bash
cd /home/soda/vla
bash scripts/start_inference_server.sh
```

#### 터미널 2: 로그 모니터링
```bash
cd /home/soda/vla
bash scripts/monitor_inference_server.sh
```

### Option 3: RoboVLMs 스타일 API 서버 (forward_continuous 사용)

RoboVLMs의 `forward_continuous()` 방식을 사용하는 버전:

```bash
cd /home/soda/vla
python3 api_server_robovlms.py
```

**특징:**
- `robovlms_mobile_vla_inference.py`와 완전 통합
- 이미지 버퍼 관리 (window_size 지원)
- abs_action 전략 지원
- 메모리 최적화 (FP16)

## 🔧 환경 변수 설정

### 필수 환경 변수

```bash
# API Key (자동 생성되지만, 미리 설정 권장)
export VLA_API_KEY="your-generated-api-key"

# 체크포인트 경로 (커스텀 지정 시)
export VLA_CHECKPOINT_PATH="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
```

### 선택적 환경 변수 (RoboVLMs 스타일)

```bash
# Window size (이미지 버퍼 크기, 기본값: 2)
export VLA_WINDOW_SIZE=2

# Chunk size (예측할 액션 수, 기본값: 10)
export VLA_CHUNK_SIZE=10

# abs_action 전략 사용 (기본값: true)
export VLA_USE_ABS_ACTION=true

# 정규화 해제 전략 (scale/minmax/safe, 기본값: safe)
export VLA_DENORM_STRATEGY=safe
```

## 📊 API 엔드포인트

### Health Check (인증 불필요)
```bash
curl http://localhost:8000/health
```

### 모델 정보 조회 (API Key 필요)
```bash
curl -H "X-API-Key: $VLA_API_KEY" http://localhost:8000/model/info
```

### 추론 요청 (API Key 필요)
```bash
# 이미지를 base64로 인코딩
IMAGE_B64=$(base64 -w 0 test_image.jpg)

# 추론 요청
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $VLA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE_B64\",
    \"instruction\": \"Navigate to the left bottle\",
    \"use_abs_action\": true
  }"
```

### 이미지 버퍼 리셋 (RoboVLMs 스타일만)
```bash
curl -X POST http://localhost:8000/reset \
  -H "X-API-Key: $VLA_API_KEY"
```

### 성능 벤치마크
```bash
curl -H "X-API-Key: $VLA_API_KEY" \
  "http://localhost:8000/benchmark?iterations=100"
```

## 🎮 Jetson에서 API 호출

Jetson 클라이언트에서 API 서버 호출 예시:

```python
import requests
import base64
from pathlib import Path

# Billy 서버 주소 (Tailscale)
BILLY_API = "http://100.XXX.XXX.XXX:8000"
API_KEY = "your-api-key"

# 이미지 읽기
with open("camera_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 추론 요청
response = requests.post(
    f"{BILLY_API}/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "image": image_b64,
        "instruction": "Navigate to the left box",
        "use_abs_action": True
    }
)

result = response.json()
action = result["action"]  # [linear_x, linear_y]
print(f"예측된 액션: {action}")
```

## 🔍 트러블슈팅

### 1. 체크포인트를 찾을 수 없음
```bash
# 체크포인트 경로 확인
ls -lh runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/

# 환경 변수로 명시적 지정
export VLA_CHECKPOINT_PATH="/home/soda/vla/ROS_action/last.ckpt"
```

### 2. GPU 메모리 부족
```bash
# GPU 상태 확인
nvidia-smi

# 다른 프로세스 종료
pkill -f python

# 메모리 정리 후 재시작
```

### 3. API Key 오류
```bash
# 로그에서 생성된 API Key 확인
tail -f logs/api_server.log | grep "생성된 API Key"

# 환경 변수 설정
export VLA_API_KEY="확인한-키-값"
```

### 4. 모델 로딩 실패
```bash
# RoboVLMs 경로 확인
ls -la RoboVLMs_upstream/

# Python 경로 확인
python3 -c "import sys; print('\n'.join(sys.path))"

# 필요시 경로 수동 추가
export PYTHONPATH="/home/soda/vla/RoboVLMs_upstream:$PYTHONPATH"
```

## 📈 성능 메트릭

### 최신 모델 (Chunk5 Epoch6)
- **검증 손실**: 0.067 (최저)
- **방향 정확도**: 100%
- **안정성**: 96%
- **평균 추론 시간**: ~120ms (GPU: RTX A5000)

### 기대 성능
```
평균 FPS: ~8-10 fps
최대 지연: <200ms
GPU 메모리: ~6-8GB (FP16)
```

## 📚 관련 문서

- [VLA 멀티 서버 설정](/home/soda/vla/docs/VLA_MULTI_SERVER_SETUP.md)
- [API 서버 관리 스크립트](/home/soda/vla/scripts/manage_api_server.sh)
- [RoboVLMs 추론 코드](/home/soda/vla/src/robovlms_mobile_vla_inference.py)

## ✅ 체크리스트

### 서버 시작 전
- [ ] GPU 사용 가능 확인 (`nvidia-smi`)
- [ ] 체크포인트 파일 존재 확인
- [ ] Python 환경 활성화
- [ ] 환경 변수 설정 (선택적)

### 서버 시작 후
- [ ] Health check 성공 (`curl http://localhost:8000/health`)
- [ ] 모델 로드 완료 확인 (로그)
- [ ] API Key 저장
- [ ] Jetson에서 연결 테스트

---

**작성**: 2025-12-24
**업데이트**: billy 서버 최신 커밋 반영
**버전**: 2.0
