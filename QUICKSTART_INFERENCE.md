# 🚀 Mobile VLA 추론 서버 빠른 시작

## 📋 준비사항
- [x] 최신 체크포인트 확인 (6.4GB, val_loss=0.067)
- [x] GPU 사용 가능 (NVIDIA RTX A5000)
- [x] Python 환경 활성화
- [x] RoboVLMs 설치

## 🎯 실행 순서

### 1단계: 멀티 터미널 서버 시작 (권장)

```bash
cd /home/soda/vla
bash scripts/run_multi_terminal.sh
```

이 명령어 하나로:
- ✅ 체크포인트 자동 선택 (최신 우선)
- ✅ GPU 상태 확인
- ✅ API 서버 시작
- ✅ 로그 실시간 모니터링
- ✅ 터미널 자동 분할 (tmux)

**화면 구성:**
```
┌─────────────────────────────────────┐
│ API 서버 (상단)                      │
│ - 모델 로딩: ✅                      │
│ - 서버 실행 중: http://0.0.0.0:8000 │
├─────────────────────────────────────┤
│ 로그 모니터링 (하단)                 │
│ - 요청 로그 실시간 표시              │
│ - 에러/경고 색상 강조                │
└─────────────────────────────────────┘
```

### 2단계: 서버 테스트

**새 터미널에서:**

```bash
# Health check
curl http://localhost:8000/health

# 전체 테스트 (API Key 필요)
python3 scripts/test_inference_server.py --api-key "로그에서-확인한-키"

# 또는 환경변수로
export VLA_API_KEY="로그에서-확인한-키"
python3 scripts/test_inference_server.py --test all
```

### 3단계: 실제 추론 테스트

```bash
# 이미지로 추론 테스트
python3 scripts/test_inference_server.py \
  --test predict \
  --image "your_image.jpg" \
  --instruction "Navigate to the left bottle"

# 성능 벤치마크
python3 scripts/test_inference_server.py --test benchmark
```

## 🔑 API Key 확인 방법

서버 시작 시 로그에서 확인:
```
⚠️  VLA_API_KEY 환경 변수가 없습니다!
생성된 API Key: jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc
다음 명령어로 저장하세요:
export VLA_API_KEY="jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc"
```

**영구 저장:**
```bash
echo 'export VLA_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

## 🛠️ 고급 옵션

### RoboVLMs 스타일 서버 (forward_continuous)

```bash
# 환경변수 설정
export VLA_CHECKPOINT_PATH="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
export VLA_WINDOW_SIZE=2
export VLA_CHUNK_SIZE=10

# RoboVLMs 서버 시작
python3 api_server_robovlms.py
```

### 수동 실행 (두 터미널)

**터미널 1:**
```bash
bash scripts/start_inference_server.sh
```

**터미널 2:**
```bash
bash scripts/monitor_inference_server.sh
```

## 📊 기대 성능

```
✅ GPU: NVIDIA RTX A5000
✅ 모델: Chunk5 Epoch6 (val_loss=0.067)
✅ 평균 지연: ~120ms
✅ FPS: ~8-10
✅ GPU 메모리: ~6-8GB
```

## 🔄 tmux 사용법

```bash
# 세션 접속
tmux attach -t vla_inference

# 패널 이동
Ctrl+b, 위/아래 화살표

# Detach (백그라운드 실행)
Ctrl+b, d

# 세션 종료
tmux kill-session -t vla_inference
```

## 🐛 트러블슈팅

### GPU 메모리 부족
```bash
# 프로세스 확인 및 종료
nvidia-smi
pkill -f python
```

### 포트 충돌 (8000)
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000

# 종료
kill -9 <PID>
```

### 체크포인트 경로 오류
```bash
# 수동 지정
export VLA_CHECKPOINT_PATH="/home/soda/vla/ROS_action/last.ckpt"
```

## 📞 Jetson에서 연결

```python
import requests
import base64

BILLY_API = "http://100.xxx.xxx.xxx:8000"
API_KEY = "your-api-key"

# 이미지 인코딩
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 추론 요청
response = requests.post(
    f"{BILLY_API}/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "image": image_b64,
        "instruction": "Navigate to the left box"
    }
)

action = response.json()["action"]
print(f"액션: {action}")
```

## 📚 추가 문서

- [상세 가이드](docs/INFERENCE_SERVER_GUIDE.md)
- [멀티 서버 설정](docs/VLA_MULTI_SERVER_SETUP.md)
- [RoboVLMs 추론](src/robovlms_mobile_vla_inference.py)

---

**준비 완료! 🎉**

이제 아래 명령어로 시작하세요:
```bash
bash scripts/run_multi_terminal.sh
```
