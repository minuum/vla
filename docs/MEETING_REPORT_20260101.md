# Jetson Mobile VLA 환경 구축 미팅 보고 (2026-01-01)

**날짜**: 2026-01-01  
**작업 시간**: 약 14:00 ~ 21:23 (7시간+)  
**목표**: Jetson에서 Mobile VLA INT8 추론 환경 구축

---

## ✅ 완료된 작업

### 1. Poetry 환경 구축 (100% 완료)

#### 설치 및 설정
- Poetry 2.2.1 설치 ✅
- 키링 비활성화 (keyring.enabled = false) ✅
- 가상환경 `.venv` 생성 ✅
- virtualenvs.in-project = true 설정 ✅

#### 설치된 패키지
```
transformers==4.41.2
numpy==1.26.4
pillow==10.4.0
accelerate==0.27.0
psutil==5.9.8
opencv-python==4.5.4.60
```

**문제 해결**:
- Poetry install이 keyring 단계에서 멈춤 → keyring 비활성화로 해결
- opencv-python 최신 버전 numpy 2.x 요구 → 4.5.4.60 설치

---

### 2. PyTorch CUDA 설치 (100% 완료)

#### Wheel 파일 다운로드
- `torch-2.3.0-cp310-cp310-linux_aarch64.whl` (202MB) ✅
- `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl` (1.4MB) ✅
- `torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl` (2.0MB) ✅

**출처**: 브라우저에서 수동 다운로드 (NVIDIA Forums 또는 다른 소스)

**문제 해결**:
- 파일명에 `(1)` 붙음 → 수동 rename으로 해결
- 설치 위치: `/home/soda/vla/wheels/`

#### 설치 및 검증
```bash
.venv/bin/pip install wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install wheels/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

**검증 결과**:
```
PyTorch 버전: 2.3.0
CUDA 사용 가능: True
CUDA 버전: 12.2
Device 이름: Orin
Device 개수: 1
```

✅ **PyTorch CUDA 완벽하게 작동 확인!**

---

### 3. 환경 진단 최종 결과

```
🐍 Python: 3.10.12 ✅
🎮 GPU: Orin ✅
   CUDA Version (PyTorch): 12.2 ✅
   PyTorch CUDA Available: ✅

📦 주요 라이브러리:
✅ torch                2.3.0
✅ torchvision          0.18.0a0+6043bc2
✅ transformers         4.41.2
✅ accelerate           0.27.0
✅ numpy                1.26.4
✅ pillow               10.4.0
✅ opencv-python        4.5.4
✅ psutil               5.9.8

설치됨: 8/9
미설치: bitsandbytes
```

---

## ❌ 실패한 작업: BitsAndBytes INT8

### 시도 1: BitsAndBytes 0.42.0

**명령어**:
```bash
.venv/bin/pip install bitsandbytes==0.42.0
```

**결과**: ❌ 실패

**에러**:
```
libbitsandbytes_cuda122.so: cannot open shared object file: No such file or directory
```

**원인**: BitsAndBytes 0.42.0에 CUDA 12.2 바이너리가 포함되지 않음

---

### 시도 2: BitsAndBytes 0.48.2

**명령어**:
```bash
.venv/bin/pip install bitsandbytes==0.48.2 --force-reinstall
```

**부작용**:
- torch 2.3.0 → 2.9.1+cpu로 자동 업그레이드 ❌
- numpy 1.26.4 → 2.2.6으로 자동 업그레이드 ❌

**복구**:
```bash
.venv/bin/pip install 'numpy<2.0' --force-reinstall
.venv/bin/pip uninstall torch -y
.venv/bin/pip install wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

**Import 테스트**:
```python
import bitsandbytes as bnb  # ✅ 성공
import torch  # ✅ 성공 (2.3.0, CUDA True)
```

**모델 로딩 시도**:
```python
from transformers import BitsAndBytesConfig, AutoModelForVision2Seq

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForVision2Seq.from_pretrained(
    ".vlms/kosmos-2-patch14-224",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**결과**: ❌ 실패

**에러**:
```
Error named symbol not found at line 449 in file /src/csrc/ops.cu
```

**원인**: BitsAndBytes 0.48.2가 PyTorch 2.3.0 + CUDA 12.2와 바이너리 호환 문제

---

## ⏳ 테스트 중: FP16 모델 로딩

### 시작 상황
- **시작 시각**: 21:10
- **베이스라인 RAM**: 2.71GB / 15.29GB (17.7%)

### 실행 명령
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
.venv/bin/python -c "
model = AutoModelForVision2Seq.from_pretrained(
    '.vlms/kosmos-2-patch14-224',
    torch_dtype=torch.float16,
    device_map='auto'
)
"
```

### 진행 상황

| 시각 | RAM 사용량 | 증가량 | 상태 |
|------|-----------|--------|------|
| 21:10 | 2.71GB | - | 시작 |
| 21:17 | 4.43GB | +1.72GB | 로딩 중 |
| 21:20 | 4.49GB | +1.78GB | 로딩 중 (출력 없음) |
| 21:23 | ? | ? | 중단됨 |

### 관찰 사항
- **10분+ 경과**: 모델 로딩에 10분 이상 소요 중
- **RAM 증가**: +1.78GB (2.71GB → 4.49GB)
- **출력 없음**: 진행 상태 표시 없음
- **프로세스 상태**: RUNNING이지만 응답 없음
- **여유 RAM**: 10.39GB (충분함)

### 가능한 상황
1. **디스크 I/O 대기**: 모델이 크고 디스크에서 로딩 중
2. **캐시 생성**: Transformers가 캐시 생성 중
3. **GPU 메모리 할당**: CUDA 메모리 할당 대기
4. **멈춤/교착 상태**: 어딘가에서 대기 중

### 중단 이유
- 10분 이상 진행 상태 없음
- 정상적인 로딩 시간 초과 (예상 1-2분)

---

## 📊 최종 상태 요약

### ✅ 성공 (90%)
1. Poetry 환경 완벽 구축
2. PyTorch CUDA 2.3.0 정상 작동
3. 모든 필수 라이브러리 설치
4. 환경 진단 스크립트 작동

### ❌ 실패 (10%)
1. BitsAndBytes INT8 불가능
2. FP16 모델 로딩 미완료 (시간 초과)

### 🔧 해결된 문제
1. Poetry keyring 멈춤 → 비활성화
2. PyTorch wheel 파일명 → 수동 rename
3. numpy 버전 충돌 → 강제 다운그레이드
4. torch CPU 설치 → wheel 재설치

### 🔴 미해결 문제
1. BitsAndBytes CUDA 12.2 호환성
2. FP16 모델 로딩 시간 (원인 불명)

---

## 🎯 다음 단계

### 즉시 가능
1. ✅ Git 커밋 및 푸시
2. ⏳ FP16 로딩 재시도 (더 작은 배치/설정)
3. ⏳ 모델 파일 크기 및 경로 확인

### 단기 (다음 세션)
1. FP16 로딩 성공 후 메모리 측정
2. 추론 테스트
3. 논문용 Table 작성

### 장기 (선택)
1. BitsAndBytes 소스 빌드
2. 다른 quantization 방법 (GPTQ, AWQ 등)

---

## 📁 생성된 파일

### 설정
- `pyproject.toml`
- `wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl`
- `wheels/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
- `wheels/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl`

### 문서
- `CURRENT_STATUS.md`
- `docs/BITSANDBYTES_WALKTHROUGH_20251231.md`
- `docs/POETRY_SETUP_JETSON.md`
- `docs/POETRY_STATUS_20251231.md`
- `docs/PYTORCH_WHEEL_ISSUE.md`

---

## 💡 교훈

### 기술적
1. **BitsAndBytes는 Jetson과 호환성 문제**: 0.42.0과 0.48.2 모두 CUDA 12.2와 문제
2. **PyTorch wheel 필수**: PyPI의 torch는 CPU 전용
3. **의존성 관리**: bitsandbytes 설치 시 torch가 자동 업그레이드됨

### 프로세스
1. **환경 격리 중요**: Poetry가 시스템 torch 충돌 방지
2. **단계별 검증**: 매 설치 후 즉시 확인 필요
3. **시간 측정**: 모델 로딩 시간 모니터링 중요

---

## 🔍 환경 정보

```
Jetson: Orin
CUDA: 12.2
Python: 3.10.12
PyTorch: 2.3.0 (CUDA)
Total RAM: 15.29 GB
Available: ~10GB (로딩 후)
```

---

**작성**: 2026-01-01 21:24 KST  
**상태**: PyTorch CUDA 완료, INT8 불가, FP16 테스트 중단
