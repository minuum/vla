# 🚀 Jetson Mobile VLA 환경 구축 완료 상태 (2025-12-31)

**다른 IP/세션에서 이어서 작업하기 위한 완전한 가이드**

---

## ✅ 완료된 작업

### 1. 환경 진단 및 측정 (100% 완료)

#### 측정 스크립트 (4개)
```bash
scripts/diagnose_jetson_environment.py       # 환경 진단
scripts/measure_baseline_memory.py           # 베이스라인 메모리
scripts/measure_chunk_performance.py         # Chunk 성능 비교
scripts/measure_model_memory.py              # 모델 메모리 타임라인
```

#### 측정 결과
- **베이스라인 메모리**: 2.27GB/15.29GB (12.71GB 여유)
- **Chunk 10 vs 5**: API 호출 46.2% 감소 (7회 vs 13회)
- **환경**: Python 3.10.12, CUDA 12.2, GPU Orin

### 2. BitsAndBytes 문제 분석 (100% 완료)

**6가지 시도 및 원인 파악**:
1. BitsAndBytes 0.48.2: CUDA 12.2 바이너리 호환 문제 ❌
2. BitsAndBytes 0.42.0: CUDA 12.2 바이너리 미포함 ❌
3. FP16 로딩: PyTorch가 CPU 전용으로 설치됨 ❌
4. PyTorch 다운그레이드: PyPI는 모두 CPU 전용 ❌
5. 시스템 torch 발견: `/usr/local`에 2.7.0+cpu 설치됨 🔴
6. LD_LIBRARY_PATH 누락: CUDA 경로 없음 🔴

**근본 원인 3가지**:
- 시스템 torch 2.7.0+cpu (제거 불가능)
- LD_LIBRARY_PATH에 CUDA 경로 누락
- PyPI torch는 모두 CPU 전용

**문서**:
- `docs/BITSANDBYTES_WALKTHROUGH_20251231.md`
- `docs/MEETING_FEEDBACK_RESULTS_20251231.md`

### 3. Poetry 환경 구축 (95% 완료)

#### 설치 완료
- Poetry 2.2.1 ✅
- 키링 비활성화 (keyring.enabled = false) ✅
- 가상환경 `.venv` 생성 ✅

#### 설치된 패키지
```
transformers==4.41.2
numpy==1.26.4
pillow==10.4.0
accelerate==0.27.0
psutil==5.9.8
opencv-python==4.5.4.60
```

#### 설정 파일
- `pyproject.toml`: Poetry 의존성 정의
- `activate_poetry.sh`: 환경 활성화 헬퍼
- `setup_poetry_jetson.sh`: 자동 설정 스크립트

#### 문서
- `docs/POETRY_SETUP_JETSON.md`: 상세 가이드
- `docs/POETRY_STATUS_20251231.md`: 완료 상태

---

## ⏳ 다음 단계 (5% 남음)

### 1. PyTorch CUDA Wheel 다운로드 및 설치

**필요한 Wheel**:
```
torch-2.3.0-cp310-cp310-linux_aarch64.whl
torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
```

**다운로드 위치**:
- NVIDIA Forums: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
- JetPack 6.0 섹션 → torch 2.3.0 선택

**파일 저장 위치**:
```bash
/home/soda/vla/wheels/
```

**설치 방법**:
```bash
cd /home/soda/vla
export PATH="$HOME/.local/bin:$PATH"

# Poetry 환경에 설치
.venv/bin/pip install wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
```

### 2. CUDA 환경 변수 설정

**매번 필요** (세션마다):
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
```

**자동화** (.envrc 사용):
```bash
cd /home/soda/vla
source .envrc  # (activate_poetry.sh가 생성)
```

### 3. 환경 확인

```bash
# Poetry 환경 활성화
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Python 실행 (Poetry 환경)
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

**예상 출력**:
```
PyTorch: 2.3.0
CUDA: True
Device: Orin
```

### 4. 모델 로딩 테스트

```bash
# FP16으로 시작 (안전)
.venv/bin/python scripts/measure_model_memory.py

# 성공 시 INT8 시도
.venv/bin/pip install bitsandbytes==0.42.0
.venv/bin/python scripts/measure_model_memory.py --int8
```

---

## 🔄 다른 IP/세션에서 시작하기

### Step 1: Repository Clone (새 머신인 경우)

```bash
cd /home/soda
git clone <repository-url> vla
cd vla
git checkout feature/inference-integration
git pull
```

### Step 2: Poetry 확인

```bash
# Poetry 설치 확인
export PATH="$HOME/.local/bin:$PATH"
poetry --version

# 없으면 설치
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 3: 가상환경 있는지 확인

```bash
cd /home/soda/vla

# .venv 존재 확인
ls -la .venv

# 없으면 재생성
poetry install --no-interaction
.venv/bin/pip install opencv-python==4.5.4.60
```

### Step 4: PyTorch Wheel 다운로드

```bash
# Wheels 디렉토리 확인
ls -la wheels/

# 없으면 다운로드 (NVIDIA Forums에서)
mkdir -p wheels
cd wheels
# 수동 다운로드: torch-2.3.0-cp310-cp310-linux_aarch64.whl
# 수동 다운로드: torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
```

### Step 5: 환경 활성화 및 테스트

```bash
cd /home/soda/vla

# 환경 변수
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# PyTorch 설치 (wheel 있는 경우)
.venv/bin/pip install wheels/torch-*.whl
.venv/bin/pip install wheels/torchvision-*.whl

# 환경 진단
.venv/bin/python scripts/diagnose_jetson_environment.py
```

---

## 📊 현재 진행률

| 카테고리 | 완료율 | 상태 |
|---------|--------|------|
| 환경 진단 | 100% | ✅ 완료 |
| 측정 스크립트 | 100% | ✅ 완료 |
| BitsAndBytes 분석 | 100% | ✅ 완료 |
| Poetry 설정 | 95% | ✅ 거의 완료 |
| PyTorch 설치 | 0% | ⏳ Wheel 다운로드 필요 |
| 모델 로딩 | 0% | ⏳ PyTorch 후 |

**전체**: 약 75% 완료

---

## 🔑 핵심 명령어 요약

```bash
# 프로젝트로 이동
cd /home/soda/vla

# PATH 설정
export PATH="$HOME/.local/bin:$PATH"

# CUDA 환경 변수
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Poetry 환경에서 Python 실행
.venv/bin/python <스크립트>

# 환경 진단
.venv/bin/python scripts/diagnose_jetson_environment.py

# 베이스라인 측정
.venv/bin/python scripts/measure_baseline_memory.py
```

---

## 📁 주요 파일 위치

### 스크립트
- `scripts/diagnose_jetson_environment.py`
- `scripts/measure_baseline_memory.py`
- `scripts/measure_chunk_performance.py`
- `scripts/measure_model_memory.py`

### 설정
- `pyproject.toml` - Poetry 의존성
- `.venv/` - Poetry 가상환경
- `wheels/` - PyTorch wheel 저장 (다운로드 필요)

### 문서
- `docs/BITSANDBYTES_WALKTHROUGH_20251231.md`
- `docs/POETRY_SETUP_JETSON.md`
- `docs/POETRY_STATUS_20251231.md`
- `docs/PYTORCH_WHEEL_ISSUE.md`

### 로그
- `logs/poetry_install_final.log`
- `logs/baseline_memory.json`
- `logs/chunk_performance_comparison.json`

---

## 🎯 최종 목표

1. ✅ Poetry 환경 구축
2. ⏳ PyTorch CUDA 2.3.0 설치
3. ⏳ FP16 모델 로딩 테스트
4. ⏳ (선택) INT8 quantization 테스트
5. ⏳ 논문용 메모리 측정 Table 작성

---

**마지막 업데이트**: 2025-12-31 16:08 KST  
**Git Commit**: 9379e4c0  
**Branch**: feature/inference-integration
