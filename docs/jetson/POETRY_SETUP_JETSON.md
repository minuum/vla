# Poetry 환경 구축 가이드 (Jetson)

**작성일**: 2025-12-31  
**목적**: 시스템 torch와 충돌 없이 독립된 개발 환경 구축

---

## 🎯 왜 Poetry인가?

### 문제 상황
- 시스템에 torch 2.7.0 (CPU 전용)이 `/usr/local`에 설치됨
- pip로 제거 불가능 (권한 문제)
- LD_LIBRARY_PATH에 CUDA 경로 누락

### Poetry 해결책
✅ 시스템 패키지와 완전히 독립된 가상환경  
✅ 프로젝트별 의존성 관리 (`pyproject.toml`)  
✅ 버전 충돌 방지 (lock 파일)  
✅ 환경이 꼬이면 `.venv` 폴더만 삭제하면 끝

---

## 🚀 빠른 시작

### 1단계: 자동 설정 실행

```bash
cd /home/soda/vla
./setup_poetry_jetson.sh
```

이 스크립트가 자동으로:
- Poetry 설치 확인/설치
- PyTorch wheel 다운로드
- 환경 변수 설정 (.envrc)
- Poetry 환경 생성 및 패키지 설치

### 2단계: 환경 활성화

```bash
source .envrc        # CUDA 환경 변수 로드
poetry shell         # Poetry 가상환경 활성화
```

### 3단계: 확인

```bash
# PyTorch CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 환경 진단
python scripts/diagnose_jetson_environment.py
```

---

## 📦 수동 설정 (자동 스크립트 실패 시)

### 1. Poetry 설치

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Poetry 설정

```bash
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true
```

### 3. PyTorch Wheel 다운로드

```bash
mkdir -p wheels
cd wheels

# Torch
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

# TorchVision
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

cd ..
```

### 4. 환경 변수 설정

```bash
cat > .envrc << 'EOF'
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$PWD:$PWD/src:$PWD/RoboVLMs:$PYTHONPATH
EOF

source .envrc
```

### 5. Poetry 환경 생성 및 패키지 설치

```bash
# 가상환경 생성
poetry install

# 가상환경 활성화
poetry shell

# PyTorch 설치
pip install wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

# 나머지 패키지
pip install transformers==4.41.2 bitsandbytes==0.42.0 accelerate
pip install 'numpy<2.0' pillow opencv-python psutil tqdm
```

---

## 🔍 환경 확인

### Poetry 환경 내에서:

```bash
# 1. Python 버전
python --version  # 3.10.12

# 2. PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# 출력: 2.3.0, True

# 3. BitsAndBytes
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
# 출력: 0.42.0

# 4. 환경 진단
python scripts/diagnose_jetson_environment.py
```

---

## 🎨 사용 방법

### 일반 사용

```bash
# 프로젝트 디렉토리로 이동
cd /home/soda/vla

# 환경 변수 로드
source .envrc

# Poetry 환경 활성화
poetry shell

# 스크립트 실행
python scripts/measure_baseline_memory.py
python scripts/measure_chunk_performance.py
```

### 새 터미널에서 (매번 필요)

```bash
cd /home/soda/vla
source .envrc
poetry shell
```

### zsh 자동화 (선택)

`~/.zshrc`에 추가:

```bash
# VLA Poetry 환경
alias vla='cd /home/soda/vla && source .envrc && poetry shell'
```

사용:
```bash
vla  # 한 번에 프로젝트 디렉토리 이동, 환경 변수 로드, Poetry 활성화
```

---

## 🐛 트러블슈팅

### Q: "torch 2.7.0+cpu is required" 에러

**A**: 시스템 torch와 충돌. Poetry 환경이 제대로 활성화되지 않았습니다.

```bash
# 확인
which python
# 출력이 /home/soda/vla/.venv/bin/python이어야 함

# 해결
deactivate  # 기존 환경 비활성화
poetry shell  # Poetry 환경 재활성화
```

### Q: "CUDA not available" 

**A**: 환경 변수 누락

```bash
# .envrc 다시 로드
source .envrc

# 확인
echo $LD_LIBRARY_PATH
# /usr/local/cuda-12.2/lib64가 포함되어야 함
```

### Q: Poetry 환경이 꼬였어요

**A**: 환경 재생성

```bash
# 기존 환경 삭제
rm -rf .venv

# 다시 생성
poetry install
```

---

## 📊 설정 파일

### pyproject.toml
- Poetry 의존성 정의
- 버전 고정

### poetry.lock
- 정확한 버전 잠금
- 재현 가능한 환경

### .envrc
- CUDA 환경 변수
- PYTHONPATH 설정

---

## ✅ 체크리스트

설정 후 확인:

- [ ] `poetry --version` 실행됨
- [ ] `source .envrc` 후 `echo $LD_LIBRARY_PATH`에 CUDA 경로 포함
- [ ] `poetry shell` 후 `which python`이 `.venv/bin/python`
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` = True
- [ ] `python scripts/diagnose_jetson_environment.py` 성공

---

**다음 단계**: Poetry 환경에서 모델 로딩 테스트

```bash
poetry shell
python scripts/measure_model_memory.py
```
