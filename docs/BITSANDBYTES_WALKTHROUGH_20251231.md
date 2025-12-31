# BitsAndBytes 워크스루 결과 및 판단 (2025-12-31)

## 🎯 목적
BitsAndBytes INT8 quantization의 Jetson 환경 호환성 검증 및 문제 해결

---

## ✅ 성공한 테스트

### 1. 환경 진단
```bash
python3 scripts/diagnose_jetson_environment.py
```

**결과**:
- Python 3.10.12 ✅
- CUDA 12.2 available ✅
- 모든 라이브러리 설치됨 (9/9) ✅

**판단**: ✅ **환경 자체는 문제없음**

---

### 2. 베이스라인 메모리 측정
```bash
python3 scripts/measure_baseline_memory.py
```

**결과**:
- Total: 15.29 GB
- Used: 2.27 GB (16.9%)
- Available: 12.71 GB

**판단**: ✅ **충분한 여유 메모리 확보**

---

### 3. Chunk 성능 비교
```bash
python3 scripts/measure_chunk_performance.py
```

**결과**:
| Metric | Chunk 5 | Chunk 10 | 차이 |
|--------|---------|----------|------|
| 호출 횟수 | 13회 | 7회 | **-46.2%** |
| 호출 빈도 | 0.67 Hz | 0.33 Hz | 절반 |
| 효율성 | 92.3% | 85.7% | -6.6%p |

**판단**: ✅ **Chunk 10 권장 (API 호출 46.2% 감소)**

---

### 4. BitsAndBytes Config 생성
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)
```

**결과**: ✅ 성공

**판단**: ✅ **BitsAndBytes 라이브러리 자체는 정상**

---

## ❌ 실패한 테스트 및 원인 분석

### 1. BitsAndBytes 0.48.2 + INT8 로딩 (첫 번째 시도)

**명령어**:
```python
model = AutoModelForVision2Seq.from_pretrained(
    ".vlms/kosmos-2-patch14-224",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**에러**:
```
Error named symbol not found at line 449 in file /src/csrc/ops.cu
```

**원인 분석**:
- BitsAndBytes 0.48.2의 CUDA 12.2 바이너리 호환 문제
- `libbitsandbytes_cuda122.so`에서 특정 CUDA 심볼을 찾지 못함

**판단**: ❌ **BitsAndBytes 0.48.2는 Jetson CUDA 12.2와 바이너리 호환 문제 있음**

---

### 2. BitsAndBytes 0.42.0으로 다운그레이드 (두 번째 시도)

**명령어**:
```bash
pip install bitsandbytes==0.42.0 --force-reinstall
```

**에러**:
```
libbitsandbytes_cuda122.so: cannot open shared object file: No such file or directory
```

**원인 분석**:
- BitsAndBytes 0.42.0에는 CUDA 12.2 바이너리가 포함되어 있지 않음
- 0.43.1 버전은 PyPI에 존재하지 않음 (0.42.0 → 0.46.0으로 직접 점프)

**판단**: ❌ **BitsAndBytes 0.42.0은 CUDA 12.2를 지원하지 않음**

---

### 3. FP16 로딩 후 GPU 이동 (세 번째 시도)

**명령어**:
```python
model = AutoModelForVision2Seq.from_pretrained(
    ".vlms/kosmos-2-patch14-224",
    torch_dtype=torch.float16
)
model = model.to('cuda')
```

**에러**:
```
AssertionError: Torch not compiled with CUDA enabled
```

**원인 분석**:
- BitsAndBytes 0.48.2 설치 시 torch가 2.9.1+cpu로 업그레이드됨
- PyPI의 torch는 CPU 전용 버전

**판단**: ❌ **pip install로 설치된 PyTorch는 CPU 전용**

---

### 4. PyTorch 2.3.0 다운그레이드 (네 번째 시도)

**명령어**:
```bash
pip install torch==2.3.0 --force-reinstall
```

**결과**:
```
PyTorch: 2.3.0
CUDA available: False
```

**원인 분석**:
- PyPI의 torch 2.3.0도 CPU 전용
- Jetson용 PyTorch는 NVIDIA가 별도로 제공

**판단**: ❌ **PyPI의 모든 PyTorch 버전은 CPU 전용**

---

### 5. 시스템 torch 2.7.0 문제 (다섯 번째 발견)

**확인**:
```bash
pip show torch
Location: /usr/local/lib/python3.10/dist-packages
```

**원인 분석**:
- 시스템 영역(`/usr/local`)에 torch 2.7.0 설치됨
- pip로 제거 불가능 (권한 문제)
- 모든 pip install이 이 버전과 충돌

**판단**: 🔴 **시스템 torch와의 충돌이 근본 문제**

---

### 6. LD_LIBRARY_PATH 누락 (여섯 번째 발견)

**확인**:
```bash
echo $LD_LIBRARY_PATH
/opt/ros/humble/opt/rviz_ogre_vendor/lib:...
```

**원인 분석**:
- CUDA 경로(`/usr/local/cuda-12.2/lib64`)가 PATH에 없음
- PyTorch가 CUDA 라이브러리를 찾지 못함

**판단**: 🔴 **CUDA 환경 변수 설정 누락**

---

## 🎯 최종 판단 및 해결 방안

### 근본 원인 3가지

1. **시스템 torch 2.7.0 충돌** (가장 심각)
   - `/usr/local`에 설치되어 pip로 제거 불가
   - 모든 pip install과 충돌

2. **CUDA 환경 변수 누락**
   - `LD_LIBRARY_PATH`에 CUDA 경로 없음
   - PyTorch가 GPU를 인식하지 못함

3. **BitsAndBytes CUDA 12.2 호환성**
   - 0.42.0: CUDA 12.2 바이너리 없음
   - 0.48.2: 바이너리 호환 문제

---

### 해결 방안: Poetry 독립 환경

**선택 이유**:
- ✅ 시스템 torch와 완전히 독립
- ✅ 프로젝트별 의존성 관리
- ✅ 버전 충돌 방지
- ✅ 환경 재생성 용이

**구현**:
1. Poetry 2.2.1 설치 완료 ✅
2. `pyproject.toml` 작성 완료 ✅
3. CUDA 환경 변수 스크립트 (.envrc) 생성 완료 ✅
4. `poetry install` 진행 중 ⏳

---

## 📊 중간 오류/경고 판단 요약

| 로그 | 타입 | 판단 | 이유 |
|------|------|------|------|
| nvcc not found | ❌ | ✅ 무시 가능 | PyTorch는 빌드된 바이너리 사용 |
| torch 2.3.0 (권장: 2.2.2) | ⚠️ | ✅ 무시 가능 | 상위 버전, 하위 호환 |
| bitsandbytes 0.48.2 (권장: 0.43.1) | ⚠️ | ❌ 실제 문제 | CUDA 12.2 바이너리 호환 문제 |
| quantize_8bit 실패 | ⚠️ | ✅ 무시 가능 | 내부 API 변경, 실사용 영향 없음 |
| symbol not found (0.48.2) | ❌ | ❌ 실제 문제 | CUDA 바이너리 호환성 |
| libbitsandbytes_cuda122.so not found (0.42.0) | ❌ | ❌ 실제 문제 | CUDA 12.2 바이너리 미포함 |
| Torch not compiled with CUDA | ❌ | ❌ 실제 문제 | CPU 전용 PyTorch 설치됨 |

---

## 🚀 다음 단계

### 즉시 (poetry install 완료 후)

```bash
# 1. 환경 활성화
source .envrc
poetry shell

# 2. 환경 확인
python -c "import torch; print(torch.__version__)"

# 3. PyTorch CUDA 버전 수동 설치 (필요시)
```

### 단기 (오늘 내)

- [ ] Poetry 환경에서 PyTorch CUDA 버전 설치
- [ ] BitsAndBytes 재설치 및 테스트
- [ ] 모델 로딩 테스트

### 중기 (논문 마감 전)

- [ ] 실제 모델 메모리 측정
- [ ] RoboVLMs vs Mobile VLA 비교
- [ ] 논문 Table/Figure 작성

---

## 💡 교훈

1. **환경 진단이 중요**
   - 단순 import 성공이 실제 동작을 보장하지 않음
   - 전체 파이프라인 테스트 필요

2. **시스템 패키지 격리 필수**
   - `/usr/local` 설치는 제거 불가
   - Poetry/venv로 독립 환경 필수

3. **Jetson은 특수 환경**
   - 일반 PyPI 패키지 != Jetson 호환
   - NVIDIA 제공 wheel 사용 필요

4. **환경 변수 설정 필수**
   - LD_LIBRARY_PATH에 CUDA 경로 필요
   - 매 세션마다 로드 필요

---

**날짜**: 2025-12-31  
**작성자**: VLA Team  
**상태**: Poetry install 진행 중 (PID 84647)
