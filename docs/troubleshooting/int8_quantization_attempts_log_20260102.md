# Jetson Orin INT8 Quantization 시도 기록

**일시**: 2026-01-02  
**목표**: BitsAndBytes INT8 quantization으로 Kosmos-2 모델 로딩

---

## 🔄 시도 내역

### 1. 초기 상태 (실패)
```
PyTorch: 2.7.0+cpu
transformers: 4.41.2
bitsandbytes: 0.48.2 (pip 설치)
accelerate: 1.7.0
```

**문제**:
- PyTorch가 CPU 전용 → CUDA 미지원
- BitsAndBytes CUDA 라이브러리 없음

### 2. BitsAndBytes 소스 빌드 (성공)
```bash
cd /tmp/bitsandbytes_jetson
git checkout 0.43.1
cmake .. -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="87"
make -j8
```

**결과**:
- ✅ `libbitsandbytes_cuda122.so` (4.0 MB) 빌드 성공
- ✅ BitsAndBytesConfig 생성 가능

### 3. PyTorch CUDA 설치 (성공)
```bash
# JetPack 6.0 + CUDA 12.2 공식 빌드
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
sudo pip3 uninstall -y torch
pip3 install --user torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

**결과**:
- ✅ PyTorch: 2.3.0 (CUDA 12.2 지원)
- ✅ GPU 인식됨

### 4. INT8 로딩 시도 (실패)
```python
model = Kosmos2ForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
```

**에러**:
```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
```

**스택 트레이스**:
```
transformers/modeling_utils.py:3820 → dispatch_model()
accelerate/big_modeling.py:502 → model.to(device)
transformers/modeling_utils.py:2702 → raise ValueError
```

### 5. Accelerate 업그레이드 (실패)
```bash
pip install --upgrade accelerate==1.12.0
```

**결과**:
- ✅ Accelerate: 1.7.0 → 1.12.0
- ❌ 동일한 에러 발생 (버그 미수정)

---

## 🔍 근본 원인 분석

### 문제 코드 위치

**accelerate/big_modeling.py:502**
```python
if device != "disk":
    model.to(device)  # ← BitsAndBytes 예외 처리 없음!
```

**transformers/modeling_utils.py:3820**
```python
if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
    dispatch_model(model, **device_map_kwargs)  # ← 항상 호출됨
```

### 왜 실패하는가?

1. **transformers**가 `from_pretrained()` 완료 후 `dispatch_model()` 호출
2. **accelerate**의 `dispatch_model()`이 `device_map`에 따라 `model.to(device)` 실행
3. **BitsAndBytes** 모델은 이미 GPU에 로드되어 `.to()` 호출 불가

### device_map 동작

| device_map 값 | 동작 | INT8 호환 |
|---------------|------|----------|
| `None` (기본) | CPU→GPU 자동 배치 시도 | ❌ `.to()` 호출 |
| `"auto"` | 자동 분산 배치 | ❌ `.to()` 호출 |
| `"balanced"` | 균형 분산 | ❌ `.to()` 호출 |
| `{}` (빈 dict) | 에러 발생 | ❌ |
| `{"": 0}` | GPU 0 명시 | ❌ `.to()` 호출 |

**결론**: 모든 경우에 `dispatch_model`이 `.to()` 호출

---

## 📦 현재 설치된 버전

```
Python: 3.10.12
PyTorch: 2.3.0 (CUDA 12.2) ✅
transformers: 4.41.2
bitsandbytes: 0.43.1 (소스 빌드) ✅
accelerate: 1.12.0
CUDA: 12.2.140 ✅
```

---

## 💡 해결 가능한 방법

### Option 1: FP16 사용 (✅ 검증됨)
```python
model = Kosmos2ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```
- 메모리: 3GB
- 안정적

### Option 2: Transformers 패치
`transformers/modeling_utils.py` 3820번 라인 수정:
```python
# BitsAndBytes 모델은 dispatch 건너뜀
if (not is_fsdp_enabled() and not is_deepspeed_zero3_enabled() 
    and not (hf_quantizer and hf_quantizer.quantization_config.quant_method == "bitsandbytes")):
    dispatch_model(model, **device_map_kwargs)
```

### Option 3: Accelerate 패치
`accelerate/big_modeling.py` 502번 라인 수정:
```python
# BitsAndBytes 모델 체크
from transformers.utils import is_bitsandbytes_available
if device != "disk":
    if not (is_bitsandbytes_available() and hasattr(model, 'is_quantized')):
        model.to(device)
```

### Option 4: 이전 버전 조합
호환되는 버전 찾기 (Poetry로 테스트)

---

## 🎯 다음 단계

1. Poetry 환경 확인
2. 호환 버전 조합 탐색
3. 또는 FP16으로 진행

---

**상태**: INT8 불가 (라이브러리 호환성), FP16 가능 ✅
