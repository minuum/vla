# Jetson PyTorch 환경 구성 완료

**일시**: 2026-01-02 08:16  
**상태**: ✅ 완료

---

## ✅ 구성 완료된 항목

### 1. Jetson PyTorch Wheels (성공한 버전)

**위치**: `/home/soda/vla/RoboVLMs/wheels/`

```
torch-2.3.0-jetson.whl (202 MB)
torchvision-0.18.0-jetson.whl (1.4 MB)
torchaudio-2.3.0-jetson.whl (2.0 MB)
```

**검증**:
```python
PyTorch: 2.3.0 ✅
CUDA available: True ✅
CUDA version: 12.2 ✅
Device: Orin ✅
```

### 2. Poetry 설정 업데이트

**RoboVLMs/pyproject.toml**:
```toml
torch = { path = "wheels/torch-2.3.0-jetson.whl" }
torchvision = { path = "wheels/torchvision-0.18.0-jetson.whl" }
torchaudio = { path = "wheels/torchaudio-2.3.0-jetson.whl" }
```

### 3. Git 설정

**.gitignore**:
```
# Jetson PyTorch wheels (large files - 206MB)
RoboVLMs/wheels/*.whl
Robo+/Mobile_VLA/wheels/*.whl
```

**wheels/README.md**: 재다운로드 가이드 포함 ✅

---

## 📦 설치된 라이브러리 (최종)

```
Python: 3.10.12
PyTorch: 2.3.0 (CUDA 12.2, Jetson 공식)
transformers: 4.41.2
accelerate: 1.12.0
bitsandbytes: 0.43.1 (소스 빌드)
CUDA: 12.2.140
```

---

## 🎯 INT8 Quantization 결론

### 시도 결과: ❌ 실패

**근본 원인**:
- `accelerate/big_modeling.py:502`에서 `model.to(device)` 무조건 호출
- BitsAndBytes INT8/4bit 모델은 `.to()` 불가
- transformers 4.41.2 + accelerate 1.12.0 조합에 버그 존재

**문서**:
- [int8_quantization_attempts_log_20260102.md](file:///home/soda/vla/docs/troubleshooting/int8_quantization_attempts_log_20260102.md)
- [dependencies_analysis_20260102.md](file:///home/soda/vla/docs/dependencies_analysis_20260102.md)

### 대안: ✅ FP16 사용

```python
model = Kosmos2ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```

**메모리**:
- INT8: ~2 GB (목표, 현재 불가)
- FP16: ~3 GB (사용 가능) ✅
- FP32: ~6 GB

**절감률**: 50% (FP32 → FP16)

---

## 🚀 다음 단계

### Option 1: FP16으로 Phase 2 진행 (권장)

```bash
cd /home/soda/vla
python3 jetson_local_complete_inference.py
# FP16 모드로 모델 로딩 및 추론
```

**장점**:
- ✅ 즉시 가능
- ✅ 안정적 (검증됨)
- ✅ 메모리 절감 충분 (50%)
- ✅ 논문 데이터로 사용 가능

### Option 2: Poetry 환경 테스트 (선택)

```bash
cd /home/soda/vla/RoboVLMs
poetry install
poetry shell
python -c "import torch; print(torch.__version__)"
```

**목적**: 격리된 환경에서 재테스트

---

## 📝 완료된 문서

1. ✅ [int8_quantization_attempts_log_20260102.md](file:///home/soda/vla/docs/troubleshooting/int8_quantization_attempts_log_20260102.md)
   - 5단계 시도 기록
   - 근본 원인 분석
   - 4가지 해결 방안

2. ✅ [dependencies_analysis_20260102.md](file:///home/soda/vla/docs/dependencies_analysis_20260102.md)
   - 의존성 비교
   - Poetry vs 현재 설정
   - 권장 진행 순서

3. ✅ [RoboVLMs/wheels/README.md](file:///home/soda/vla/RoboVLMs/wheels/README.md)
   - Wheel 파일 정보
   - 재다운로드 가이드
   - 검증 결과

---

## 🎊 성과

1. ✅ **nvcc 발견 및 설정**
2. ✅ **BitsAndBytes CUDA 빌드 성공**
3. ✅ **Jetson PyTorch CUDA 설치 성공**
4. ✅ **GPU 인식 및 정상 동작**
5. ✅ **FP16 모델 로딩 검증**
6. ✅ **Wheels 영구 저장 및 문서화**

---

**최종 권장**: FP16으로 Phase 2 진행 ✅
