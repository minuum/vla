# BitsAndBytes INT8 문제 및 해결 방안

**일시**: 2026-01-02  
**문제**: Jetson Orin에서 BitsAndBytes INT8 모델 로딩 실패

---

## ❌ 발생한 에러

```
Error named symbol not found at line 449 in file /src/csrc/ops.cu
```

### 원인 분석

1. **BitsAndBytes 버전**: 0.48.2 (권장 0.43.1보다 높음)
2. **CUDA 바이너리**: `libbitsandbytes_cuda120.so` 존재
3. **Jetson CUDA**: 12.2 (PyTorch 기준)
4. **문제**: 사전 빌드된 CUDA 커널이 Jetson Orin의 ARM64 + CUDA 조합과 호환되지 않음

### 상세 정보

```bash
✅ BitsAndBytes: 0.48.2
📁 설치 경로: ~/.local/lib/python3.10/site-packages/bitsandbytes

🔧 CUDA 라이브러리:
   - libbitsandbytes_cuda120.so
   - libbitsandbytes_cuda122.so
   - libbitsandbytes_cuda125.so
   - libbitsandbytes_cuda126.so
   - libbitsandbytes_cuda130.so
```

**판단**: x86_64용으로 빌드된 바이너리, Jetson ARM64에서 CUDA 커널 심볼 해석 실패

---

## ✅ 해결 방안

### Option 1: FP16 사용 (채택)

**장점**:
- ✅ 즉시 사용 가능
- ✅ Jetson에서 안정적
- ✅ 추론 속도 빠름

**단점**:
- ⚠️ INT8 대비 2배 메모리 사용

**구현**:
```python
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # FP16
    device_map="auto",
    low_cpu_mem_usage=True
)
```

**메모리 비교**:
- INT8: ~1-1.5GB (이론상, 작동 안 함)
- FP16: ~2-3GB (실제 사용)
- FP32: ~4-6GB

---

### Option 2: BitsAndBytes 소스 빌드 (추후)

Jetson용 BitsAndBytes를 직접 빌드하는 방법:

```bash
# CUDA Toolkit 설치 (JetPack에 포함)
sudo apt install cuda-toolkit-12-2

# BitsAndBytes 0.43.1 소스 빌드
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout 0.43.1

# ARM64 + CUDA 12.2 빌드
CUDA_VERSION=122 make cuda12x
python setup.py install
```

**주의**: 빌드 시간 오래 걸림 (30분~1시간)

---

### Option 3: TensorRT 양자화 (최적)

Jetson에 최적화된 양자화 방법:

```python
import tensorrt as trt
# TensorRT INT8 calibration
# Jetson 전용 최적화
```

**장점**:
- ✅ Jetson 전용 최적화
- ✅ INT8 지원
- ✅ 가장 빠른 추론 속도

**단점**:
- ⚠️ 복잡한 설정
- ⚠️ 학습 시간 필요

---

## 📊 현재 상태

### 측정 가능한 항목

| 항목 | INT8 | FP16 | 상태 |
|------|------|------|------|
| BitsAndBytes Config | ✅ | - | 생성 가능 |
| 모델 로딩 | ❌ | ✅ | FP16만 가능 |
| 메모리 측정 | ❌ | ✅ | 진행 중 |
| 추론 속도 | ❌ | ⏳ | 측정 예정 |

### 논문용 데이터

**원래 계획**:
- RoboVLMs FP32: ~10GB
- Mobile VLA INT8: ~2GB
- **절감률: 80%**

**실제 가능**:
- RoboVLMs FP32: ~10GB (측정 가능)
- Mobile VLA FP16: ~3GB (측정 중)
- **절감률: 70%**

**결론**: FP16도 충분히 의미 있는 메모리 절감

---

## 🎯 다음 단계

1. ✅ FP16 모델 로딩 테스트 진행 중
2. ⏳ 메모리 사용량 측정
3. ⏳ 추론 속도 벤치마크
4. ⏳ 논문 Table 작성 (FP16 기준)

---

## 📚 참고 자료

1. **BitsAndBytes Jetson 이슈**:
   - https://github.com/TimDettmers/bitsandbytes/issues/XXX
   - ARM64 빌드 가이드

2. **Jetson 최적화**:
   - TensorRT 공식 문서
   - NVIDIA Jetson Developer Guide

3. **대안 프로젝트**:
   - OpenVLA Jetson 배포
   - Mobile-ALOHA Jetson 설정

---

**상태**: FP16으로 우회 진행 중 ⏳
