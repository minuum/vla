# NVIDIA 포럼 검색 결과 및 해결 방안

**검색일**: 2026-01-02  
**문제**: Jetson Orin에서 BitsAndBytes INT8 quantization 실패

---

## 🔍 NVIDIA 포럼 및 커뮤니티 검색 결과

### 핵심 발견사항

**문제 확인**: ✅ **동일한 이슈가 여러 사용자에게 발생**

**에러**:
```
ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models
```

**근본 원인**:
1. **BitsAndBytes가 이미 GPU에 모델 배치** → 추가 `.to()` 호출 불필요
2. **accelerate.dispatch_model이 자동으로 `.to()` 호출** → 충돌 발생
3. **transformers + accelerate + bitsandbytes 버전 조합 문제**

---

## ✅ 커뮤니티 해결 방법

### Solution 1: device_map 제거 (가장 많이 언급됨)

```python
# ❌ 실패
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"  # ← 이것이 문제!
)

# ✅ 성공
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    quantization_config=bnb_config
    # device_map 제거!
)
```

**하지만**: 우리는 이미 시도했고 실패했음 (transformers 4.41.2가 dispatch_model을 강제 호출)

### Solution 2: 호환 버전 다운그레이드

**HuggingFace 커뮤니티 성공 사례**:

```
transformers==4.30.0  # 4.35 미만
accelerate==0.21.0    # 1.0 미만
bitsandbytes==0.40.0  # 0.42 미만
```

**이유**: 
- transformers 4.30은 BitsAndBytes 특별 처리 포함
- accelerate 0.21은 quantized 모델 체크 있음
- 이후 버전(특히 1.0+)에서 회귀 버그 발생

### Solution 3: TensorRT 사용 (NVIDIA 공식 권장)

**Jetson 최적화**:
- NVIDIA TensorRT가 Jetson에 최적화됨
- INT8 quantization 공식 지원
- BitsAndBytes보다 빠름

---

## 🎯 Poetry 환경 테스트 계획

### Phase 1: 호환 버전 조합 테스트

**목표**: transformers 4.30 + accelerate 0.21 조합

```bash
cd /home/soda/vla/RoboVLMs
poetry add transformers@4.30.0 accelerate@0.21.0
poetry install
poetry shell
python test_int8.py
```

### Phase 2: 중간 버전 테스트

**목표**: 최신 기능 유지하면서 호환성 찾기

조합 후보:
1. transformers==4.33.0 + accelerate==0.23.0
2. transformers==4.35.0 + accelerate==0.25.0

### Phase 3: 현재 버전 유지 + Monkey Patch

**목표**: accelerate.dispatch_model 패치

```python
# accelerate/big_modeling.py의 dispatch_model 함수 패치
import accelerate.big_modeling
original_dispatch = accelerate.big_modeling.dispatch_model

def patched_dispatch(model, **kwargs):
    # BitsAndBytes 모델은 건너뜀
    if hasattr(model, 'is_loaded_in_8bit') or hasattr(model, 'is_loaded_in_4bit'):
        return model
    return original_dispatch(model, **kwargs)

accelerate.big_modeling.dispatch_model = patched_dispatch
```

---

## 📝 다음 단계

1. ✅ **Poetry 환경 확인 및 준비**
2. ⏳ **transformers 4.30 + accelerate 0.21 설치**
3. ⏳ **INT8 quantization 테스트**
4. ⏳ **성공 시 버전 고정, 실패 시 FP16**

---

**참고 자료**:
- HuggingFace Discussions: `.to is not supported` 이슈
- NVIDIA Forums: Jetson INT8 quantization 가이드
- GitHub Issues: transformers + bitsandbytes 호환성
