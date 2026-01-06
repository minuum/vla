# VLA Component Tests - Quick Guide

로봇 단에서 실행 가능한 VLA 컴포넌트별 단위 테스트

## 🧪 Available Tests

### 1. Vision Encoder Test
Vision encoder가 이미지를 제대로 처리하는지 확인
```bash
python3 scripts/test_vla_components.py --test vision
```

**확인 항목:**
- 이미지 전처리 (720x1280 → 224x224)
- Vision feature extraction
- 출력 shape, dtype, range
- NaN/Inf 값 체크

---

### 2. Language Encoder Test (Korean)
한국어 instruction이 제대로 인코딩되는지 확인
```bash
python3 scripts/test_vla_components.py --test language
```

**확인 항목:**
- 한국어 tokenization
- 한국어 vs 영어 토큰 수 비교
- Text embedding 생성
- Token ID 확인

---

### 3. Inference Sanity Check
추론 결과의 정당성 확인
```bash
python3 scripts/test_vla_components.py --test inference
```

**확인 항목:**
- 출력 범위 체크 (-100 ~ 100)
- NaN/Inf 체크
- Deterministic output (같은 입력 → 같은 출력)
- **Instruction sensitivity (Left vs Right 구분)**

---

### 4. Data Collector Compatibility
Data collector와의 호환성 확인
```bash
python3 scripts/test_vla_components.py --test compatibility
```

**확인 항목:**
- Action 범위 비교 (Data collector vs Model output)
- Gain 계산 검증
- 형식 호환성

---

## 🚀 Run All Tests

모든 테스트를 한 번에 실행:
```bash
python3 scripts/test_vla_components.py --test all
```

**예상 소요 시간:** ~2-3분
- Model loading: ~30-50초
- Each test: ~10-20초

---

## 📊 Expected Output

```
=======================================================================================
🧪 Running All VLA Component Tests
=======================================================================================

🚀 VLA Component Test Setup
...
✅ Model loaded in 47.32s

=======================================================================================
🖼️  Test 1: Vision Encoder
...
✅ Vision encoding successful!
   Processing time: 45.23ms
   ✓ No NaN values
   ✓ No Inf values

=======================================================================================
🔤 Test 2: Language Encoder (Korean)
...
[Test 1] Instruction: "가장 왼쪽 외곽으로 돌아 컵까지 가세요"
   ✅ Encoding successful!
   Tokens: 28

=======================================================================================
🧠 Test 3: Inference Sanity Check
...
✅ Inference successful!
   ✓ Values in reasonable range
   ✓ No NaN or Inf values
   ✓ Deterministic output
   ✓ Model is instruction-sensitive

=======================================================================================
📊 Test Summary
   Vision              : ✅ PASS
   Language            : ✅ PASS
   Inference           : ✅ PASS
   Compatibility       : ✅ PASS

   Total: 4/4 tests passed

🎉 All tests passed!
```

---

## 🔧 Troubleshooting

### GPU Memory 부족
```bash
# INT8 모드로 테스트
python3 scripts/test_vla_components.py --test all --checkpoint <path> --use-int8
```

### 특정 테스트만 실패
개별 테스트 재실행하여 디버깅:
```bash
python3 scripts/test_vla_components.py --test inference --verbose
```

---

## 📝 Test Results Interpretation

### Vision Encoder
- **Processing time < 50ms**: ✅ Good
- **No NaN/Inf**: ✅ Critical
- **Output shape = [B, N, D]**: ✅ Expected

### Language Encoder
- **Korean tokens > English tokens**: ✅ Expected (한국어는 음절 단위)
- **Token IDs 범위**: 보통 0-50000 사이

### Inference Sanity
- **X axis (forward/backward)**:
  - Positive = Forward (기대)
  - Negative = Backward (문제)
- **Y axis (left/right)**:
  - Positive = Left
  - Negative = Right
- **Instruction sensitivity**:
  - Left vs Right instruction에서 Y값 달라야 함 (중요!)

---

## 🎯 When to Run

1. **새 모델 체크포인트 로드 후**
2. **코드 수정 후**
3. **실제 주행 전 검증**
4. **문제 디버깅 시**

---

**Created:** 2026-01-07  
**Purpose:** Pre-deployment component validation
