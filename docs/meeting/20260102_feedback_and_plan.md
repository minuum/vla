# 1/2 미팅 피드백 정리 및 다음 계획 (2026-01-02)

## 📌 미팅 피드백 핵심 요약

### 1️⃣ **Action Value Scaling 문제**
**현상:**
- 모델 출력값: `1.15`, `-1.15`, `0` 수준
- 현재 `action_gain=60.0`로 증폭 중

**피드백:**
> 추론 시 업스케일링 자체는 문제 없음. 중요한 것은 **방향성(부호)**

**의미:**
- 스케일 자체보다 **좌표계 정합성**이 우선
- `x < 0`(후진)이 나오는 이유 규명 필요

---

### 2️⃣ **Step-by-Step Unit Test 부재**
**요구사항:**
- **18 steps를 7.2~8초 내 완료** (1 step = 0.4초)
- 각 스텝마다 **기대값(Expected Value)** 존재
- 현재: ~10초 소요 (추론 오버헤드 포함)

**필요 조치:**
- 각 스텝 실행 전 **Expected Range 검증**
- Validation 실패 시 **재추론** 로직

---

### 3️⃣ **오이동 방지 (Expectation Validation)**
**문제:**
- 로봇이 전혀 다른 궤적으로 움직임
- 0, 0 액션일 때 뒤로 가는 현상 (Omniwheel 특성)

**해결책:**
> 각 스텝마다 expectation을 주고, 기존 설정과 다르면 다시 추론할 수 있도록 조건문 필요

**구현:**
```python
if not validate_action(expected, actual):
    retry_inference()
```

---

### 4️⃣ **목표(Target) 명확화**
**현재:** `"Navigate to the target"` (모호)

**개선:** `"Navigate to the bottle"` or `"Navigate to the left bottle"`

**이유:**
- Target의 의미 파악이 불명확
- Window size 2로 추론하는데, 목표와 **상관 있게(contextual)** 추론되어야 함

---

## 🎯 다음 주 계획 (1/3 ~ 1/7)

### Phase 1: 분석 (1/3~1/4)
1. **좌표계 검증**
   - 데이터셋 `linear_x`, `linear_y` 원시값 확인
   - `data_collector.py` vs `inference_node.py` 좌표 정합성 체크
   - 부호 반전 필요 여부 판단

2. **현재 성능 측정**
   - 18 steps 소요 시간 정확히 측정
   - 추론 시간 vs 실행 시간 분리

### Phase 2: 구현 (1/4~1/6)
1. **Expected Value Validator 추가**
   - `validate_action(expected, actual)` 메서드 구현
   - JSON으로 Step별 Expected Range 정의
   - 재추론 로직 추가

2. **Instruction 구체화**
   - Default: `"Navigate to the left bottle"`
   - 시나리오 1~4 모두 "bottle" 명시

3. **좌표 변환 수정** (필요 시)
   - Action 부호 반전 적용
   - A/B 테스트로 검증

4. **성능 최적화**
   - 불필요한 대기 시간 제거
   - 목표: 8초 이내 완료

### Phase 3: 검증 (1/6~1/7)
1. **통합 테스트**
   - 전진 방향 확인
   - 7.5초 내 완료 확인
   - Validation 로직 작동 확인

2. **미팅 자료 준비**
   - Before/After 궤적 비교
   - 개선 내용 정리

---

## 📅 타임라인

| 날짜 | 목표 | 예상 시간 |
|------|------|-----------|
| **1/3 (금)** | Phase 1 완료 (분석) | 2h |
| **1/4 (토)** | Validator 구현 | 3h |
| **1/5 (일)** | Instruction + 좌표 수정 | 2h |
| **1/6 (월)** | 통합 테스트 + 문서화 | 3h |
| **1/7 (화)** | **교수님 미팅** | - |

---

## 🔧 주요 구현 항목

### A. Expected Value Validation
```python
# Config 예시 (JSON)
{
  "scenario_1": {
    "step_validation": {
      "x_range": [0.5, 2.0],  // 전진 기대
      "y_range": [-0.5, 2.0], // 좌측 편향
      "retry_on_fail": true
    }
  }
}

# 검증 로직
def validate_action(action, step):
    if not (x_min <= action[0] <= x_max):
        return False  # 재추론 필요
    return True
```

### B. Instruction Refinement
```python
# Before
"Navigate to the target"

# After
scenarios = {
    '1': "Navigate to the left bottle",
    '2': "Navigate to the right bottle",
    '3': "Navigate around two boxes to the left bottle",
    '4': "Navigate around two boxes to the right bottle"
}
```

### C. Coordinate Fix (필요 시)
```python
# 분석 결과 x축 반전 필요 시
action[0] = -action[0]  # X축 부호 반전
```

---

## 📊 예상 결과

### Before (현재)
- 이동 방향: 142도 (좌측 후방)
- 소요 시간: ~10초
- Validation: 없음

### After (목표)
- 이동 방향: 90도 (전진)
- 소요 시간: 7~8초
- Validation: Step별 검증 + 재추론

---

## 📝 커밋 메시지

```
feat: Add step-by-step validation and coordinate fixes

BREAKING CHANGE: Inference behavior changed

Changes:
- Add expected value validation for each step
- Refine instructions to specify "bottle" instead of "target"
- Fix coordinate system (X-axis inversion if needed)
- Optimize inference loop for 8sec target

Related:
- Meeting feedback 2026-01-02
- Target: 1/7 demo ready

Files:
- mobile_vla_inference_node.py: Add validate_action method
- config/expected_trajectory_config.json: Define expected ranges
- robovlms_mobile_vla_inference.py: Verify denormalization logic
```
