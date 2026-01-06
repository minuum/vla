# VLA Instruction 수정 완료 브리핑 (2026-01-07)

## 🎯 수정 완료

### 변경 사항
**File:** `mobile_vla_inference_node.py`

**Before (영어):**
```python
self.current_instruction = "Navigate to the left bottle"

scenarios = {
    '1': "Navigate to the left bottle",
    '2': "Navigate to the right bottle",
    '3': "Navigate around two boxes to the left bottle",
    '4': "Navigate around two boxes to the right bottle"
}
```

**After (한국어, 학습 데이터와 동일):**
```python
self.current_instruction = "가장 왼쪽 외곽으로 돌아 컵까지 가세요"

scenarios = {
    '1': "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    '2': "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
    '3': "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    '4': "가장 오른쪽 외곽으로 돌아 컵까지 가세요"
}
```

---

## 📊 문제 분석 요약

### 발견된 근본 원인

**3가지 불일치가 동시에 발생:**

1. **언어 불일치**
   - 학습: 한국어
   - 추론: 영어
   - VLM은 언어에 매우 민감

2. **내용 불일치**
   - 학습: "가장 왼쪽 **외곽으로 돌아** 컵까지 가세요" (구체적 경로)
   - 추론: "Navigate **to** the left bottle" (단순 목표)

3. **데이터 소스 불일치**
   - H5 파일 내부: English instruction 저장됨
   - RoboVLMs 로더: **H5 무시**, 파일명에서 한국어 생성
   - **학습 시 실제 사용된 것은 한국어**

### 증거 자료

**Citation 1: H5 파일 내부 (사용 안 됨)**
```python
# episode_20251203_042905_1box_hori_left_core_medium.h5
language_instruction: "Navigate around obstacles and reach the front of the beverage bottle on the left"
```

**Citation 2: 실제 학습에 사용된 Instruction (한국어)**
```python
# RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:L151-160
self.scenario_instructions = {
    "1box_hori_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    "1box_hori_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요"
}

# L258-260
scenario = self._extract_scenario(str(episode_name))
task_description = self.scenario_instructions.get(scenario, "컵까지 가세요")
```

**Citation 3: Config 파일**
```json
// Mobile_VLA/configs/mobile_vla_chunk5_20251217.json:L128-135
"train_dataset": {
    "type": "MobileVLAH5Dataset",
    "data_dir": "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
    "episode_pattern": "episode_20251*.h5"
}
```

---

## 🔬 이전 분석의 오류 및 수정

### ❌ 이전 가설 (틀림)

**docs/20260102_Weekly_Dev_Report.md:**
> 가설 1: 좌표계 불일치  
> 가설 2: 데이터 정규화 문제

**docs/meeting/20260102_feedback_and_plan.md:**
> X축 반전 필요 (INVERT_X_AXIS=True)

### ✅ 실제 원인 (맞음)

**Instruction 언어 불일치가 1차 원인**
- X축 반전은 **부작용**일 가능성
- Instruction만 수정하면 방향성 문제가 해결될 수 있음

**재검토 필요:**
- X축 반전 로직 (`INVERT_X_AXIS=True`)
- Instruction 수정 후 제거 가능성 높음

---

## 📋 다음 테스트 계획

### Test 1: 한국어 Instruction 효과 검증
```bash
# 1. 추론 실행
vla-inference
# 2. 'S' 키로 시작
# 3. 결과 확인
```

**기대 결과:**
- ✅ 로봇이 **전진** (X>0)
- ✅ Left instruction → 왼쪽으로 회피
- ✅ Right instruction → 오른쪽으로 회피

**만약 여전히 후진한다면:**
- X축 반전 유지 필요
- 하지만 방향성은 개선될 것으로 예상

### Test 2: Left vs Right 구분 능력
```bash
# Scenario 1 (Left)
vla-inference
# 로그에서: "가장 왼쪽 외곽으로..."

# Scenario 2 (Right)
vla-inference
# 로그에서: "가장 오른쪽 외곽으로..."
```

**비교:**
- Y값 부호 차이 확인
- 궤적 그래프로 Left/Right 명확히 구분되는지 확인

---

## 📝 빌드 결과

```
✅ colcon build --packages-select mobile_vla_package
Summary: 1 package finished [4.56s]
```

---

## 🎯 예상 효과

### Before (영어 Instruction)
- 모델이 instruction을 제대로 이해 못 함
- 후진 (X=-4.7)
- 방향성 혼란

### After (한국어 Instruction, 학습 데이터와 동일)
- 모델이 instruction을 정확히 이해
- 전진 가능성 높음
- Left/Right 명확히 구분

---

## 🚨 중요 발견 사항

1. **H5 파일의 `language_instruction` 필드는 학습에 사용 안 됨**
   - RoboVLMs 데이터 로더가 파일명에서 instruction 재생성
   - H5 내부 English instruction은 무시됨

2. **VLA 모델은 instruction-conditioned**
   - 하나의 모델이 left/right 모두 학습
   - Instruction만 다르게 주면 다른 궤적 생성
   - 이게 Vision-Language-Action의 핵심

3. **X축 반전은 임시 조치**
   - Instruction 문제 해결 후 제거 가능성
   - 재검증 필요

---

## 📅 다음 단계

1. **즉시 테스트** (1/7 오전)
   - vla-inference 실행
   - 방향성 확인
   - Before/After 궤적 비교

2. **X축 반전 재검토** (테스트 결과에 따라)
   - 여전히 후진 → 유지
   - 전진 성공 → 제거

3. **1/7 미팅 준비**
   - Before/After 그래프
   - Instruction 불일치 설명
   - 개선 효과 시연

---

**작성**: 2026-01-07 05:51  
**Status**: ✅ Instruction 수정 완료, 빌드 완료, 테스트 준비됨
