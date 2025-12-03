# 가이드 자동 저장 기능 복구 (2025-11-27)

**작성일**: 2025-11-27  
**상태**: 수정 완료

---

## 🔍 문제 개요

이전에는 가이드 패턴을 수집할 때 자동으로 저장되었지만, 최근 변경사항으로 인해 가이드가 이미 존재하는 경우 자동 저장이 비활성화되었습니다.

### 문제 상황

1. **이전 동작**: Core 패턴 에피소드 종료 시 항상 가이드 자동 저장
2. **현재 동작**: `record_core_pattern`이 False이면 저장하지 않음
3. **영향**: 가이드가 이미 있는 경우 새로운 패턴으로 갱신되지 않음

---

## 🔧 원인 분석

### 변경 전후 비교

**이전 코드 (예상)**:
```python
# Core 패턴이면 항상 저장
if scenario and ("_core_" in self.episode_name or self.episode_name.endswith("_core")):
    if len(self.current_episode_keys) > 0:
        # 항상 저장
        ...
```

**현재 코드 (문제)**:
```python
# record_core_pattern이 True일 때만 저장
if scenario and ("_core_" in self.episode_name or self.episode_name.endswith("_core")):
    if self.record_core_pattern and len(self.current_episode_keys) > 0:
        # record_core_pattern이 False이면 저장 안 함
        ...
```

### `record_core_pattern` 설정 로직

```python
# start_episode_with_pattern_and_distance (라인 2852)
combo_key = self._combined_key(scenario_id, pattern_type, distance_level)
has_combo = combo_key in self.core_patterns
has_scenario_only = scenario_id in self.core_patterns
# 이미 가이드가 있으면 False로 설정 (문제!)
self.record_core_pattern = (not has_combo and not has_scenario_only) or self.overwrite_core
```

**문제점**: 가이드가 이미 있으면 `record_core_pattern`이 False로 설정되어, 에피소드 종료 시 가이드가 갱신되지 않습니다.

---

## 🛠️ 해결 방법

### 수정 내용

`stop_episode` 함수에서 `record_core_pattern` 조건을 제거하고, `current_episode_keys`가 있으면 항상 저장하도록 변경했습니다.

**수정 전**:
```python
if self.record_core_pattern and len(self.current_episode_keys) > 0:
    # 저장 로직
```

**수정 후**:
```python
# 자동 저장: record_core_pattern이 True이거나, current_episode_keys가 있으면 항상 저장
# (이전에는 가이드가 이미 있으면 저장하지 않았지만, 이제는 항상 최신 패턴으로 갱신)
if len(self.current_episode_keys) > 0:
    # 저장 로직
```

### 변경 사항

1. **조건 제거**: `self.record_core_pattern` 조건 제거
2. **항상 저장**: `current_episode_keys`가 있으면 항상 저장
3. **로그 개선**: 저장 시 가이드 문자열도 함께 출력

---

## 📊 영향 범위

### 긍정적 영향

1. **자동 갱신**: Core 패턴 에피소드 종료 시 항상 최신 패턴으로 가이드 갱신
2. **일관성 유지**: 실제 수집된 패턴과 가이드가 항상 일치
3. **사용자 편의성**: 수동으로 가이드를 갱신할 필요 없음

### 주의사항

- **덮어쓰기**: 기존 가이드가 자동으로 덮어쓰기됨
- **의도하지 않은 변경**: 사용자가 의도하지 않은 패턴으로 갱신될 수 있음
- **해결책**: `overwrite_core` 토글이 있지만, 현재는 항상 저장하도록 변경됨

---

## 🔗 관련 파일

- **수정 파일**: `ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py`
  - `stop_episode` 함수 (라인 1530-1560)
- **관련 변수**:
  - `self.record_core_pattern`: 가이드 저장 플래그 (현재는 사용하지 않음)
  - `self.current_episode_keys`: 현재 에피소드의 키 시퀀스
  - `self.core_patterns`: 저장된 가이드 패턴 딕셔너리

---

## ✅ 검증 방법

1. **Core 패턴 에피소드 수집**:
   - N 키 → 시나리오 선택 → Core 패턴 선택 → 거리 선택
   - 에피소드 수집 완료 (M 키)
   - 로그에서 "💾 핵심 패턴 자동 저장" 메시지 확인

2. **가이드 파일 확인**:
   - `mobile_vla_dataset/core_patterns.json` 파일 확인
   - 해당 조합 키의 가이드가 최신 패턴으로 갱신되었는지 확인

3. **기존 가이드 덮어쓰기 확인**:
   - 기존 가이드가 있는 조합으로 에피소드 수집
   - 에피소드 종료 후 가이드가 갱신되었는지 확인

---

## 📝 참고사항

- 이전에는 `record_core_pattern`이 False일 때 저장하지 않았지만, 이제는 항상 저장합니다.
- `overwrite_core` 토글은 현재 사용되지 않지만, 향후 기능 확장 시 활용 가능합니다.
- Variant 패턴은 자동 저장되지 않습니다 (Core 패턴만 자동 저장).

---

**마지막 업데이트**: 2025-11-27

