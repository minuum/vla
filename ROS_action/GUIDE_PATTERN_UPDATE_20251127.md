# 가이드 패턴 업데이트 이력

**작성일**: 2025-11-27  
**이유**: 현재 데이터셋의 실제 패턴 분포와 코드의 하드코딩된 가이드 불일치

---

## 🔍 문제 발견

### 현재 상황
- **코드 하드코딩 가이드**: `W W W A Q Q Q Q Q Q Q Q W W W W Q` (마지막 Q)
- **실제 데이터셋 패턴**: 84/85개 (98.8%)가 `W W W A Q Q Q Q Q Q Q Q W W W W W` (마지막 W)
- **core_patterns.json**: 이미 올바른 패턴(`W W W A Q Q Q Q Q Q Q Q W W W W W`) 저장됨

### 원인 분석

#### 커밋 이력
- **커밋 8a52d8cd** (2025-11-26): `feat: Update core patterns guide from dataset analysis`
  - Legacy 데이터셋(247개) 기반으로 가이드 설정
  - Legacy 데이터셋의 최다 패턴: `W W W A Q Q Q Q Q Q Q Q W W W W Q` (119/120, 99.2%)
  - 이 패턴이 코드에 하드코딩됨

#### 현재 데이터셋 (바퀴 교체 후)
- **총 파일 수**: 85개
- **패턴 분포**:
  - `W W W A Q Q Q Q Q Q Q Q W W W W W`: 84개 (98.8%)
  - `W W W A Q Q Q Q Q Q Q Q W W W W Q`: 1개 (1.2%)

### 문제점
1. Legacy 데이터셋 기반으로 설정된 가이드가 현재 데이터셋과 불일치
2. 바퀴 교체 후 로봇 성능 변화로 인해 패턴이 변경됨
3. 코드의 하드코딩된 기본 가이드가 실제 데이터와 다름

---

## ✅ 해결 방법

### 변경 사항
1. **코드 수정**: `mobile_vla_data_collector.py`의 하드코딩된 기본 가이드 업데이트
   - `1box_left` 기본 가이드: `W W W A Q Q Q Q Q Q Q Q W W W W Q` → `W W W A Q Q Q Q Q Q Q Q W W W W W`
   - 두 곳 수정:
     - `guide_mode == "dataset"` 모드의 `default_guides`
     - 수동 모드의 `default_guides` (fallback)

2. **core_patterns.json**: 이미 올바른 패턴으로 저장되어 있음 (변경 불필요)

### 영향 범위
- **데이터셋 모드**: `guide_mode == "dataset"`일 때 사용되는 기본 가이드
- **수동 모드**: `core_patterns.json`에 패턴이 없을 때 사용되는 fallback 가이드

---

## 📊 데이터셋 분석 결과

### 전체 패턴 분포 (85개 파일)
```
W W W A Q Q Q Q Q Q Q Q W W W W W: 84회 (98.8%)
W W W A Q Q Q Q Q Q Q Q W W W W Q: 1회 (1.2%)
```

### 시나리오별 분포
- 모든 파일이 `1box_hori_left_core_medium` 패턴
- 84개가 `W W W A Q Q Q Q Q Q Q Q W W W W W` 패턴 사용
- 1개만 `W W W A Q Q Q Q Q Q Q Q W W W W Q` 패턴 사용

---

## 🔄 변경 이력

### 2025-11-27
- **문제 발견**: 코드 가이드와 실제 데이터셋 패턴 불일치
- **원인 확인**: Legacy 데이터셋 기반 가이드가 현재 데이터셋과 다름
- **해결**: 코드의 하드코딩된 기본 가이드를 현재 데이터셋 패턴으로 업데이트

---

## 📝 참고사항

1. **가이드 모드 우선순위**:
   - `guide_mode == "dataset"`: 하드코딩된 기본 가이드 사용
   - `guide_mode == "manual"`: `core_patterns.json`에서 로드한 가이드 사용 (없으면 fallback 기본 가이드)

2. **바퀴 교체 영향**:
   - 바퀴 교체 후 로봇 성능 변화로 인해 패턴이 변경됨
   - Legacy 데이터셋: `W W W A Q Q Q Q Q Q Q Q W W W W Q` (마지막 Q)
   - 현재 데이터셋: `W W W A Q Q Q Q Q Q Q Q W W W W W` (마지막 W)

3. **데이터 일관성**:
   - 현재 수집 중인 데이터와 가이드가 일치하도록 수정
   - 향후 데이터 수집 시 올바른 가이드 사용

---

**마지막 업데이트**: 2025-11-27

