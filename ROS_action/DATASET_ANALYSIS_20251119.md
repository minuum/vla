# 데이터셋 분석 결과 (2025-11-19)

## 📊 전체 통계

- **총 에피소드**: 231개
- **고유 Trajectory 종류**: 16개
- **시나리오 종류**: 2개 (1box_left, 1box_right)

## 📋 시나리오별 수집 통계

### 1box_left
- **총 103개 에피소드**

#### Trajectory 분포:
1. **W W W W A Q Q Q Q Q Q Q W W W W E** (79개) - 가장 많이 수집된 패턴
2. **W W A Q Q Q Q Q Q A W W W W W W W** (10개)
3. **W W W A A Q Q Q Q Q Q Q W W W W W** (6개)
4. **W W W A Q Q Q Q Q Q Q W W W W W W** (2개)
5. **W W W** (2개) - 불완전한 데이터
6. 기타 개별 trajectory들 (각 1개씩)

### 1box_right
- **총 128개 에피소드**

#### Trajectory 분포:
1. **W W W D E E E E E E W W W W Q Q Q** (64개) - 가장 많이 수집된 패턴
2. **W W W W D E E E E E E E W W W W Q** (39개)
3. **W W W W D E E E E E W W Q Q Q Q Q** (21개)
4. **W W W W D D E E E E W W W Q Q Q Q** (1개)
5. **W W W D D E E E E E W W W W Q Q Q** (1개)

## 🔍 주요 발견 사항

1. **1box_right 시나리오**가 더 많이 수집됨 (128개 vs 103개)
2. **주요 trajectory 패턴**:
   - 1box_left: `W W W W A Q Q Q Q Q Q Q W W W W E` (79개)
   - 1box_right: `W W W D E E E E E E W W W W Q Q Q` (64개)
3. **불완전한 데이터**:
   - 1box_left에 3개 액션만 있는 에피소드 2개
   - 1box_left에 5개 액션만 있는 에피소드 1개
   - 1box_left에 액션이 없는 에피소드 1개

## 📝 수집 현황

### 1box_hori_right_core_medium
- 현재 가이드: `W W W W D E E E E E E E W W Q Q W` (core_patterns.json 기준)
- 실제 수집된 주요 trajectory:
  - `W W W D E E E E E E W W W W Q Q Q` (64개) - 가장 많이 수집됨
  - `W W W W D E E E E E E E W W W W Q` (39개)
  - `W W W W D E E E E E W W Q Q Q Q Q` (21개)

**가이드와 실제 수집된 trajectory가 다름** - 가이드 수정이 필요할 수 있음

