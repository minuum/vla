# 📊 궤적 패턴 분포 분석 보고서

**분석 일시:** 2025-11-20  
**총 에피소드:** 244개  
**분석 대상:** `1box_left__core__medium`, `1box_right__core__medium`

---

## 🎯 수집 목표 설정

### 목표 구조
- **시나리오당 목표:** 250개 (총 1000개 목표)
- **패턴 분배:**
  - Core: 150개 (60%)
  - Variant: 100개 (40%)
- **거리별 분배 (Core):**
  - close: 50개
  - medium: 75개
  - far: 25개
- **거리별 분배 (Variant):**
  - close: 25개
  - medium: 25개
  - far: 50개

---

## 📊 현재 수집 현황

### 1. `1box_left__core__medium`

**총 수집:** 120개

**패턴 분포:**
- **주요 패턴 (119개, 99.2%):** `W W W A Q Q Q Q Q Q Q Q W W W W Q`
- **다른 패턴 (1개, 0.8%):** `W W W A Q Q Q Q Q Q Q Q W W W W W`

**목표:** 75개  
**상태:** ✅ **목표 달성!** (초과: 44개)

**분석:**
- 거의 모든 에피소드(99.2%)가 동일한 궤적 패턴을 따름
- 1개만 다른 패턴 (마지막 액션이 Q → W로 변경)
- 일관성 매우 높음

### 2. `1box_right__core__medium`

**총 수집:** 124개

**패턴 분포:**
- **주요 패턴 (124개, 100%):** `W W W D E E E E E E W W W W Q Q Q`
- **다른 패턴:** 없음

**목표:** 75개  
**상태:** ✅ **목표 달성!** (초과: 49개)

**분석:**
- 모든 에피소드가 완전히 동일한 궤적 패턴
- 완벽한 일관성 (100%)

---

## 💡 권장 사항

### 1. 주요 패턴으로 통일 수집

현재 두 시나리오 모두 **목표를 초과 달성**했으며, 주요 패턴이 압도적으로 많습니다.

**주요 패턴:**
- `1box_left__core__medium`: `W W W A Q Q Q Q Q Q Q Q W W W W Q` (119개)
- `1box_right__core__medium`: `W W W D E E E E E E W W W W Q Q Q` (124개)

### 2. 다른 패턴 처리 방안

**`1box_left__core__medium`의 다른 패턴 (1개):**
- 궤적: `W W W A Q Q Q Q Q Q Q Q W W W W W`
- 차이점: 마지막 액션이 Q → W로 변경
- **권장:** 이 1개는 제외하고 주요 패턴만 사용하는 것을 권장

### 3. 추가 수집 필요량

현재 **core_medium은 목표를 초과 달성**했으므로 추가 수집 불필요합니다.

**다음 단계:**
- `core_close`: 50개 목표 (현재 수집량 확인 필요)
- `core_far`: 25개 목표 (현재 수집량 확인 필요)
- `variant` 패턴들: 각 거리별 목표 확인 필요

---

## 📋 궤적 패턴 상세 분석

### `1box_left__core__medium` 주요 패턴
```
W W W A Q Q Q Q Q Q Q Q W W W W Q
```

**해석:**
- 초반: 전진(W) 3회
- 중간: 좌회전(A) 1회
- 대각선: 전진+좌(Q) 6회
- 후반: 전진(W) 4회
- 마지막: 대각선 전진+좌(Q) 1회

### `1box_right__core__medium` 주요 패턴
```
W W W D E E E E E E W W W W Q Q Q
```

**해석:**
- 초반: 전진(W) 3회
- 중간: 우회전(D) 1회
- 대각선: 전진+우(E) 6회
- 후반: 전진(W) 4회
- 마지막: 대각선 전진+좌(Q) 3회

---

## ✅ 결론

1. **일관성 평가:** 두 시나리오 모두 매우 높은 일관성을 보임
   - `1box_left`: 99.2% 일관성
   - `1box_right`: 100% 일관성

2. **목표 달성:** `core_medium`은 목표(75개)를 초과 달성
   - `1box_left`: 119개 (목표 대비 159%)
   - `1box_right`: 124개 (목표 대비 165%)

3. **권장 사항:**
   - 주요 패턴으로 통일하여 수집 계속 진행
   - 다른 패턴은 제외하고 일관성 유지
   - `core_close`, `core_far`, `variant` 패턴 수집 진행

---

## 🔍 참고 프로젝트

분석에 참고한 파일들:
- `ROS_action/analyze_trajectories.py`: 궤적 추출 및 분석
- `docs/reports/CORE_PATTERN_ANALYSIS_20251114.md`: 이전 패턴 분석 보고서
- `ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py`: 데이터 수집 로직

