# 데이터셋 수집 통계 (2025-11-19)

## 📊 전체 요약

- **총 에피소드**: 334개
- **고유 Trajectory 패턴**: 16개
- **시나리오 종류**: 2개

---

## 📋 시나리오별 수집 현황

### 1. 1box_left
- **총 수집**: 105개

### 2. 1box_right  
- **총 수집**: 229개

---

## 🎯 Trajectory 패턴별 통계 (전체 순위)

### 전체 에피소드 기준 순위

| 순위 | 개수 | Trajectory 패턴 | 길이 | 시나리오 분포 |
|------|------|----------------|------|---------------|
| **1위** | **167개** | `W W W D E E E E E E W W W W Q Q Q` | 17 | 1box_right |
| **2위** | **79개** | `W W W W A Q Q Q Q Q Q Q W W W W E` | 17 | 1box_left |
| **3위** | **39개** | `W W W W D E E E E E E E W W W W Q` | 17 | 1box_right |
| **4위** | **21개** | `W W W W D E E E E E W W Q Q Q Q Q` | 17 | 1box_right |
| **5위** | **10개** | `W W A Q Q Q Q Q Q A W W W W W W W` | 17 | 1box_left |
| **6위** | **6개** | `W W W A A Q Q Q Q Q Q Q W W W W W` | 17 | 1box_left |
| **7위** | **2개** | `W W W` | 3 | 1box_left (불완전) |
| **8위** | **2개** | `W W W A Q Q Q Q Q Q Q W W W W W W` | 17 | 1box_left |
| **9위** | **1개** | `W W W A Q Q Q Q Q Q W W W W W E E` | 17 | 1box_left |
| **10위** | **1개** | `W W A Q Q Q Q Q Q W W W W W E E W` | 17 | 1box_left |
| **11위** | **1개** | `W W W W A A Q Q Q Q Q Q Q W W W W` | 17 | 1box_left |
| **12위** | **1개** | `` (빈 패턴) | 0 | 1box_left (오류) |
| **13위** | **1개** | `W W W A A` | 5 | 1box_left (불완전) |
| **14위** | **1개** | `W W W W A Q Q Q Q Q Q Q W W W W W` | 17 | 1box_left |
| **15위** | **1개** | `W W W W D D E E E E W W W Q Q Q Q` | 17 | 1box_right |
| **16위** | **1개** | `W W W D D E E E E E W W W W Q Q Q` | 17 | 1box_right |

---

## 📊 시나리오별 상세 Trajectory 분포

### 🎯 1box_left (105개)

| 순위 | 개수 | Trajectory 패턴 | 비율 |
|------|------|----------------|------|
| **1위** | **79개** | `W W W W A Q Q Q Q Q Q Q W W W W E` | 75.2% |
| **2위** | **10개** | `W W A Q Q Q Q Q Q A W W W W W W W` | 9.5% |
| **3위** | **6개** | `W W W A A Q Q Q Q Q Q Q W W W W W` | 5.7% |
| **4위** | **2개** | `W W W` | 1.9% (불완전) |
| **5위** | **2개** | `W W W A Q Q Q Q Q Q Q W W W W W W` | 1.9% |
| **6위** | **1개** | `W W W A Q Q Q Q Q Q W W W W W E E` | 1.0% |
| **7위** | **1개** | `W W A Q Q Q Q Q Q W W W W W E E W` | 1.0% |
| **8위** | **1개** | `W W W W A A Q Q Q Q Q Q Q W W W W` | 1.0% |
| **9위** | **1개** | `` (빈 패턴) | 1.0% (오류) |
| **10위** | **1개** | `W W W A A` | 1.0% (불완전) |
| **11위** | **1개** | `W W W W A Q Q Q Q Q Q Q W W W W W` | 1.0% |

**주요 패턴**: `W W W W A Q Q Q Q Q Q Q W W W W E` (79개, 75.2%)

### 🎯 1box_right (229개)

| 순위 | 개수 | Trajectory 패턴 | 비율 |
|------|------|----------------|------|
| **1위** | **167개** | `W W W D E E E E E E W W W W Q Q Q` | 72.9% |
| **2위** | **39개** | `W W W W D E E E E E E E W W W W Q` | 17.0% |
| **3위** | **21개** | `W W W W D E E E E E W W Q Q Q Q Q` | 9.2% |
| **4위** | **1개** | `W W W W D D E E E E W W W Q Q Q Q` | 0.4% |
| **5위** | **1개** | `W W W D D E E E E E W W W W Q Q Q` | 0.4% |

**주요 패턴**: `W W W D E E E E E E W W W W Q Q Q` (167개, 72.9%)

---

## 🔍 주요 발견 사항

### 1. 가이드 vs 실제 수집 패턴 비교

#### 1box_hori_right__core__medium
- **현재 가이드** (core_patterns.json): `W W W W D E E E E E E E W W Q Q W`
- **실제 수집 1위**: `W W W D E E E E E E W W W W Q Q Q` (167개, 72.9%)
- **실제 수집 2위**: `W W W W D E E E E E E E W W W W Q` (39개, 17.0%)
- **실제 수집 3위**: `W W W W D E E E E E W W Q Q Q Q Q` (21개, 9.2%)

**분석**: 가이드와 실제 수집된 패턴이 다름. 가장 많이 수집된 패턴이 가이드와 다르므로 가이드 수정 고려 필요.

### 2. 데이터 품질 이슈

- **불완전한 데이터**: 4개
  - 1box_left: 3개 액션만 있는 에피소드 2개
  - 1box_left: 5개 액션만 있는 에피소드 1개
  - 1box_left: 액션이 없는 에피소드 1개

### 3. 패턴 집중도

- **1box_left**: 1위 패턴이 75.2%로 높은 집중도
- **1box_right**: 1위 패턴이 72.9%로 높은 집중도
- 두 시나리오 모두 주요 패턴에 집중되어 있음

---

## 📈 수집 진행률

- **1box_left**: 105개 / 250개 목표 (42.0%)
- **1box_right**: 229개 / 250개 목표 (91.6%)
- **전체**: 334개 / 1000개 목표 (33.4%)

---

## 💡 권장 사항

1. **가이드 수정**: 1box_hori_right_core_medium의 가이드를 가장 많이 수집된 패턴으로 업데이트 고려
2. **데이터 품질**: 불완전한 데이터 4개 확인 및 필요시 재수집
3. **1box_left 수집**: 현재 42% 진행률로 추가 수집 필요

