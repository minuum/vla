# Mobile VLA Data Collector - 함수 및 기능 문서

**파일**: `mobile_vla_data_collector.py`  
**마지막 업데이트**: 2025-01-13

---

## 📋 목차

1. [주요 기능 개요](#주요-기능-개요)
2. [이동 제어 함수](#이동-제어-함수)
3. [데이터 수집 함수](#데이터-수집-함수)
4. [자동화 함수](#자동화-함수)
5. [가이드 관리 함수](#가이드-관리-함수)
6. [UI 및 상태 관리 함수](#ui-및-상태-관리-함수)

---

## 주요 기능 개요

### 시스템 아키텍처

```
사용자 입력 (키보드)
    ↓
handle_key_input() ← 메인 키 입력 처리
    ↓
├── N 키: 수동 데이터 수집
│   ├── 타이머 시작 (0.4초)
│   ├── 이동 명령 발행
│   ├── 데이터 수집
│   └── stop_movement_timed() → stop_movement_internal() (8회 정지)
│
├── B 키: 자동 복귀
│   └── execute_auto_return() (18프레임, 타이머 + 8회 정지)
│
├── A 키: 자동 측정
│   └── execute_auto_measurement() (타이머 + 8회 정지)
│
└── H/U 키: 가이드 편집/갱신
    ├── H: 수동 편집 (show_guide_edit_menu)
    └── U: 마지막 에피소드로 갱신 (save_edited_guide)
```

### 정지 메커니즘 (통일됨)

모든 이동 제어 함수는 동일한 정지 메커니즘을 사용합니다:

1. **타이머 시작** (0.4초 후 자동 정지)
2. **이동 명령 발행**
3. **타이머 만료 시 stop_movement_timed() 호출**
   - stop_movement_internal(): 5회 정지 신호 발행
   - 추가 3회 정지 신호 발행
   - **총 8회 정지 명령 보장**

---

## 이동 제어 함수

### 1. `handle_key_input(key: str)`

**역할**: 키보드 입력을 처리하고 로봇 이동 명령을 발행  
**위치**: Line ~700-827

#### 주요 동작

1. **WASD/QEZC 키 처리** (이동 키)
   ```python
   - W: 전진 (linear_x: 1.15)
   - S: 후진 (linear_x: -1.15)
   - A: 좌측 (linear_y: 1.15)
   - D: 우측 (linear_y: -1.15)
   - Q: 전진+좌측 (diagonal)
   - E: 전진+우측 (diagonal)
   - Z: 후진+좌측 (diagonal)
   - C: 후진+우측 (diagonal)
   ```

2. **정지 메커니즘**
   - 기존 타이머 취소 및 강제 정지 (3회)
   - 새 타이머 시작 (0.4초, `threading.Lock` 사용)
   - 이동 명령 발행
   - 데이터 수집 (collecting=True일 때)

3. **스페이스바** (수동 정지)
   - `stop_movement_internal(collect_data=True)` 호출

#### 특징
- **타이머 우선 시작**: 이미지 수집 블로킹과 무관하게 정지 보장
- **Thread-safe**: `movement_lock` 사용

---

### 2. `stop_movement_timed()`

**역할**: 타이머 콜백 함수, 자동 정지 실행  
**위치**: Line ~828-896

#### 주요 동작

1. 현재 액션 상태 확인 (이미 정지 상태면 스킵)
2. 타이머 유효성 확인 (`movement_lock` 사용)
3. `stop_movement_internal(collect_data=False)` 호출
4. **추가 정지 신호 3회 발행** (0.05초 간격)
5. **총 8회 정지 명령 보장**

#### 로깅
- 타이머 콜백 실행 시작/종료
- 현재 액션 상태
- 정지 신호 발행 횟수

---

### 3. `stop_movement_internal(collect_data: bool)`

**역할**: 내부 정지 함수, 실제 정지 명령 발행  
**위치**: Line ~898-945

#### 주요 동작

1. 이미 정지 상태면 스킵
2. **정지 신호 5회 발행** (0.05초 간격)
3. **안정화 대기 0.1초**
4. 데이터 수집 (collect_data=True일 때)

#### 파라미터
- `collect_data`: True면 정지 시점 데이터 수집

---

### 4. `publish_cmd_vel(action: Dict, source: str)`

**역할**: ROS2 `/cmd_vel` 토픽으로 Twist 메시지 발행  
**위치**: Line ~947-985

#### 주요 동작

1. Twist 메시지 생성
2. ROS2 토픽 발행
3. 하드웨어 제어 (ROBOT_AVAILABLE=True일 때)
   - `pop.driving.set_velocity()` 호출

#### 파라미터
- `action`: {"linear_x", "linear_y", "angular_z"}
- `source`: 로깅용 소스 식별자

---

## 자동화 함수

### 5. `execute_auto_return(return_actions: List[Dict])`

**역할**: 에피소드 종료 후 시작 위치로 자동 복귀  
**위치**: Line ~2968-3015

#### 주요 동작

1. **초기화**: 1회 정지 신호 발행 + 0.1초 안정화
2. **17개 액션 실행** (정규화됨)
   - 각 액션마다:
     - 이동 명령 발행
     - 0.4초 대기 (연속 실행)
3. **최종 정지**: 1회 정지 신호 발행 + 0.1초 안정화

#### 특징
- **17개 액션 정규화**: 부족하면 STOP_ACTION으로 패딩, 초과하면 잘라냄
- **간단한 연속 실행**: 정지 신호 최소화, 타이머 의존 제거
- **별도 스레드 실행**: 메인 키 입력 스레드와 독립
- **예상 소요 시간**: 17 × 0.4초 = 6.8초

#### 변경 이력
- 2025-01-13: 18프레임 정규화 추가, N 키와 동일한 정지 메커니즘 적용
- 2025-01-13: 자동 연속 실행을 위해 정지 신호 최소화 (복귀만)
- 2025-01-13: 17개 액션으로 변경 (초기 프레임 1개 + 17개 액션 = 18프레임)

---

### 6. `start_auto_return()`

**역할**: 자동 복귀 시작 (에피소드 데이터 역순 변환)  
**위치**: Line ~2916-2964

#### 주요 동작

1. 에피소드 데이터에서 start_action만 추출
2. 각 액션을 반대 방향으로 변환 (`get_reverse_action`)
3. 역순으로 정렬
4. **18프레임으로 정규화**
5. `execute_auto_return()` 스레드 시작

#### 18프레임 정규화 로직
```python
if len(return_actions) < 18:
    # STOP_ACTION으로 패딩
    return_actions.extend([STOP_ACTION] * (18 - len(return_actions)))
elif len(return_actions) > 18:
    # 첫 18개만 사용
    return_actions = return_actions[:18]
```

---

### 7. `execute_auto_measurement(scenario_id, pattern_type, distance_level)`

**역할**: 가이드 기반 자동 측정 실행  
**위치**: Line ~3111-3226

#### 주요 동작

1. 가이드 키 시퀀스 가져오기 (`get_core_pattern_guide_keys`)
2. 에피소드 시작
3. **각 키를 순차적으로 실행** (18개)
   - 기존 타이머 취소 및 강제 정지 (3회)
   - 타이머 시작 (0.4초)
   - 이동 명령 발행
   - 데이터 수집
   - 타이머 대기 (0.4초)
   - 안정화 대기 (0.3초)
4. 에피소드 종료
5. 반복 측정 확인 (`check_and_continue_repeat_measurement`)

#### 특징
- **N 키와 동일한 정지 메커니즘**: 타이머 + 8회 정지
- **SPACE 키 처리**: `stop_movement_internal(collect_data=True)` 호출
- **별도 스레드 실행**: `auto_measurement_mode` 플래그로 제어

#### 예상 소요 시간
```python
시간 = len(guide_keys) × (0.4초 + 0.3초)
     = 18개 × 0.7초 = 12.6초
```

#### 변경 이력
- 2025-01-13: N 키와 동일한 정지 메커니즘 적용 (0.4초 + 0.3초)

---

## 가이드 관리 함수

### 8. `show_guide_edit_menu()`

**역할**: 가이드 수동 편집 메뉴 표시 및 키 입력 처리  
**위치**: Line ~2100-2140

#### 주요 동작

1. 현재 입력된 가이드 키 표시 (최대 18개)
2. 키 입력 허용:
   - **WASD/QEZC/RT/SPACE**: 가이드 키 추가
   - **백스페이스**: 마지막 키 삭제
   - **Enter**: 저장 (`save_edited_guide`)
   - **X**: 취소

#### 제약
- 최대 18개 키까지 입력 가능
- 18개 초과 시 경고 메시지

---

### 9. `save_edited_guide(scenario_id, pattern_type, distance_level, keys)`

**역할**: 편집된 가이드를 저장  
**위치**: Line ~2142-2180

#### 주요 동작

1. 18키로 정규화 (부족하면 SPACE로 패딩)
2. 조합 키 생성: `{scenario}__{pattern}__{distance}`
3. `core_patterns` 딕셔너리에 저장
4. `core_patterns.json` 파일에 저장

#### 예시
```json
{
  "1box_left__core__medium": ["W", "W", "Q", "Q", "SPACE", ...],
  "1box_left__core__far": ["W", "W", "W", "Q", "SPACE", ...]
}
```

---

### 10. `get_core_pattern_guide_keys(scenario_id, pattern_type, distance_level)`

**역할**: 조합에 해당하는 가이드 키 시퀀스 가져오기  
**위치**: Line ~2182-2200

#### 주요 동작

1. 조합 키 생성: `{scenario}__{pattern}__{distance}`
2. `core_patterns` 딕셔너리에서 검색
3. 없으면 빈 리스트 반환

#### 반환값
```python
["W", "W", "Q", "Q", "SPACE", "SPACE", ...] # 18개 키
```

---

## 데이터 수집 함수

### 11. `start_episode_with_pattern_and_distance(scenario_id, pattern_type, distance_level)`

**역할**: 에피소드 시작 (시나리오, 패턴, 거리 설정)  
**위치**: Line ~1500-1580

#### 주요 동작

1. 에피소드 데이터 초기화
2. 시나리오, 패턴, 거리 설정
3. `episode_start` 이벤트 수집
4. collecting 플래그 활성화

---

### 12. `collect_data_point_with_action(event_type, action)`

**역할**: 액션 발생 시점의 데이터 포인트 수집  
**위치**: Line ~1582-1650

#### 주요 동작

1. 이미지 가져오기 (`get_latest_image_via_service`)
2. 타임스탬프 기록
3. 액션 정보 저장
4. episode_data에 추가

#### 수집 데이터
```python
{
  "image": np.array,
  "action": {"linear_x", "linear_y", "angular_z"},
  "action_event_type": "episode_start" | "start_action" | "stop_action",
  "timestamp": float,
  "time_period": "dawn" | "morning" | "evening" | "night"
}
```

---

### 13. `save_episode_data()`

**역할**: 에피소드 종료 시 HDF5 파일로 저장  
**위치**: Line ~1800-1950

#### 주요 동작

1. 데이터 검증 (18개 미만이면 경고)
2. HDF5 파일 생성
3. 데이터셋 저장:
   - `/observations/images`: (N, H, W, C)
   - `/action`: (N, 3)
   - `/action_event_type`: (N,)
   - 메타데이터 (시나리오, 패턴, 거리, 시간대 등)
4. 통계 업데이트
5. **마지막 완료 에피소드 액션 저장** (U 키 기능용)

#### 파일명 형식
```
episode_{timestamp}_{scenario}_{pattern}_{distance}.h5
예: episode_20251113_180812_1box_left_core_medium.h5
```

#### 변경 이력
- 2025-01-13: `last_completed_episode_actions` 저장 로직 추가

---

## UI 및 상태 관리 함수

### 14. `show_repeat_count_selection()`

**역할**: 반복 횟수 입력 화면 표시  
**위치**: Line ~2050-2098

#### 주요 동작

1. 현재 가이드 표시 (Core 패턴일 때)
2. 옵션 표시:
   - **숫자 (1-9)**: 반복 횟수 설정
   - **H 키**: 가이드 편집 (`show_guide_edit_menu`)
   - **U 키**: 마지막 에피소드로 가이드 갱신 (가능할 때만)
   - **X 키**: 취소

#### U 키 표시 조건
```python
if (selected_pattern_type == "core" and 
    len(last_completed_episode_actions) > 0):
    # U 키 옵션 표시
```

#### 변경 이력
- 2025-01-13: U 키 옵션 추가

---

### 15. `reset_to_initial_state()`

**역할**: 모든 상태 변수를 초기 상태로 리셋  
**위치**: Line ~2300-2380

#### 주요 동작

1. 에피소드 데이터 초기화
2. 선택 모드 플래그 리셋
3. 가이드 편집 모드 리셋
4. 반복 측정 상태 리셋
5. **마지막 완료 에피소드 액션 리셋**
6. 자동 복귀/측정 플래그 리셋

#### 리셋되는 변수
```python
- episode_data
- collecting
- scenario_selection_mode
- guide_edit_mode
- guide_edit_keys
- repeat_count_mode
- is_repeat_measurement_active
- last_completed_episode_actions  # 추가됨
- auto_return_active
- auto_measurement_active
```

---

### 16. `create_progress_bar(current, target, width=15)`

**역할**: ASCII 진행률 바 생성  
**위치**: Line ~2400-2430

#### 예시
```python
create_progress_bar(25, 250, width=15)
# 출력: "█░░░░░░░░░░░░░░ 25/250 (10.0%)"

create_progress_bar(100, 100, width=10)
# 출력: "██████████ 100/100 (100.0%)"
```

---

### 17. `show_measurement_task_table()`

**역할**: 측정 가능한 태스크와 종류를 표로 표시  
**위치**: Line ~3028-3064

#### 주요 동작

1. 시나리오별 진행률 표시 (4개)
2. 패턴 타입별 목표 표시 (2개)
3. 거리 레벨별 목표 표시 (3개)
4. 조합별 통계 표시 (4×2×3=24개)

#### 출력 예시
```
📋 시나리오 (4개):
   1: 1box_left - 목표: 250개 | █░░░░░░░░░░░░░░ 25/250 (10.0%)
   2: 1box_right - 목표: 250개 | ░░░░░░░░░░░░░░░ 0/250 (0.0%)
   ...

🎯 패턴 타입 (2개):
   CORE: 핵심 패턴 (Core) - 목표: 600개
   VARIANT: 변형 패턴 (Variant) - 목표: 400개

📍 거리 레벨 (3개):
   CLOSE: 근거리 - 샘플/시나리오: 75개
   MEDIUM: 중거리 - 샘플/시나리오: 100개
   FAR: 원거리 - 샘플/시나리오: 75개
```

---

## 헬퍼 함수

### 18. `get_reverse_action(action: Dict)`

**역할**: 액션의 반대 방향 반환 (복귀용)  
**위치**: Line ~2900-2914

#### 동작
```python
원본 액션: {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0}
반대 액션: {"linear_x": -1.15, "linear_y": -0.0, "angular_z": -0.0}
```

---

### 19. `get_latest_image_via_service()`

**역할**: ROS2 서비스로 최신 이미지 가져오기  
**위치**: Line ~1050-1080

#### 주요 동작

1. `/get_image` 서비스 호출
2. 타임아웃: **2.0초** (10초에서 단축됨)
3. Image 메시지를 OpenCV 이미지로 변환 (`CvBridge`)
4. 실패 시 검은 이미지 반환 (480×640×3)

#### 변경 이력
- 2025-01-13: 타임아웃 10초 → 2초 단축

---

### 20. `resync_scenario_progress()`

**역할**: 디스크의 H5 파일을 스캔하여 통계 재동기화  
**위치**: Line ~1950-2050

#### 주요 동작

1. `mobile_vla_dataset/` 디렉터리 스캔
2. H5 파일의 메타데이터 읽기
3. 시나리오별/패턴별/거리별 카운트 집계
4. `scenario_progress.json` 저장

---

## 📊 주요 상태 변수

### 이동 제어
```python
self.current_action: Dict[str, float]      # 현재 액션
self.movement_timer: threading.Timer       # 자동 정지 타이머
self.movement_lock: threading.Lock         # 타이머 동기화 락
self.STOP_ACTION = {                       # 정지 액션
    "linear_x": 0.0,
    "linear_y": 0.0,
    "angular_z": 0.0
}
```

### 데이터 수집
```python
self.collecting: bool                      # 수집 중 플래그
self.episode_data: List[Dict]              # 현재 에피소드 데이터
self.last_completed_episode_actions: List[str]  # 마지막 완료 에피소드 키 시퀀스
```

### 가이드 관리
```python
self.guide_edit_mode: bool                 # 가이드 편집 모드
self.guide_edit_keys: List[str]            # 편집 중인 가이드 키
self.core_patterns: Dict[str, List[str]]   # 조합별 가이드 저장
```

### 반복 측정
```python
self.is_repeat_measurement_active: bool    # 반복 측정 활성화
self.target_repeat_count: int              # 목표 반복 횟수
self.current_repeat_index: int             # 현재 반복 인덱스
```

### 자동화
```python
self.auto_return_active: bool              # 자동 복귀 활성화
self.auto_measurement_active: bool         # 자동 측정 활성화
```

---

## 🔄 주요 워크플로우

### 1. 수동 데이터 수집 (N 키)

```
1. N 키 입력
2. 시나리오 선택 (1-4)
3. 패턴 선택 (C/V)
4. 거리 선택 (J/K/L)
5. 반복 횟수 입력 (1-9)
   - 옵션: H (가이드 편집), U (가이드 갱신)
6. 에피소드 시작
7. WASD/QEZC 키로 이동
   - 각 키마다 타이머 + 8회 정지
8. P 키로 에피소드 종료
9. H5 파일 저장
10. 자동 복귀 가능 (B 키)
11. N 키로 다음 반복 또는 X로 종료
```

### 2. 자동 측정 (A 키)

```
1. A 키 입력
2. 시나리오 선택 (1-4)
3. 패턴 선택 (C - Core만 가능)
4. 거리 선택 (J/K/L)
5. 가이드 확인 (없으면 중단)
6. 에피소드 자동 실행 (18키)
   - 각 키마다 타이머 + 8회 정지
7. H5 파일 자동 저장
8. 자동 복귀 가능 (B 키)
```

### 3. 가이드 편집 (H 키)

```
1. N → 시나리오 → 패턴 (Core) → 거리 선택
2. H 키 입력
3. WASD/QEZC/RT/SPACE 키로 가이드 입력 (최대 18개)
4. 백스페이스로 삭제
5. Enter로 저장 또는 X로 취소
6. 18키로 자동 정규화 (SPACE 패딩)
7. core_patterns.json에 저장
```

### 4. 가이드 갱신 (U 키)

```
1. 에피소드 완료 (P 키)
2. N 키로 새 에피소드
3. 같은 시나리오/패턴/거리 선택
4. U 키 입력
5. 마지막 에피소드의 WASD/QEZC 키 추출
6. 18키로 정규화 후 가이드로 저장
7. core_patterns.json에 저장
```

### 5. 자동 복귀 (B 키)

```
1. 에피소드 완료 (P 키)
2. B 키 입력
3. 에피소드 액션 역순 + 반대 방향 변환
4. 18프레임으로 정규화
5. 각 액션 실행 (타이머 + 8회 정지)
6. 시작 위치 복귀 완료
```

---

## 🛠️ 변경 이력

### 2025-01-13

#### ✅ Done
1. **이동 시간 조정**
   - 키 입력: 0.3초 → 0.4초
   - 자동 측정: 0.31초 → 0.4초
   - 복귀: 0.3초 → 0.4초

2. **복귀 기능 개선**
   - 18프레임 정규화 추가 (STOP_ACTION 패딩)
   - N 키와 동일한 정지 메커니즘 적용 (타이머 + 8회 정지)
   - 예상 소요 시간 수정: 18 × 0.4초 = 7.2초

3. **자동 측정 개선**
   - N 키와 동일한 정지 메커니즘 적용 (타이머 + 8회 정지)
   - 예상 소요 시간 수정: 18 × 0.7초 = 12.6초

4. **U 키 가이드 갱신 기능 추가**
   - 마지막 완료 에피소드 액션 자동 저장
   - 반복 횟수 입력 화면에서 U 키로 갱신 가능

5. **정지 문제 해결**
   - 타이머 시작 순서 변경 (블로킹 전 시작)
   - 이미지 서비스 타임아웃 단축 (10초 → 2초)

#### 🔄 In Progress
- 없음

#### 📝 Todo
1. 카메라 초기화 실패 문제 해결
2. 데이터 수집 진행 (현재 25/1000)

---

## 📞 문의 및 수정

이 문서는 코드 변경 시 자동으로 업데이트됩니다.  
문제나 개선 사항이 있으면 이슈를 생성해주세요.

**마지막 수정**: 2025-01-13  
**작성자**: AI Assistant

