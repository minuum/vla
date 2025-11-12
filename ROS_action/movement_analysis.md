# 이동 신호 멈추지 않는 현상 분석

## 로그 분석 결과

### 문제 발생 시점
- **573번 줄**: A 키 입력, 타이머 상태 확인 (이미 종료됨) - 시간: 01:46:15.995
- **574번 줄**: 수집 진행 6/18, 다음 키: Q - 시간: 01:46:26.041 (**약 11초 후!**)
- **575번 줄**: **A 키에 대한 새 타이머 생성 및 시작** (이상함 - 다음 키는 Q인데 A 키 타이머를 생성?)

### 문제점

1. **시간 간격 문제**
   - 573번 줄과 574번 줄 사이에 약 11초의 시간 간격이 있음
   - 이는 키 입력 처리 중 블로킹이 발생했거나, 다른 작업이 지연을 일으킨 것으로 보임

2. **키 입력 불일치**
   - 574번 줄에서 "다음 키: Q"라고 표시되었는데
   - 575번 줄에서는 "키입력:A"로 새 타이머를 생성
   - 이는 키 입력 처리 로직에서 이전 키(A)가 다시 처리되거나, 키 입력 버퍼에 문제가 있을 수 있음

3. **타이머와 키 입력 동기화 문제**
   - 타이머 콜백이 실행되어 정지 명령을 발행했지만
   - 키 입력 처리 시점에서 이전 키(A)가 다시 처리되는 것으로 보임

## 이동 방식 및 라이브러리 관계

### 1. ROS2 메시지 발행 구조

```
mobile_vla_data_collector.py
  └─ publish_cmd_vel()
      ├─ Twist 메시지 생성
      ├─ self.cmd_pub.publish(twist)  # ROS2 토픽 발행
      └─ 하드웨어 제어 (ROBOT_AVAILABLE일 때)
          └─ self.driver.move() / self.driver.spin() / self.driver.stop()
```

**ROS2 Publisher 설정:**
```python
self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
```
- 토픽: `/cmd_vel`
- 메시지 타입: `geometry_msgs/msg/Twist`
- 큐 크기: 10

### 2. 타이머 기반 이동 제어

```
키 입력 (handle_key_input)
  ├─ 기존 타이머 취소 (있으면)
  ├─ 강제 정지 (이전 액션이 있으면)
  ├─ 새 액션 시작 (publish_cmd_vel)
  └─ 타이머 시작 (0.3초 후 자동 정지)
      └─ stop_movement_timed() 콜백
          └─ stop_movement_internal()
              └─ 정지 명령 3회 발행 + 추가 정지 2회 = 총 5회
```

### 3. 정지 명령 발행 과정

```python
# stop_movement_internal()에서:
for i in range(3):
    self.publish_cmd_vel(self.STOP_ACTION, source=f"stop_internal_{i+1}")
    time.sleep(0.02)  # 각 신호 사이 딜레이

# stop_movement_timed()에서 추가로:
for i in range(2):
    self.publish_cmd_vel(self.STOP_ACTION, source=f"timer_extra_stop_{i+1}")
    time.sleep(0.01)
```

**총 5회의 정지 명령이 발행됨**

## 문제 원인 분석

### 가능한 원인들

1. **ROS2 메시지 버퍼 문제**
   - ROS2는 비동기적으로 메시지를 발행함
   - 여러 번 발행해도 실제 하드웨어에 전달되지 않을 수 있음
   - 특히 빠른 연속 명령 시 일부가 손실될 수 있음

2. **타이머 콜백과 키 입력 처리 간의 경쟁 조건 (Race Condition)**
   - 타이머 콜백이 실행되는 동안 새로운 키 입력이 들어오면
   - 타이머 취소가 제대로 되지 않을 수 있음
   - 또는 키 입력 처리 중 타이머 콜백이 실행되면 상태가 꼬일 수 있음

3. **키 입력 버퍼 문제**
   - `get_key()` 함수가 블로킹 방식으로 동작
   - 키 입력이 지연되거나 버퍼에 남아있을 수 있음
   - 특히 빠른 연속 입력 시 문제가 발생할 수 있음

4. **하드웨어 제어 지연**
   - `self.driver.move()` 등의 하드웨어 제어가 블로킹될 수 있음
   - 하드웨어 응답이 느리면 다음 명령이 지연될 수 있음

## 해결 방안

### 1. 정지 명령 발행 강화

현재 5회 발행하지만, 더 확실하게 하기 위해:
- 정지 명령 발행 횟수 증가 (5회 → 7-10회)
- 발행 간 딜레이 증가 (0.02초 → 0.05초)
- 정지 명령 발행 후 추가 안정화 대기 시간 증가

### 2. 타이머와 키 입력 동기화 개선

- 타이머 취소 시 락(lock) 사용
- 키 입력 처리 전 현재 액션 상태 확인 강화
- 타이머 콜백 실행 중 키 입력 처리 차단

### 3. ROS2 메시지 발행 확인

- `publish()` 호출 후 실제 발행 여부 확인
- 발행 실패 시 재시도 로직 추가

### 4. 키 입력 버퍼 정리

- 새 키 입력 전 이전 키 입력 버퍼 정리
- 키 입력 처리 전 상태 확인 강화

## 권장 수정 사항

1. **정지 명령 발행 강화**
   ```python
   # stop_movement_internal()에서:
   for i in range(5):  # 3회 → 5회
       self.publish_cmd_vel(self.STOP_ACTION, source=f"stop_internal_{i+1}")
       time.sleep(0.05)  # 0.02초 → 0.05초
   
   # 추가 안정화 대기
   time.sleep(0.1)  # 0.03초 → 0.1초
   ```

2. **타이머 취소 시 락 사용**
   ```python
   import threading
   self.movement_lock = threading.Lock()
   
   # 타이머 취소 시:
   with self.movement_lock:
       if self.movement_timer is not None:
           self.movement_timer.cancel()
           self.movement_timer = None
   ```

3. **키 입력 처리 전 상태 확인 강화**
   ```python
   # 키 입력 처리 전:
   if self.movement_timer is not None and self.movement_timer.is_alive():
       # 타이머가 실행 중이면 강제 정지
       self.stop_movement_internal(collect_data=False)
       time.sleep(0.1)  # 안정화 대기
   ```


