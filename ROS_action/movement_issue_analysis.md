# 로봇 이동이 멈추지 않는 문제 - 종합 분석

## 로그 분석 결과

### 타임라인 (966-988번 줄)

```
06:39:29.426 - [A 키] 수집 진행 6/18, 다음 키: Q
06:39:29.426 - [A 키] 새 타이머 생성
06:39:29.727 - [A 키 타이머] 콜백 실행 (0.3초 후)
06:39:29.730 - [STOP] 정지 명령 시작
06:39:29.731-30.013 - [STOP] 정지 신호 5회 발행 (각 0.05초 간격)
06:39:30.114-30.283 - [STOP] 추가 정지 신호 3회 발행
06:39:30.283 - [STOP] 타이머 기반 정지 완료 (총 8회)
06:39:31.269 - [Q 키] 타이머 종료 확인 (약 1초 후)
06:39:41.315 - [Q 키] 수집 진행 7/18 (약 10초 후!!!)
```

### 핵심 문제점

**987번 줄(06:39:31.269)과 988번 줄(06:39:41.315) 사이에 약 10초의 시간 간격**

이는 키 입력 처리 중 어딘가에서 블로킹이 발생했음을 의미합니다.

## 가능한 원인들 (우선순위 순)

### 1. 🔴 ROS2 메시지 발행은 성공했지만 실제 로봇이 정지 명령을 받지 못함

**증거:**
- 코드상으로는 정지 명령이 8회 발행됨
- 타이머 콜백이 정상 실행됨
- 로그에 에러가 없음
- 하지만 실제 로봇은 멈추지 않음

**가능한 세부 원인:**
1. `/cmd_vel` 토픽을 구독하는 **로봇 제어 노드가 실행 중이 아님** 또는 제대로 동작하지 않음
2. ROS2 토픽 통신에 **네트워크 지연** 또는 **메시지 손실**
3. 로봇 제어 노드가 메시지를 받았지만 **하드웨어에 전달하지 못함**
4. ROS2 **QoS 설정 불일치** (Publisher와 Subscriber의 QoS가 맞지 않음)

**확인 방법:**
```bash
# 1. /cmd_vel 토픽 확인
ros2 topic list | grep cmd_vel
ros2 topic info /cmd_vel

# 2. 실시간 메시지 확인
ros2 topic echo /cmd_vel

# 3. 로봇 제어 노드 실행 확인
ros2 node list

# 4. QoS 설정 확인
ros2 topic info /cmd_vel --verbose
```

---

### 2. 🔴 하드웨어 제어 명령이 실제로 실행되지 않음

**증거:**
- `ROBOT_AVAILABLE = False`일 가능성 (로그에 하드웨어 제어 로그 없음)
- 또는 `self.driver.stop()`이 호출되지 않음

**가능한 세부 원인:**
1. `ROBOT_AVAILABLE` 플래그가 `False`로 설정되어 있어 하드웨어 제어가 스킵됨
2. `self.driver` 객체가 `None`이거나 초기화되지 않음
3. `self.driver.stop()` 호출 시 **에러가 발생했지만 catch되어 로그에 나타나지 않음**
4. Pop 로봇 라이브러리(`pop.driving`)의 버그 또는 통신 문제

**확인 방법:**
```python
# publish_cmd_vel 함수에서 하드웨어 제어 부분 확인
# 라인 1000-1026 근처
```

---

### 3. 🟡 `collect_data_point_with_action()` 함수에서 블로킹 발생

**증거:**
- 987번 줄과 988번 줄 사이의 10초 간격
- 키 입력 처리 중 블로킹 가능성

**가능한 세부 원인:**
1. **이미지 수집** (`get_latest_image_via_service`) 시 타임아웃 또는 지연
2. 서비스 호출이 응답을 받지 못하고 10초 대기
3. `rclpy.spin_until_future_complete`가 블로킹됨

**확인 방법:**
- `collect_data_point_with_action` 함수 내부 로깅 추가
- 이미지 서비스 호출 시간 측정

---

### 4. 🟡 ROS2 `publish()` 자체가 블로킹됨

**증거:**
- `self.cmd_pub.publish(twist)` 호출 시 블로킹 가능성

**가능한 세부 원인:**
1. ROS2 Publisher의 큐가 가득 차서 블로킹 (`queue_size=10`)
2. ROS2 Executor가 과부하 상태
3. 네트워크 지연으로 publish가 블로킹

**확인 방법:**
```python
# publish_cmd_vel 함수에서 publish 시간 측정
ros_publish_start = time.time()
self.cmd_pub.publish(twist)
ros_publish_time = time.time() - ros_publish_start
if ros_publish_time > 0.1:  # 100ms 이상이면 경고
    self.get_logger().warn(f"⚠️ ROS publish 지연: {ros_publish_time*1000:.2f}ms")
```

---

### 5. 🟡 타이머 콜백과 키 입력 처리 간의 데드락

**증거:**
- 락(Lock) 사용으로 인한 데드락 가능성

**가능한 세부 원인:**
1. `self.movement_lock`을 타이머 콜백과 키 입력 처리에서 동시에 획득하려다 데드락
2. 락을 획득한 상태에서 블로킹 함수 호출 (예: `time.sleep`, `publish`)

**확인 방법:**
- 락 획득/해제 로깅 추가
- 락 타임아웃 설정

---

### 6. 🟢 로봇 하드웨어 자체 문제

**가능한 세부 원인:**
1. **배터리 부족**: 모터가 제대로 동작하지 않음
2. **모터 제어기 오류**: 하드웨어 컨트롤러가 멈춤
3. **통신 케이블 문제**: USB/시리얼 통신 단선
4. **펌웨어 버그**: 로봇 펌웨어가 정지 명령을 무시

**확인 방법:**
- 로봇 LED 상태 확인
- 배터리 전압 확인
- 로봇 재부팅

---

### 7. 🟢 ROS2 DDS 통신 문제

**가능한 세부 원인:**
1. DDS 미들웨어 버그
2. 네트워크 설정 문제
3. 방화벽 또는 포트 차단

**확인 방법:**
```bash
# DDS 통신 상태 확인
ros2 doctor
ros2 doctor --report
```

---

### 8. 🟢 시스템 리소스 문제

**가능한 세부 원인:**
1. CPU 과부하
2. 메모리 부족
3. 디스크 I/O 병목

**확인 방법:**
```bash
top
htop
iostat
```

---

## 우선 확인 사항 (순서대로)

### 1단계: ROS2 토픽 및 노드 확인
```bash
# 터미널 1: 데이터 수집 노드 실행 중인 상태에서
# 터미널 2에서 실행:

# 1. 실행 중인 노드 확인
ros2 node list

# 2. /cmd_vel 토픽 확인
ros2 topic list | grep cmd_vel

# 3. /cmd_vel 토픽 정보 확인
ros2 topic info /cmd_vel --verbose

# 4. 실시간 메시지 모니터링 (키를 누를 때 메시지가 나타나는지 확인)
ros2 topic echo /cmd_vel
```

### 2단계: 하드웨어 제어 로깅 강화
```python
# publish_cmd_vel 함수 수정하여 하드웨어 제어 로그 추가
# 라인 1000-1026 근처

if ROBOT_AVAILABLE and self.driver:
    self.get_logger().info(f"🔧 [HARDWARE] 제어 시작: {action}")
    try:
        # ... 기존 코드 ...
        self.get_logger().info(f"✅ [HARDWARE] 제어 성공")
    except Exception as e:
        self.get_logger().error(f"❌ [HARDWARE] 제어 실패: {e}")
else:
    self.get_logger().warn(f"⚠️ [HARDWARE] 사용 불가: ROBOT_AVAILABLE={ROBOT_AVAILABLE}, driver={self.driver}")
```

### 3단계: 이미지 수집 시간 측정
```python
# collect_data_point_with_action 함수에 시간 측정 추가

def collect_data_point_with_action(self, action_event_type: str, action: Dict[str, float]):
    start_time = time.time()
    self.get_logger().info(f"🔍 [DATA] collect_data_point_with_action 시작: {action_event_type}")
    
    # 이미지 수집
    image_start = time.time()
    image = self.get_latest_image_via_service()
    image_time = time.time() - image_start
    self.get_logger().info(f"🔍 [DATA] 이미지 수집 완료: {image_time*1000:.2f}ms")
    
    # ... 나머지 코드 ...
    
    total_time = time.time() - start_time
    self.get_logger().info(f"🔍 [DATA] collect_data_point_with_action 완료: {total_time*1000:.2f}ms")
```

### 4단계: 락 획득/해제 로깅
```python
# handle_key_input에서 락 사용 부분 수정

self.get_logger().info(f"🔒 [LOCK] 락 획득 시도...")
with self.movement_lock:
    self.get_logger().info(f"🔒 [LOCK] 락 획득 성공")
    # ... 기존 코드 ...
self.get_logger().info(f"🔓 [LOCK] 락 해제 완료")
```

---

## 결론 및 권장 조치

### 가장 가능성 높은 원인 (우선순위)

1. **ROS2 토픽 통신 문제** (60% 확률)
   - 로봇 제어 노드가 실행 중이 아님
   - 또는 /cmd_vel 토픽을 구독하지 않음

2. **하드웨어 제어 미동작** (30% 확률)
   - ROBOT_AVAILABLE = False
   - 또는 self.driver가 제대로 초기화되지 않음

3. **이미지 수집 블로킹** (10% 확률)
   - get_latest_image_via_service가 타임아웃

### 즉시 실행할 디버깅

1. **ROS2 토픽 확인** (1단계)
2. **하드웨어 제어 로깅 추가** (2단계)
3. **시간 측정 로깅 추가** (3단계)

이 디버깅을 통해 정확한 원인을 파악할 수 있습니다.

