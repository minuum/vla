# 로봇 멈추지 않는 문제 해결 방안

## 문제 원인 (확인됨)

**이미지 서비스 응답 대기로 인한 10초 블로킹**

### 발생 순서
1. Q 키 입력
2. 정지 명령 8회 발행 (성공)
3. 새 이동 명령 발행 (Q 키 액션)
4. `collect_data_point_with_action()` 호출
5. **`get_latest_image_via_service()` 호출 → 10초 블로킹**
6. 10초 동안 키 입력 처리 불가능
7. **로봇은 마지막 이동 명령(Q 키)을 계속 실행**

## 해결 방안 3가지

### 방안 1: 이미지 수집 전 명시적 정지 (권장) ⭐

```python
def collect_data_point_with_action(self, action_event_type: str, action: Dict[str, float]):
    # 🔴 이미지 수집 전에 반드시 정지
    if action_event_type == "start_action":
        # 이동 명령을 발행하기 전에 정지 상태 확인
        # (이미 정지 상태여야 함)
        pass
    
    # 이미지 수집 (블로킹 가능)
    image = self.get_latest_image_via_service()
    
    # ... 나머지 코드
```

**장점**: 간단하고 효과적
**단점**: 이미지 수집 중 블로킹은 여전히 발생

### 방안 2: 이미지 서비스 타임아웃 단축

```python
# 현재: timeout_sec=10.0
rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

# 변경: timeout_sec=2.0 (2초로 단축)
rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
```

**장점**: 블로킹 시간 단축
**단점**: 이미지 수집 실패 가능성 증가

### 방안 3: 키 입력과 이미지 수집 분리 (근본적 해결)

```python
# 키 입력 처리를 별도 스레드에서 실행
def handle_key_input(self, key: str):
    # ... 기존 로직 ...
    
    # 이미지 수집을 별도 스레드에서 실행
    if self.collecting:
        threading.Thread(
            target=self.collect_data_point_with_action_async,
            args=(action_event_type, action)
        ).start()
```

**장점**: 근본적 해결, 키 입력 즉시 처리
**단점**: 복잡한 구현, 동기화 문제 가능

## 추가 확인 사항

### 1. 이미지 서비스 노드 실행 확인
```bash
# ROS2 노드 목록
ros2 node list

# 카메라 관련 노드 찾기
ros2 node list | grep -i camera
ros2 node list | grep -i image

# 서비스 목록
ros2 service list | grep image
```

### 2. 이미지 서비스가 실행되지 않는 경우
- 카메라 노드 시작 필요
- 또는 이미지 서비스를 사용하지 않도록 코드 수정

## 권장 해결 순서

1. **즉시 적용**: 타임아웃 단축 (10초 → 2초)
2. **근본 해결**: 이미지 서비스 노드 확인 및 재시작
3. **장기 해결**: 키 입력과 이미지 수집 분리

## 코드 수정 (타임아웃 단축)

```python
# 라인 1054 수정
# 변경 전:
rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

# 변경 후:
rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
```

이 변경으로 최악의 경우에도 2초만 블로킹되며, 로봇이 계속 이동하는 현상이 크게 줄어듭니다.

