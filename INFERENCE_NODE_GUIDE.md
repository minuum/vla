# Mobile VLA 추론 노드 실행 가이드

## ✅ 완료된 작업

### 1. **로컬 전용 API 서버**
```python
api_server_local.py
- API Key 인증 제거
- Jetson 로컬 전용
- FastAPI 기반
```

### 2. **ROS2 추론 노드**
```python
mobile_vla_inference_node.py
- mobile_vla_data_collector.py 구조 기반
- Camera service 통합
- 실시간 추론 → 로봇 제어
```

### 3. **테스트 스크립트**
```python
test_live_inference.py
- 더미 이미지 생성
- 단일/연속 추론 테스트
- 통계 분석
```

## 🚀 실행 방법

### Step 1: ROS2 패키지 빌드

```bash
cd /home/soda/vla/ROS_action
colcon build --packages-select mobile_vla_package
source install/local_setup.bash
```

### Step 2: Camera Server 시작 (터미널 1)

```bash
source /opt/ros/humble/setup.bash
cd /home/soda/vla/ROS_action
source install/local_setup.bash
ros2 run camera_pub camera_publisher_continuous
```

### Step 3: VLA 추론 노드 시작 (터미널 2)

```bash
source /opt/ros/humble/setup.bash
cd /home/soda/vla/ROS_action
source install/local_setup.bash
ros2 run mobile_vla_package vla_inference_node
```

## 🎮 키보드 제어

추론 노드 실행 후:

| 키 | 기능 |
|----|------|
| `S` | 추론 시작/중지 |
| `1` | 시나리오 1: "Navigate around obstacles and reach the left bottle" |
| `2` | 시나리오 2: "Navigate around obstacles and reach the right bottle" |
| `3` | 시나리오 3: "Navigate around two boxes and reach the left bottle" |
| `4` | 시나리오 4: "Navigate around two boxes and reach the right bottle" |
| `P` | 통계 표시 (평균 지연, FPS 등) |
| `Ctrl+C` | 종료 |

## 📊 추론 흐름

```
┌─────────────────────────────────────────┐
│  1. 카메라 이미지 취득 (get_image_service) │
│     ↓                                    │
│  2. ImageBuffer에 추가 (window_size=2) │
│     ↓                                    │
│  3. Mobile VLA 추론 (로컬)               │
│     - forward_continuous()             │
│     - abs_action 전략                  │
│     ↓                                    │
│  4. 정규화 해제 (denormalize_action)    │
│     ↓                                    │
│  5. 로봇 제어 (/cmd_vel Twist)          │
└─────────────────────────────────────────┘
```

## 🔧 설정

### 추론 설정 (mobile_vla_inference_node.py)
```python
checkpoint_path: 최신 체크포인트
window_size: 2
fwd_pred_next_n: 10
use_abs_action: True
denormalize_strategy: "safe"
max_linear_x: 0.5  # 안전 속도
max_linear_y: 0.5
inference_interval: 0.3  # 300ms 주기
```

## 📝 로그 예시

```
[INFO] 🤖 Mobile VLA 추론 노드 준비 완료!
[INFO] 📋 조작 방법:
[INFO]    S: 추론 시작/중지
[INFO]    1-4: 시나리오 선택
[INFO]    P: 통계 표시
[INFO] 🚀 추론 시작!
[INFO] 📝 지시문: Navigate around obstacles and reach the left bottle
[INFO] ⏳ 버퍼 채우는 중... (1/2)
[INFO] ⏳ 버퍼 채우는 중... (2/2)
[INFO] ✅ [1] 액션: [0.245, 0.180] | 지연: 234.5ms | 방향: 1.0
[INFO] ✅ [2] 액션: [0.312, 0.156] | 지연: 189.2ms | 방향: 1.0
...
```

## 📊 통계 (P 키)

```
============================================================
📊 추론 통계
============================================================
총 추론 횟수: 25
평균 지연: 210.3 ms
최대 지연: 345.1 ms
최소 지연: 165.8 ms
평균 FPS: 4.75
현재 지시문: Navigate around obstacles and reach the left bottle
============================================================
```

## 🎯 다음 단계

1. **카메라 서버 확인**: `ros2 service list | grep get_image`
2. **추론 노드 빌드**: `colcon build --packages-select mobile_vla_package`
3. **추론 노드 실행**: `ros2 run mobile_vla_package vla_inference_node`
4. **추론 시작**: `S` 키 누르기
5. **동작 확인**: 로봇이 추론 결과에 따라 움직이는지 확인

---

**준비 완료! 이제 실행하세요:**
```bash
# 터미널 1
ros2 run camera_pub camera_publisher_continuous

# 터미널 2
ros2 run mobile_vla_package vla_inference_node
```
