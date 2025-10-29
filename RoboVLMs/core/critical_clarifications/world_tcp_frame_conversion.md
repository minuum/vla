# Critical Issue #2: World Frame vs TCP Frame 변환의 물리적 의미

## 문제점

"World frame의 action을 TCP frame으로 변환"한다고 했지만, **왜 변환하는지**, **언제 변환하는지** 명확하지 않음.

## 정확한 사실

### 변환 시점과 목적

```python
# collater() - 배치 생성 시 변환 수행
def collater(self, sample):
    # ... (전처리)
    robot_obs = torch.from_numpy(
        np.array([np.stack(s["robot_obs"]) for s in sample])
    )[:, :-1]
    
    # TCP frame으로 변환 (옵션)
    if self.tcp_rel:  # tcp_rel=True일 때만 변환
        action_tensors = world_to_tcp_frame(action_tensors, robot_obs)
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:857-858`

### 변환 이유 (물리적 의미)

| **상황** | **World Frame** | **TCP Frame** |
|----------|-----------------|---------------|
| **로봇이 정면을 향함** | "오른쪽으로 10cm" = (+0.1, 0, 0) | "오른쪽으로 10cm" = (+0.1, 0, 0) |
| **로봇이 180도 회전** | "오른쪽으로 10cm" = **(-0.1, 0, 0)** | "오른쪽으로 10cm" = **(+0.1, 0, 0)** |

**핵심**:
- **World Frame**: 로봇 자세에 따라 같은 명령이 다른 절대 좌표로 변환됨
- **TCP Frame**: 로봇 자세와 무관하게 "end-effector 기준 상대 이동"이 일정함
- **일반화 성능**: TCP frame이 월등히 높음 (다른 자세에서도 동일한 동작)

### 실제 사용 여부

```python
# Config 설정
{
    "tcp_rel": false  # 대부분의 CALVIN 실험에서 False
}
```

**이유**:
- CALVIN 데이터셋은 **이미 rel_actions (TCP frame)으로 저장되어 있음**
- `tcp_rel=True`는 **World frame action을 TCP frame으로 변환**할 때만 필요
- RoboVLMs는 CALVIN의 기본 rel_actions를 그대로 사용

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:550-574`

## 물리적 의미 상세

### World Frame의 문제점

```python
# 로봇이 정면을 향한 상태
robot_pose = [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw
command = "오른쪽으로 10cm 이동"
world_action = [0.1, 0, 0, 0, 0, 0, 0]  # +x 방향

# 로봇이 180도 회전한 상태
robot_pose = [0, 0, 0, 0, 0, 180]  # yaw만 180도 회전
command = "오른쪽으로 10cm 이동"
world_action = [-0.1, 0, 0, 0, 0, 0, 0]  # -x 방향 (다른 값!)
```

### TCP Frame의 장점

```python
# 로봇이 정면을 향한 상태
robot_pose = [0, 0, 0, 0, 0, 0]
command = "오른쪽으로 10cm 이동"
tcp_action = [0.1, 0, 0, 0, 0, 0, 0]  # TCP 기준 +x 방향

# 로봇이 180도 회전한 상태
robot_pose = [0, 0, 0, 0, 0, 180]
command = "오른쪽으로 10cm 이동"
tcp_action = [0.1, 0, 0, 0, 0, 0, 0]  # TCP 기준 +x 방향 (동일!)
```

## 변환 수학적 과정

### World to TCP Frame 변환

```python
def world_to_tcp_frame(action, robot_obs):
    # 1. 현재 로봇 자세에서 World → TCP 변환 행렬 생성
    world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
    tcp_T_world = torch.inverse(world_T_tcp)
    
    # 2. Translation 변환 (World → TCP)
    pos_w_rel = action[..., :3]  # World frame에서의 이동량
    pos_tcp_rel = tcp_T_world @ pos_w_rel  # TCP frame으로 변환
    
    # 3. Rotation 변환 (World → TCP)
    orn_w_rel = action[..., 3:6] * 0.01  # Downscaling
    world_T_tcp_new = euler_angles_to_matrix(
        robot_obs[..., 3:6] + orn_w_rel, convention="XYZ"
    )
    tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
    orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ")
    orn_tcp_rel *= 100  # Upscaling
    
    # 4. 최종 rel_action 생성
    action_tcp = torch.cat([
        pos_tcp_rel,      # TCP frame 기준 이동
        orn_tcp_rel,      # TCP frame 기준 회전
        action[..., -1:]  # Gripper (변환 불필요)
    ], dim=-1)
    
    return action_tcp
```

## 정리

### 변환의 목적
1. **일반화 성능 향상**: 로봇 자세와 무관한 일관된 액션
2. **직관적 제어**: "오른쪽으로 이동"이 항상 같은 의미
3. **학습 효율성**: 동일한 동작을 다른 자세에서도 학습 가능

### 실제 사용
- **CALVIN 데이터셋**: 이미 TCP frame (rel_actions)으로 저장
- **RoboVLMs**: CALVIN의 rel_actions를 그대로 사용
- **tcp_rel=False**: 변환 없이 직접 사용

### 핵심 포인트
- **World Frame**: 절대 좌표, 로봇 자세에 따라 값이 변함
- **TCP Frame**: 상대 좌표, 로봇 자세와 무관하게 일정
- **변환 필요성**: World frame 데이터를 TCP frame으로 변환할 때만 필요
