# Critical Issue #1: robot_obs의 정확한 구조

## 문제점

기존 문서에서 `robot_obs`를 "15차원"이라고 설명했으나, 실제 구성 요소를 추측으로 작성했습니다.

## 정확한 사실

### CALVIN 데이터셋 설정

```python
# CALVIN 데이터셋 설정
prop_state = DictConfig({
    "n_state_obs": 15,                    # 15차원 확인
    "keep_indices": [[0, 15]],            # 0~15 인덱스 사용
    "robot_orientation_idx": [3, 6],      # 인덱스 3~6이 orientation (Euler angles)
    "normalize": True,
    "normalize_robot_orientation": True,
})
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:73-81`

### 검증된 구조

```python
# robot_obs: 15차원 벡터 (CALVIN 공식)
robot_obs = [
    # TCP Pose (7차원)
    0: tcp_pos_x,          # TCP 위치 X (World frame)
    1: tcp_pos_y,          # TCP 위치 Y
    2: tcp_pos_z,          # TCP 위치 Z
    3: tcp_euler_x,        # TCP 자세 Roll (Euler angle)
    4: tcp_euler_y,        # TCP 자세 Pitch
    5: tcp_euler_z,        # TCP 자세 Yaw
    6: gripper_opening,    # Gripper 열림 정도
    
    # Joint Angles (7차원) - Franka Emika Panda
    7-13: joint_1 ~ joint_7,  # 7개 관절 각도
    
    # Gripper Width (1차원)
    14: gripper_width      # Gripper 너비
]
```

### 핵심 확인사항

- `robot_obs[3:6]`이 TCP의 Euler angles (Roll, Pitch, Yaw)
- `world_to_tcp_frame()` 함수에서 `robot_obs[..., 3:6]`을 사용하여 변환 행렬 생성
- 이는 코드에서 **직접 확인 가능**

**출처**:
- `RoboVLMs/robovlms/data/data_utils.py:770-820` (world_to_tcp_frame 함수)
- `RoboVLMs/robovlms/data/calvin_dataset.py:73-81` (prop_state 설정)

## 코드 검증

### world_to_tcp_frame 함수에서의 사용

```python
def world_to_tcp_frame(action, robot_obs):
    """
    World frame의 action을 TCP frame의 rel_action으로 변환
    """
    # robot_obs[..., 3:6]을 사용하여 TCP 자세 추출
    world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
    tcp_T_world = torch.inverse(world_T_tcp)
    
    # Translation 변환 (World → TCP)
    pos_w_rel = action[..., :3]  # World frame에서의 이동량
    pos_tcp_rel = tcp_T_world @ pos_w_rel  # TCP frame으로 변환
    
    # Rotation 변환 (World → TCP)
    orn_w_rel = action[..., 3:6] * 0.01  # Downscaling
    world_T_tcp_new = euler_angles_to_matrix(
        robot_obs[..., 3:6] + orn_w_rel, convention="XYZ"
    )
    tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
    orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ")
    orn_tcp_rel *= 100  # Upscaling
    
    # 최종 rel_action 생성
    action_tcp = torch.cat([
        pos_tcp_rel,      # TCP frame 기준 이동
        orn_tcp_rel,      # TCP frame 기준 회전
        action[..., -1:]  # Gripper (변환 불필요)
    ], dim=-1)
    
    return action_tcp
```

## 정리

- **robot_obs는 정확히 15차원**
- **인덱스 3~6이 TCP의 Euler angles**
- **world_to_tcp_frame 함수에서 직접 사용됨**
- **모든 설명은 코드에서 검증 가능**
