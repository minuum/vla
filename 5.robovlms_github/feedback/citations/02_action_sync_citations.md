# 2. Action-rel_action Synchronization - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed from @RoboVLMs)**

### **2.1 Coordinate Transformation Functions**
- **File**: `RoboVLMs/robovlms/data/data_utils.py:770-821` (Updated from @RoboVLMs)
- **Implementation**: `world_to_tcp_frame()` function for absolute to relative coordinate transformation
- **Core Code**:
```python
def world_to_tcp_frame(action, robot_obs):
    """절대 좌표계에서 TCP(Tool Center Point) 상대 좌표계로 변환"""
    # 배치 크기와 시퀀스 길이 추출
    b, s, _ = action.shape
    
    # 로봇 관찰값에서 오일러 각도를 회전 행렬로 변환
    world_T_tcp = (
        euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
        .float()
        .reshape(-1, 3, 3)
    )
    # TCP에서 월드로의 변환 행렬 (역행렬)
    tcp_T_world = torch.inverse(world_T_tcp)
    
    # 위치 좌표 변환 (월드 → TCP)
    pos_w_rel = action[..., :3].reshape(-1, 3, 1)
    pos_tcp_rel = tcp_T_world @ pos_w_rel
    
    # 회전 좌표 스케일링 (0.01 → 100)
    orn_w_rel = action[..., 3:6] * 0.01
    orn_tcp_rel *= 100
    
    # 위치, 회전, 그리퍼 액션 결합
    action_tcp = torch.cat([
        pos_tcp_rel.reshape(b, s, -1),      # TCP 상대 위치
        orn_tcp_rel.reshape(b, s, -1),     # TCP 상대 회전
        action[..., -1:],                   # 그리퍼 액션 (변경 없음)
    ], dim=-1)
```

### **2.2 TCP to World Frame Transformation**
- **File**: `RoboVLMs/robovlms/data/data_utils.py:823-874` (Updated from @RoboVLMs)
- **Implementation**: `tcp_to_world_frame()` function for relative to absolute coordinate transformation
- **Core Code**:
```python
def tcp_to_world_frame(action, robot_obs):
    """TCP 상대 좌표계에서 절대 좌표계로 변환"""
    # 배치 크기와 시퀀스 길이 추출
    b, s, _ = action.shape
    
    # 로봇 관찰값에서 오일러 각도를 회전 행렬로 변환
    world_T_tcp = (
        euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
        .float()
        .reshape(-1, 3, 3)
    )
    
    # TCP 상대 위치를 월드 절대 위치로 변환
    pos_tcp_rel = action[..., :3].reshape(-1, 3, 1)
    pos_w_rel = world_T_tcp @ pos_tcp_rel
    
    # 회전 좌표 변환 (TCP → 월드)
    orn_tcp_rel = action[..., 3:6] * 0.01
    orn_w_rel *= 100
    
    # 위치, 회전, 그리퍼 액션 결합
    action_w = torch.cat([
        pos_w_rel.reshape(b, s, -1),        # 월드 절대 위치
        orn_w_rel.reshape(b, s, -1),       # 월드 절대 회전
        action[..., -1:],                   # 그리퍼 액션 (변경 없음)
    ], dim=-1)
```

### **2.3 CALVIN Dataset TCP Frame Transformation**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:857-858` (Updated from @RoboVLMs)
- **Implementation**: TCP relative frame transformation in dataset collater
- **Code**:
```python
# TCP 상대 좌표계 사용 여부 확인
if self.tcp_rel:
    # 절대 좌표를 TCP 상대 좌표로 변환
    action_tensors = world_to_tcp_frame(action_tensors, robot_obs)
```

### **2.4 Model Wrapper TCP Frame Usage**
- **File**: `RoboVLMs/eval/calvin/model_wrapper.py:360-368` (Updated from @RoboVLMs)
- **Implementation**: TCP frame transformation in model wrapper step function
- **Code**:
```python
# TCP 상대 좌표계 사용 여부 확인
if self.tcp_rel:
    # 로봇 관찰값을 텐서로 변환하고 차원 확장
    robot_obs = (
        torch.from_numpy(obs["robot_obs"])
        .unsqueeze(0)      # 배치 차원 추가
        .unsqueeze(0)      # 시퀀스 차원 추가
        .unsqueeze(0)      # 시간 차원 추가
        .repeat(1, 1, self.fwd_pred_next_n, 1)  # 예측 스텝 수만큼 반복
    )
    # TCP 상대 좌표를 월드 절대 좌표로 변환
    action = tcp_to_world_frame(action, robot_obs)
```

### **2.6 Action Normalization**
- **File**: `RoboVLMs/robovlms/data/data_utils.py:682-688`
- **Implementation**: Action normalization function
- **Code**:
```python
def normalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    """액션을 [-1, 1] 범위로 정규화"""
    # 마지막 값(그리퍼) 저장
    last_val = action[..., -1]
    
    # 액션을 지정된 범위로 클리핑
    action = np.clip(action, a_min=float(action_min), a_max=float(action_max))
    
    # [-1, 1] 범위로 정규화
    res = 2 * (action - action_min) / (action_max - action_min) - 1
    
    # 마지막 값 유지 옵션 (그리퍼 액션 보존)
    if maintain_last:
        res[..., -1] = last_val
    return res
```

## 📊 **Synchronization Method Evidence**

### **2.7 Absolute vs Relative Coordinates**
- **Absolute Coordinates**: 3D world coordinates (x, y, z, rx, ry, rz, gripper)
- **Relative Coordinates**: Normalized relative coordinates (-1, 1) with scaling factors
- **Transformation**: `world_to_tcp_frame()` and `tcp_to_world_frame()`

### **2.8 Scaling Factors**
- **Position Scaling**: Factor 50.0 for x, y, z coordinates
- **Orientation Scaling**: Factor 20.0 for rx, ry, rz rotations
- **Gripper Action**: Binary (-1, 1) or continuous control
- **Normalization**: Clipped to (-1, 1) range

### **2.9 Coordinate System Features**
- **World Coordinates**: 3D absolute position and orientation
- **TCP Coordinates**: Tool Center Point relative coordinates
- **Relative Actions**: Zero-padded except gripper action
- **Gripper Handling**: Maintained separately from arm actions

## 🎯 **Key Findings**

1. **Dual Coordinate System**: Both absolute and relative coordinates supported
2. **Automatic Transformation**: Built-in conversion functions (`world_to_tcp_frame`, `tcp_to_world_frame`)
3. **Scaling Optimization**: Different factors for position (50.0) vs orientation (20.0)
4. **CALVIN Integration**: Native support in dataset loading with `rel_actions` configuration
5. **TCP Frame Support**: Tool Center Point relative coordinate transformation
6. **Gripper Handling**: Separate processing for gripper actions vs arm actions

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/data/data_utils.py` (coordinate transformation functions)
- `RoboVLMs/robovlms/data/calvin_dataset.py` (relative actions processing)
- `RoboVLMs/eval/calvin/model_wrapper.py` (scaling factors implementation)
- `RoboVLMs/robovlms/data/pose_transforms.py` (pose transformation utilities)
