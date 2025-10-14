# 13. 2D/3D Coordinate Synchronization - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed)**

### **13.1 Coordinate Transformation Functions**
- **File**: `RoboVLMs/robovlms/data/data_utils.py:770-872`
- **Implementation**: `world_to_tcp_frame()` and `tcp_to_world_frame()` functions
- **Code**:
```python
def world_to_tcp_frame(action, robot_obs):
    """절대 좌표계에서 TCP 상대 좌표계로 변환"""
    # 절대 좌표 → 상대 좌표 변환
    pos_w_rel = action[..., :3].reshape(-1, 3, 1)    # 위치 좌표 (x, y, z)
    pos_tcp_rel = tcp_T_world @ pos_w_rel            # TCP 상대 위치로 변환
    
    # 회전 스케일링 (0.01 → 100)
    orn_w_rel = action[..., 3:6] * 0.01              # 회전 좌표 스케일링
    orn_tcp_rel *= 100                               # 회전 좌표 역스케일링
    
    # 위치, 회전, 그리퍼 액션 결합
    action_tcp = torch.cat([
        pos_tcp_rel.reshape(b, s, -1),      # TCP 상대 위치
        orn_tcp_rel.reshape(b, s, -1),     # TCP 상대 회전
        action[..., -1:],                   # 그리퍼 액션 (변경 없음)
    ], dim=-1)
```

### **13.2 Coordinate System Synchronization**
- **File**: `5.robovlms_github/feedback/action_image_text_syncing.md:330-352`
- **Implementation**: 2D/3D coordinate synchronization method
- **Code**:
```python
# 2D와 3D 좌표 동기화
# 상대 월드 좌표를 (-1, 1)로 정규화하고 스케일링 팩터 50으로 클리핑

# 위치 스케일링: 스케일링 팩터 50
position_scaled = position * 50              # 위치 좌표 50배 스케일링

# 회전 스케일링: 스케일링 팩터 20
orientation_scaled = orientation * 20        # 회전 좌표 20배 스케일링
```

### **13.3 CALVIN Dataset Action Normalization**
- **File**: `5.robovlms_github/feedback/action_image_text_syncing.md:96-99`
- **Implementation**: CALVIN dataset action normalization
- **Code**:
```python
# rel_action (상대 좌표)
tcp position (3): x,y,z in relative world coordinates     # TCP 위치 (3차원)
normalized and clipped to (-1, 1) with scaling factor 50  # (-1, 1) 정규화, 스케일링 팩터 50
tcp orientation (3): euler angles x,y,z in relative world coordinates  # TCP 회전 (3차원)
normalized and clipped to (-1, 1) with scaling factor 20  # (-1, 1) 정규화, 스케일링 팩터 20
gripper_action (1): binary (close = -1, open = 1)        # 그리퍼 액션 (이진값)
```

## 📊 **Coordinate Synchronization Evidence**

### **13.4 Absolute vs Relative Coordinates**
- **Absolute Coordinates**: 3D world coordinates (x, y, z, rx, ry, rz, gripper)
- **Relative Coordinates**: Normalized relative coordinates (-1, 1) with scaling factors
- **Transformation**: `world_to_tcp_frame()` and `tcp_to_world_frame()`

### **13.5 Scaling Factors**
- **Position Scaling**: Factor 50 for position coordinates
- **Orientation Scaling**: Factor 20 for rotation coordinates
- **Gripper Action**: Binary (-1, 1) for gripper control
- **Normalization**: Clipped to (-1, 1) range

### **13.6 Coordinate System Features**
- **World Coordinates**: 3D absolute position and orientation
- **TCP Coordinates**: Tool Center Point relative coordinates
- **Normalization**: Consistent (-1, 1) range across all dimensions
- **Scaling**: Different factors for position vs orientation

## 🎯 **Key Findings**

1. **Dual Coordinate System**: Both absolute and relative coordinates supported
2. **Automatic Transformation**: Built-in conversion functions
3. **Scaling Optimization**: Different factors for position vs orientation
4. **CALVIN Integration**: Native support in dataset loading

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/data/data_utils.py`
- `5.robovlms_github/feedback/action_image_text_syncing.md`
- `RoboVLMs/robovlms/data/calvin_dataset.py`
- `RoboVLMs/robovlms/data/pose_transforms.py`
