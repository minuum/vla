# 🚀 RoboVLMs → Mobile VLA Task 변경 계획

## 🎯 **1차 목표: RoboVLMs Task를 로봇팔에서 모바일 로봇으로 변경**

### **📊 현재 상황 분석**

#### **RoboVLMs (로봇팔 조작)**
```python
# 기존 RoboVLMs 액션 공간
robovlms_action_space = {
    "end_effector_pos": [x, y, z],        # 3D 위치
    "end_effector_rot": [rx, ry, rz],     # 3D 회전
    "gripper_state": [open/close]         # 그리퍼 상태
}
# 총 7D 액션 공간
```

#### **Mobile VLA (모바일 로봇 내비게이션)**
```python
# 현재 Mobile VLA 액션 공간
mobile_vla_action_space = {
    "linear_x": [-2.0, 2.0],     # 전진/후진 속도 (m/s)
    "linear_y": [-1.0, 1.0],     # 좌우 이동 속도 (m/s)  
    "angular_z": [-3.14, 3.14],  # 회전 속도 (rad/s)
}
# 총 3D 액션 공간 (현재 구현됨)
```

## 🔄 **Task 변경 전략**

### **1. 액션 공간 매핑**

#### **A. 기존 3D → 4D 확장**
```python
# 현재 3D 액션 공간
current_action = [linear_x, linear_y, angular_z]

# 확장된 4D 액션 공간 (RoboVLMs 스타일)
enhanced_action = [
    linear_x,      # 전진/후진 속도
    linear_y,      # 좌우 이동 속도
    angular_z,     # 회전 속도
    action_type    # 액션 타입 (0:이동, 1:회전, 2:정지, 3:특수)
]
```

#### **B. 액션 타입별 세분화**
```python
action_types = {
    0: "move_forward",      # 전진 이동
    1: "move_backward",     # 후진 이동
    2: "turn_left",         # 좌회전
    3: "turn_right",        # 우회전
    4: "move_left",         # 좌측 이동
    5: "move_right",        # 우측 이동
    6: "stop",              # 정지
    7: "special_action"     # 특수 액션 (can tracking 등)
}
```

### **2. 데이터 처리 파이프라인 변경**

#### **A. 기존 데이터 로더 수정**
```python
# 현재: mobile_vla_data_collector.py 기반
class MobileNavigationDataset:
    def __getitem__(self, idx):
        return {
            "images": self.episodes[idx]["images"],                    # [T, H, W, 3]
            "actions": self.episodes[idx]["actions"],                  # [T, 3] → [T, 4]
            "action_event_types": self.episodes[idx]["action_event_types"], # [T]
            "scenario": self.episodes[idx]["scenario"],                # "1box_vert_left" 
            "language": self.korean_instructions[scenario]             # "왼쪽으로 돌아서 컵까지 가세요"
        }
```

#### **B. RoboVLMs 스타일 데이터 로더 추가**
```python
# 새로운: RoboVLMs 스타일 데이터 로더
class RoboVLMsMobileDataset:
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return {
            "rgb": episode["images"],                    # [T, H, W, 3]
            "action": episode["actions"],                # [T, 4] (확장된 액션)
            "language": episode["language"],             # 자연어 명령
            "scenario": episode["scenario"],             # 시나리오 컨텍스트
            "action_chunk": self.get_action_chunk(idx)   # 액션 청킹
        }
    
    def get_action_chunk(self, idx, chunk_size=8):
        """RoboVLMs 스타일 액션 청킹"""
        episode = self.episodes[idx]
        actions = episode["actions"]
        
        # 액션 청킹 (연속된 액션을 그룹화)
        chunks = []
        for i in range(0, len(actions), chunk_size):
            chunk = actions[i:i+chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(chunk)
        
        return chunks
```

### **3. 모델 아키텍처 변경**

#### **A. 기존 Enhanced 모델 확장**
```python
# 현재: EnhancedKosmos2CLIPHybrid
class EnhancedKosmos2CLIPHybrid(nn.Module):
    def __init__(self, action_dim=3):  # 3D → 4D로 변경
        super().__init__()
        self.action_dim = action_dim  # 4
        
        # 기존 구조 유지
        self.kosmos2_model = Kosmos2ForConditionalGeneration.from_pretrained(...)
        self.clip_model = CLIPModel.from_pretrained(...)
        self.vision_resampler = MobileOptimizedVisionResampler(...)
        self.clip_normalization = MobileOptimizedCLIPNormalization(...)
        
        # 액션 출력 레이어 수정
        self.action_head = nn.Linear(hidden_dim, action_dim)  # 3 → 4
```

#### **B. RoboVLMs 스타일 Policy Head 추가**
```python
# 새로운: RoboVLMs 스타일 Policy Head
class RoboVLMsMobilePolicyHead(nn.Module):
    def __init__(self, input_dim, action_dim=4, chunk_size=8):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # 액션 예측 헤드
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # 액션 청킹 헤드
        self.chunk_head = nn.Linear(input_dim, action_dim * chunk_size)
        
        # 액션 타입 분류 헤드
        self.action_type_head = nn.Linear(input_dim, len(action_types))
    
    def forward(self, features):
        # 단일 액션 예측
        single_action = self.action_head(features)
        
        # 액션 청크 예측
        action_chunk = self.chunk_head(features)
        action_chunk = action_chunk.view(-1, self.chunk_size, self.action_dim)
        
        # 액션 타입 예측
        action_type = self.action_type_head(features)
        
        return {
            "single_action": single_action,
            "action_chunk": action_chunk,
            "action_type": action_type
        }
```

### **4. 학습 전략 변경**

#### **A. 다중 손실 함수**
```python
class RoboVLMsMobileLoss(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # 단일 액션 손실
        single_loss = self.mse_loss(predictions["single_action"], targets["actions"])
        
        # 액션 청크 손실
        chunk_loss = self.mse_loss(predictions["action_chunk"], targets["action_chunks"])
        
        # 액션 타입 손실
        type_loss = self.ce_loss(predictions["action_type"], targets["action_types"])
        
        # 총 손실
        total_loss = single_loss + 0.5 * chunk_loss + 0.3 * type_loss
        
        return {
            "total_loss": total_loss,
            "single_loss": single_loss,
            "chunk_loss": chunk_loss,
            "type_loss": type_loss
        }
```

#### **B. 학습 스케줄링**
```python
# 단계별 학습 전략
training_stages = {
    "Stage 1": {
        "description": "기본 4D 액션 학습",
        "action_dim": 4,
        "chunk_size": 1,
        "epochs": 5
    },
    "Stage 2": {
        "description": "액션 청킹 학습",
        "action_dim": 4,
        "chunk_size": 4,
        "epochs": 5
    },
    "Stage 3": {
        "description": "전체 RoboVLMs 스타일 학습",
        "action_dim": 4,
        "chunk_size": 8,
        "epochs": 10
    }
}
```

## 🎯 **구현 계획**

### **Week 1: 액션 공간 확장**
1. **액션 차원 확장**: 3D → 4D
2. **액션 타입 추가**: 8가지 액션 타입 정의
3. **데이터 로더 수정**: 4D 액션 지원

### **Week 2: RoboVLMs 스타일 데이터 로더**
1. **액션 청킹 구현**: 연속 액션 그룹화
2. **RoboVLMsMobileDataset 클래스**: 새로운 데이터 로더
3. **데이터 검증**: 기존 데이터와 호환성 확인

### **Week 3: 모델 아키텍처 변경**
1. **Enhanced 모델 수정**: 4D 액션 출력
2. **RoboVLMsMobilePolicyHead 추가**: 새로운 정책 헤드
3. **다중 출력 지원**: 단일/청크/타입 예측

### **Week 4: 학습 및 검증**
1. **다중 손실 함수**: 3가지 손실 조합
2. **단계별 학습**: 3단계 학습 전략
3. **성능 비교**: 기존 모델과 성능 비교

## 📊 **예상 성능 개선**

### **현재 vs 목표**
| 지표 | 현재 (3D) | 목표 (4D) | 개선율 |
|------|-----------|-----------|--------|
| **MAE** | 0.2121 | 0.15-0.18 | 15-30% |
| **Success Rate** | 0% | 50-70% | +50-70%p |
| **액션 정확도** | 78.8% | 85-90% | 6-11%p |
| **청크 정확도** | N/A | 60-80% | 새로운 지표 |

### **RoboVLMs 스타일 장점**
1. **액션 청킹**: 연속된 액션의 일관성 향상
2. **액션 타입**: 명확한 액션 분류
3. **다중 출력**: 다양한 예측 방식
4. **확장성**: 향후 복잡한 작업 지원

## 🚀 **다음 단계**

1. **즉시 시작**: 액션 공간 3D → 4D 확장
2. **데이터 수정**: 기존 데이터셋에 액션 타입 추가
3. **모델 수정**: Enhanced 모델의 액션 출력 차원 변경
4. **학습 시작**: 4D 액션으로 재학습

이 계획을 통해 RoboVLMs의 강력한 액션 청킹과 다중 출력 방식을 모바일 로봇에 적용할 수 있습니다!
