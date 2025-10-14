# 5. 7 DOF Synchronization Method and Learning - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed from @RoboVLMs)**

### **5.1 Action Space Configuration**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1344-1382` (Updated from @RoboVLMs)
- **Implementation**: Action space configuration and routing
- **Code**:
```python
def forward_action(self, vision_x, lang_x, attention_mask=None, ...):
    """액션 공간에 따른 포워드 라우팅"""
    # 액션 공간 설정 확인 (연속/이산)
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    if action_space == "discrete":
        # 이산 액션 공간 처리
        return self.forward_discrete(
            vision_x=vision_x,           # 비전 입력
            lang_x=lang_x,               # 언어 입력
            attention_mask=attention_mask, # 어텐션 마스크
            action_labels=action_labels,  # 액션 레이블
            action_mask=action_mask,      # 액션 마스크
            # ... 기타 파라미터들
        )
    else:
        # 연속 액션 공간 처리
        return self.forward_continuous(
            vision_x=vision_x,           # 비전 입력
            lang_x=lang_x,               # 언어 입력
            attention_mask=attention_mask, # 어텐션 마스크
            action_labels=action_labels,  # 액션 레이블
            action_mask=action_mask,      # 액션 마스크
            # ... 기타 파라미터들
        )
```

### **5.2 Action Parser Implementation**
- **File**: `RoboVLMs/vla_test/robovlm_action_parser.py:15-102` (Updated from @RoboVLMs)
- **Implementation**: Action space enum and parser configuration
- **Code**:
```python
class ActionSpace(Enum):
    """액션 공간 타입"""
    CONTINUOUS = "continuous"  # 연속 액션 공간
    DISCRETE = "discrete"      # 이산 액션 공간

class RoboVLMActionParser:
    """RoboVLMs 액션 파서 (7 DOF 지원)"""
    def __init__(self, 
                 action_space: ActionSpace = ActionSpace.CONTINUOUS,
                 action_dim: int = 6,  # 6 DOF + 1 gripper = 7 DOF
                 bins: int = 256,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 prediction_horizon: int = 1):
        
        self.action_space = action_space    # 액션 공간 타입
        self.action_dim = action_dim        # 7 DOF 설정 (6 DOF 팔 + 1 DOF 그리퍼)
        self.bins = bins                    # 이산화 시 사용할 빈 수
        self.min_action = min_action        # 액션 최소값 (-1.0)
        self.max_action = max_action        # 액션 최대값 (1.0)
        
        # 이산 액션 공간을 위한 빈 생성
        if action_space == ActionSpace.DISCRETE:
            # 액션 범위를 빈으로 분할
            self.action_bins = np.linspace(min_action, max_action, bins)
            # 각 빈의 중심값 계산
            self.bin_centers = (self.action_bins[:-1] + self.action_bins[1:]) / 2.0
```

### **5.3 Discrete Action Decoder**
- **File**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:173-227` (Updated from @RoboVLMs)
- **Implementation**: `DiscreteDecoder` class for discrete action processing
- **Code**:
```python
class DiscreteDecoder(BasePolicyHead):
    """이산 액션 디코더 (7 DOF)"""
    def __init__(
        self,
        hidden_size,              # 히든 상태 크기
        action_dim,               # 액션 차원 (7 DOF)
        action_space="continuous", # 액션 공간 타입
        down_sample="pooling",     # 다운샘플링 방법
        latent=1,                 # 잠재 차원
        cont_token_nun=1,         # 연속 토큰 수
        n_bin=256,                # 이산화 빈 수
        min_action=-1,            # 액션 최소값
        max_action=1,             # 액션 최대값
        tokenizer=None,           # 토크나이저
        **kwargs,
    ):
        super().__init__(
            hidden_size, action_dim, action_space, down_sample, latent, **kwargs
        )
        self.cont_token_num = cont_token_nun  # 연속 토큰 수
        self.n_bin = n_bin                    # 이산화 빈 수
        self.min_action = min_action          # 액션 최소값
        self.max_action = max_action          # 액션 최대값

        # 액션 토크나이저 import 및 초기화
        from robovlms.model.policy_head.action_tokenizer import ActionTokenizer

        self.action_tokenizer = ActionTokenizer(
            tokenizer,                    # 토크나이저
            bins=self.n_bin,              # 빈 수
            min_action=self.min_action,   # 최소 액션값
            max_action=self.max_action,   # 최대 액션값
        )
```

### **5.2.1 Discrete vs Continuous Action Space 차이점**

#### **Continuous Action Space (연속 액션 공간)**
- **정의**: 실수 값으로 표현되는 연속적인 액션
- **예시**: `[0.5, -0.3, 0.8, 0.2, -0.1, 0.7, 0.0]` (7 DOF)
- **특징**: 
  - 정밀한 제어 가능
  - 직접적인 로봇 제어
  - 메모리 효율적
- **사용 사례**: 정밀한 로봇 조작, 연속적인 움직임

#### **Discrete Action Space (이산 액션 공간)**
- **정의**: 토큰으로 표현되는 이산적인 액션
- **예시**: `[45, 23, 67, 12, 89, 34, 0]` (토큰 ID)
- **특징**:
  - 토큰 기반 표현
  - 언어 모델과 호환
  - 시퀀스 모델링 용이
- **사용 사례**: 언어 모델 기반 제어, 시퀀스 예측

#### **변환 과정**
```python
# Continuous → Discrete 변환
continuous_action = [0.5, -0.3, 0.8]  # 연속 값
discrete_tokens = tokenizer.encode(continuous_action)  # [45, 23, 67]

# Discrete → Continuous 변환  
discrete_tokens = [45, 23, 67]  # 토큰 ID
continuous_action = tokenizer.decode(discrete_tokens)  # [0.5, -0.3, 0.8]
```

#### **7 DOF 동기화 과정**
1. **Continuous**: 7차원 실수 벡터 직접 사용
2. **Discrete**: 7차원 실수 벡터 → 7개 토큰 → 시퀀스 예측
3. **동기화**: 시간적 일관성 유지
4. **학습**: End-to-end 파인튜닝

### **5.2.2 Continuous vs Discrete 사용 사례**

#### **Continuous Action Space 사용 사례**

##### **1. 정밀한 로봇 조작**
- **상황**: 미세한 움직임이 필요한 작업
- **예시**: 수술용 로봇, 정밀 조립, 미세 조작
- **장점**: 연속적인 제어, 정밀도 높음
- **코드 예시**:
```python
# 정밀한 그리퍼 제어
gripper_action = 0.75  # 75% 닫힘 (연속값)
arm_position = [0.123, -0.456, 0.789]  # 정밀한 위치
```

##### **2. 실시간 제어**
- **상황**: 실시간 피드백이 필요한 작업
- **예시**: 동적 환경에서의 조작, 실시간 추적
- **장점**: 직접적인 제어, 지연 시간 최소화
- **코드 예시**:
```python
# 실시간 제어
current_action = [x, y, z, rx, ry, rz, gripper]  # 직접 제어
robot.execute_action(current_action)  # 즉시 실행
```

##### **3. 물리 시뮬레이션**
- **상황**: 물리 엔진과의 연동
- **예시**: MuJoCo, PyBullet 시뮬레이션
- **장점**: 물리 법칙과 자연스러운 연동
- **코드 예시**:
```python
# 시뮬레이션 환경
action = env.action_space.sample()  # 연속 액션 샘플링
obs, reward, done, info = env.step(action)  # 물리 시뮬레이션
```

#### **Discrete Action Space 사용 사례**

##### **1. 언어 모델 기반 제어**
- **상황**: VLM과의 통합이 필요한 경우
- **예시**: RoboVLMs, RT-2, PaLM-E
- **장점**: 언어 모델과 자연스러운 통합
- **코드 예시**:
```python
# 언어 모델과 통합
text_prompt = "Pick up the red block"
action_tokens = model.generate(text_prompt)  # [45, 23, 67, 12, 89, 34, 0]
action = tokenizer.decode(action_tokens)  # [0.5, -0.3, 0.8, 0.2, -0.1, 0.7, 0.0]
```

##### **2. 시퀀스 모델링**
- **상황**: 시간적 의존성이 중요한 작업
- **예시**: 장기 계획, 복잡한 태스크 시퀀스
- **장점**: 시퀀스 모델의 강력한 표현력 활용
- **코드 예시**:
```python
# 시퀀스 예측
action_sequence = model.predict_sequence(obs_history)  # [token1, token2, ..., token7]
for token in action_sequence:
    action = tokenizer.decode(token)
    robot.execute_action(action)
```

##### **3. 이산화된 제어**
- **상황**: 제한된 액션 공간이 필요한 경우
- **예시**: 게임 AI, 제한된 환경에서의 학습
- **장점**: 탐색 공간 축소, 학습 안정성
- **코드 예시**:
```python
# 이산화된 액션 공간
discrete_actions = [0, 1, 2, 3, 4, 5, 6]  # 7개 이산 액션
selected_action = discrete_actions[3]  # 4번째 액션 선택
```

#### **5.2.3 선택 기준**

##### **Continuous를 선택하는 경우**
- ✅ **정밀한 제어**가 필요한 경우
- ✅ **실시간 제어**가 중요한 경우
- ✅ **물리 시뮬레이션**과 연동하는 경우
- ✅ **직접적인 로봇 제어**가 필요한 경우

##### **Discrete를 선택하는 경우**
- ✅ **언어 모델과 통합**하는 경우
- ✅ **시퀀스 모델링**이 중요한 경우
- ✅ **이산화된 제어**가 필요한 경우
- ✅ **토큰 기반 표현**이 유리한 경우

##### **RoboVLMs에서의 선택**
- **기본 설정**: `"action_space": "continuous"` (대부분의 설정 파일)
- **Discrete 사용**: 특정 실험에서만 사용
- **이유**: 정밀한 로봇 제어가 주 목적이기 때문

### **5.3 Action Head Initialization**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:425-468`
- **Implementation**: `_init_heads()` function
- **Code**:
```python
def _init_heads(self):
    action_head = None
    if self.act_head_configs is not None:
        import robovlms.model.policy_head as action_heads
        
        _kwargs = copy.deepcopy(self.act_head_configs)
        _kwargs.update(
            dict(
                hidden_size=self.hidden_size,
                fwd_pred_next_n=self.fwd_pred_next_n,
                window_size=self.window_size,
                n_bin=self.act_head_configs.get("n_bin", 256),
                min_action=self.act_head_configs.get("min_action", -1),
                max_action=self.act_head_configs.get("max_action", 1),
            )
        )
        _cls = getattr(action_heads, _kwargs.pop("type"))
        self.latent_num = self.act_head_configs.get("latent", 1)
        action_head = _cls(**_kwargs)
    
    return action_head, fwd_decoder, clip_norm_head
```

## 📊 **Learning Process Evidence**

### **5.4 Action Sequence Learning**
- **Sequence Length**: Configurable window size (8, 16, 32)
- **Action Chunking**: Multi-step action prediction
- **Temporal Modeling**: History-aware action generation

### **5.5 7 DOF Synchronization**
- **Position (3 DOF)**: X, Y, Z coordinates
- **Orientation (3 DOF)**: Euler angles (X, Y, Z)
- **Gripper (1 DOF)**: Binary or continuous control
- **Total**: 7 DOF synchronized action space

## 🎯 **Key Findings**

1. **Discrete Action Space**: Tokenized action representation
2. **Sequence Prediction**: Multi-step action generation
3. **7 DOF Support**: Complete robot arm control
4. **Configurable**: Flexible action space configuration

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/model/backbone/base_backbone.py`
- `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py`
- `RoboVLMs/configs/calvin_finetune/*.json` (9 files)
- `RoboVLMs/configs/oxe_training/*.json` (4 files)
