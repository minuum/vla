# RoboVLMs Code Analysis

## 프레임워크 구조

### 핵심 컴포넌트

#### 1. BaseRoboVLM 클래스
```python
class BaseRoboVLM:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.image_processor = None
        self.hidden_size = None
        self.word_embedding = None
        self.text_tower = None
        self.vision_tower = None
    
    @property
    def image_processor(self):
        """이미지 전처리기 반환"""
        pass
    
    @property
    def hidden_size(self):
        """VLM 백본의 히든 크기 반환"""
        pass
    
    @property
    def word_embedding(self):
        """단어 임베딩 반환"""
        pass
    
    @property
    def text_tower(self):
        """텍스트 처리 컴포넌트 반환"""
        pass
    
    @property
    def vision_tower(self):
        """비전 처리 컴포넌트 반환"""
        pass
    
    @property
    def model(self):
        """VLM 백본 반환"""
        pass
    
    def model_encode_images(self, images):
        """이미지를 비전 토큰으로 인코딩"""
        pass
```

#### 2. VLM 통합 예시 (PaliGemma)
```python
class RoboPaligemma(BaseRoboVLM):
    @property
    def image_processor(self):
        return self.model.processor
    
    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size
    
    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens
    
    @property
    def text_tower(self):
        return self.model.language_model.model
    
    @property
    def vision_tower(self):
        return self.model.vision_tower
    
    @property
    def model(self):
        return self.backbone
    
    def model_encode_images(self, images):
        image_outputs = self.model.vision_tower(images)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.model.config.hidden_size**0.5)
        return image_features
```

## VLA 구조 구현

### 1. One-Step Models

#### 연속 액션 모델
```python
class OneStepContinuousVLA(BaseRoboVLM):
    def forward(self, vision_x, lang_x, attention_mask=None):
        # VLM 백본으로 learnable 토큰 예측
        learnable_token = self.model(vision_x, lang_x, attention_mask)
        
        # MLP로 액션 벡터 예측
        action_sequence = self.action_head(learnable_token)
        
        return {"action": action_sequence}
```

#### 이산 액션 모델
```python
class OneStepDiscreteVLA(BaseRoboVLM):
    def forward(self, vision_x, lang_x, attention_mask=None):
        # VLM 백본으로 액션 토큰 예측
        action_tokens = self.model(vision_x, lang_x, attention_mask)
        
        return {"action": action_tokens}
```

### 2. Interleaved Models

```python
class InterleavedContinuousVLA(BaseRoboVLM):
    def forward(self, vision_x, lang_x, attention_mask=None):
        # 관찰-액션 시퀀스 구성
        sequence = self.create_interleaved_sequence(vision_x, lang_x)
        
        # VLM 백본으로 시퀀스 처리
        learnable_tokens = self.model(sequence)
        
        # MLP로 액션 예측
        action_sequence = self.action_head(learnable_tokens[-1])
        
        return {"action": action_sequence}
    
    def create_interleaved_sequence(self, vision_x, lang_x):
        """관찰과 액션을 교차 형식으로 구성"""
        sequence = []
        for i in range(len(vision_x)):
            sequence.append(vision_x[i])  # [OBS]
            sequence.append(self.learnable_token)  # [LRN]
        return sequence
```

### 3. Policy Head Models

```python
class PolicyHeadContinuousVLA(BaseRoboVLM):
    def __init__(self, config):
        super().__init__(config)
        self.policy_head = self.create_policy_head(config)
    
    def forward(self, vision_x, lang_x, attention_mask=None):
        # 각 시간 단계에서 단일 단계 멀티모달 표현 생성
        learnable_tokens = []
        for i in range(len(vision_x)):
            obs = vision_x[i]
            lang = lang_x[i] if isinstance(lang_x, list) else lang_x
            
            # VLM으로 learnable 토큰 생성
            learnable_token = self.model(obs, lang, attention_mask)
            learnable_tokens.append(learnable_token)
        
        # 정책 헤드로 히스토리 정보 모델링 및 액션 예측
        action_sequence = self.policy_head(learnable_tokens)
        
        return {"action": action_sequence}
    
    def create_policy_head(self, config):
        """정책 헤드 생성 (RNN, Transformer, Diffusion 등)"""
        if config.policy_head_type == "rnn":
            return nn.LSTM(config.hidden_size, config.action_dim)
        elif config.policy_head_type == "transformer":
            return nn.TransformerEncoder(...)
        elif config.policy_head_type == "diffusion":
            return DiffusionPolicyHead(...)
```

## 액션 처리

### 1. 액션 정규화
```python
class ActionNormalizer:
    def __init__(self, action_stats):
        self.action_stats = action_stats  # 1st, 99th quantile
    
    def normalize(self, actions):
        # Quantile 기반 클램핑
        clamped_actions = torch.clamp(
            actions, 
            self.action_stats['1st'], 
            self.action_stats['99th']
        )
        
        # 정규화 [-1, 1]
        normalized_actions = 2 * (clamped_actions - self.action_stats['1st']) / \
                           (self.action_stats['99th'] - self.action_stats['1st']) - 1
        
        return normalized_actions
    
    def denormalize(self, normalized_actions):
        # 역정규화
        actions = (normalized_actions + 1) / 2 * \
                 (self.action_stats['99th'] - self.action_stats['1st']) + \
                 self.action_stats['1st']
        return actions
```

### 2. 액션 이산화
```python
class ActionDiscretizer:
    def __init__(self, action_stats, num_bins=256, offset=10):
        self.action_stats = action_stats
        self.num_bins = num_bins
        self.offset = offset
    
    def discretize(self, normalized_actions):
        # 각 차원을 256개 빈으로 이산화
        discrete_actions = []
        for i in range(normalized_actions.shape[-1]):
            bin_width = (self.action_stats['99th'][i] - self.action_stats['1st'][i]) / self.num_bins
            bin_indices = ((normalized_actions[:, i] - self.action_stats['1st'][i]) / bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            discrete_actions.append(bin_indices + self.offset)
        
        return torch.stack(discrete_actions, dim=-1)
    
    def detokenize(self, discrete_tokens):
        # 이산 토큰을 연속 액션으로 변환
        continuous_actions = []
        for i in range(discrete_tokens.shape[-1]):
            bin_indices = discrete_tokens[:, i] - self.offset
            bin_width = (self.action_stats['99th'][i] - self.action_stats['1st'][i]) / self.num_bins
            continuous_action = self.action_stats['1st'][i] + bin_indices.float() * bin_width
            continuous_actions.append(continuous_action)
        
        return torch.stack(continuous_actions, dim=-1)
```

## 손실 함수

### 1. 연속 액션 손실
```python
class ContinuousActionLoss:
    def __init__(self, pose_weight=1.0, gripper_weight=1.0):
        self.pose_weight = pose_weight
        self.gripper_weight = gripper_weight
    
    def compute_loss(self, pred_actions, target_actions):
        # 포즈 (첫 6차원): MSE 손실
        pose_loss = F.mse_loss(
            pred_actions[..., :6], 
            target_actions[..., :6]
        )
        
        # 그리퍼 (마지막 차원): BCE 손실
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred_actions[..., -1:], 
            target_actions[..., -1:]
        )
        
        total_loss = self.pose_weight * pose_loss + self.gripper_weight * gripper_loss
        return total_loss
```

### 2. 이산 액션 손실
```python
class DiscreteActionLoss:
    def compute_loss(self, pred_tokens, target_tokens):
        # 각 차원별 Cross-Entropy 손실
        losses = []
        for i in range(pred_tokens.shape[-1]):
            loss = F.cross_entropy(
                pred_tokens[..., i], 
                target_tokens[..., i]
            )
            losses.append(loss)
        
        return sum(losses)
```

## 훈련 파이프라인

### 1. 데이터 로더
```python
class VLADataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.data = self.load_data()
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 이미지 전처리
        images = self.process_images(sample['images'])
        
        # 언어 지시
        language_instruction = sample['language_instruction']
        
        # 액션 처리
        actions = self.process_actions(sample['actions'])
        
        return {
            'images': images,
            'language': language_instruction,
            'actions': actions
        }
```

### 2. 훈련 루프
```python
class VLATrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Forward pass
            outputs = self.model(
                vision_x=batch['images'],
                lang_x=batch['language'],
                attention_mask=batch.get('attention_mask')
            )
            
            # Loss computation
            loss = self.compute_loss(outputs, batch['actions'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

## 평가 파이프라인

### 1. 시뮬레이션 평가
```python
class SimulationEvaluator:
    def __init__(self, model, env_config):
        self.model = model
        self.env_config = env_config
    
    def evaluate_episode(self, task_instruction):
        """단일 에피소드 평가"""
        obs = self.env.reset()
        total_reward = 0
        
        for step in range(self.max_steps):
            # 액션 예측
            with torch.no_grad():
                action = self.model.predict_action(obs, task_instruction)
            
            # 환경에서 액션 실행
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward, info
```

### 2. 실제 로봇 평가
```python
class RealRobotEvaluator:
    def __init__(self, model, robot_config):
        self.model = model
        self.robot_config = robot_config
        self.robot = self.initialize_robot()
    
    def evaluate_task(self, task_instruction, num_rollouts=5):
        """작업 평가"""
        success_count = 0
        
        for rollout in range(num_rollouts):
            success = self.run_single_rollout(task_instruction)
            if success:
                success_count += 1
        
        return success_count / num_rollouts
```

## 핵심 학습 아이디어

### 1. VLM → VLA 변환
- **최소한의 수정**: 기존 VLM 구조 최대한 보존
- **액션 컴포넌트 주입**: VLM에 액션 예측 능력 추가
- **멀티모달 융합**: 시각, 언어, 액션 정보 통합

### 2. 히스토리 정보 활용
- **Interleaved**: 관찰과 액션을 교차 형식으로 처리
- **Policy Head**: 별도 정책 헤드에서 히스토리 융합
- **시퀀스 모델링**: RNN, Transformer 등 활용

### 3. 액션 공간 처리
- **연속 액션**: MSE + BCE 손실
- **이산 액션**: Cross-Entropy 손실
- **정규화**: Quantile 기반 정규화

### 4. 일반화 전략
- **Vision-Language 사전 훈련**: 필수 요소
- **Cross-embodiment 데이터**: Post-training 활용
- **다양한 설정**: Unseen 환경에서 평가
