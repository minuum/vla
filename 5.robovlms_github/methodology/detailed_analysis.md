# RoboVLMs Methodology 상세 분석

## GitHub Repository 정보
- **Repository**: [RoboVLMs](https://github.com/robovlms/robovlms)
- **Paper**: [Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models](https://arxiv.org/abs/2412.14058)
- **Website**: [robovlms.github.io](https://robovlms.github.io)

## Vision-Language Models (VLMs)

### 1. 기본 구조
**GitHub Code Reference**: `model/backbone/base_backbone.py:45-67`
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
```

**수식 표현**:
```
l̂ = VLM(I, l_prompt)
```
- I: 입력 이미지
- l_prompt: 텍스트 프롬프트
- l̂: VLM이 생성한 텍스트 출력

**훈련 손실**:
```
L_VLM = CrossEntropy(l̂, l_target)
```

### 2. 아키텍처 분류

#### Encoder-Decoder 구조
**GitHub Code Reference**: `model/backbone/encoder_decoder.py:12-28`
```python
class EncoderDecoderVLM(BaseRoboVLM):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
    
    def forward(self, images, text):
        # 인코더로 특징 추출
        encoded_features = self.encoder(images, text)
        
        # 디코더로 출력 생성
        output = self.decoder(encoded_features)
        
        return output
```

**특징**:
- 인코더(특징 추출) + 디코더(자동회귀 생성)
- 인코더와 디코더 간 cross-attention을 통한 특징 융합
- 입력 모달리티에 대한 상세한 이해
- 대표 모델: Flamingo, OFA

#### Decoder-Only 구조
**GitHub Code Reference**: `model/backbone/decoder_only.py:15-32`
```python
class DecoderOnlyVLM(BaseRoboVLM):
    def __init__(self, config):
        super().__init__(config)
        self.unified_transformer = self.create_unified_transformer()
    
    def forward(self, images, text):
        # 시각과 텍스트 토큰 연결
        multimodal_tokens = self.concatenate_tokens(images, text)
        
        # 통합 트랜스포머로 처리
        output = self.unified_transformer(multimodal_tokens)
        
        return output
```

**특징**:
- 통합된 트랜스포머 프레임워크
- 시각과 텍스트 토큰을 연결하여 self-attention으로 융합
- 유연성과 확장성
- 대표 모델: GPT-4V, LLaVA

## Vision-Language-Action Models (VLAs)

### 1. 기본 정의
**GitHub Code Reference**: `model/vla/base_vla.py:18-35`
```python
class BaseVLA:
    def __init__(self, config):
        self.config = config
        self.vlm_backbone = None
        self.action_head = None
        self.history_modeling = None
    
    def forward(self, observations, language_instruction):
        """VLA의 기본 forward pass"""
        # 멀티모달 표현 생성
        multimodal_repr = self.vlm_backbone(observations, language_instruction)
        
        # 액션 예측
        actions = self.action_head(multimodal_repr)
        
        return actions
```

**수식 표현**:
```
a_{t:t+L-1} = VLA(o_{t-H+1:t}, l_prompt)
```
- a_{t:t+L-1}: 예측된 7차원 액션 시퀀스
- L: 액션 시퀀스 길이
- H: 히스토리 관찰 길이

### 2. 액션 전처리

#### 액션 정규화
**GitHub Code Reference**: `model/action_spaces/action_normalization.py:12-28`
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
```

**수식**:
```python
# Quantile 기반 클램핑
a_i' = min(a_i^{99th}, max(a_i^{1st}, a_i))

# 정규화
ã_i = 2 × (a_i' - a_i^{1st}) / (a_i^{99th} - a_i^{1st}) - 1
```

#### 액션 이산화
**GitHub Code Reference**: `model/action_spaces/action_discretization.py:15-35`
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
```

### 3. 액션 예측

#### 연속 액션 예측
**GitHub Code Reference**: `model/action_spaces/continuous_action.py:18-32`
```python
class ContinuousActionPrediction:
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

**수식**:
```python
L_VLA = Σ_{i=t}^{t+L-1} [MSE(â_i,pose, ã_i,pose) + λ × BCE(a_i,gripper, ã_i,gripper)]
```

#### 이산 액션 예측
**GitHub Code Reference**: `model/action_spaces/discrete_action.py:12-25`
```python
class DiscreteActionPrediction:
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

**수식**:
```python
L_VLA = Σ_{i=t}^{t+L-1} Σ_{j=1}^{7} CE([ACT]_i^j, ã_i^j)
```

## VLA 구조 분류

### 1. One-Step Models
**GitHub Code Reference**: `model/architectures/one_step.py:15-35`
```python
class OneStepVLA(BaseVLA):
    def __init__(self, config):
        super().__init__(config)
        self.action_head = self.create_action_head()
    
    def forward(self, current_observation, language_instruction):
        """현재 시간 단계의 관찰만 사용"""
        # VLM으로 멀티모달 표현 생성
        multimodal_repr = self.vlm_backbone(current_observation, language_instruction)
        
        # 액션 예측
        action_sequence = self.action_head(multimodal_repr)
        
        return {"action": action_sequence}
```

**수식**:
```
â_{t:t+L-1} = VLA(o_t, l_prompt)
```

#### 연속 액션 모델
**GitHub Code Reference**: `model/architectures/one_step_continuous.py:18-32`
```python
class OneStepContinuousVLA(OneStepVLA):
    def forward(self, vision_x, lang_x, attention_mask=None):
        # VLM 백본으로 learnable 토큰 예측
        learnable_token = self.model(vision_x, lang_x, attention_mask)
        
        # MLP로 액션 벡터 예측
        action_sequence = self.action_head(learnable_token)
        
        return {"action": action_sequence}
```

**수식**:
```python
[LRN] = VLM(o_t, l_prompt)
â_{t:t+L-1} = MLP([LRN])
```

#### 이산 액션 모델
**GitHub Code Reference**: `model/architectures/one_step_discrete.py:12-25`
```python
class OneStepDiscreteVLA(OneStepVLA):
    def forward(self, vision_x, lang_x, attention_mask=None):
        # VLM 백본으로 액션 토큰 예측
        action_tokens = self.model(vision_x, lang_x, attention_mask)
        
        return {"action": action_tokens}
```

**수식**:
```python
[ACT]_{t:t+L-1}^{1:7} = VLM(o_t, l_prompt)
```

### 2. Interleaved-Continuous-Action Models
**GitHub Code Reference**: `model/architectures/interleaved.py:18-42`
```python
class InterleavedContinuousVLA(BaseVLA):
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

**수식**:
```python
O_t = ([OBS]_{t-H+1}, [LRN]), ..., ([OBS]_t, [LRN])
[LRN]_{t-H+1:t} = VLM(O_t)
â_{t:t+L-1} = MLP([LRN]_t)
```

### 3. Policy-Head-Continuous-Action Models
**GitHub Code Reference**: `model/architectures/policy_head.py:15-42`
```python
class PolicyHeadContinuousVLA(BaseVLA):
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

**수식**:
```python
o_t = ([OBS]_t, [LRN])
[LRN]_t = VLM(o_t, l_prompt)
a_{t:t+L-1} = h([LRN]_{t-H+1}, ..., [LRN]_t)
```

## RoboVLMs 프레임워크

### 1. 핵심 특징
**GitHub Code Reference**: `framework/robovlms.py:12-28`
```python
class RoboVLMs:
    def __init__(self, config):
        self.config = config
        self.supported_backbones = [
            'KosMos', 'Flamingo', 'LLaVA', 'Qwen-VL',
            'PaliGemma', 'InstructBLIP', 'BLIP-2', 'Otter'
        ]
        self.supported_architectures = [
            'one_step_continuous', 'one_step_discrete',
            'interleaved_continuous', 'policy_head_continuous'
        ]
    
    def create_vla(self, backbone_name, architecture_name):
        """VLM을 VLA로 변환"""
        backbone = self.get_backbone(backbone_name)
        architecture = self.get_architecture(architecture_name)
        return self.combine_components(backbone, architecture)
```

**특징**:
- **30줄 이내 코드**로 VLM을 VLA로 변환
- **8개 VLM 백본** 지원
- **4가지 VLA 구조** 지원
- **유연한 통합**: 새로운 VLM 쉽게 추가

### 2. VLM 통합 예시
**GitHub Code Reference**: `model/backbone/robopaligemma.py:18-42`
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

### 3. 구현 원리
**GitHub Code Reference**: `framework/implementation_principle.py:15-35`
```python
class ImplementationPrinciple:
    def __init__(self):
        self.principle = {
            'minimal_modification': '기존 VLM 구조 최대한 보존',
            'action_component_injection': 'VLM에 액션 예측 능력 추가',
            'multimodal_fusion': '시각, 언어, 액션 정보 통합'
        }
    
    def transform_vlm_to_vla(self, vlm_model):
        """VLM을 VLA로 변환하는 핵심 원리"""
        # 1. VLM의 핵심 속성 설정
        vlm_attributes = self.set_vlm_attributes(vlm_model)
        
        # 2. 멀티모달 특징 융합 메커니즘 정의
        fusion_mechanism = self.define_fusion_mechanism()
        
        # 3. 액션 예측을 위한 추가 컴포넌트 통합
        action_components = self.add_action_components()
        
        return self.combine_all_components(vlm_attributes, fusion_mechanism, action_components)
```

## 실험 설정

### 1. 하이퍼파라미터
**GitHub Code Reference**: `config/hyperparameters.py:12-28`
```python
class HyperparameterConfig:
    def __init__(self):
        self.configs = {
            'calvin_performance': {
                'batch_size': 128,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'warmup_ratio': 0.25,
                'total_epochs': 5
            },
            'simplerenv_performance': {
                'batch_size': 128,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'warmup_ratio': 0.25,
                'total_iterations': 50000
            },
            'real_robot': {
                'batch_size': 128,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'warmup_ratio': 0.25,
                'total_epochs': 5
            }
        }
```

### 2. 훈련 환경
**GitHub Code Reference**: `training/training_environment.py:18-35`
```python
class TrainingEnvironment:
    def __init__(self):
        self.hardware = {
            'gpus': '4 x 8 A100 GPU cluster',
            'memory': '80GB per GPU',
            'storage': 'NVMe SSD'
        }
        
        self.software = {
            'python': '3.8.10',
            'pytorch': '>=2.0',
            'cuda': '11.8'
        }
    
    def setup_training_environment(self):
        """훈련 환경 설정"""
        # CUDA 설정
        torch.cuda.set_device(0)
        
        # 분산 훈련 설정
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        return model
```

### 3. 체크포인트 선택 전략
**GitHub Code Reference**: `training/checkpoint_selection.py:12-28`
```python
class CheckpointSelection:
    def __init__(self):
        self.strategy = {
            'calvin': '5 epochs, final model',
            'simplerenv': '100K iterations, best model in 10K intervals',
            'real_robot': '5 epochs, final model'
        }
    
    def select_checkpoint(self, model, validation_metrics):
        """체크포인트 선택 전략"""
        # 로봇 정책의 성능이 오프라인 평가 지표에 완전히 의존하지 않음
        # 장기간 롤아웃에서의 복합 오류로 인한 어려움
        
        # 고정 에포크/반복 수 사용
        if self.config.experiment_type == 'calvin':
            return self.get_final_model(model, epochs=5)
        elif self.config.experiment_type == 'simplerenv':
            return self.get_best_model(model, iterations=100000)
        else:
            return self.get_final_model(model, epochs=5)
```

## 결론

RoboVLMs의 방법론은 VLM을 VLA로 변환하는 체계적이고 효율적인 접근법을 제시합니다. 핵심은 기존 VLM의 강력한 멀티모달 이해 능력을 보존하면서 액션 예측 능력을 추가하는 것입니다.

### 핵심 원리
1. **최소한의 수정**: 기존 VLM 구조 최대한 보존
2. **액션 컴포넌트 주입**: VLM에 액션 예측 능력 추가
3. **멀티모달 융합**: 시각, 언어, 액션 정보 통합
4. **체계적 실험**: 공정한 비교를 위한 통합 환경
