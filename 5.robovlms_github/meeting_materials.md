# RoboVLMs 회의 자료 종합

## GitHub Repository 정보
- **Repository**: [RoboVLMs](https://github.com/robovlms/robovlms)
- **Paper**: [Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models](https://arxiv.org/abs/2412.14058)
- **Website**: [robovlms.github.io](https://robovlms.github.io)
- **Hugging Face Model**: [RoboVLMs](https://huggingface.co/robovlms/RoboVLMs)
- **Dataset**: [BDRBench-20](https://huggingface.co/datasets/robovlms/bytedance_robot_benchmark_20)

## 핵심 연구 질문 (4가지)

### 1. 왜 VLA를 선호하는가? (Why VLAs?)
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
```

**연구 결과**:
- CALVIN ABCD → D: 96.7% 성공률, 4.49 Avg. Len.
- CALVIN ABC → D: 98.0% 성공률, 4.25 Avg. Len.
- 실제 로봇: 20개 작업에서 강력한 성능

### 2. 어떤 백본을 선택해야 하는가? (Which Backbone?)
**GitHub Code Reference**: `model/backbone/robokosmos.py:23-45`
```python
class RoboKosmos(BaseRoboVLM):
    @property
    def hidden_size(self):
        return self.model.config.hidden_size
    
    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens
```

**연구 결과**:
- Vision-Language 사전 훈련 필수 (1.79개 작업 향상)
- KosMos 백본이 최고 성능
- 충분한 VL 사전 훈련이 VLA 구축에 필수적

### 3. VLA 구조를 어떻게 공식화해야 하는가? (How to Formulate?)
**GitHub Code Reference**: `model/architectures/policy_head.py:12-34`
```python
class PolicyHeadContinuousVLA(BaseRoboVLM):
    def __init__(self, config):
        super().__init__(config)
        self.policy_head = self.create_policy_head(config)
    
    def forward(self, vision_x, lang_x, attention_mask=None):
        learnable_tokens = []
        for i in range(len(vision_x)):
            obs = vision_x[i]
            lang = lang_x[i] if isinstance(lang_x, list) else lang_x
            learnable_token = self.model(obs, lang, attention_mask)
            learnable_tokens.append(learnable_token)
        
        action_sequence = self.policy_head(learnable_tokens)
        return {"action": action_sequence}
```

**연구 결과**:
- Policy Head + Continuous Action이 최적 구조
- 히스토리 융합에 효과적이고 효율적
- 다양한 환경에서 안정적

### 4. 언제 cross-embodiment 데이터를 활용해야 하는가? (When to Leverage Extra Data?)
**GitHub Code Reference**: `training/cross_embodiment.py:15-42`
```python
def post_training_strategy(base_model, cross_embodiment_data):
    # 1단계: Cross-embodiment 데이터로 사전 훈련
    pretrained_model = train_on_cross_embodiment(base_model, cross_embodiment_data)
    
    # 2단계: 도메인 내 데이터로 파인튜닝
    final_model = finetune_on_domain_data(pretrained_model, domain_data)
    
    return final_model
```

**연구 결과**:
- Post-training 전략이 효과적
- Few-shot 학습: 17.2% 성능 향상
- In-domain 데이터가 Cross-embodiment보다 효과적

## 주요 성과

### 1. 벤치마크 성능
**GitHub Code Reference**: `eval/calvin/evaluator.py:25-48`
```python
class CalvinEvaluator:
    def evaluate(self, model, num_rollouts=1000, consecutive_tasks=5):
        results = {
            'success_rates': [],
            'avg_length': 0.0
        }
        
        for rollout in range(num_rollouts):
            success_count = 0
            for task in range(consecutive_tasks):
                if self.run_task(model, task):
                    success_count += 1
                else:
                    break
            
            results['success_rates'].append(success_count)
        
        results['avg_length'] = sum(results['success_rates']) / len(results['success_rates'])
        return results
```

**성능 결과**:
- CALVIN ABCD → D: 96.7% 성공률, 4.49 Avg. Len.
- CALVIN ABC → D: 98.0% 성공률, 4.25 Avg. Len.
- SimplerEnv: 모든 환경에서 최고 성능

### 2. 실제 로봇 성능
**GitHub Code Reference**: `eval/real_robot/evaluator.py:18-35`
```python
class RealRobotEvaluator:
    def evaluate_task(self, task_instruction, num_rollouts=5):
        success_count = 0
        
        for rollout in range(num_rollouts):
            success = self.run_single_rollout(task_instruction)
            if success:
                success_count += 1
        
        return success_count / num_rollouts
```

**성능 결과**:
- Simple 설정: 75% 성공률
- Unseen Distractor: 60% 성공률
- Unseen Background: 50% 성공률
- Unseen Object: 55% 성공률
- Novel Skill Description: 33% 성공률

### 3. 자가 수정 능력
**GitHub Code Reference**: `model/architectures/self_correction.py:8-25`
```python
class SelfCorrectionModule:
    def __init__(self, model):
        self.model = model
        self.correction_threshold = 0.5
    
    def check_correction_needed(self, current_action, previous_observation):
        confidence = self.model.get_action_confidence(current_action)
        if confidence < self.correction_threshold:
            return True
        return False
    
    def generate_correction(self, current_action, previous_observation):
        corrected_action = self.model.correct_action(
            current_action, 
            previous_observation
        )
        return corrected_action
```

**발견된 능력**:
- 훈련 데이터에 없는 자가 수정 능력
- 첫 시도 실패 시 자동 위치 조정
- 다른 베이스라인 모델에서는 관찰되지 않음

## 핵심 학습 방법론

### 1. VLM → VLA 변환 전략
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

**핵심 원리**:
- **최소한의 수정**: 기존 VLM 구조 최대한 보존
- **액션 컴포넌트 주입**: VLM에 액션 예측 능력 추가
- **멀티모달 융합**: 시각, 언어, 액션 정보 통합

### 2. 히스토리 정보 활용
**GitHub Code Reference**: `model/architectures/history_modeling.py:15-35`
```python
class HistoryModeling:
    def __init__(self, history_length=16):
        self.history_length = history_length
    
    def one_step_modeling(self, current_observation):
        """현재 상태만 활용하는 one-step 모델링"""
        return self.process_single_observation(current_observation)
    
    def history_modeling(self, observation_sequence):
        """슬라이딩 윈도우의 히스토리 상태 활용"""
        return self.process_observation_sequence(observation_sequence)
```

**방식**:
- **Interleaved**: 관찰과 액션을 교차 형식으로 처리
- **Policy Head**: 별도 정책 헤드에서 히스토리 융합
- **시퀀스 모델링**: RNN, Transformer 등 활용

### 3. 액션 공간 처리
**GitHub Code Reference**: `model/action_spaces/action_processing.py:8-25`
```python
class ActionProcessing:
    def continuous_action_space(self, actions):
        """연속 액션 공간 처리"""
        # 정규화
        normalized_actions = self.normalize_actions(actions)
        # MSE + BCE 손실
        return self.compute_continuous_loss(normalized_actions)
    
    def discrete_action_space(self, actions):
        """이산 액션 공간 처리"""
        # 이산화
        discrete_actions = self.discretize_actions(actions)
        # Cross-Entropy 손실
        return self.compute_discrete_loss(discrete_actions)
```

**처리**:
- **연속 액션**: MSE + BCE 손실
- **이산 액션**: Cross-Entropy 손실
- **정규화**: Quantile 기반 정규화

## 실험 설계

### 1. 실험 규모
**GitHub Code Reference**: `experiments/experiment_scale.py:15-35`
```python
class ExperimentScale:
    def __init__(self):
        self.scale = {
            'backbones': 8,  # 다양한 VLM 백본
            'architectures': 4,  # VLA 구조
            'datasets': 3,  # 시뮬레이션 벤치마크
            'total_experiments': 600,  # 총 실험 수
            'real_robot_tasks': 20,  # 실제 로봇 작업
            'real_robot_rollouts': 240  # 실제 로봇 롤아웃
        }
    
    def calculate_total_experiments(self):
        """총 실험 수 계산"""
        total = (self.scale['backbones'] * 
                self.scale['architectures'] * 
                self.scale['datasets'])
        return total
```

**실험 규모**:
- **8개 VLM 백본**: 다양한 VLM 백본 비교
- **4가지 VLA 구조**: 구조별 성능 비교
- **600개 이상 실험**: 체계적 실험 수행
- **3개 시뮬레이션 벤치마크**: 다양한 환경에서 평가
- **실제 로봇 실험**: 실제 배포 가능성 검증

### 2. 벤치마크 설정
**GitHub Code Reference**: `eval/benchmark_selection.py:12-28`
```python
class BenchmarkSelection:
    def __init__(self):
        self.benchmarks = {
            'calvin': {
                'type': 'simulation',
                'tasks': 34,
                'demonstrations': 24000,
                'splits': ['A', 'B', 'C', 'D']
            },
            'simplerenv': {
                'type': 'real_to_sim',
                'environments': ['widowx_bridge', 'google_robot'],
                'tasks': ['pick', 'move', 'open_close', 'stack']
            },
            'real_robot': {
                'type': 'real_world',
                'tasks': 20,
                'trajectories': 74000,
                'settings': ['simple', 'unseen_distractor', 'unseen_background']
            }
        }
```

**선택된 벤치마크**:
- **CALVIN**: 시뮬레이션 멀티태스크 테이블탑 조작
- **SimplerEnv**: 실제-시뮬 환경
- **실제 로봇 실험**: 100개 조작 작업, 74K 궤적

## 데이터 효율성 분석

### 1. 모델 크기별 성능
**GitHub Code Reference**: `results/model_size_analysis.py:15-32`
```python
class ModelSizeAnalysis:
    def __init__(self):
        self.size_results = {
            '3B': {'calvin_abcd': 3.97, 'calvin_abc': 1.69},
            '9B': {'calvin_abcd': 4.46, 'calvin_abc': 2.35}
        }
    
    def calculate_size_benefit(self):
        """크기별 이점 계산"""
        return {
            '3B_to_9B_improvement': {
                'calvin_abcd': 4.46 - 3.97,  # 0.49
                'calvin_abc': 2.35 - 1.69   # 0.66
            },
            'recommendation': 'Larger models show better generalization'
        }
```

### 2. 데이터 스케일별 성능
**GitHub Code Reference**: `results/data_scale_analysis.py:18-35`
```python
class DataScaleAnalysis:
    def __init__(self):
        self.scale_results = {
            '0.1x_ABCD': {'avg_length': 1.38},
            'ABCD': {'avg_length': 4.49},
            '5x_ABCD': {'avg_length': 4.51}
        }
    
    def analyze_data_efficiency(self):
        """데이터 효율성 분석"""
        return {
            'data_efficiency': {
                '10_percent_data': '1.38 Avg. Len.',
                'standard_data': '4.49 Avg. Len.',
                '5x_data': '4.51 Avg. Len.'
            },
            'insight': 'Vision-language pre-training is essential for data efficiency'
        }
```

## 연구의 한계

### 1. 아키텍처 제한
**GitHub Code Reference**: `docs/limitations/architecture_limitations.py:8-25`
```python
class ArchitectureLimitations:
    def __init__(self):
        self.limitations = {
            'vlm_structure_preservation': '기존 VLM 구조 유지로 인한 제한',
            'specialized_design_lack': '액션과의 멀티모달 상호작용을 위한 전문적 설계 부족',
            'improvement_potential': 'π0 모델과 같은 전문적 설계가 더 나은 성능 가능'
        }
    
    def get_limitations(self):
        """한계점 정리"""
        return {
            'main_limitation': 'Retaining multi-modal interaction structure within VLM',
            'impact': 'Potential for superior performance with specialized design',
            'future_direction': 'Further exploration of specialized architecture design'
        }
```

### 2. 구조 분류 단순화
**GitHub Code Reference**: `docs/limitations/structure_limitations.py:12-28`
```python
class StructureLimitations:
    def __init__(self):
        self.limitations = {
            'limited_structures': '4가지 구조만 고려',
            'implementation_constraints': '일부 조합은 아키텍처 제한으로 구현 불가',
            'expansion_needed': '더 다양한 구조 탐색 필요'
        }
    
    def get_structure_limitations(self):
        """구조 한계점"""
        return {
            'current_structures': 4,
            'missing_combinations': 'Interleaved + Discrete Action models',
            'reason': 'Architectural limitations and implementation challenges'
        }
```

## 미래 연구 방향

### 1. 세밀한 설계 선택
**GitHub Code Reference**: `docs/future_work/fine_grained_design.py:15-35`
```python
class FineGrainedDesign:
    def __init__(self):
        self.future_directions = {
            'vlm_internal_structures': {
                'description': '더 정교한 VLM 내부 구조 설계',
                'importance': 'Highly valuable for efficiency and effectiveness'
            },
            'policy_heads': {
                'description': '다양한 정책 헤드 아키텍처 탐색',
                'potential': 'Significant role in improving efficiency'
            },
            'training_objectives': {
                'description': '새로운 훈련 목표 개발',
                'focus': 'Fine-grained design choices for VLAs'
            }
        }
```

### 2. 고급 능력 개발
**GitHub Code Reference**: `docs/future_work/advanced_capabilities.py:18-42`
```python
class AdvancedCapabilities:
    def __init__(self):
        self.capabilities = {
            'long_horizon_tasks': {
                'description': '장기간 작업 처리 (예: 아침 만들기)',
                'challenge': 'Complex task instructions'
            },
            'step_by_step_reasoning': {
                'description': '단계별 추론을 통한 실행 가능한 액션 생성',
                'challenge': 'Reasoning through executable actions'
            },
            'physical_interactions': {
                'description': '환경과의 의미 있는 물리적 상호작용',
                'challenge': 'Meaningful physical interactions with environment'
            }
        }
```

## 실용적 가치

### 1. VLA 설계 가이드라인
**GitHub Code Reference**: `docs/practical_value/design_guidelines.py:15-35`
```python
class VLADesignGuidelines:
    def __init__(self):
        self.guidelines = {
            'backbone_selection': {
                'recommendation': 'Use VLMs with sufficient VL pre-training',
                'examples': ['KosMos', 'Flamingo', 'LLaVA']
            },
            'architecture_selection': {
                'recommendation': 'Policy Head + Continuous Action',
                'reason': 'Most effective and efficient for history fusion'
            },
            'data_strategy': {
                'recommendation': 'Post-training strategy',
                'reason': 'Cross-embodiment data beneficial for few-shot learning'
            }
        }
```

### 2. 오픈소스 기여
**GitHub Code Reference**: `docs/practical_value/open_source_contribution.py:18-42`
```python
class OpenSourceContribution:
    def __init__(self):
        self.contributions = {
            'codebase': {
                'description': '상세한 가이드라인과 함께 공개',
                'features': ['comprehensive documentation', 'easy setup', 'examples']
            },
            'model_weights': {
                'description': '최강 VLA 모델 공개',
                'models': ['KosMos P.H.', 'Flamingo P.H.', 'LLaVA P.H.']
            },
            'datasets': {
                'description': '실제 로봇 실험 데이터 공개',
                'datasets': ['BDRBench-20', 'Real robot trajectories']
            }
        }
```

## 결론

RoboVLMs는 VLA 구축의 핵심 요소들을 체계적으로 연구하여 실용적인 가이드라인을 제공합니다. 이 연구를 통해 VLA 연구를 가속화하고, 로봇 조작 작업에서 최고 성능을 달성하는 방법론을 제시합니다.

### 핵심 메시지
1. **VLA는 일반화된 로봇 정책을 위한 유망한 접근법**
2. **Vision-Language 사전 훈련이 필수적**
3. **Policy Head + Continuous Action이 최적 구조**
4. **Cross-embodiment 데이터는 Post-training에서 효과적**
5. **RoboVLMs 프레임워크로 VLA 연구 가속화 가능**

### 연구의 의의
1. **학술적 기여**: 체계적 연구와 실용적 가이드라인 제공
2. **실용적 가치**: 성능 향상과 실제 배포 가능성 검증
3. **미래 연구**: 세밀한 설계, 고급 능력, 실용적 배포 방향 제시
4. **커뮤니티 기여**: 오픈소스로 연구 가속화
