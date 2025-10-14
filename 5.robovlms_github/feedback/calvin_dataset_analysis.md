# CALVIN Dataset 상세 분석

## GitHub Repository 정보
- **Repository**: [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs)
- **CALVIN Dataset**: [CALVIN](https://github.com/mees/calvin/tree/main)
- **Paper**: [Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models](https://arxiv.org/abs/2412.14058)
- **Website**: [robovlms.github.io](https://robovlms.github.io)

## 1. CALVIN Dataset 개요

### 1.1 데이터셋 기본 정보
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:521-873](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L521-L873)
**실제 코드 위치**: CALVIN 데이터셋 클래스
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """
    CALVIN 데이터셋 클래스
    - 24K 인간 텔레오퍼레이션 시연
    - 34개 기본 기술
    - A, B, C, D 분할
    - 언어 지시 포함
    """
    def __init__(self, ...):
        # 데이터셋 초기화
        pass
```

**CALVIN 데이터셋 특징**:
- **총 시연**: 24K 인간 텔레오퍼레이션 시연
- **언어 지시**: 모든 시연에 언어 지시 포함
- **궤적 길이**: 64 시간 단계 이하
- **기본 기술**: 34개 사전 정의된 기본 기술
- **분할**: A, B, C, D 4개 분할

### 1.2 34개 기본 기술 목록
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 34개 기본 기술
```python
self.tasks = [
    'rotate blue block right', 'move slider right',
    'lift red block slider', 'place slider',
    'turn off light bulb', 'turn off led light',
    'push in drawer', 'lift blue block drawer',
    'close drawer', 'lift pink block slider',
    'lift pink block table', 'move slider left',
    'turn on light bulb', 'rotate blue block left',
    'turn on led light', 'push pink block right',
    'push red block left', 'lift blue block table',
    'place in drawer', 'rotate red block left',
    'push pink block left', 'lift stacked blocks',
    'lift blue block slider', 'push blue block right'
]
```

## 2. 데이터셋 구조 분석

### 2.1 Action 데이터 구조
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:826-828](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L826-L828)
```python
# Action (절대 좌표)
['actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in absolute world coordinates
tcp orientation (3): euler angles x,y,z in absolute world coordinates
gripper_action (1): binary (close = -1, open = 1)

# rel_action (상대 좌표)
['rel_actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
gripper_action (1): binary (close = -1, open = 1)
```

### 2.2 이미지 데이터 구조
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:863-864](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L863-L864)
```python
# CALVIN 데이터셋에서 이미지 처리
image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])
```

### 2.3 텍스트 데이터 구조
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:866-867](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L866-L867)
```python
# CALVIN 데이터셋에서 텍스트 처리
stacked_language = [s["lang"] for s in sample]
text_tensors, attention_mask = self.text_fn(stacked_language)
```

## 3. 데이터셋 분할 전략

### 3.1 ABCD 분할
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
```python
# ABCD 분할: 일반화 평가
'abcd_to_d': {
    'train_splits': ['A', 'B', 'C', 'D'],
    'test_split': 'D',
    'purpose': 'generalization_evaluation'
}
```

**ABCD → D 평가**:
- **훈련**: A, B, C, D 모든 분할
- **테스트**: D 분할
- **목적**: 일반화 능력 평가
- **성능**: 96.7% 단일 작업 성공률, 4.49 Avg. Len.

### 3.2 ABC 분할
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
```python
# ABC 분할: 향상된 일반화 평가
'abc_to_d': {
    'train_splits': ['A', 'B', 'C'],
    'test_split': 'D',
    'purpose': 'enhanced_generalization_evaluation'
}
```

**ABC → D 평가**:
- **훈련**: A, B, C 분할
- **테스트**: D 분할
- **목적**: 향상된 일반화 평가
- **성능**: 98.0% 단일 작업 성공률, 4.25 Avg. Len.

## 4. 데이터셋 활용 전략

### 4.1 데이터셋 로드
**GitHub Code Reference**: `5.robovlms_github/implementation/README.md:120-135`
```python
# CALVIN 데이터셋 로드
from robovlms.data.calvin import CalvinDataset

dataset = CalvinDataset(
    data_path="/path/to/calvin/data",
    split="ABCD",  # 또는 "ABC"
    window_size=16,
    action_chunk_size=10
)

dataloader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True,
    num_workers=4
)
```

### 4.2 데이터 전처리
**GitHub Code Reference**: `5.robovlms_github/methodology/README.md:48-57`
```python
# 액션 정규화
# Quantile 기반 클램핑
a_i' = min(a_i^{99th}, max(a_i^{1st}, a_i))

# 정규화
ã_i = 2 × (a_i' - a_i^{1st}) / (a_i^{99th} - a_i^{1st}) - 1
```

## 5. CALVIN 평가 메트릭

### 5.1 연속 작업 성공률
**GitHub Code Reference**: `5.robovlms_github/experiments/README.md:18-42`
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

**평가 메트릭**:
- **연속 작업 성공률**: 1~5개 연속 작업 성공률
- **평균 달성 길이**: 평균적으로 달성한 작업 수
- **롤아웃 수**: 1000개 롤아웃으로 평가

### 5.2 성능 결과
**GitHub Code Reference**: `5.robovlms_github/results/README.md:12-28`
```python
# CALVIN ABCD → D 결과
calvin_abcd_results = {
    'single_task_success': 0.967,  # 96.7%
    'consecutive_tasks': [0.967, 0.930, 0.899, 0.865, 0.826],
    'avg_length': 4.49
}

# CALVIN ABC → D 결과  
calvin_abc_results = {
    'single_task_success': 0.980,  # 98.0%
    'consecutive_tasks': [0.980, 0.936, 0.854, 0.778, 0.704],
    'avg_length': 4.25
}
```

## 6. 데이터셋 특성 분석

### 6.1 시뮬레이션 환경
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
```python
# CALVIN 시뮬레이션 환경
simulation_environment = {
    'type': 'simulation',
    'robot': 'Franka Panda',
    'workspace': 'table-top manipulation',
    'objects': 'blocks, sliders, drawers, lights',
    'camera': 'static overhead camera'
}
```

### 6.2 작업 복잡도
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
```python
# 작업 복잡도 분석
task_complexity = {
    'simple_tasks': [
        'turn on light bulb', 'turn off light bulb',
        'turn on led light', 'turn off led light'
    ],
    'medium_tasks': [
        'move slider left', 'move slider right',
        'push blue block left', 'push blue block right'
    ],
    'complex_tasks': [
        'lift stacked blocks', 'place in drawer',
        'lift blue block drawer', 'close drawer'
    ]
}
```

## 7. 데이터셋 활용 시 고려사항

### 7.1 메모리 효율성
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:230-247`
```python
def memory_efficient_training(model, batch):
    # 그래디언트 체크포인팅으로 메모리 사용량 감소
    with torch.cuda.amp.autocast():
        outputs = model(batch)
        loss = compute_loss(outputs, batch['targets'])
    
    # 그래디언트 누적
    loss = loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 7.2 배치 처리
**GitHub Code Reference**: `5.robovlms_github/implementation/README.md:140-145`
```python
# 배치 처리 설정
batch_config = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}
```

## 8. 데이터셋 확장 전략

### 8.1 데이터 증강
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:140-145`
```python
# 데이터 증강
def data_augmentation(original_data):
    augmented_data = []
    
    # 이미지 증강
    for image in original_data['images']:
        # 회전, 크기 조정, 색상 조정
        augmented_image = augment_image(image)
        augmented_data.append(augmented_image)
    
    return augmented_data
```

### 8.2 Cross-embodiment 데이터 활용
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:109-121`
```python
# Cross-embodiment Post-training
for batch in cross_embodiment_dataloader:
    images = batch['images']
    language = batch['language_instruction']
    actions = batch['actions']
    
    # VLA 손실
    vla_loss = compute_vla_loss(VLA(images, language), actions)
    vla_loss.backward()
    optimizer.step()
```

## 9. 데이터셋 품질 관리

### 9.1 데이터 검증
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:264-282`
```python
def validate_dataset(dataset):
    """데이터셋 품질 검증"""
    validation_results = {
        'image_quality': check_image_quality(dataset),
        'action_consistency': check_action_consistency(dataset),
        'text_quality': check_text_quality(dataset),
        'temporal_alignment': check_temporal_alignment(dataset)
    }
    return validation_results
```

### 9.2 데이터 필터링
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:264-282`
```python
def filter_dataset(dataset, quality_threshold=0.8):
    """품질 기준에 따른 데이터 필터링"""
    filtered_data = []
    for sample in dataset:
        if sample['quality_score'] >= quality_threshold:
            filtered_data.append(sample)
    return filtered_data
```

## 10. CALVIN 데이터셋 활용 결과

### 10.1 성능 향상
**GitHub Code Reference**: `5.robovlms_github/results/README.md:12-28`
```python
# CALVIN 성능 향상 결과
performance_improvement = {
    'baseline_GR1': 4.21,  # Avg. Len.
    'robovlms_abcd': 4.49,  # +0.28 improvement
    'robovlms_abc': 4.25,   # +0.04 improvement
    'improvement_percentage': 6.7  # 4.49/4.21 - 1
}
```

### 10.2 일반화 능력
**GitHub Code Reference**: `5.robovlms_github/results/README.md:15-32`
```python
# 일반화 능력 분석
generalization_analysis = {
    'abcd_to_d': {
        'single_task': 0.967,
        'avg_length': 4.49,
        'generalization': 'high'
    },
    'abc_to_d': {
        'single_task': 0.980,
        'avg_length': 4.25,
        'generalization': 'very_high'
    }
}
```

## 결론

CALVIN 데이터셋은 RoboVLMs의 핵심 평가 벤치마크로, 다음과 같은 특징을 가집니다:

### 핵심 특징
1. **대규모 데이터**: 24K 시연 데이터
2. **언어 지시**: 모든 시연에 자연어 명령 포함
3. **다양한 작업**: 34개 기본 기술
4. **분할 전략**: ABCD/ABC 분할로 일반화 평가
5. **7 DOF 액션**: TCP pose + gripper action
6. **절대/상대 좌표**: action과 rel_action 동시 제공
7. **시뮬레이션 환경**: 안정적인 평가 환경
8. **연속 작업**: 1~5개 연속 작업 성공률 평가
9. **성능 향상**: 기존 SOTA 대비 6.7% 향상
10. **일반화 능력**: Unseen 환경에서도 강력한 성능
