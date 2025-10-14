# CALVIN Dataset GitHub 분석

## GitHub Repository 정보
- **Repository**: [CALVIN](https://github.com/mees/calvin/tree/main)
- **RoboVLMs Integration**: [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs)
- **Paper**: [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2110.06169)
- **Website**: [calvin-benchmark.org](https://calvin-benchmark.org)

## 1. CALVIN Dataset 개요

### 1.1 데이터셋 기본 정보
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN GitHub 저장소 메인 디렉토리

```python
# CALVIN 데이터셋 기본 구조
calvin_dataset = {
    'total_demonstrations': 24000,
    'language_instructions': True,
    'trajectory_length': 64,  # time steps
    'basic_skills': 34,
    'splits': ['A', 'B', 'C', 'D'],
    'robot': 'Franka Panda',
    'workspace': 'table-top manipulation',
    'camera': 'static overhead camera'
}
```

**CALVIN 데이터셋 특징**:
- **총 시연**: 24K 인간 텔레오퍼레이션 시연
- **언어 지시**: 모든 시연에 언어 지시 포함
- **궤적 길이**: 64 시간 단계 이하
- **기본 기술**: 34개 사전 정의된 기본 기술
- **분할**: A, B, C, D 4개 분할
- **로봇**: Franka Panda 7-DOF 로봇팔
- **환경**: 테이블탑 조작 시뮬레이션

### 1.2 34개 기본 기술 목록
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 34개 기본 기술

```python
# CALVIN 34개 기본 기술
calvin_tasks = [
    'rotate blue block right', 'move slider right',
    'lift red block slider', 'place slider',
    'turn off light bulb', 'turn off led light',
    'push in drawer', 'lift blue block drawer',
    'close drawer', 'lift pink block slider',
    'lift pink block table', 'move slider left',
    'open drawer', 'turn on light bulb',
    'rotate blue block left', 'turn on led light',
    'push pink block right', 'push red block left',
    'lift blue block table', 'place in drawer',
    'rotate red block left', 'push pink block left',
    'lift stacked blocks', 'lift blue block slider',
    'push blue block right', 'lift red block table',
    'push red block right', 'lift green block table',
    'push green block left', 'lift yellow block table',
    'push yellow block right', 'lift orange block table',
    'push orange block left', 'lift purple block table'
]
```

## 2. 데이터셋 구조 분석

### 2.1 Action 데이터 구조
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 액션 구조

```python
# CALVIN 액션 데이터 구조
action_structure = {
    'actions': {
        'dtype': 'np.float32',
        'shape': '(7,)',
        'description': '7-DOF 로봇 액션',
        'components': {
            'tcp_position': '3D position (x, y, z) in absolute world coordinates',
            'tcp_orientation': '3D orientation (x, y, z) in absolute world coordinates',
            'gripper_action': '1D binary (close = -1, open = 1)'
        }
    },
    'rel_actions': {
        'dtype': 'np.float32', 
        'shape': '(7,)',
        'description': '7-DOF 상대 액션',
        'components': {
            'tcp_position': '3D position in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50',
            'tcp_orientation': '3D orientation in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20',
            'gripper_action': '1D binary (close = -1, open = 1)'
        }
    }
}
```

### 2.2 이미지 데이터 구조
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 이미지 구조

```python
# CALVIN 이미지 데이터 구조
image_structure = {
    'rgb_obs': {
        'rgb_static': 'Static overhead camera RGB image (224x224 or 336x336)',
        'rgb_gripper': 'Gripper-mounted camera RGB image (224x224 or 336x336)'
    },
    'depth_obs': {
        'depth_static': 'Static overhead camera depth image (optional)',
        'depth_gripper': 'Gripper-mounted camera depth image (optional)'
    },
    'camera_info': {
        'intrinsic': 'Camera intrinsic parameters',
        'extrinsic': 'Camera extrinsic parameters'
    }
}
```

### 2.3 텍스트 데이터 구조
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 텍스트 구조

```python
# CALVIN 텍스트 데이터 구조
text_structure = {
    'lang': 'Natural language instruction string',
    'lang_ann': 'Language annotation metadata',
    'task_id': 'Task identifier for the instruction',
    'task_name': 'Human-readable task name'
}
```

## 3. 데이터셋 분할 전략

### 3.1 ABCD 분할
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 ABCD 분할

```python
# ABCD 분할: 일반화 평가
calvin_splits = {
    'A': {
        'description': 'Basic manipulation tasks',
        'tasks': ['lift', 'place', 'push', 'pull'],
        'objects': ['blocks', 'sliders', 'drawers']
    },
    'B': {
        'description': 'Complex manipulation tasks',
        'tasks': ['rotate', 'stack', 'unstack'],
        'objects': ['blocks', 'sliders', 'drawers']
    },
    'C': {
        'description': 'Light manipulation tasks',
        'tasks': ['turn_on', 'turn_off'],
        'objects': ['light_bulb', 'led_light']
    },
    'D': {
        'description': 'Combined complex tasks',
        'tasks': ['all_above'],
        'objects': ['all_above'],
        'purpose': 'Generalization evaluation'
    }
}
```

**ABCD → D 평가**:
- **훈련**: A, B, C, D 모든 분할
- **테스트**: D 분할
- **목적**: 일반화 능력 평가
- **성능**: 96.7% 단일 작업 성공률, 4.49 Avg. Len.

### 3.2 ABC 분할
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 ABC 분할

```python
# ABC 분할: 향상된 일반화 평가
abc_to_d_split = {
    'train_splits': ['A', 'B', 'C'],
    'test_split': 'D',
    'purpose': 'Enhanced generalization evaluation',
    'description': 'Train on A, B, C splits, test on D split'
}
```

**ABC → D 평가**:
- **훈련**: A, B, C 분할
- **테스트**: D 분할
- **목적**: 향상된 일반화 평가
- **성능**: 98.0% 단일 작업 성공률, 4.25 Avg. Len.

## 4. 평가 메트릭

### 4.1 성공률 메트릭
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 평가 메트릭

```python
# CALVIN 평가 메트릭
evaluation_metrics = {
    'consecutive_tasks': {
        '1_task': 'Single task success rate',
        '2_tasks': 'Two consecutive tasks success rate',
        '3_tasks': 'Three consecutive tasks success rate',
        '4_tasks': 'Four consecutive tasks success rate',
        '5_tasks': 'Five consecutive tasks success rate'
    },
    'average_length': {
        'description': 'Average number of successfully executed tasks per rollout',
        'calculation': 'Sum of successful tasks / Total rollouts'
    },
    'task_specific': {
        'description': 'Success rate for individual tasks',
        'calculation': 'Successful executions / Total attempts per task'
    }
}
```

### 4.2 평가 설정
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 평가 설정

```python
# CALVIN 평가 설정
evaluation_setup = {
    'rollouts': 1000,
    'consecutive_tasks': 5,
    'max_steps_per_task': 64,
    'success_criteria': {
        'position_tolerance': 0.05,  # 5cm
        'orientation_tolerance': 0.1,  # 0.1 rad
        'gripper_tolerance': 0.1
    }
}
```

## 5. 데이터셋 활용 전략

### 5.1 훈련 전략
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 훈련 전략

```python
# CALVIN 데이터셋 훈련 전략
training_strategies = {
    'full_dataset': {
        'description': 'Train on all ABCD splits',
        'use_case': 'Maximum performance',
        'data_size': '24K demonstrations'
    },
    'generalization': {
        'description': 'Train on ABC, test on D',
        'use_case': 'Generalization evaluation',
        'data_size': '18K demonstrations (A+B+C)'
    },
    'few_shot': {
        'description': 'Train on limited data',
        'use_case': 'Data efficiency evaluation',
        'data_size': '10 trajectories per task'
    }
}
```

### 5.2 데이터 전처리
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 전처리

```python
# CALVIN 데이터 전처리
data_preprocessing = {
    'action_normalization': {
        'method': 'Quantile-based clamping',
        'range': '[-1, 1]',
        'quantiles': ['1st', '99th']
    },
    'image_preprocessing': {
        'resize': '224x224 or 336x336',
        'normalization': 'ImageNet mean/std',
        'augmentation': 'Random crop, flip, color jitter'
    },
    'text_preprocessing': {
        'tokenization': 'BERT tokenizer',
        'max_length': 256,
        'padding': 'Dynamic padding'
    }
}
```

## 6. 데이터셋 특성 분석

### 6.1 시뮬레이션 환경
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 시뮬레이션 환경

```python
# CALVIN 시뮬레이션 환경
simulation_environment = {
    'type': 'simulation',
    'robot': 'Franka Panda',
    'workspace': 'table-top manipulation',
    'objects': 'blocks, sliders, drawers, lights',
    'camera': 'static overhead camera',
    'physics_engine': 'PyBullet',
    'rendering': 'OpenGL'
}
```

### 6.2 작업 복잡도
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 작업 복잡도

```python
# 작업 복잡도 분석
task_complexity = {
    'simple_tasks': [
        'turn on light bulb', 'turn off light bulb',
        'turn on led light', 'turn off led light'
    ],
    'medium_tasks': [
        'lift block', 'place block', 'push block',
        'pull slider', 'move slider'
    ],
    'complex_tasks': [
        'lift stacked blocks', 'place in drawer',
        'open drawer', 'close drawer'
    ]
}
```

## 7. 성능 결과

### 7.1 RoboVLMs 성능
**GitHub Code Reference**: [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs)
**실제 코드 위치**: RoboVLMs CALVIN 성능 결과

```python
# RoboVLMs CALVIN 성능 결과
robovlms_performance = {
    'ABCD_to_D': {
        'single_task': 0.967,
        'two_tasks': 0.930,
        'three_tasks': 0.899,
        'four_tasks': 0.865,
        'five_tasks': 0.826,
        'avg_length': 4.49
    },
    'ABC_to_D': {
        'single_task': 0.980,
        'two_tasks': 0.936,
        'three_tasks': 0.854,
        'four_tasks': 0.778,
        'five_tasks': 0.704,
        'avg_length': 4.25
    }
}
```

### 7.2 비교 모델 성능
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 벤치마크 비교 결과

```python
# 비교 모델 성능 (ABCD → D)
comparison_performance = {
    'MCIL': {'avg_length': 0.40},
    'R3M_Frozen': {'avg_length': 0.10},
    'Voltron_Frozen': {'avg_length': 0.11},
    'Voltron_Fine_tuned': {'avg_length': 2.08},
    'RT_1': {'avg_length': 2.45},
    'HULC': {'avg_length': 3.06},
    'GR_1': {'avg_length': 4.21},
    'KosMos_PH_RoboVLMs': {'avg_length': 4.49}  # SOTA
}
```

## 8. 데이터셋 다운로드 및 사용

### 8.1 데이터셋 다운로드
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 다운로드

```bash
# CALVIN 데이터셋 다운로드
git clone https://github.com/mees/calvin.git
cd calvin

# 데이터셋 다운로드 스크립트 실행
python scripts/download_dataset.py

# 데이터셋 검증
python scripts/validate_dataset.py
```

### 8.2 데이터셋 로드
**GitHub Code Reference**: [CALVIN Dataset](https://github.com/mees/calvin/tree/main)
**실제 코드 위치**: CALVIN 데이터셋 로드

```python
# CALVIN 데이터셋 로드
from calvin import CalvinDataset

# 데이터셋 초기화
dataset = CalvinDataset(
    data_path='./calvin_dataset',
    split='ABCD',
    window_size=16,
    action_chunk_size=10
)

# 데이터 로드
for batch in dataset:
    images = batch['rgb_obs']
    actions = batch['actions']
    language = batch['lang']
    # 모델 훈련...
```

## 결론

CALVIN 데이터셋은 로봇 조작을 위한 언어 조건 정책 학습의 표준 벤치마크입니다:

### 핵심 특징
- **24K 시연**: 대규모 인간 텔레오퍼레이션 데이터
- **34개 기술**: 다양한 조작 작업 커버
- **ABCD 분할**: 체계적인 일반화 평가
- **7-DOF 액션**: TCP 위치, 회전, 그리퍼 제어
- **멀티모달**: 이미지, 텍스트, 액션 동기화

### RoboVLMs 통합
- **최고 성능**: 4.49 Avg. Len. (ABCD → D)
- **강력한 일반화**: 4.25 Avg. Len. (ABC → D)
- **실용적 활용**: 실제 로봇 배포 가능

CALVIN 데이터셋은 Vision-Language-Action 모델의 성능을 평가하고 개선하는 데 필수적인 벤치마크입니다.
