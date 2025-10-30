# Mobile VLA 파인튜닝 태스크 정의

## 개요

Mobile VLA (Vision-Language-Action) 시스템의 핵심 태스크는 **모바일 로봇의 장애물 회피 네비게이션**입니다. 이 문서는 파인튜닝해야 하는 구체적인 태스크와 요구사항을 명확히 정의합니다.

## 태스크 정의

### 🎯 **핵심 목표**
모바일 로봇이 카메라 이미지와 언어 명령을 입력받아, 장애물을 피해 목표 지점까지 자율적으로 이동하는 능력을 학습

### 📋 **태스크 시나리오**

#### **8가지 네비게이션 시나리오**

| 시나리오 | 설명 | 장애물 | 경로 | 목표 |
|---------|------|--------|------|------|
| `1box_vert_left` | 1개 박스, 세로 배치, 왼쪽 경로 | 1개 박스 (세로) | 왼쪽 우회 | 컵 도달 |
| `1box_vert_right` | 1개 박스, 세로 배치, 오른쪽 경로 | 1개 박스 (세로) | 오른쪽 우회 | 컵 도달 |
| `1box_hori_left` | 1개 박스, 가로 배치, 왼쪽 경로 | 1개 박스 (가로) | 왼쪽 우회 | 컵 도달 |
| `1box_hori_right` | 1개 박스, 가로 배치, 오른쪽 경로 | 1개 박스 (가로) | 오른쪽 우회 | 컵 도달 |
| `2box_vert_left` | 2개 박스, 세로 배치, 왼쪽 경로 | 2개 박스 (세로) | 왼쪽 우회 | 컵 도달 |
| `2box_vert_right` | 2개 박스, 세로 배치, 오른쪽 경로 | 2개 박스 (세로) | 오른쪽 우회 | 컵 도달 |
| `2box_hori_left` | 2개 박스, 가로 배치, 왼쪽 경로 | 2개 박스 (가로) | 왼쪽 우회 | 컵 도달 |
| `2box_hori_right` | 2개 박스, 가로 배치, 오른쪽 경로 | 2개 박스 (가로) | 오른쪽 우회 | 컵 도달 |

#### **언어 명령 예시**
```python
language_commands = {
    "1box_vert_left": "Navigate around the single box obstacle by going left",
    "1box_vert_right": "Navigate around the single box obstacle by going right", 
    "1box_hori_left": "Navigate around the single box obstacle by going left",
    "1box_hori_right": "Navigate around the single box obstacle by going right",
    "2box_vert_left": "Navigate around two box obstacles by going left",
    "2box_vert_right": "Navigate around two box obstacles by going right",
    "2box_hori_left": "Navigate around two box obstacles by going left",
    "2box_hori_right": "Navigate around two box obstacles by going right"
}
```

## 액션 공간 정의

### **2D 연속 액션 공간**

```python
action_space = {
    'linear_x': {
        'range': [-1.15, 1.15],  # m/s
        'description': 'Forward/backward velocity',
        'keyboard': 'W/S keys'
    },
    'linear_y': {
        'range': [-1.15, 1.15],  # m/s  
        'description': 'Left/right velocity',
        'keyboard': 'A/D keys'
    },
    'angular_z': {
        'range': [-1.15, 1.15],  # rad/s
        'description': 'Rotation velocity',
        'keyboard': 'R/T keys'
    },
    'action_type': {
        'range': [0, 3],  # discrete
        'description': 'Action type classification',
        'values': {
            0: 'movement',
            1: 'rotation', 
            2: 'stop',
            3: 'special'
        }
    }
}
```

### **키보드 입력 매핑**

| 키 | 액션 | linear_x | linear_y | angular_z | 설명 |
|----|------|----------|----------|-----------|------|
| **W** | 전진 | 1.15 | 0.0 | 0.0 | 앞으로 이동 |
| **S** | 후진 | -1.15 | 0.0 | 0.0 | 뒤로 이동 |
| **A** | 좌측 | 0.0 | 1.15 | 0.0 | 왼쪽으로 이동 |
| **D** | 우측 | 0.0 | -1.15 | 0.0 | 오른쪽으로 이동 |
| **Q** | 좌상 대각선 | 1.15 | 1.15 | 0.0 | 전진+좌측 |
| **E** | 우상 대각선 | 1.15 | -1.15 | 0.0 | 전진+우측 |
| **Z** | 좌하 대각선 | -1.15 | 1.15 | 0.0 | 후진+좌측 |
| **C** | 우하 대각선 | -1.15 | -1.15 | 0.0 | 후진+우측 |
| **R** | 좌회전 | 0.0 | 0.0 | 1.15 | 반시계 방향 회전 |
| **T** | 우회전 | 0.0 | 0.0 | -1.15 | 시계 방향 회전 |
| **SPACE** | 정지 | 0.0 | 0.0 | 0.0 | 모든 움직임 정지 |

## 입력 데이터 구조

### **시각적 입력**
```python
# 카메라 이미지
images: {
    'shape': [T, 720, 1280, 3],  # 시계열 RGB 이미지
    'dtype': 'uint8',
    'normalization': 'ImageNet standard',
    'preprocessing': 'resize to 224x224'
}
```

### **언어적 입력**
```python
# 텍스트 명령
language: {
    'type': 'natural language instruction',
    'examples': [
        "Navigate around the single box obstacle by going left",
        "Navigate around two box obstacles by going right"
    ],
    'tokenization': 'Kosmos-2 tokenizer',
    'max_length': 256
}
```

### **로봇 상태**
```python
# 로봇 관측 상태
robot_state: {
    'shape': [T, 15],  # 시계열 로봇 상태
    'components': [
        'position_x', 'position_y', 'position_z',
        'velocity_x', 'velocity_y', 'velocity_z', 
        'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
        'orientation_quaternion_x', 'orientation_quaternion_y', 
        'orientation_quaternion_z', 'orientation_quaternion_w',
        'battery_level', 'system_status'
    ]
}
```

## 데이터셋 특성

### **수집된 데이터 통계**
- **총 에피소드**: 72개 (2025-08-15 수집)
- **에피소드 길이**: 18 프레임 (고정)
- **이미지 해상도**: 720 × 1280 × 3 (RGB)
- **액션 차원**: 3차원 (X, Y, Z)
- **Z축 특성**: 모든 액션에서 Z=0 (2D 평면 이동)

### **시나리오별 분포**

#### **1박스 시나리오 (24개)**
- `1box_vert_left`: 8개
- `1box_vert_right`: 8개  
- `1box_hori_left`: 8개
- `1box_hori_right`: 8개

#### **2박스 시나리오 (24개)**
- `2box_vert_left`: 8개
- `2box_vert_right`: 8개
- `2box_hori_left`: 8개  
- `2box_hori_right`: 8개

### **액션 패턴 분석**
```python
# 실제 사용된 액션 패턴 예시
action_patterns = {
    '1box_vert_left': ['W', 'W', 'W', 'A', 'A', 'W', 'W', 'D', 'D'],
    '1box_vert_right': ['W', 'W', 'D', 'D', 'W', 'W', 'W', 'A', 'A'],
    '2box_vert_left': ['W', 'W', 'A', 'A', 'A', 'W', 'W', 'D', 'D', 'D'],
    '2box_vert_right': ['W', 'D', 'D', 'D', 'W', 'W', 'W', 'A', 'A', 'A']
}
```

## 파인튜닝 대상 컴포넌트

### **1. VLM 백본 (Kosmos-2)**
```python
# Full Fine-tuning 대상
vlm_components = {
    'vision_encoder': {
        'purpose': '장애물/목표 인식',
        'input': 'RGB images (224x224)',
        'output': 'visual features'
    },
    'text_encoder': {
        'purpose': '언어 명령 이해', 
        'input': 'natural language instructions',
        'output': 'text features'
    },
    'multimodal_fusion': {
        'purpose': '시각+언어 정보 융합',
        'input': 'visual + text features',
        'output': 'multimodal representation'
    },
    'lrn_token': {
        'purpose': '액션 예측을 위한 학습 가능한 토큰',
        'type': 'learnable parameter',
        'dimension': 1024
    }
}
```

### **2. LSTM Policy Head**
```python
# 액션 예측 헤드
policy_head = {
    'lstm_layers': {
        'count': 4,
        'hidden_size': 1024,
        'purpose': '시계열 히스토리 모델링'
    },
    'action_decoder': {
        'input_dim': 1024,
        'output_dim': 4,  # [linear_x, linear_y, angular_z, action_type]
        'purpose': '2D 액션 공간 예측'
    },
    'history_buffer': {
        'window_size': 8,
        'purpose': '과거 관측/액션 기억'
    }
}
```

### **3. 학습 목표**
```python
# Loss Function 구성
loss_components = {
    'action_loss': {
        'type': 'MSE Loss',
        'target': '액션 예측 정확도',
        'weight': 1.0
    },
    'navigation_loss': {
        'type': 'Success Rate Loss', 
        'target': '목표 도달 성공률',
        'weight': 0.5
    },
    'obstacle_avoidance_loss': {
        'type': 'Collision Penalty',
        'target': '장애물 회피 성능', 
        'weight': 0.3
    }
}

total_loss = (
    action_loss * 1.0 +
    navigation_loss * 0.5 + 
    obstacle_avoidance_loss * 0.3
)
```

## 파인튜닝 목표

### **1. 시각적 이해 능력**
- **장애물 인식**: 박스의 위치, 크기, 형태 파악
- **목표 식별**: 컵의 위치와 도달 가능성 판단
- **공간 이해**: 장애물과 목표 간의 관계성 파악
- **경로 계획**: 효율적인 이동 경로 시각화

### **2. 언어 이해 능력**
- **명령 해석**: "Navigate around obstacles" 명령 이해
- **방향성 파악**: "left" vs "right" 경로 선택
- **복잡성 처리**: 다중 장애물 환경에서의 방향성 이해
- **맥락 이해**: 시나리오별 적절한 행동 선택

### **3. 액션 예측 능력**
- **2D 평면 이동**: 효율적인 경로 계획
- **속도 조절**: 장애물 회피를 위한 적절한 속도 조절
- **방향 제어**: 목표 도달을 위한 정확한 방향 제어
- **시계열 모델링**: 과거 관측을 통한 미래 액션 예측

## 성능 평가 지표

### **1. 정확도 지표**
- **액션 예측 정확도**: MSE Loss
- **목표 도달 성공률**: Success Rate
- **장애물 회피 성공률**: Collision Avoidance Rate

### **2. 효율성 지표**
- **경로 효율성**: 최단 경로 대비 실제 경로 비율
- **시간 효율성**: 목표 도달까지 소요 시간
- **에너지 효율성**: 총 이동 거리

### **3. 안정성 지표**
- **충돌 회피율**: 장애물과의 충돌 방지 성능
- **경로 안정성**: 급격한 방향 전환 최소화
- **일관성**: 동일 시나리오에서의 일관된 성능

## 실행 방법

### **파인튜닝 실행**
```bash
# Mobile VLA 파인튜닝 실행
python train_mobile_vla.py \
  --config configs/mobile_vla/train_mobile_vla_full_ft.json \
  --data_dir /home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset
```

### **추론 실행**
```bash
# Mobile VLA 추론 실행
python eval/mobile_vla/inference_wrapper.py \
  --checkpoint runs/mobile_vla/checkpoints/mobile_vla-best.ckpt \
  --config configs/mobile_vla/train_mobile_vla_full_ft.json
```

---

*이 문서는 Mobile VLA 파인튜닝 태스크의 완전한 정의를 제공합니다. 실제 구현은 `RoboVLMs/docs/MOBILE_VLA_GUIDE.md`를 참조하세요.*
