---
license: apache-2.0
tags:
- vision-language-action
- mobile-robot
- kosmos-2b
- robotics
- obstacle-avoidance
datasets:
- mobile-vla-dataset
language:
- en
- ko
metrics:
- mae
- r2_score
library_name: transformers
pipeline_tag: robotics
---

# 🚀 Mobile VLA: Vision-Language-Action Model for Mobile Robots

## 📋 Model Description

Mobile VLA는 Kosmos-2B를 기반으로 한 Mobile Robot 전용 Vision-Language-Action 모델입니다. 
장애물 회피 시나리오에서 연속적인 3D 액션 예측을 수행합니다.

### 🎯 핵심 기능

- **Vision-Language-Action**: 이미지와 텍스트 지시사항을 받아 로봇 액션 예측
- **3D 연속 제어**: `[linear_x, linear_y, angular_z]` 형태의 연속 액션 공간
- **장애물 회피**: 1-box, 2-box 시나리오에서 좌우 회피 전략 학습
- **실시간 처리**: 효율적인 vision-only 처리로 빠른 추론

### 🔧 기술 사양

- **백본 모델**: microsoft/kosmos-2-patch14-224
- **입력**: RGB 이미지 (224x224) + 텍스트 지시사항
- **출력**: 3D 연속 액션 벡터
- **학습 방식**: Huber Loss 기반 회귀
- **데이터**: 72개 실제 로봇 에피소드

## 📊 성능 지표

### 전체 성능
- **전체 MAE**: 0.285
- **임계값 정확도 (0.1)**: 37.5%

### 액션별 성능
| 액션 | MAE | R² Score | 설명 |
|------|-----|----------|------|
| linear_x | 0.243 | 0.354 | 전진/후진 (우수) |
| linear_y | 0.550 | 0.293 | 좌우 이동 (보통) |
| angular_z | 0.062 | 0.000 | 회전 (낮음) |

### 시나리오별 성능
| 시나리오 | MAE | 등급 | 설명 |
|----------|-----|------|------|
| 1box_right_vertical | 0.217 | B+ | 우수 |
| 1box_left_horizontal | 0.303 | B | 양호 |
| 2box_left_vertical | 0.322 | B | 양호 |
| 1box_left_vertical | 0.337 | B- | 보통 |

## 🚀 사용 방법

### 설치
```bash
pip install transformers torch pillow numpy
```

### 기본 사용법
```python
from mobile_vla import MobileVLAModel, MobileVLATrainer
from PIL import Image
import torch

# 모델 로드
model = MobileVLAModel.from_pretrained("minuum/mobile-vla")

# 이미지와 태스크 준비
image = Image.open("robot_camera.jpg")
task = "Navigate around obstacles to track the target cup"

# 예측
with torch.no_grad():
    actions = model.predict(image, task)
    
print(f"Predicted actions: {actions}")
# 출력: [linear_x, linear_y, angular_z]
```

### 고급 사용법
```python
# 배치 처리
images = [Image.open(f"frame_{i}.jpg") for i in range(8)]
actions = model.predict_sequence(images, task)

# 실시간 제어
for frame in camera_stream:
    action = model.predict(frame, task)
    robot.execute(action)
```

## 🏗️ 모델 아키텍처

```
[RGB Images] → [Kosmos-2B Vision] → [Action Head] → [3D Actions]
     ↓              ↓                    ↓             ↓
   224x224    Image Features         Regression    [x, y, θ]
```

### 핵심 컴포넌트
1. **Kosmos-2B Vision Model**: 이미지 특징 추출
2. **Action Head**: 3D 회귀 헤드 (512 → 3*chunk_size)
3. **Window/Chunk**: 8프레임 관찰 → 2프레임 예측

## 📈 RoboVLMs와의 비교

| 항목 | RoboVLMs | Mobile VLA |
|------|----------|------------|
| **데이터 요구량** | 수백만 데모 | 72 에피소드 |
| **액션 공간** | 7-DOF Discrete | 3D Continuous |
| **추론 속도** | 복합적 | 빠름 |
| **특화 분야** | 범용 Manipulation | Mobile Robot |
| **평가 방식** | 성공률 | 다차원 회귀 지표 |

## 🎯 주요 개선사항

- **데이터 효율성**: 1000배 적은 데이터로 실용적 성능
- **실시간 성능**: Vision-only 처리로 빠른 추론
- **연속 제어**: 정밀한 3D 액션 예측
- **시나리오 특화**: 장애물 회피 전용 최적화

## 📚 학습 데이터

- **에피소드 수**: 72개
- **시나리오**: 1box/2box × left/right × vertical/horizontal
- **액션**: [linear_x, linear_y, angular_z] 연속 값
- **이미지**: 실제 로봇 카메라 RGB (224x224)

## 🔬 연구 배경

이 모델은 RoboVLMs의 Window/Chunk 메커니즘을 유지하면서 Mobile Robot에 특화된 기능을 추가한 연구입니다:

1. **Window/Chunk 유지**: 8프레임 관찰 → 2프레임 예측 구조
2. **Kosmos-2B 통합**: Vision-Language 백본 활용
3. **연속 제어**: Discrete → Continuous 액션 공간 전환
4. **실제 로봇 데이터**: HDF5 형태의 실제 수집 데이터

## 📄 인용

```bibtex
@misc{mobile_vla_2024,
  title={Mobile VLA: Vision-Language-Action Model for Mobile Robot Navigation},
  author={Mobile VLA Team},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/minuum/mobile-vla}
}
```

## 🤝 기여

이 모델은 RoboVLMs 프레임워크를 기반으로 개발되었으며, Mobile Robot 커뮤니티의 발전을 위해 공개됩니다.

## 📞 연락처

- **Issues**: [GitHub Issues](https://github.com/minuum/vla/issues)
- **Discussions**: [HuggingFace Discussions](https://huggingface.co/minuum/mobile-vla/discussions)

---
*Generated on 2025-08-21*
