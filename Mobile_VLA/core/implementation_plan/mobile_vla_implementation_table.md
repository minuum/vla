# Mobile VLA 실기능 구현 계획표

## 1. 전체 구현 로드맵

| **단계** | **구현 항목** | **우선순위** | **예상 기간** | **상태** | **관련 문서** |
|----------|---------------|--------------|---------------|----------|---------------|
| **1단계** | 데이터 수집 환경 테스트 | 🔴 높음 | 1일 | 진행중 | `current_status_and_plan.md` |
| **2단계** | Mobile VLA 모델 구조 구현 | 🔴 높음 | 3일 | 대기 | `robovlms_vs_mobile_vla.md` |
| **3단계** | 학습 파이프라인 구현 | 🟡 중간 | 5일 | 대기 | `lstm_layer_learning.md` |
| **4단계** | 추론 시스템 구현 | 🟡 중간 | 4일 | 대기 | `real_world_data_collection.md` |
| **5단계** | 성능 평가 시스템 구현 | 🟢 낮음 | 3일 | 대기 | `finetuning_analysis/` |

## 2. 데이터 수집 시스템 구현

| **구현 항목** | **세부 작업** | **기술 스택** | **예상 시간** | **상태** | **참고 문서** |
|---------------|---------------|---------------|---------------|----------|---------------|
| **환경 테스트** | Branch b131fb5 동작 확인 | ROS2, Python | 2시간 | 진행중 | `current_status_and_plan.md` |
| **카메라 서비스** | camera_service_server 실행 | ROS2, OpenCV | 1시간 | 대기 | `KOSMOS2_CAMERA_INTEGRATION_GUIDE.md` |
| **VLA 컬렉터** | vla_collector 실행 | ROS2, HDF5 | 1시간 | 대기 | `current_status_and_plan.md` |
| **데이터 검증** | 실시간 품질 확인 | Python, NumPy | 2시간 | 대기 | `adjusted_collection_strategy.md` |

## 3. Mobile VLA 모델 구조 구현

| **구현 항목** | **세부 작업** | **기술 스택** | **예상 시간** | **상태** | **참고 문서** |
|---------------|---------------|---------------|---------------|----------|---------------|
| **2D 액션 공간** | X, Y, Gripper (3차원) | PyTorch | 4시간 | 대기 | `MOBILE_VLA_TASK_DEFINITION.md` |
| **LoRA Fine-tuning** | VLM 백본 동결, LoRA 파라미터 | PEFT, Transformers | 8시간 | 대기 | `robovlms_vs_mobile_vla.md` |
| **LSTM Policy Head** | 2층 LSTM, 512 hidden size | PyTorch, LSTM | 6시간 | 대기 | `lstm_layer_learning.md` |
| **Multi-modal Fusion** | [LRN] Token, Attention | Transformers | 4시간 | 대기 | `action_synchronization.md` |

## 4. 학습 파이프라인 구현

| **구현 항목** | **세부 작업** | **기술 스택** | **예상 시간** | **상태** | **참고 문서** |
|---------------|---------------|---------------|---------------|----------|---------------|
| **데이터 로더** | HDF5 데이터셋 로더 | PyTorch, H5py | 6시간 | 대기 | `real_world_data_collection.md` |
| **Loss 함수** | MSE + BCE Loss | PyTorch | 2시간 | 대기 | `lstm_layer_learning.md` |
| **Optimizer** | AdamW + CosineAnnealingLR | PyTorch | 2시간 | 대기 | `lstm_layer_learning.md` |
| **학습 루프** | Training/Validation Loop | PyTorch Lightning | 8시간 | 대기 | `end_to_end_learning.md` |

## 5. 추론 시스템 구현

| **구현 항목** | **세부 작업** | **기술 스택** | **예상 시간** | **상태** | **참고 문서** |
|---------------|---------------|---------------|---------------|----------|---------------|
| **실시간 추론** | 실시간 액션 예측 | PyTorch, ROS2 | 6시간 | 대기 | `real_world_data_collection.md` |
| **Jetson 최적화** | TorchScript, FP16 | TorchScript | 4시간 | 대기 | `DOCKER_JETSON_GUIDE.md` |
| **Docker 컨테이너** | Jetson L4T 기반 | Docker, L4T | 4시간 | 대기 | `DOCKER_JETSON_GUIDE.md` |
| **메모리 최적화** | 16GB 제한 내 최적화 | PyTorch | 4시간 | 대기 | `robovlms_vs_mobile_vla.md` |

## 6. 성능 평가 시스템 구현

| **구현 항목** | **세부 작업** | **기술 스택** | **예상 시간** | **상태** | **참고 문서** |
|---------------|---------------|---------------|---------------|----------|---------------|
| **벤치마크** | 성능 측정 스크립트 | Python, NumPy | 4시간 | 대기 | `evaluation/performance_analysis/` |
| **메트릭 측정** | 정확도, 속도, 메모리 | Python | 2시간 | 대기 | `evaluation/performance_analysis/` |
| **양자화 테스트** | FP16, INT8 양자화 | TensorRT, ONNX | 6시간 | 대기 | `FINAL_TENSORRT_ANALYSIS.md` |
| **성능 분석** | 결과 시각화 | Matplotlib, Seaborn | 2시간 | 대기 | `evaluation/performance_analysis/` |

## 7. 구현 우선순위 및 일정

### 7.1 1주차 (즉시 시작)

| **날짜** | **작업** | **예상 시간** | **상태** |
|----------|----------|---------------|----------|
| **오늘** | 데이터 수집 환경 테스트 | 4시간 | 진행중 |
| **내일** | Mobile VLA 모델 구조 구현 시작 | 8시간 | 대기 |
| **수요일** | 2D 액션 공간 구현 | 4시간 | 대기 |
| **목요일** | LoRA Fine-tuning 구현 | 8시간 | 대기 |
| **금요일** | LSTM Policy Head 구현 | 6시간 | 대기 |

### 7.2 2주차

| **날짜** | **작업** | **예상 시간** | **상태** |
|----------|----------|---------------|----------|
| **월요일** | Multi-modal Fusion 구현 | 4시간 | 대기 |
| **화요일** | 데이터 로더 구현 | 6시간 | 대기 |
| **수요일** | Loss 함수 및 Optimizer 구현 | 4시간 | 대기 |
| **목요일** | 학습 루프 구현 | 8시간 | 대기 |
| **금요일** | 학습 파이프라인 테스트 | 4시간 | 대기 |

### 7.3 3주차

| **날짜** | **작업** | **예상 시간** | **상태** |
|----------|----------|---------------|----------|
| **월요일** | 실시간 추론 구현 | 6시간 | 대기 |
| **화요일** | Jetson 최적화 | 4시간 | 대기 |
| **수요일** | Docker 컨테이너 구현 | 4시간 | 대기 |
| **목요일** | 메모리 최적화 | 4시간 | 대기 |
| **금요일** | 추론 시스템 테스트 | 4시간 | 대기 |

### 7.4 4주차

| **날짜** | **작업** | **예상 시간** | **상태** |
|----------|----------|---------------|----------|
| **월요일** | 벤치마크 구현 | 4시간 | 대기 |
| **화요일** | 메트릭 측정 구현 | 2시간 | 대기 |
| **수요일** | 양자화 테스트 | 6시간 | 대기 |
| **목요일** | 성능 분석 구현 | 2시간 | 대기 |
| **금요일** | 전체 시스템 통합 테스트 | 8시간 | 대기 |

## 8. 기술 스택 및 의존성

### 8.1 핵심 기술 스택

| **카테고리** | **기술** | **버전** | **용도** |
|--------------|----------|----------|----------|
| **딥러닝** | PyTorch | 2.0+ | 모델 구현 |
| **VLM** | Transformers | 4.30+ | Kosmos2 모델 |
| **Fine-tuning** | PEFT | 0.4+ | LoRA 구현 |
| **로봇 제어** | ROS2 | Humble | 실시간 제어 |
| **컨테이너** | Docker | 20.10+ | 배포 |
| **Jetson** | L4T | 35.2+ | 하드웨어 최적화 |

### 8.2 의존성 파일

```python
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
torchvision>=0.15.0
torchaudio>=2.0.0
einops>=0.6.1
pillow>=9.5.0
h5py>=3.8.0
numpy>=1.21.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorrt>=8.6.0
onnx>=1.14.0
```

## 9. 성공 지표

### 9.1 기술적 지표

| **지표** | **목표값** | **측정 방법** | **현재값** |
|----------|------------|---------------|------------|
| **추론 속도** | 10 FPS 이상 | 실시간 측정 | 미측정 |
| **메모리 사용량** | 14GB 이하 | Jetson 모니터링 | 미측정 |
| **정확도** | 75% 이상 | 벤치마크 테스트 | 미측정 |
| **학습 시간** | 5시간 이하 | 학습 로그 | 미측정 |

### 9.2 기능적 지표

| **지표** | **목표값** | **측정 방법** | **현재값** |
|----------|------------|---------------|------------|
| **데이터 수집** | 1000개 에피소드 | 수집 로그 | 0개 |
| **모델 학습** | 수렴 완료 | Loss 곡선 | 미완료 |
| **실시간 추론** | 안정적 동작 | 연속 테스트 | 미완료 |
| **Docker 배포** | 정상 동작 | 컨테이너 테스트 | 미완료 |

## 10. 위험 요소 및 대응 방안

### 10.1 기술적 위험

| **위험 요소** | **확률** | **영향도** | **대응 방안** |
|---------------|----------|------------|---------------|
| **Jetson 메모리 부족** | 높음 | 높음 | 모델 크기 축소, 양자화 |
| **학습 수렴 실패** | 중간 | 높음 | 하이퍼파라미터 튜닝 |
| **실시간 성능 부족** | 중간 | 중간 | TorchScript 최적화 |
| **Docker 호환성** | 낮음 | 중간 | L4T 기반 이미지 사용 |

### 10.2 일정 위험

| **위험 요소** | **확률** | **영향도** | **대응 방안** |
|---------------|----------|------------|---------------|
| **데이터 수집 지연** | 중간 | 높음 | 자동화 스크립트 개발 |
| **모델 구현 복잡성** | 높음 | 높음 | 단계별 구현, 테스트 |
| **성능 최적화 어려움** | 중간 | 중간 | 전문가 컨설팅 |
| **통합 테스트 실패** | 낮음 | 높음 | 모듈별 단위 테스트 |

## 11. 다음 단계

### 11.1 즉시 실행 (오늘)

1. **데이터 수집 환경 테스트** (2시간)
   ```bash
   git checkout b131fb5
   ros2 launch camera_service_server camera_service.launch.py
   ros2 launch vla_collector vla_collector.launch.py
   ```

2. **첫 에피소드 수집** (2시간)
   - 5개 에피소드 테스트 수집
   - 데이터 품질 확인

### 11.2 1주차 목표

1. **Mobile VLA 모델 구조 완성**
2. **기본 학습 파이프라인 구축**
3. **100개 에피소드 수집 완료**

### 11.3 장기 목표

1. **1000개 에피소드 수집 완료**
2. **완전한 Mobile VLA 시스템 구축**
3. **Jetson에서 안정적 동작 확인**

## 12. 참고 자료

- `RoboVLMs/core/data_collection/current_status_and_plan.md`: 데이터 수집 계획
- `RoboVLMs/core/comparison_analysis/robovlms_vs_mobile_vla.md`: 아키텍처 비교
- `RoboVLMs/core/finetuning_analysis/lstm_layer_learning.md`: LSTM 학습
- `RoboVLMs/core/finetuning_analysis/real_world_data_collection.md`: 데이터 수집
- `Mobile_VLA/docs/DOCKER_JETSON_GUIDE.md`: Jetson 가이드
- `Mobile_VLA/evaluation/performance_analysis/`: 성능 평가
