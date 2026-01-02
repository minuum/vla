# Mobile VLA - Vision-Language-Action for Mobile Robot

> 음료수 병 탐색을 위한 VLA 모델 개발 및 Jetson 온디바이스 배포

**최종 업데이트**: 2026-01-02  
**Status**: ✅ Phase 2 완료 (Jetson Inference) | 🔄 모델 재학습 필요

---

## 📋 Quick Start

### Jetson Orin에서 추론 실행
```bash
# 1. 환경 활성화
source ~/.vla_phase4_aliases
source ROS_action/install/setup.bash

# 2. 카메라 서버 실행 (Terminal 1)
vla-camera

# 3. 추론 노드 실행 (Terminal 2)
vla-inference
# 'S' 키로 추론 시작, 1-4 키로 시나리오 변경
```

### 분석 도구
```bash
# 좌표계 검증
python3 scripts/verify_coordinate_system.py --samples 10

# 추론 데이터 시각화
python3 scripts/visualize_inference_log.py docs/memory_analysis/inference_data_*.json
```

---

## 🎯 프로젝트 현황

### ✅ 완료된 작업
- **모델 학습**: Frozen VLM + Policy Head (Val Loss: 0.010)
- **Jetson 환경 구축**: PyTorch 2.1.0 + Transformers 4.35.0
- **ROS2 추론 노드**: 18-frame stepwise inference 완료
- **메모리 프로파일링**: 자동 JSON 저장 + 시각화 도구
- **좌표계 분석**: 데이터셋 분석 스크립트 완성

### ⚠️ 알려진 이슈

#### 1. 모델 방향성 문제 (CRITICAL)
**현상:**
- 모델이 후진(x<0) 방향으로 일관되게 출력
- 데이터: X>0=전진, 모델 출력: X<0=후진

**분석 결과** (`verify_coordinate_system.py`):
```
Dataset: X=+1.15 (100% 전진)
Model Output: X=-4.74 (후진)
```

**임시 조치**:
- `mobile_vla_inference_node.py`에 `INVERT_X_AXIS=True` 플래그 추가
- ⚠️ 근본 해결책 아님, 모델 재학습 필요

**다음 계획** (1/7 미팅까지):
1. 학습 파이프라인의 좌표 변환 로직 검증
2. 올바른 좌표계로 모델 재학습 (1-2 epochs)
3. End-to-End 방향성 테스트

#### 2. Instruction 모호성
**문제**: "Navigate to the target" (모호)  
**해결**: "Navigate to the left/right bottle" (구체화 완료)

---

## 📁 주요 디렉토리

```
vla/
├── ROS_action/
│   ├── src/mobile_vla_package/        # ROS2 추론 노드
│   │   └── mobile_vla_inference_node.py
│   └── mobile_vla_dataset/            # 훈련 데이터 (237 episodes)
│
├── src/
│   └── robovlms_mobile_vla_inference.py  # 추론 엔진
│
├── scripts/
│   ├── verify_coordinate_system.py    # 좌표계 분석 도구
│   └── visualize_inference_log.py     # 궤적 시각화
│
├── docs/
│   ├── 20260102_Weekly_Dev_Report.md  # 주간 리포트
│   ├── meeting/                       # 미팅 자료
│   └── memory_analysis/               # 추론 데이터
│
└── runs/
    └── mobile_vla_chunk5_20251217/
        └── epoch_epoch=06-val_loss=val_loss=0.067.ckpt  # 현재 모델
```

---

## 🔧 기술 스택

### On-Device (Jetson AGX Orin)
- **OS**: Jetson Linux (r36.4.0)
- **Python**: 3.10
- **PyTorch**: 2.1.0 (JP 6.0 compatible)
- **Transformers**: 4.35.0
- **ROS2**: Humble

### 모델
- **Backbone**: Kosmos-2 (Frozen)
- **Policy**: Trainable MLP Head
- **Precision**: FP16
- **VRAM**: ~3.12GB

---

## 📊 성능

### 추론 성능
- **Window Size**: 2 frames
- **Chunk Size**: 2 actions
- **Latency**: ~150ms/step (Jetson Orin)
- **메모리**: CPU ~2GB, GPU ~3.2GB

### 주행 성능
- **Target**: 18 steps in 7.2~8초
- **Current**: ~10초 (추론 오버헤드 포함)
- **Accuracy**: 방향성 이슈로 미측정 → 재학습 후 평가 예정

---

## 📅 로드맵

### Phase 1: 학습 ✅
- [x] 데이터셋 수집 (237 episodes)
- [x] Frozen VLM 학습
- [x] Best Model 선정

### Phase 2: Jetson 배포 ✅
- [x] PyTorch + Transformers 환경 구축
- [x] ROS2 Inference Node 개발
- [x] 메모리 프로파일링 시스템 구축
- [x] 좌표계 분석 도구 개발

### Phase 3: 모델 수정 🔄 (진행중)
- [ ] 좌표 변환 로직 검증
- [ ] 모델 재학습 (올바른 좌표계)
- [ ] Step-by-Step Validation 구현
- [ ] 성능 최적화 (8초 목표)

### Phase 4: 실주행 테스트 (예정)
- [ ] Closed-loop 주행
- [ ] Success Rate 측정
- [ ] 논문 작성

---

## 📖 문서

### 핵심 문서
- [20260102_Weekly_Dev_Report.md](docs/20260102_Weekly_Dev_Report.md) - 주간 개발 리포트
- [implementation_plan.md](.gemini/antigravity/brain/.../implementation_plan.md) - 개선 계획
- [meeting/20260102_feedback_and_plan.md](docs/meeting/20260102_feedback_and_plan.md) - 미팅 피드백

### 가이드
- Jetson 환경: `docs/JETSON_PYTORCH_SETUP_COMPLETE_20260102.md`
- ROS2 추론: `docs/guides/phase2_session_guide.md`

---

## 🛠️ Development

### Git Workflow
```bash
# 현재 브랜치: feature/inference-integration
git status

# 커밋
git add -A
git commit -m "feat: Add coordinate system fix and instruction refinement"

# 푸시
git push origin feature/inference-integration
```

### 커밋 컨벤션
- `feat:` 새로운 기능
- `fix:` 버그 수정
- `docs:` 문서 업데이트
- `refactor:` 코드 리팩토링
- `test:` 테스트 추가

---

## 👥 Team
- **Soda** - Jetson 배포, 추론 노드 개발
- **Billy** - 모델 학습, 데이터셋 구축

---

## 📚 References
- **RoboVLMs**: https://github.com/robotics-survey/RoboVLMs
- **Kosmos-2**: https://huggingface.co/microsoft/kosmos-2-patch14-224
- **PyTorch for Jetson**: https://forums.developer.nvidia.com/t/pytorch-for-jetson
