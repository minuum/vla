# 15. RoboVLMs 학습 및 추론 파이프라인 - 문서 구조

##  개요

RoboVLMs의 학습과 추론 과정을 다루는 문서 시리즈입니다. 문서가 길어 3개 파일로 분리되었습니다.

---

##  문서 구조

### **15_1. VLM Fine-tuning과 LSTM Layer 학습: 데이터 수집 및 학습 파이프라인**
- **내용**: VLA 구조 분류, Action Space 설정, 실제 데이터 수집, VLM/LSTM 학습 과정
- **파일**: `15_1_data_collection_training_pipeline.md`
- **라인 수**: ~1450 lines

**주요 섹션**:
- 0. VLA 구조와 Action Space 설정 (4가지 VLA 구조, Continuous vs Discrete, RoboVLMs 선택)
- 1. Real-World 데이터 수집 과정 (CALVIN, RoboVLMs 데이터셋)
- 2. VLM Fine-tuning 아키텍처 (vision_tower, text_tower, Trainable Parameters)
- 3. LSTM Decoder 아키텍처 (Policy Head 구조)
- 4. 데이터 전처리 과정 (Image, Action, Text)
- 5. VLM Fine-tuning 과정 (학습 흐름)
- 6. LSTM Layer 학습 과정 (Teacher Forcing, Hidden State)
- 7. 전체 학습 파이프라인 (Forward/Backward Pass)

---

### **15_2. 학습/추론 변수 및 벤치마크 분석**
- **내용**: 학습/추론 변수, 환경 변수, CALVIN/SimplerEnv 벤치마크 상세 분석
- **파일**: `15_2_training_inference_variables_benchmarks.md`
- **라인 수**: ~800 lines

**주요 섹션**:
- 7. 학습 변수 (window_size, fwd_pred_next_n, batch_size, learning_rate 등)
- 8. 추론 변수 (inference mode, single image processing, autoregressive generation)
- 9. 벤치마크 분석 (Real-World Experiments, CALVIN 34개 스킬, SimplerEnv)

---

### **16. Action-Image-Text Synchronization 완전 가이드**  **NEW**
- **내용**: Action/Image/Text 동기화 메커니즘, rel_action 설명, 학습 과정 Q&A
- **파일**: `16_action_image_text_sync_complete_guide.md`
- **라인 수**: ~710 lines

**주요 섹션**:
- 1. 기본 개념 이해 (action vs rel_action, VLM Finetuning vs LoRA, Embedded Token)
- 2. CALVIN 데이터셋 구조 (robot_obs, rel_actions, Window Size)
- 3. 학습 과정 (VLM과 Action Head 동시 학습, Forward Pass, Loss Function)
- 4. 실제 학습 설정 (Kosmos + LSTM Config, Trainable Parameters)
- **핵심 Q&A 요약**: 사용자 질문 11개에 대한 명확한 답변

---

##  각 문서의 목적

| **문서** | **목적** | **대상 독자** |
|---------|---------|-------------|
| **15_1** | VLA 구조 이해 + 학습 파이프라인 전체 흐름 | VLA 전반을 공부하려는 사람 |
| **15_2** | 학습/추론 설정 + 벤치마크 이해 | 실험 재현 및 성능 비교하려는 사람 |
| **16** | Action/Image/Text 동기화 메커니즘 | rel_action, 동기화 문제를 해결하려는 사람 |

---

## 📖 읽는 순서 권장

### **처음 공부하는 경우**:
1. **16번 먼저 읽기** (기본 개념 + 질문 중심)
   - action vs rel_action 이해
   - VLM Finetuning 개념
   - Token Synchronization 메커니즘
   
2. **15_1번 읽기** (전체 아키텍처 + 학습 과정)
   - VLA 4가지 구조 비교
   - VLM + LSTM 아키텍처 상세
   - 학습 파이프라인 전체 흐름
   
3. **15_2번 읽기** (실험 설정 + 벤치마크)
   - Hyperparameter 의미
   - CALVIN/SimplerEnv 평가 방법

---

### **특정 목적이 있는 경우**:

#### "rel_action이 뭐야? action과 어떻게 다른 거야?"
→ **16번 Part 1.1** 읽기

#### "VLM과 LSTM이 동시에 학습되는 거야?"
→ **16번 Part 3.3** 읽기

#### "CALVIN 34개 스킬이 정확히 뭐야?"
→ **15_2번 Section 9.2** 읽기

#### "RoboVLMs는 왜 Continuous Action을 선택했어?"
→ **15_1번 Section 0.10.4** 읽기

#### "학습할 때 window_size를 어떻게 설정해야 해?"
→ **15_2번 Section 7.1** 읽기

---

## 🔗 문서 간 연결

```
[16] Action-Image-Text Sync 완전 가이드
  ↓ (기본 개념 이해 후)
[15_1] VLM Fine-tuning과 LSTM Layer 학습
  ↓ (아키텍처 이해 후)
[15_2] 학습/추론 변수 및 벤치마크
  ↓ (실험 재현 준비 완료)
```

---

##  핵심 결론 (3개 문서 통합)

### **RoboVLMs의 선택**:
```
Policy-Head-Continuous-Action 구조
= VLM (특징 추출) + LSTM (히스토리 + 액션 예측)
+ Full Fine-tuning (전체 파라미터 학습)
+ Relative Actions (TCP frame 기준)
→ CALVIN Avg. Len. 4.49 (전체 1위)
```

### **학습 과정**:
```
Image + Text + [LRN] Token
  ↓ VLM (Multi-modal Fusion)
Fused [LRN]
  ↓ LSTM (Temporal Reasoning)
7-DOF rel_action
  ↓ Loss (MSE + BCE)
End-to-End Backpropagation
```

### **핵심 기술**:
- **rel_action**: TCP frame 기준 상대 변화량 (일반화 성능 향상)
- **[LRN] Token**: Multi-modal 정보 융합 매개체 (학습 가능)
- **End-to-End**: VLM과 LSTM 동시 학습 (최적화)

---

##  참고 자료

- RoboVLMs 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"
- CALVIN 논문: "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks"
- RoboVLMs GitHub: https://github.com/robovlms/RoboVLMs
- Hugging Face Transformers Documentation: VLM 백본 (Kosmos-2, PaliGemma 등)
