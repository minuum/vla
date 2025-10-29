# RoboVLMs 핵심 개념 가이드

## 📚 디렉토리 구조

```
core/
├── concepts/           # 핵심 개념
│   └── action_synchronization.md
├── architecture/       # 아키텍처
│   └── vlm_lstm_integration.md
├── data_flow/          # 데이터 플로우
│   └── calvin_dataset_flow.md
├── training/           # 학습 과정
│   └── end_to_end_learning.md
└── README.md          # 이 파일
```

## 🎯 핵심 질문과 답변

### Q1. VLM Finetuning (F-FT)과 LoRA는 무엇인가?

**A**: 
- **Full-FT**: VLM 전체 파라미터 재학습 (RoboVLMs 사용)
- **LoRA**: 저차원 행렬만 학습 (메모리 효율적, 성능 약간 낮음)

**자세한 내용**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md#2-vlm-fine-tuning-fft과-lora)

### Q2. action과 rel_action은 어떻게 다르고, 어떻게 동기화되는가?

**A**: 
- **action**: World frame 절대 좌표
- **rel_action**: TCP frame 상대 변화량 (RoboVLMs 사용)
- **변환**: `world_to_tcp_frame()` 함수로 변환

**자세한 내용**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md#1-action-vs-rel_action-핵심-차이점)

### Q3. 7-DOF 로봇팔 움직임이 어떻게 표현되고 학습되는가?

**A**: Translation(3) + Rotation(3) + Gripper(1) = 7차원 벡터

**자세한 내용**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md#12-7-dof-상대-액션-구조)

### Q4. Image, Text, Action이 어떻게 동시에 학습되는가?

**A**: 
- 모두 **Token**으로 변환 → VLM Attention으로 융합
- **[LRN] Token**이 Multi-modal 정보 통합

**자세한 내용**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md#3-embedded-token-multi-modal-fusion의-핵심)

### Q5. Embedded Token이 무엇이고 어떻게 동기화되는가?

**A**: 
- **[LRN]**: 학습 가능한 Action Token
- VLM을 통과하며 Image + Text 정보를 융합

**자세한 내용**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md#32-action-token-lrn-embedding으로-multi-modal-정보-융합)

### Q6. CALVIN 데이터셋은 어떻게 구성되어 있는가?

**A**: 
- Image(2개) + Text + robot_obs(15차원) + rel_actions(7차원)
- 24K demonstrations, 34 basic skills

**자세한 내용**: [`data_flow/calvin_dataset_flow.md`](data_flow/calvin_dataset_flow.md#1-calvin-데이터셋-개요)

### Q7. 실제 학습 과정에서 VLM과 Action Head는 동시에 학습되는가?

**A**: **예!** End-to-End로 모든 파라미터 동시 학습

**자세한 내용**: [`training/end_to_end_learning.md`](training/end_to_end_learning.md#1-학습-파이프라인-전체-흐름)

## 🏗️ 전체 아키텍처 개요

### 시스템 구성도

```
Input Data
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Multi-modal Input                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Image     │  │    Text     │  │  [LRN]      │    │
│  │ (2 cameras) │  │ (language)  │  │ (learnable) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                  VLM Backbone                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Vision    │  │    Text     │  │  Attention  │    │
│  │  Encoder    │  │  Encoder    │  │   Layers    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                Multi-modal Fusion                       │
│              (Self-Attention Mechanism)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│              Fused [LRN] Token Output                   │
│            (Image + Text + Action Info)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                Policy Head (LSTM)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   LSTM      │  │   Linear    │  │   Output    │    │
│  │  (History)  │  │   Layers    │  │ (7-DOF)     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                7-DOF Action Prediction                  │
│        [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]      │
└─────────────────────────────────────────────────────────┘
```

**자세한 내용**: [`architecture/vlm_lstm_integration.md`](architecture/vlm_lstm_integration.md#1-전체-아키텍처-개요)

## 📊 데이터 플로우

### CALVIN 데이터셋 처리

```
CALVIN Dataset
    ↓
Episode Loading (24K demonstrations)
    ↓
Multi-modal Data Extraction
    ├── RGB Images (rgb_static, rgb_gripper)
    ├── Robot State (robot_obs)
    ├── Actions (rel_actions)
    └── Language (language)
    ↓
Data Preprocessing
    ├── Image: Resize, Normalize, Augmentation
    ├── Action: Normalize [-1, 1]
    └── Language: Tokenization
    ↓
Sequence Sampling (window_size=8)
    ↓
Batch Creation (batch_size=8)
    ↓
Model Input
```

**자세한 내용**: [`data_flow/calvin_dataset_flow.md`](data_flow/calvin_dataset_flow.md#8-데이터-플로우-요약)

## 🔄 학습 과정

### End-to-End 학습 플로우

```
1. 데이터 로드
   ↓
2. Image → Vision Tokens (VLM Vision Encoder)
   ↓
3. Text → Text Tokens (VLM Tokenizer)
   ↓
4. [LRN] Token 추가
   ↓
5. Multi-modal Fusion (VLM Backbone)
   ↓
6. [LRN] Token 추출
   ↓
7. LSTM에 [LRN] 입력
   ↓
8. 7-DOF Action 예측
   ↓
9. Loss 계산 (MSE + BCE)
   ↓
10. Backpropagation (VLM + LSTM 동시 업데이트)
```

**자세한 내용**: [`training/end_to_end_learning.md`](training/end_to_end_learning.md#1-학습-파이프라인-전체-흐름)

## 🎯 핵심 개념 요약

### 1. Action Synchronization
- **절대 액션 vs 상대 액션**: World frame vs TCP frame
- **7-DOF 표현**: Translation(3) + Rotation(3) + Gripper(1)
- **정규화**: [-1, 1] 범위로 클리핑

### 2. VLM Integration
- **Multi-modal Fusion**: Text + Vision + Action 토큰 융합
- **[LRN] Token**: 학습 가능한 액션 토큰
- **End-to-End 학습**: VLM과 LSTM 동시 학습

### 3. Data Processing
- **CALVIN 데이터셋**: 24K demonstrations, 34 skills
- **전처리**: 이미지 정규화, 액션 정규화, 토큰화
- **시퀀스 처리**: 8프레임 윈도우 크기

### 4. Training Process
- **Loss Function**: MSE (pose) + BCE (gripper)
- **Gradient Flow**: Loss → LSTM → VLM → Vision/Text Encoder
- **파라미터 업데이트**: 모든 모듈 동시 학습

## 📚 문서 가이드

### 개념 이해 순서
1. **`concepts/action_synchronization.md`**: 기본 개념 이해
2. **`architecture/vlm_lstm_integration.md`**: 아키텍처 구조 파악
3. **`data_flow/calvin_dataset_flow.md`**: 데이터 처리 과정
4. **`training/end_to_end_learning.md`**: 학습 과정 상세

### 빠른 참조
- **Action vs Rel_Action**: [`concepts/action_synchronization.md#1`](concepts/action_synchronization.md#1-action-vs-rel_action-핵심-차이점)
- **VLM + LSTM 구조**: [`architecture/vlm_lstm_integration.md#1`](architecture/vlm_lstm_integration.md#1-전체-아키텍처-개요)
- **CALVIN 데이터셋**: [`data_flow/calvin_dataset_flow.md#1`](data_flow/calvin_dataset_flow.md#1-calvin-데이터셋-개요)
- **학습 과정**: [`training/end_to_end_learning.md#1`](training/end_to_end_learning.md#1-학습-파이프라인-전체-흐름)

## 🔗 관련 파일

### 코드 파일
- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: 기본 VLM + LSTM 통합
- `RoboVLMs/robovlms/data/calvin_dataset.py`: CALVIN 데이터 로더
- `RoboVLMs/robovlms/data/data_utils.py`: 데이터 유틸리티 함수
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 학습 설정

### 설정 파일
- `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json`: Kosmos-2 Full-FT 설정
- `RoboVLMs/configs/calvin_finetune/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json`: PaLI-Gemma Full-FT 설정

## 📖 논문 참조

- **RoboVLMs 논문**: Section B (VLA Models), Section C (VLA Structures)
- **CALVIN 논문**: "CALVIN: A Benchmark for Multimodal Language-Conditioned Imitation Learning for Long-Horizon Robot Manipulation Tasks"

## 🚀 시작하기

1. **개념 이해**: [`concepts/action_synchronization.md`](concepts/action_synchronization.md)부터 시작
2. **아키텍처 파악**: [`architecture/vlm_lstm_integration.md`](architecture/vlm_lstm_integration.md)로 구조 이해
3. **데이터 처리**: [`data_flow/calvin_dataset_flow.md`](data_flow/calvin_dataset_flow.md)로 데이터 플로우 파악
4. **학습 과정**: [`training/end_to_end_learning.md`](training/end_to_end_learning.md)로 학습 원리 이해

---

**이 문서는 RoboVLMs 프로젝트의 핵심 개념들을 체계적으로 정리한 가이드입니다. 각 섹션은 독립적으로 읽을 수 있도록 구성되어 있으며, 상호 참조를 통해 전체적인 이해를 도울 수 있습니다.**
