# Critical Clarifications 가이드

## 📚 디렉토리 구조

```
critical_clarifications/
├── README.md                           # 이 파일
├── robot_obs_structure.md              # robot_obs의 정확한 구조
├── world_tcp_frame_conversion.md       # World Frame vs TCP Frame 변환
├── lrn_token_processing.md             # [LRN] Token의 정확한 위치와 처리
├── simultaneous_learning.md            # "동시 학습"의 정확한 의미
├── action_chunk_processing.md          # Action Chunk의 정확한 처리
├── window_size_boundary.md             # Window Size 경계 처리 분석
└── vlm_finetuning_purpose.md           # VLM Fine-tuning 목적 분석
```

## 🎯 Critical Issues 요약

### Issue #1: robot_obs의 정확한 구조
**문제**: robot_obs를 "15차원"이라고 했지만, 실제 구성 요소를 추측으로 작성
**해결**: CALVIN 데이터셋 설정에서 명시적으로 정의된 구조 확인
- **TCP Pose (7차원)**: 위치(3) + 자세(3) + 그리퍼(1)
- **Joint Angles (7차원)**: Franka Emika Panda 7개 관절
- **Gripper Width (1차원)**: 그리퍼 너비

### Issue #2: World Frame vs TCP Frame 변환
**문제**: "World frame의 action을 TCP frame으로 변환"한다고 했지만, 왜 변환하는지 불명확
**해결**: 물리적 의미와 실제 사용 여부 명확화
- **변환 이유**: 로봇 자세와 무관한 일관된 액션 표현
- **실제 사용**: CALVIN은 이미 rel_actions로 저장되어 변환 불필요

### Issue #3: [LRN] Token의 정확한 위치와 처리
**문제**: "[LRN] 토큰이 마지막에 추가된다"고 했지만, 정확한 처리 과정 불명확
**해결**: 생성-복제-삽입-추출 과정 상세 설명
- **생성**: 단일 learnable parameter
- **복제**: 배치별로 복제
- **삽입**: EOS 토큰 앞에 삽입
- **추출**: VLM 출력의 마지막 토큰

### Issue #4: "동시 학습"의 정확한 의미
**문제**: "VLM과 LSTM이 동시에 학습된다"고 했지만, 정확히 무엇이 업데이트되는지 불명확
**해결**: Gradient flow와 실제 업데이트되는 파라미터 명확화
- **End-to-End 학습**: Loss에서 VLM까지 gradient 전파
- **Single Pass**: 한 번의 forward/backward로 모든 모듈 학습
- **Joint Optimization**: 모든 파라미터가 동시에 최적화

### Issue #5: Action Chunk의 정확한 처리
**문제**: "Action chunk를 예측한다"고 했지만, 정확한 의미와 생성 방식 불명확
**해결**: Sliding window 방식과 MPC 구조 상세 설명
- **Ground Truth**: Sliding window로 각 시간 단계마다 "현재~미래 N개" 추출
- **모델 예측**: LSTM이 각 시간 단계마다 N개 액션 동시 예측
- **MPC 구조**: Training은 병렬, Inference는 첫 번째만 실행

### Issue #6: Window Size 경계 처리
**문제**: Window size를 벗어난 시점에서 액션 예측을 어떻게 처리하는지 불명확
**해결**: 데이터 확장과 unfold 처리 방식 명확화
- **데이터 확장**: `window_size + fwd_pred_next_n` 길이로 준비
- **unfold 적용**: `fwd_pred_next_n` 크기로 sliding window 생성
- **첫 번째 제거**: Boundary effect 방지

### Issue #7: VLM Fine-tuning 목적
**문제**: LRN 토큰이 이미 학습된 VLM에서 나온다면, 무엇을 Fine-tuning하는지 불명확
**해결**: LRN 토큰의 정체와 Fine-tuning 목적 명확화
- **LRN 토큰**: 새로 추가된 learnable parameter
- **Fine-tuning 목적**: LRN 토큰의 의미 학습과 multimodal fusion
- **Domain Adaptation**: 로봇 제어 도메인에 특화

## 🔍 핵심 검증 사항

### 코드 근거 확인
모든 설명은 다음 파일에서 **직접 확인 가능**:
- `robovlms/data/calvin_dataset.py`: 데이터 구조, 전처리
- `robovlms/data/data_utils.py`: World ↔ TCP 변환
- `robovlms/model/backbone/base_backbone.py`: VLM + LSTM 통합
- `robovlms/model/policy_head/lstm_decoder.py`: LSTM 구조
- `configs/calvin_finetune/*.json`: 실험 설정

### 교수 평가 체크리스트
- [x] robot_obs의 15차원 구조가 명확하게 설명되었는가?
- [x] World ↔ TCP frame 변환의 물리적 의미가 설명되었는가?
- [x] [LRN] 토큰의 생성-복제-추출 과정이 명확한가?
- [x] "동시 학습"의 의미가 gradient flow로 설명되었는가?
- [x] Action chunk의 생성 방식이 sliding window로 명확한가?
- [x] Window size 경계 처리가 명확하게 설명되었는가?
- [x] VLM Fine-tuning의 목적이 LRN 토큰 학습으로 명확한가?
- [x] 모든 설명에 코드 출처가 명시되어 있는가?

## 📖 사용 방법

### 빠른 참조
- **robot_obs 구조**: [`robot_obs_structure.md`](robot_obs_structure.md)
- **Frame 변환**: [`world_tcp_frame_conversion.md`](world_tcp_frame_conversion.md)
- **LRN 토큰**: [`lrn_token_processing.md`](lrn_token_processing.md)
- **동시 학습**: [`simultaneous_learning.md`](simultaneous_learning.md)
- **Action Chunk**: [`action_chunk_processing.md`](action_chunk_processing.md)
- **Window Size**: [`window_size_boundary.md`](window_size_boundary.md)
- **VLM Fine-tuning**: [`vlm_finetuning_purpose.md`](vlm_finetuning_purpose.md)

### 상세 학습
각 문서는 독립적으로 읽을 수 있도록 구성되어 있으며, 상호 참조를 통해 전체적인 이해를 도울 수 있습니다.

## 🎯 핵심 결론

### 검증된 사실들
1. **robot_obs 구조**: `prop_state` config에서 명시적으로 정의됨
2. **[LRN] Token**: 단일 파라미터, 배치별 복제, 마지막 위치 추출
3. **동시 학습**: End-to-End backpropagation, 모든 파라미터 한 번에 업데이트
4. **Action Chunk**: Sliding window로 생성, LSTM에서 한 번에 예측
5. **Window Size**: 데이터 확장과 unfold 처리로 경계 문제 해결
6. **VLM Fine-tuning**: LRN 토큰의 의미 학습과 multimodal fusion

### 코드 근거
모든 설명은 실제 코드에서 **직접 확인 가능**하며, 각 문서에 출처가 명시되어 있습니다.

---

**이 문서는 RoboVLMs 프로젝트의 Critical Issues를 체계적으로 정리한 가이드입니다. 각 이슈는 독립적으로 해결되어 있으며, 코드 근거와 함께 명확하게 설명되어 있습니다.**
