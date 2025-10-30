# 📚 RoboVLMs 논문 VLA Structures 섹션 분석

> **인용**: 논문 "C. VLA Structures" 섹션

## 🎯 **1. VLA 구조 개요**

### **4가지 VLA 구조 분류**
> **인용**: "As shown in Fig. 12, there are mainly four kinds of VLA structures categorized by action space and history aggregating method, namely one-step-continuous-action models, one-step-discrete-action models, interleaved-continuous-action models, and policy-head-continuous-action models." (논문 C. VLA Structures 섹션)

#### **분류 기준**
- **액션 공간**: 연속 액션 vs 이산 액션
- **히스토리 집계 방법**: One-step vs Interleaved vs Policy Head

#### **4가지 구조**
1. **One-step-continuous-action models**: 단계별 연속 액션 모델
2. **One-step-discrete-action models**: 단계별 이산 액션 모델
3. **Interleaved-continuous-action models**: 인터리브 연속 액션 모델
4. **Policy-head-continuous-action models**: 정책 헤드 연속 액션 모델

### **RoboVLMs 프레임워크의 유연성**
> **인용**: "Note that our proposed framework RoboVLMs could transfer VLMs to arbitrary VLA structures with no effort." (논문 C. VLA Structures 섹션)

- **유연성**: VLM을 임의의 VLA 구조로 변환 가능
- **노력 없음**: 추가 노력 없이 변환 가능
- **범용성**: 다양한 VLA 구조 지원

## 🔧 **2. One-Step Models**

### **One-Step 모델의 정의**
> **인용**: "One-step models predict future action sequences using only the observation at the current time step t, i.e., a history length of 1." (논문 C. VLA Structures 섹션)

#### **특징**
- **히스토리 길이**: 1 (현재 시간 단계만 사용)
- **입력**: 현재 시간 단계 t의 관측값만 사용
- **출력**: 미래 액션 시퀀스 예측

#### **수학적 공식**
> **인용**: "ˆat:t+L−1 = VLA(ot, lprompt) , (9)" (논문 C. VLA Structures 섹션)

- **ˆat:t+L−1**: 예측된 액션 시퀀스
- **ot**: 현재 시간 단계 t의 관측값
- **lprompt**: 언어 프롬프트

### **One-Step 모델의 두 가지 변형**

#### **1) 연속 액션 모델 (Continuous-action model)**

##### **구조 및 공식**
> **인용**: "In the continuous-action formulation, the VLM model first predicts a learnable token [LRN] using the VLM backbone. This is achieved by fusing visual and language tokens (in an encoder-decoder architecture) or concatenating multi-modal tokens (in a decoder-only architecture). An MLP is then used to predict the action vector:" (논문 C. VLA Structures 섹션)

##### **공식 (10)**
> **인용**: "[LRN] = VLM(ot, lprompt) ,  ˆat:t+L−1 = MLP([LRN]) (10)" (논문 C. VLA Structures 섹션)

- **[LRN]**: 학습 가능한 토큰
- **VLM**: VLM 백본을 통한 토큰 예측
- **MLP**: 액션 벡터 예측을 위한 MLP

##### **아키텍처별 처리**
- **Encoder-Decoder**: 시각적 토큰과 언어 토큰 융합
- **Decoder-Only**: 멀티모달 토큰 연결

##### **대표 모델들**
> **인용**: "The one-step continuous-action models include ACT [53], BC-Z [19], MVP [37], R3M [34], VIMA [20], 3D Diffuser [21], RoboMamba [29], and π0 [4]." (논문 C. VLA Structures 섹션)

- **ACT [53]**: 대표적인 One-step 연속 액션 모델
- **BC-Z [19]**: 대표적인 One-step 연속 액션 모델
- **MVP [37]**: 대표적인 One-step 연속 액션 모델
- **R3M [34]**: 대표적인 One-step 연속 액션 모델
- **VIMA [20]**: 대표적인 One-step 연속 액션 모델
- **3D Diffuser [21]**: 대표적인 One-step 연속 액션 모델
- **RoboMamba [29]**: 대표적인 One-step 연속 액션 모델
- **π0 [4]**: 대표적인 One-step 연속 액션 모델

#### **2) 이산 액션 모델 (Discrete-action model)**

##### **구조 및 공식**
> **인용**: "For discrete action prediction, we directly follow the straightforward next-word prediction same as VLMs, where actions are discretized into tokens like texts:" (논문 C. VLA Structures 섹션)

##### **공식 (11)**
> **인용**: "[ACT]1:7  t:t+L−1 = VLM(ot, lprompt) , (11)" (논문 C. VLA Structures 섹션)

- **[ACT]1:7 t:t+L−1**: 7차원 액션 토큰 시퀀스
- **VLM**: VLM을 통한 직접 토큰 예측
- **방법**: VLM의 다음 단어 예측과 동일한 방식

##### **대표 모델들**
> **인용**: "The one-step discrete-action models include RT-1 [6], RT-2 [7], 3D-VLA [55], LAPA [50], OpenVLA [22], and EmbodiedCOT [52]." (논문 C. VLA Structures 섹션)

- **RT-1 [6]**: 대표적인 One-step 이산 액션 모델
- **RT-2 [7]**: 대표적인 One-step 이산 액션 모델
- **3D-VLA [55]**: 대표적인 One-step 이산 액션 모델
- **LAPA [50]**: 대표적인 One-step 이산 액션 모델
- **OpenVLA [22]**: 대표적인 One-step 이산 액션 모델
- **EmbodiedCOT [52]**: 대표적인 One-step 이산 액션 모델

## 🔄 **3. Interleaved-Continuous-Action Models**

### **Interleaved 모델의 정의**
> **인용**: "Interleaved models receive observation-action sequences:" (논문 C. VLA Structures 섹션)

#### **입력 시퀀스**
> **인용**: "Ot = ([OBS]t−H+1, [LRN]), ..., ([OBS]t, [LRN]) ,  where Ot represents the input token sequence at time instant t, [OBS] denotes observation tokens and [LRN] denotes the learnable action token and is duplicated for H times and insert into Ot with an interleaved format." (논문 C. VLA Structures 섹션)

##### **시퀀스 구성**
- **Ot**: 시간 t에서의 입력 토큰 시퀀스
- **[OBS]**: 관측 토큰
- **[LRN]**: 학습 가능한 액션 토큰
- **H번 복제**: H번 복제되어 인터리브 형식으로 삽입

#### **처리 과정**
> **인용**: "The VLM backbone fuses this sequence (in a decoder-only structure) and predicts the action sequence through an MLP based on each action token:" (논문 C. VLA Structures 섹션)

##### **공식 (12)**
> **인용**: "[LRN]t−H+1:t = VLM(Ot) ,  aˆt:t+L−1 = MLP([LRN]t) , (12)" (논문 C. VLA Structures 섹션)

- **[LRN]t−H+1:t**: 히스토리 길이 H의 학습 가능한 토큰 시퀀스
- **VLM(Ot)**: VLM 백본을 통한 시퀀스 융합
- **MLP([LRN]t)**: 각 액션 토큰을 기반으로 한 액션 시퀀스 예측

#### **액션 예측 과정**
> **인용**: "The [LRN]t which is utilized to predict the action chunk aˆt:t+L−1, represents the [LRN] inserted after [OBS]t and fused with the observations before t." (논문 C. VLA Structures 섹션)

- **[LRN]t**: 액션 청크 aˆt:t+L−1 예측에 사용되는 토큰
- **위치**: [OBS]t 이후에 삽입된 [LRN]
- **융합**: t 이전의 관측값들과 융합

#### **손실 함수 및 정규화**
> **인용**: "The loss and action unnormalization procedure is identity with one-step continuous action models." (논문 C. VLA Structures 섹션)

- **손실 함수**: One-step 연속 액션 모델과 동일
- **정규화**: 액션 비정규화 절차도 동일

#### **추론 과정**
> **인용**: "At time instant t of inference, the input sequence contains only the current observation [OBS]t and the language instruction lprompt, we add the learnable token [ACT] at the end of the input sequence and pass the sequence to the VLM to predict the action. After the robot executes the predicted action, we add the new observation [OBS]t+1 and language instruction lprompt to the input sequence to predict the action in the current step." (논문 C. VLA Structures 섹션)

##### **추론 단계**
1. **입력 구성**: 현재 관측값 [OBS]t + 언어 지시사항 lprompt
2. **토큰 추가**: 학습 가능한 토큰 [ACT]를 시퀀스 끝에 추가
3. **VLM 처리**: VLM에 시퀀스 전달하여 액션 예측
4. **로봇 실행**: 예측된 액션 실행
5. **새 관측값**: 새로운 관측값 [OBS]t+1 추가
6. **반복**: 다음 단계 예측을 위해 반복

#### **대표 모델들**
> **인용**: "The interleaved-continuous-action models include GR-1 [47], OCTO [39], GR-2 [8]. Note that the interleaved-discrete-action models like GATO [38] and RoboCat [5] are out of consideration." (논문 C. VLA Structures 섹션)

##### **포함된 모델들**
- **GR-1 [47]**: 대표적인 Interleaved 연속 액션 모델
- **OCTO [39]**: 대표적인 Interleaved 연속 액션 모델
- **GR-2 [8]**: 대표적인 Interleaved 연속 액션 모델

##### **제외된 모델들**
- **GATO [38]**: Interleaved 이산 액션 모델 (고려 대상 아님)
- **RoboCat [5]**: Interleaved 이산 액션 모델 (고려 대상 아님)

## 🎯 **4. Policy-Head-Continuous-Action Models**

### **Policy Head 모델의 정의**
> **인용**: "Unlike interleaved models, which fuse historical information within the VLM backbone, policy-head VLAs only require the VLM to provide single-step multi-modal representations at each time step t:" (논문 C. VLA Structures 섹션)

#### **Interleaved 모델과의 차이점**
- **Interleaved**: VLM 백본 내에서 히스토리 정보 융합
- **Policy Head**: VLM이 각 시간 단계 t에서 단일 단계 멀티모달 표현만 제공

#### **단일 단계 표현**
> **인용**: "ot = ([OBS]t, [LRN]) ,  [LRN]t = VLM(ot, lprompt) (13)" (논문 C. VLA Structures 섹션)

##### **공식 (13) 설명**
- **ot**: 시간 t에서의 입력
- **[OBS]t**: 시간 t의 관측 토큰
- **[LRN]**: 학습 가능한 토큰
- **[LRN]t**: VLM을 통한 단일 단계 표현

#### **정책 헤드를 통한 히스토리 모델링**
> **인용**: "Historical information is then modeled and actions are predicted through an additional policy head h, such as an RNN [10, 15, 30], transformer [14, 43], or diffusion model [9]:" (논문 C. VLA Structures 섹션)

##### **공식 (14)**
> **인용**: "at:t+L−1 = h([LRN]t−H+1, ..., [LRN]t) (14)" (논문 C. VLA Structures 섹션)

- **h**: 정책 헤드 (RNN, Transformer, Diffusion model 등)
- **[LRN]t−H+1, ..., [LRN]t**: 히스토리 길이 H의 학습 가능한 토큰 시퀀스
- **at:t+L−1**: 시퀀스 길이 L의 액션 청크

#### **정책 헤드 옵션**
- **RNN [10, 15, 30]**: 순환 신경망
- **Transformer [14, 43]**: 트랜스포머
- **Diffusion model [9]**: 확산 모델

#### **아키텍처별 차이점**

##### **Decoder-Only 백본**
> **인용**: "Note that the interleaved-continuous-action model is only available for decoder-only backbones. The policy-head-continuous-action model can be built based on VLM backbones with both encoder-decoder and decoder-only structures." (논문 C. VLA Structures 섹션)

- **Interleaved**: Decoder-only 백본에서만 사용 가능
- **Policy Head**: Encoder-decoder와 Decoder-only 구조 모두 지원

##### **Encoder-Decoder vs Decoder-Only**

###### **Encoder-Decoder 구조**
> **인용**: "The input sequence of the encoder-decoder VLM fuses only contains the text and learnable action tokens, it fuses the multi-modal input with cross-attention where the text tokens combined with the learnable tokens are the keys and values, and vision tokens are the queries." (논문 C. VLA Structures 섹션)

- **입력**: 텍스트와 학습 가능한 액션 토큰만 포함
- **융합**: Cross-attention을 통한 멀티모달 입력 융합
- **Keys/Values**: 텍스트 토큰 + 학습 가능한 토큰
- **Queries**: 비전 토큰

###### **Decoder-Only 구조**
> **인용**: "The decoder-only backbone directly concatenates the vision, language, and learnable tokens as input and utilizes self-attention to fuse the multi-modal features." (논문 C. VLA Structures 섹션)

- **입력**: 비전, 언어, 학습 가능한 토큰을 직접 연결
- **융합**: Self-attention을 통한 멀티모달 특징 융합

#### **대표 모델들**
> **인용**: "The policy-head-continuous-action models include RoboFlamingo [24], RoboUniview [27], and DeeRVLA [51]." (논문 C. VLA Structures 섹션)

- **RoboFlamingo [24]**: 대표적인 Policy Head 연속 액션 모델
- **RoboUniview [27]**: 대표적인 Policy Head 연속 액션 모델
- **DeeRVLA [51]**: 대표적인 Policy Head 연속 액션 모델

#### **추론 과정**
> **인용**: "At every inference step t, the current observation [OBS]t and language instruction lprompt along with a learnable token [LRN] is concatenated as a complete input sequence, which is further passed into the VLM backbone. After the policy head takes [LRN] and predicts the current action sequences, the robot steps with the predicted actions and obtains the new observation for the next round of prediction." (논문 C. VLA Structures 섹션)

##### **추론 단계**
1. **입력 구성**: 현재 관측값 [OBS]t + 언어 지시사항 lprompt + 학습 가능한 토큰 [LRN]
2. **VLM 처리**: VLM 백본에 입력 시퀀스 전달
3. **정책 헤드**: [LRN]을 받아 현재 액션 시퀀스 예측
4. **로봇 실행**: 예측된 액션으로 로봇 실행
5. **새 관측값**: 새로운 관측값 획득
6. **반복**: 다음 예측을 위해 반복

## 🔍 **5. VLA 구조별 비교 분석**

### **구조별 특징 비교**

| 구조 | 히스토리 처리 | 액션 공간 | VLM 백본 | 대표 모델 |
|------|---------------|-----------|----------|-----------|
| **One-Step-Continuous** | 현재 관측값만 | 연속 | Encoder-Decoder/Decoder-Only | ACT, BC-Z, MVP, R3M, VIMA, 3D Diffuser, RoboMamba, π0 |
| **One-Step-Discrete** | 현재 관측값만 | 이산 | Decoder-Only | RT-1, RT-2, 3D-VLA, LAPA, OpenVLA, EmbodiedCOT |
| **Interleaved-Continuous** | 인터리브 시퀀스 | 연속 | Decoder-Only | GR-1, OCTO, GR-2 |
| **Policy-Head-Continuous** | 정책 헤드 | 연속 | Encoder-Decoder/Decoder-Only | RoboFlamingo, RoboUniview, DeeRVLA |

### **히스토리 처리 방법 비교**

#### **One-Step Models**
- **특징**: 현재 관측값만 사용
- **장점**: 단순한 구조, 빠른 추론
- **단점**: 히스토리 정보 활용 제한

#### **Interleaved Models**
- **특징**: 인터리브된 관측-액션 시퀀스
- **장점**: 히스토리 정보의 효과적 활용
- **단점**: Decoder-only 백본에서만 사용 가능

#### **Policy Head Models**
- **특징**: 전용 정책 헤드를 통한 히스토리 모델링
- **장점**: 유연한 히스토리 처리, 다양한 백본 지원
- **단점**: 추가 정책 헤드 필요

### **액션 공간별 비교**

#### **연속 액션**
- **장점**: 높은 정밀도, 직접적인 제어
- **단점**: VLM과의 호환성 제한
- **훈련**: MSE + BCE 손실

#### **이산 액션**
- **장점**: VLM 토크나이저와 호환
- **단점**: 정밀도 제한, 전처리/후처리 필요
- **훈련**: 교차 엔트로피 손실

## 🎯 **6. 실제 로봇 플랫폼**

### **Figure 13: 실제 로봇 플랫폼**
> **인용**: "Fig. 13: The demonstration of our real robot platform. The platform is equipped with a side camera and a wrist camera." (논문 C. VLA Structures 섹션)

#### **플랫폼 구성**
- **로봇**: Kinova Gen 3
- **카메라**: Side Camera + Wrist Camera
- **작업 공간**: 다양한 가정용 객체들

#### **카메라 시스템**
- **Side Camera**: 로봇 상단에 위치한 전체 작업 공간 관측
- **Wrist Camera**: 엔드 이펙터에 위치한 상호작용 영역 관측
- **이중 관측**: 전체 맥락과 상세 상호작용 모두 포착

## 🚀 **7. VLA 구조의 발전 방향**

### **구조적 발전**
1. **하이브리드 구조**: 다양한 구조의 장점 결합
2. **효율적 히스토리**: 더 효율적인 히스토리 처리 방법
3. **전용 아키텍처**: 로봇 제어에 특화된 구조

### **기술적 발전**
1. **정책 헤드**: RNN, Transformer, Diffusion model 등 다양한 선택지
2. **멀티모달 융합**: Cross-attention, Self-attention 등 다양한 융합 방법
3. **액션 표현**: 연속/이산 액션의 장점을 결합한 새로운 표현

### **응용 확장**
1. **다양한 로봇**: 다양한 로봇 플랫폼 적용
2. **복잡한 작업**: 더 복잡한 로봇 작업 처리
3. **실시간 제어**: 실시간 로봇 제어 최적화

## 🎯 **8. 결론**

### **VLA 구조의 핵심 가치**
1. **유연한 구조**: 다양한 구조적 선택지 제공
2. **효율적 처리**: 히스토리 정보의 효과적 활용
3. **멀티모달 통합**: 시각, 언어, 액션의 효과적 통합

### **구조 선택의 기준**
1. **작업 복잡도**: 작업의 복잡도에 따른 구조 선택
2. **데이터 특성**: 사용 가능한 데이터의 특성
3. **성능 요구사항**: 성능과 효율성의 균형

### **RoboVLMs의 장점**
1. **유연성**: VLM을 임의의 VLA 구조로 변환 가능
2. **범용성**: 다양한 VLA 구조 지원
3. **편의성**: 추가 노력 없이 변환 가능

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
