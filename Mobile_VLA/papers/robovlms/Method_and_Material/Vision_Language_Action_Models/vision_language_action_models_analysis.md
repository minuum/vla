# 📚 RoboVLMs 논문 Vision-Language-Action Models 섹션 분석

> **인용**: 논문 "B. Vision-Language-Action Models" 섹션

## 🎯 **1. VLA 개요 및 정의**

### **VLA의 정의**
> **인용**: "Vision-language-action models (VLAs) are predominantly applied to robotic tasks, where they serve as a generalist robot policy π capable of handling complicated tasks." (논문 B. Vision-Language-Action Models 섹션)

#### **VLA의 특징**
- **주요 적용**: 로봇 작업에 주로 적용
- **역할**: 일반화 로봇 정책 π 역할
- **능력**: 복잡한 작업 처리 가능
- **목적**: 로봇 제어를 위한 멀티모달 모델

### **VLA의 수학적 공식화**
> **인용**: "In formal, VLAs predict action sequences based on previous observations at the current time step t: at:t+L−1 = VLA(ot−H+1:t, lprompt) , (4)" (논문 B. Vision-Language-Action Models 섹션)

#### **공식 (4) 설명**
- **at:t+L−1**: 예측된 액션 시퀀스
- **ot−H+1:t**: 이전 관측값들
- **lprompt**: 텍스트 프롬프트
- **VLA**: Vision-Language-Action Model 함수

#### **매개변수 설명**
> **인용**: "where at:t+L−1 are a sequence of predicted 7-dim actions, L is the action sequence length and H is the history observation length." (논문 B. Vision-Language-Action Models 섹션)

- **at:t+L−1**: 예측된 7차원 액션 시퀀스
- **L**: 액션 시퀀스 길이
- **H**: 히스토리 관측 길이

### **VLA vs VLM의 차이점**
> **인용**: "Different from VLMs, the observations of VLAs ot−H+1:t usually contain proprioceptive states st−H+1:t like the joint angles and end-effector positions besides the visual inputs It−H+1:t." (논문 B. Vision-Language-Action Models 섹션)

#### **VLA 관측값의 특징**
- **Proprioceptive states**: st−H+1:t (관절 각도, 엔드 이펙터 위치)
- **Visual inputs**: It−H+1:t (시각적 입력)
- **차이점**: VLM과 달리 proprioceptive 상태 포함

## 🏗️ **2. VLA 구조 분류**

### **구조 분류 기준**
> **인용**: "As is discussed in Section IV-B, we abstract and categorized VLAs into four representative structures based on 1) historical information modeling and 2) action space." (논문 B. Vision-Language-Action Models 섹션)

#### **분류 기준**
1. **Historical information modeling**: 히스토리 정보 모델링
2. **Action space**: 액션 공간

#### **4가지 대표 구조**
- **One-Step-Continuous-Action Models**: 단계별 연속 액션 모델
- **Interleaved-Continuous-Action-Models**: 인터리브 연속 액션 모델
- **One-Step-Discrete-Action Models**: 단계별 이산 액션 모델
- **Policy-Head-Continuous-Action-Models**: 정책 헤드 연속 액션 모델

### **Figure 12: VLA 구조 도식화**

> **인용**: "Fig. 12: The illustration of considered VLA formulations, including several popular designs. For example, RoboFlamingo [24] is a Policy-Head-Continuous-type VLA, RT-2 [7] and OpenVLA [22] corresponds to the One-Step-Discrete-Actiontype VLA. Octo [39] and GR [47] correspond to the Interleaved-Continuous-Action-type VLA with a fixed window size." (논문 B. Vision-Language-Action Models 섹션)

#### **대표 모델들**
- **RoboFlamingo [24]**: Policy-Head-Continuous-type VLA
- **RT-2 [7]**: One-Step-Discrete-Action-type VLA
- **OpenVLA [22]**: One-Step-Discrete-Action-type VLA
- **Octo [39]**: Interleaved-Continuous-Action-type VLA
- **GR [47]**: Interleaved-Continuous-Action-type VLA

## 🔧 **3. 액션 전처리 (Action Pre-process)**

### **액션 정규화 (Action Normalization)**
> **인용**: "For both continuous and discrete action spaces, we normalize each dimension of the 7-DoF action. Following Kim et al. [22], we count the 1st and 99th quantile of the actions in the training data and use the quantiles to clamp each dimension of the action [7]:" (논문 B. Vision-Language-Action Models 섹션)

#### **정규화 과정**
- **대상**: 연속 및 이산 액션 공간 모두
- **방법**: 7-DoF 액션의 각 차원 정규화
- **기준**: 훈련 데이터의 1번째와 99번째 분위수 사용

#### **공식 (5): 액션 클램핑**
> **인용**: "ai′ = min(ai  99th , max(ai  1st , ai)) (5)" (논문 B. Vision-Language-Action Models 섹션)

- **ai′**: 클램핑된 i번째 차원의 액션 값
- **ai**: 원본 i번째 차원의 액션 값
- **ai 1st**: 1번째 분위수
- **ai 99th**: 99번째 분위수

#### **공식 (6): 액션 정규화**
> **인용**: " ̃ai = 2 × (ai′ − ai  1st )/(ai  99th − ai  1st ) − 1 (6)" (논문 B. Vision-Language-Action Models 섹션)

- ** ̃ai**: 정규화된 i번째 차원의 액션
- **범위**: [-1, 1] 범위로 정규화
- **그리퍼 상태**: 마지막 차원은 그리퍼 열기/닫기 상태 ∈ {-1, 1}

#### **정규화된 액션**
> **인용**: " ̃a = [a ̃1, a ̃2, · · · , a ̃7] is the normalized action, each dimension is in the range of [−1, 1], the last dimension representing the open/close status of the gripper ∈ {−1, 1}." (논문 B. Vision-Language-Action Models 섹션)

- ** ̃a**: 정규화된 액션 벡터
- **차원**: 7차원
- **범위**: 각 차원이 [-1, 1] 범위
- **그리퍼**: 마지막 차원은 그리퍼 상태

### **액션 이산화 (Action Discretization)**
> **인용**: "For discrete action representation, we need to further discretize the normalized action  ̃a. Following Brohan et al. [7], Kim et al. [22], we map continuous robot actions to discrete tokens used by the VLM's tokenizer." (논문 B. Vision-Language-Action Models 섹션)

#### **이산화 과정**
- **목적**: 이산 액션 표현을 위해
- **방법**: 정규화된 액션  ̃a를 추가로 이산화
- **매핑**: 연속 로봇 액션을 VLM 토크나이저가 사용하는 이산 토큰으로 매핑

#### **이산화 방법**
> **인용**: "Specifically, we discretize each robot action dimension into one of 256 bins separately. For each dimension, we set the width of the bin to uniformly divide the interval between the quantile of 1st and 99th of actions in the training data." (논문 B. Vision-Language-Action Models 섹션)

- **빈 수**: 각 차원을 256개 빈 중 하나로 이산화
- **빈 너비**: 1번째와 99번째 분위수 간 구간을 균등하게 분할
- **방법**: 각 차원별로 독립적으로 이산화

#### **토큰 변환**
> **인용**: "Using this discretization, we transform  ̃a to a with 7 discrete integers ∈ [0 . . . 255]. To avoid damaging the original special token positions in the language tokenizer, we add an offset (default set to 10) and replace the last offset ∼ 256+offset tokens with a discretized index." (논문 B. Vision-Language-Action Models 섹션)

- **변환**:  ̃a를 7개의 이산 정수로 변환
- **범위**: [0...255] 범위의 7개 이산 정수
- **오프셋**: 기본값 10의 오프셋 추가
- **토큰 교체**: 마지막 오프셋 ~ 256+오프셋 토큰을 이산화된 인덱스로 교체

## 🎯 **4. 액션 예측 (Action Prediction)**

### **연속 액션 (Continuous Actions)**
> **인용**: "We optimize the mean square error (MSE) and binary cross entropy (BCE) for the predicted action sequence at:t+L−1 with the ground truth action sequence  ̃at:t+L−1:" (논문 B. Vision-Language-Action Models 섹션)

#### **손실 함수**
> **인용**: "lVLA =  t+L−1  X  i=t  MSE(ˆai,pose,  ̃ai,pose) + λ ∗ BCE(ai,gripper,  ̃ai,gripper) (7)" (논문 B. Vision-Language-Action Models 섹션)

##### **공식 (7) 설명**
- **lVLA**: VLA 훈련 손실
- **MSE**: 평균 제곱 오차 (첫 6개 차원)
- **BCE**: 이진 교차 엔트로피 (마지막 그리퍼 차원)
- **λ**: 균형 가중치

##### **손실 함수 구성**
- **MSE 손실**: 첫 6개 차원에 대해 계산
- **BCE 손실**: 마지막 그리퍼 차원에 대해 계산
- **예측**: ˆat:t+L−1 (예측된 액션 시퀀스)
- **정답**:  ̃at:t+L−1 (정답 액션 시퀀스)

### **이산 액션 (Discrete Actions)**
> **인용**: "Discrete-action models predict action tokens ACTi for each action dimension i. These tokens are the index of discretized bins from continuous action by dimension, which can be easily de-tokenized to recover the action vector." (논문 B. Vision-Language-Action Models 섹션)

#### **이산 액션 예측**
- **예측 대상**: 각 액션 차원 i에 대한 액션 토큰 ACTi
- **토큰 의미**: 차원별 연속 액션에서 이산화된 빈의 인덱스
- **역변환**: 쉽게 역토큰화하여 액션 벡터 복원 가능

#### **손실 함수**
> **인용**: "The optimization object has a similar cross-entropy (CE) format as text generation widely used in VLM training: lVLA =  t+L−1  X  i=t  7  X  j=1  CE([ACT]j  i ,  ̃aj  i ) (8)" (논문 B. Vision-Language-Action Models 섹션)

##### **공식 (8) 설명**
- **lVLA**: VLA 훈련 손실
- **CE**: 교차 엔트로피 (텍스트 생성과 유사한 형식)
- **[ACT]j i**: 시간 i에서 예측된 액션 토큰 [ACT]의 j번째 차원의 빈 인덱스
- ** ̃aj i**: 해당하는 정답

##### **손실 함수 구성**
- **이중 합**: 시간 t부터 t+L-1까지, 차원 1부터 7까지
- **교차 엔트로피**: 각 차원별로 교차 엔트로피 계산
- **VLM 훈련**: 텍스트 생성에서 널리 사용되는 형식과 유사

#### **추론 시 역변환**
> **인용**: "During inference time, after getting the predicted action token ACTi, we re-project the discrete tokens to the center of the corresponding bins into a continuous form for achieving tasks." (논문 B. Vision-Language-Action Models 섹션)

- **과정**: 예측된 액션 토큰 ACTi 획득 후
- **역변환**: 이산 토큰을 해당 빈의 중심으로 재투영
- **결과**: 작업 수행을 위한 연속 형태로 변환

## 🔍 **5. VLA 구조별 상세 분석**

### **One-Step-Continuous-Action Models**
- **특징**: 단일 단계에서 연속 액션 예측
- **입력**: 현재 관측값과 히스토리
- **출력**: 연속 액션 시퀀스
- **장점**: 단순한 구조, 빠른 추론

### **Interleaved-Continuous-Action-Models**
- **특징**: 인터리브된 히스토리 정보 처리
- **입력**: 인터리브된 텍스트-비전 시퀀스
- **출력**: 연속 액션
- **장점**: 히스토리 정보의 효과적 활용
- **예시**: Octo [39], GR [47]

### **One-Step-Discrete-Action Models**
- **특징**: 단일 단계에서 이산 액션 예측
- **입력**: 현재 관측값과 히스토리
- **출력**: 이산 액션 토큰
- **장점**: VLM 토크나이저와의 호환성
- **예시**: RT-2 [7], OpenVLA [22]

### **Policy-Head-Continuous-Action-Models**
- **특징**: 전용 정책 헤드를 통한 연속 액션 예측
- **입력**: VLM 출력과 히스토리 정보
- **출력**: 연속 액션
- **장점**: 히스토리 정보의 효과적 집계
- **예시**: RoboFlamingo [24]

## 📊 **6. 액션 공간별 비교**

### **연속 액션 vs 이산 액션**

| 특징 | 연속 액션 | 이산 액션 |
|------|-----------|-----------|
| **표현력** | 높은 정밀도 | 제한된 정밀도 |
| **훈련** | MSE + BCE 손실 | 교차 엔트로피 손실 |
| **추론** | 직접 사용 | 역토큰화 필요 |
| **호환성** | VLM과 제한적 | VLM 토크나이저와 호환 |
| **복잡도** | 상대적으로 단순 | 전처리/후처리 필요 |

### **손실 함수 비교**

#### **연속 액션 손실**
- **MSE**: 포즈 차원 (6차원)에 대한 평균 제곱 오차
- **BCE**: 그리퍼 차원 (1차원)에 대한 이진 교차 엔트로피
- **균형**: λ 가중치로 두 손실 균형

#### **이산 액션 손실**
- **CE**: 각 차원별 교차 엔트로피
- **형식**: VLM 텍스트 생성과 유사
- **범위**: 256개 빈에 대한 분류

## 🎯 **7. VLA의 핵심 특징**

### **멀티모달 처리**
1. **시각적 입력**: It−H+1:t (이미지 시퀀스)
2. **텍스트 입력**: lprompt (언어 지시사항)
3. **Proprioceptive 입력**: st−H+1:t (로봇 상태)

### **히스토리 모델링**
1. **One-Step**: 현재 관측값만 사용
2. **Interleaved**: 인터리브된 히스토리 시퀀스
3. **Policy Head**: 전용 헤드를 통한 히스토리 집계

### **액션 예측**
1. **연속 액션**: 직접적인 연속값 예측
2. **이산 액션**: 토큰 기반 이산값 예측
3. **시퀀스**: 다중 시간 단계 액션 예측

## 🚀 **8. VLA의 발전 방향**

### **구조적 발전**
1. **하이브리드 구조**: 연속/이산 액션의 장점 결합
2. **효율적 히스토리**: 더 효율적인 히스토리 모델링
3. **전용 아키텍처**: 로봇 제어에 특화된 구조

### **훈련 방법론**
1. **손실 함수**: 더 효과적인 손실 함수 설계
2. **정규화**: 액션 정규화 방법 개선
3. **이산화**: 더 정밀한 이산화 방법

### **응용 확장**
1. **다양한 로봇**: 다양한 로봇 플랫폼 적용
2. **복잡한 작업**: 더 복잡한 로봇 작업 처리
3. **실시간 제어**: 실시간 로봇 제어 최적화

## 🎯 **9. 결론**

### **VLA의 핵심 가치**
1. **멀티모달 통합**: 시각, 언어, 액션의 효과적 통합
2. **일반화 능력**: 다양한 로봇 작업에 대한 일반화
3. **유연한 구조**: 다양한 구조적 선택지 제공

### **구조 선택의 기준**
1. **작업 복잡도**: 작업의 복잡도에 따른 구조 선택
2. **데이터 특성**: 사용 가능한 데이터의 특성
3. **성능 요구사항**: 성능과 효율성의 균형

### **미래 발전 방향**
1. **전문화된 구조**: 로봇 제어에 특화된 구조 개발
2. **효율적 훈련**: 더 효율적인 훈련 방법론
3. **실시간 적용**: 실시간 로봇 제어 최적화

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
