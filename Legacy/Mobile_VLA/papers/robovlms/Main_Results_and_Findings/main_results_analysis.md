# RoboVLMs 논문 Main Results and Findings 분석

> 인용: 논문 II. MAIN RESULTS AND FINDINGS 섹션
> 

## 🎯 **주요 결과 및 발견사항 개요**

### **연구 목표 및 접근법**

> 인용: "The primary objective of this work is to establish VLAs as robust generalist robotic policies by thoroughly analyzing contemporary VLA architectures and identifying the key factors driving their performance." (논문 II 섹션)
> 
- **목표**: VLA를 강건한 일반화 로봇 정책으로 확립
- **방법**: 현대 VLA 아키텍처의 철저한 분석
- **핵심**: 성능을 이끄는 핵심 요인 식별

### **RoboVLMs 프레임워크 소개**

> 인용: "To this end, we introduce RoboVLMs, a unified VLA framework that enables the systematic integration and exploration of essential components in VLA design." (논문 II 섹션)
> 
- **통합 프레임워크**: VLA 설계의 핵심 구성요소 통합
- **체계적 탐색**: 필수 구성요소의 체계적 탐색 가능
- **유연한 설계**: 다양한 VLA 설계 옵션 지원

## **Table I: 연구 질문과 답변 요약**

> 인용: "TABLE I: The performance of the built VLAs based on VLMs with different image token numbers and VL pre-train data scales. The first three rows are flamingo backbones with encoder-decoder structures, the rest backbones are decoder-only structures. Note that for VLMs with multi-stage training, the data scale refers to the data amount utilized for the final stage of fine-tuning. 'UNK' denotes unknown." (논문 Table I)
> 

### **연구 질문과 답변 체계**

| Essential Questions | Research Questions | Research Findings |
| --- | --- | --- |
| **Q1: Why VLAs** | Q1.1: Are VLAs a proper choice for building generalist robot policies? | A1.1: VLA is a promising path towards generalist robot policies. |
|  | Q1.2: How does the best VLA built with RoboVLMs perform in real-world scenarios? | A1.2: The best setup VLA built with RoboVLMs appears strong effectiveness and robustness in real-world scenarios. |
| **Q2: Which Backbone** | Q2.1: Which type of VLMs are more suitable for constructing VLAs? | A2.1: Sufficient vision-language pre-trained on large vision-language datasets benefit VLAs |
| **Q3: How to Formulate** | Q3.1: What is the best-performing VLA structure? | A3.1: Continuous action space with policy head to integrate history is the best structure. |
|  | Q3.2: How do different formulations affect the generalization and data efficiency for VLAs? | A3.2: The KosMos backbone with a separate policy head for history fusion performs the best in generalization and data efficiency. |
| **Q4: When to Leverage Extra Data** | Q4.1: How do large-scale cross-embodiment datasets contribute to VLAs? | A4.1: Extra in-domain data shows beneficial; 2) Post-training further improves overall as well as few-shot performance. |

### **Table I의 핵심 내용**

- **VLM 백본 비교**: 서로 다른 이미지 토큰 수와 VL 사전 훈련 데이터 규모를 가진 VLM 기반 VLA 성능
- **구조적 차이**: 첫 3개 행은 인코더-디코더 구조의 Flamingo 백본, 나머지는 디코더 전용 구조
- **훈련 데이터**: 다단계 훈련 VLM의 경우 최종 미세 조정 단계 데이터 양 기준
- **알 수 없는 정보**: "UNK"로 표시

## **4가지 핵심 연구 질문과 답변**

**Research Questions : $Q_{n.m}$**

**Research Findings : $A_{n.m}$**

### **Q1: Why do we prefer VLAs?**

> 인용: "1) Why do we prefer VLAs?" (논문 II 섹션)

**상세 분석**: [Why do we prefer VLAs?](./Why_do_we_prefer_VLAs/why_do_we_prefer_vlas.md)에서 자세한 분석 확인

### **Q1.1: Are VLAs a proper choice for building generalist robot policies?**

> 인용: "A1.1: VLA is a promising path towards generalist robot policies." (Table I)
> 
- **답변**: VLA는 일반화 로봇 정책을 구축하는 유망한 경로
- **근거**: 사전 훈련된 VLMs의 강력한 비전-언어 표현 능력 활용

### **Q1.2: How does the best VLA built with RoboVLMs perform in real-world scenarios?**

> 인용: "A1.2: The best setup VLA built with RoboVLMs appears strong effectiveness and robustness in real-world scenarios." (Table I)
> 
- **답변**: RoboVLMs로 구축된 최적 VLA는 실제 환경에서 강력한 효과성과 강건성 보임
- **검증**: 실제 로봇 조작 시나리오에서의 성능 확인

### **Q2: Which Backbone?**

> 인용: "2) How should we formulate VLAs?" (논문 II 섹션)

**상세 분석**: [Which VLM backbone is better for VLAs?](./Which_VLM_Backbone_is_Better/which_vlm_backbone_is_better.md)에서 자세한 분석 확인

### **Q2.1: Which type of VLMs are more suitable for constructing VLAs?**

> 인용: "A2.1: Sufficient vision-language pre-trained on large vision-language datasets benefit VLAs" (Table I)
> 
- **답변**: 대규모 비전-언어 데이터셋에서 충분히 사전 훈련된 VLMs이 VLA에 유리
- **핵심**: 사전 훈련 데이터의 규모와 품질이 중요

### **Q3: How to Formulate VLAs?**

> 인용: "3) Which VLM backbone is more suitable for VLAs?" (논문 II 섹션)

**상세 분석**: [How to Formulate VLAs?](./How_to_Formulate_VLAs/how_to_formulate_vlas.md)에서 자세한 분석 확인

### **Q3.1: What is the best-performing VLA structure?**

> 인용: "A3.1: Continuous action space with policy head to integrate history is the best structure." (Table I)
> 
- **답변**: 히스토리를 통합하는 정책 헤드와 연속 액션 공간이 최적 구조
- **핵심**: Policy Head + Continuous Action Space 조합

### **Q3.2: How do different formulations affect the generalization and data efficiency for VLAs?**

> 인용: "A3.2: The KosMos backbone with a separate policy head for history fusion performs the best in generalization and data efficiency." (Table I)
> 
- **답변**: 히스토리 융합을 위한 별도 Policy를 가진 KosMos 백본이
         generalization and data efficiency에서 최고 성능
- **특징**: KosMos + Policy Head 조합의 우수성

### **Q4: When to Leverage Extra Data?**

> 인용: "4) When should we leverage cross-embodiment datasets?" (논문 II 섹션)

**상세 분석**: [When should we leverage cross-embodiment datasets?](./When_to_Leverage_Cross_Embodiment_Datasets/when_to_leverage_cross_embodiment_datasets.md)에서 자세한 분석 확인

### **Q4.1: How do large-scale cross-embodiment datasets contribute to VLAs?**

> 인용: "A4.1: Extra in-domain data shows beneficial; 2) Post-training further improves overall as well as few-shot performance." (Table I)
> 
- **답변**: 추가 도메인 내 데이터가 유익하며, 후훈련이 전체 및 few-shot 성능을 더욱 향상
- **전략**: Pre-training → Fine-tuning → Post-training 순서

## **Figure 3: 벤치마크 및 평가 환경**

> 인용: "Fig. 3: Two simulated and one real-world benchmarks. We show environment setups and example tasks involved." (논문 Figure 3)
> 

![image.png](attachment:6ca7c05e-83b5-4182-b665-ca4d2943cf03:image.png)

### **벤치마크 구성**

- **CALVIN**: Slide Window, Stack Block, Open Drawer, Open Light
- **SimplerEnv**: Put Spoon on Towel, Put Eggplant in Basket, Stack Block, Put Carrot on Plate
- **Real-World**: Open the Oven, Press Toaster, Pickup Knife, Pickup Cucumber

### **시뮬레이션 벤치마크**

> 인용: "we choose tow well-known and widely used simulation benchmarks (CALVIN [32] and SimplerEnv [40])" (논문 II 섹션)
> 

### **CALVIN [32]**

> 인용: "CALVIN [32] is a simulation benchmark for multitask table-top manipulation. The dataset contains four splits A, B, C, and D according to different scene settings and provides 34 basic tasks with 24K human teleoperated demonstrations annotated with language instructions in total." (논문 II 섹션)
> 
- **특징**: 다중 작업 테이블탑 조작 시뮬레이션 벤치마크
- **데이터**: 4개 분할 (A, B, C, D), 34개 기본 작업, 24K 인간 텔레오퍼레이션 시연
- **평가**: 1~5개 연속 작업 성공률, 평균 달성 작업 수 (Avg. Len.)

### **SimplerEnv [25]**

> 인용: "SimplerEnv [25] is designed as a suite of real-to-sim environments and enables the evaluation of robot policies in simulation." (논문 II 섹션)
> 
- **특징**: 실제-시뮬레이션 환경으로 설계
- **목적**: 시뮬레이션에서 로봇 정책 평가 가능
- **비교**: Google Robot, Bridge V2 등 실제 환경과 비교 가능한 아레나

### Real-World

> 인용: "Real Robot Benchmark [8] consists of over 70K teleoperated human trajectories used to fine-tune robot policies, covering 105 manipulation tasks." (논문 II 섹션)
> 

#### **Real Robot Benchmark 상세 정보**
> **인용**: "To evaluate the performance of models on this benchmark, we adopt the approach outlined in [23], testing each model on one Simple setting and four challenging Unseen settings. Examples of these settings are shown in Fig.4. In total, we evaluate each VLA across 20 tasks, with 5 settings per task and 3 rollouts per setting, reporting the average success rate for each setting. A detailed description of the benchmarks is provided in Appendix K and Appendix D. All tasks included in these benchmarks are driven by single-arm robots, leading to a 7-DoF action - the 6D pose 2 of the gripper and one-dimensional open/close status. Robot observation is accessible from proprioceptive sensory information, visual observation, and language input." (논문 II 섹션)

- **데이터 규모**: 70K 이상의 텔레오퍼레이션 궤적, 105개 조작 작업
- **평가 방법**: [23]에서 제시된 접근법 채택
- **평가 설정**: 1개 Simple 설정 + 4개 Unseen 설정
- **작업 구성**: 20개 작업, 작업당 5개 설정, 설정당 3회 롤아웃
- **성능 지표**: 각 설정에 대한 평균 성공률 보고

#### **로봇 시스템 사양**
- **로봇 타입**: 단일 팔 로봇 (single-arm robots)
- **액션 공간**: 7-DoF (7자유도)
  - **6D pose**: 그리퍼의 6차원 포즈 (위치 + 회전)
  - **1D status**: 그리퍼 열기/닫기 상태
- **관측 정보**:
  - **Proprioceptive sensory information**: 로봇의 내부 센서 정보
  - **Visual observation**: 시각적 관측
  - **Language input**: 언어 입력

#### **벤치마크 상세 설명**
- **상세 문서**: [Appendix D](../Appendix/Appendix_D_Benchmark_Details/appendix_d_benchmark_details.md)와 [Appendix K](../Appendix/Appendix_K_Rollout_Examples_in_Real_World_Experiments/appendix_k_rollout_examples.md)에서 벤치마크에 대한 상세 설명 제공
- **예시 설정**: Figure 4에서 Unseen 설정들의 예시 제시

### **실제 환경 평가**

> 인용: "we adopt the approach outlined in [23], testing each model on one Simple setting and four challenging Unseen settings." (논문 II 섹션)
> 

- **규모**: 70K 이상의 텔레오퍼레이션 궤적, 105개 조작 작업
- **평가**: 1개 Simple 설정 + 4개 Unseen 설정
- **작업**: 20개 작업, 작업당 5개 설정, 설정당 3회 롤아웃

![image.png](attachment:b94d5e95-ce79-4b13-97ed-46c4fbbc7da5:image.png)

### **Figure 4: 실제 실험 설정**

> 인용: "Fig. 4: The illustration of the experimental settings in real-world experiments. We evaluate the models in 20 tasks with five rollouts for each task, involving Unseen Distractor, Unseen Target Object and Unseen Background, Novel Skill Description." (논문 Figure 4)
> 

![image.png](attachment:33e0b625-aee9-4ba2-839b-9ef8040623d2:image.png)

- **평가 방법**: 20개 작업, 각 작업당 5회 롤아웃
- **특별 고려사항**:
    - Open Drawer 같은 작업은 unseen object 설정 테스트 제외
    - 각 롤아웃의 객체 레이아웃은 훈련 세트와 다르게 랜덤 초기화
    - Unseen target object는 picking 작업에만 적용

### **Unseen 설정들**

> 인용: "testing each model on one Simple setting and four challenging Unseen settings" (논문 II 섹션)
> 
1. **Unseen Distractors**: 보이지 않는 방해물
2. **Unseen Backgrounds**: 보이지 않는 배경
3. **Unseen Target Objects**: 보이지 않는 대상 물체
4. **Novel Skill Descriptions**: 새로운 기술 설명

## TABLE II: Simulation performances

> Simulation performances on CALVIN benchmark, all models are trained on split ABCD/ABC, and evaluated on split D. KosMos P.H. represents the VLA utilizing KosMos-2 as backbone and policy head as architecture, built with the RoboVLMs framework, and is maximally trained for 5 epochs. We will continue to use the expression of backbone and structure to represent the VLAs built with RoboVLMs in the following paper.
> 

### **CALVIN 벤치마크 성능**

> 인용: Table II의 CALVIN 벤치마크 성능 결과
> 

### **ABCD 분할 훈련 결과** [4-2-4-1. ABCD 분할](https://www.notion.so/4-2-4-1-ABCD-27210831b37580128c19ea8a16a21a41?pvs=21)

| Method | VLA? | Train | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCIL | ✗ | ABCD | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| R3M (Frozen) | ✗ | ABCD | 0.085 | 0.005 | 0.001 | 0.000 | 0.000 | 0.10 |
| Voltron (Frozen) | ✗ | ABCD | 0.101 | 0.003 | 0.001 | 0.000 | 0.000 | 0.11 |
| Voltron (Fine-tuned) | ✗ | ABCD | 0.837 | 0.566 | 0.352 | 0.208 | 0.115 | 2.08 |
| RT-1 | ✗ | ABCD | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| HULC | ✗ | ABCD | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06 |
| **GR-1** | ✓ | ABCD | **0.949** | **0.896** | **0.844** | **0.789** | **0.731** | **4.21** |
| **KosMos P.H. (RoboVLMs)** | ✓ | ABCD | **0.967** | **0.930** | **0.899** | **0.865** | **0.826** | **4.49** |

### **ABC 분할 훈련 결과 [4-2-4-2. ABC 분할](https://www.notion.so/4-2-4-2-ABC-27210831b37580e28b14cf8b519a30b3?pvs=21)**

| Method | VLA? | Train | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCIL | ✗ | ABC | 0.304 | 0.013 | 0.002 | 0.000 | 0.000 | 0.31 |
| Voltron (Frozen) | ✗ | ABC | 0.026 | 0.001 | 0.000 | 0.000 | 0.000 | 0.03 |
| Voltron (Fine-tuned) | ✗ | ABC | 0.569 | 0.272 | 0.105 | 0.038 | 0.014 | 1.00 |
| RT-1 | ✗ | ABC | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| HULC | ✗ | ABC | 0.418 | 0.165 | 0.057 | 0.019 | 0.011 | 0.67 |
| **GR-1** | ✓ | ABC | **0.854** | **0.712** | **0.596** | **0.497** | **0.401** | **3.06** |
| **KosMos P.H. (RoboVLMs)** | ✓ | ABC | **0.980** | **0.936** | **0.854** | **0.778** | **0.704** | **4.25** |

### **성능 분석**

### **ABCD 분할에서의 성능**

- **KosMos P.H. (RoboVLMs)**: 모든 연속 작업에서 최고 성능 달성
    - 1개 작업: 96.7% (GR-1 94.9% 대비 +1.8%p)
    - 5개 작업: 82.6% (GR-1 73.1% 대비 +9.5%p)
    - Avg. Len.: 4.49 (GR-1 4.21 대비 +0.28)

### **ABC 분할에서의 성능**

- **KosMos P.H. (RoboVLMs)**: 제한된 데이터에서도 우수한 성능
    - 1개 작업: 98.0% (GR-1 85.4% 대비 +12.6%p)
    - 5개 작업: 70.4% (GR-1 40.1% 대비 +30.3%p)
    - Avg. Len.: 4.25 (GR-1 3.06 대비 +1.19)

### **성능 우위 확인**

> 인용: "The built VLA model with a proper backbone and structure can outperform the state-of-the-art generalist robot policies by a large margin." (논문 II 섹션)
> 
- **SOTA 대비**: 적절한 백본과 구조를 가진 VLA 모델이 기존 SOTA 일반화 로봇 정책을 큰 폭으로 능가
- **일관성**: 다양한 벤치마크에서 일관된 성능 우위
- **실용성**: 실제 환경에서의 강건성 확인

## **연구 기여, 실용적 가치**

### **연구 기여도**

> 인용: "We hope that our findings can practically help build robust, generalizable, and well-performing VLAs." (논문 II 섹션)
> 
- **이론적 기여**: VLA 설계의 체계적 이해 제공
- **실용적 기여**: 강건하고 일반화 가능하며 성능이 우수한 VLA 구축 가이드
- **방법론적 기여**: RoboVLMs 프레임워크를 통한 통합적 접근

### **실험 설계의 체계성**

> 인용: "As shown in Tab. I, we further divide the 4 essential problems into 6 research problem, and implement successive experiments of VLAs to answer each research problem." (논문 II 섹션)
> 
- **4가지 핵심 문제**: Why, Which, How, When
- **6개 세부 연구 문제**: 체계적인 분할과 실험
- **순차적 실험**: 각 연구 문제에 대한 연속적 실험 수행
