# When should we leverage cross-embodiment datasets?

> **인용**: 논문 "When should we leverage cross-embodiment datasets?" 섹션

## 🎯 **연구 배경 및 동기**

### **최근 동향**
> **인용**: "In recent work, it has been a dominant trend to leverage large-scale cross-embodiment robot manipulation datasets to improve the performance of VLAs [4, 7, 22, 35]." (논문 섹션)

최근 연구에서 대규모 cross-embodiment 로봇 조작 데이터셋을 활용하여 VLA 성능을 향상시키는 것이 지배적인 동향입니다.

### **연구의 필요성**
> **인용**: "However, it is still not fully clear if it helps and an important question remains:" (논문 섹션)

하지만 이것이 실제로 도움이 되는지 명확하지 않으며, 중요한 질문이 남아있습니다.

### **Question 6: How do large-scale cross-embodiment datasets contribute to VLAs?**

> **인용**: "Question 6: How do large-scale cross-embodiment datasets contribute to VLAs?" (논문 섹션)

대규모 cross-embodiment 데이터셋이 VLA에 어떻게 기여하는지 탐구하는 핵심 질문입니다.

## 🔬 **연구 질문 세분화**

### **두 가지 하위 질문**
> **인용**: "To address this, we break the question into two sub-questions: 1) What types of data from large-scale cross-embodiment datasets are the most beneficial for building VLAs? 2) When and how should these data be utilized effectively?" (논문 섹션)

#### **1. 데이터 유형의 효과성**
- **질문**: 대규모 cross-embodiment 데이터셋의 어떤 유형의 데이터가 VLA 구축에 가장 유익한가?
- **목표**: 가장 효과적인 데이터 유형 식별

#### **2. 데이터 활용 전략**
- **질문**: 언제, 어떻게 이러한 데이터를 효과적으로 활용해야 하는가?
- **목표**: 최적의 데이터 활용 전략 개발

## 🧪 **실험 설계 및 방법론**

### **실험 전략**
> **인용**: "In this section, we conduct a series of experiments to investigate different strategies for using external large-scale cross-embodiment datasets." (논문 섹션)

외부 대규모 cross-embodiment 데이터셋을 활용하는 다양한 전략을 조사하기 위한 일련의 실험을 수행합니다.

### **두 가지 주요 설정**

#### **1. Pre-train (사전 훈련)**
> **인용**: "Pre-train: Pre-training the model with in-domain manipulation data and cross-embodiment datasets. This approach has been explored in RT-2 [7], OpenVLA [22], and OCTO [39]." (논문 섹션)

- **방법**: in-domain 조작 데이터와 cross-embodiment 데이터셋으로 모델 사전 훈련
- **사용 사례**: RT-2 [7], OpenVLA [22], OCTO [39]에서 탐구됨
- **특징**: 동시에 두 유형의 데이터를 활용

#### **2. Post-train (사후 훈련)**
> **인용**: "Post-train: First, training the VLMs on cross-embodiment datasets, followed by fine-tuning with in-domain manipulation tasks. This strategy has been adopted by π0 [4]." (논문 섹션)

- **방법**: 
  1. 먼저 cross-embodiment 데이터셋으로 VLM 훈련
  2. 그 다음 in-domain 조작 작업으로 파인튜닝
- **사용 사례**: π0 [4]에서 채택된 전략
- **특징**: 순차적 훈련 방식

### **실험 설정**

#### **기본 모델**
> **인용**: "Our experiments in this section use the best-performing KosMos backbone with a policy head for history fusion as the base model." (논문 섹션)

- **백본**: KosMos (최고 성능)
- **구조**: 히스토리 융합을 위한 정책 헤드
- **이유**: 이전 실험에서 최고 성능을 보인 설정

#### **Cross-Embodiment 데이터셋**
> **인용**: "We use Open X-Embodiment (OXE) [35] as the cross-embodiment dataset, which comprises a diverse range of robot manipulation data collected worldwide and is the most widely used one in recent works [4, 7, 22, 39]." (논문 섹션)

- **데이터셋**: Open X-Embodiment (OXE) [35]
- **특징**: 전 세계에서 수집된 다양한 로봇 조작 데이터
- **사용도**: 최근 연구에서 가장 널리 사용됨

#### **비교 기준선**
> **인용**: "For comparison, we also evaluate a baseline setting, Finetune, where the VLA is trained exclusively on in-domain data." (논문 섹션)

- **Finetune**: in-domain 데이터만으로 VLA 훈련
- **목적**: cross-embodiment 데이터의 효과를 비교하기 위한 기준선

### **Google Robot 환경 추가 설정**

#### **RT Partial Finetune**
> **인용**: "Additionally, for Google Robot, we include both RT Partial Finetune and RT Finetune, where RT Partial Finetune involves only the trajectories with the same task type as the evaluating tasks" (논문 섹션)

- **방법**: 평가 작업과 동일한 작업 유형의 궤적만 포함
- **특징**: 특정 작업 유형에 집중

#### **RT Finetune**
> **인용**: "and RT Finetune involves co-finetuning the policy with additional data from the same robot across different tasks." (논문 섹션)

- **방법**: 동일한 로봇의 다양한 작업에서 추가 데이터로 정책 공동 파인튜닝
- **특징**: 다양한 작업의 데이터 활용

### **Bridge 환경 설정**

#### **Bridge Finetune**
> **인용**: "For Bridge, we only evaluate Bridge Finetune which finetune the policy with the entire Bridge-V2 dataset, since the training dataset does not contain trajectories with the same instructions of the evaluated tasks." (논문 섹션)

- **방법**: 전체 Bridge-V2 데이터셋으로 정책 파인튜닝
- **이유**: 훈련 데이터셋에 평가 작업과 동일한 지시사항의 궤적이 없음

## 📊 **실험 결과 분석**

### **Figure 10: SimplerEnv에서의 Cross-Embodiment 훈련 Ablation 연구**

> **인용**: "Fig. 10: Ablation studies for cross-embodiment training on SimpleEnv. We evaluate four different training recipes." (논문 섹션)

#### **WidowX+Bridge 환경**

##### **실험 설정**
1. **Bridge Finetune**: 전체 Bridge 데이터셋으로 VLA 직접 파인튜닝 (테스트 작업 제외)
2. **OXE Pre-Train**: OXE 데이터셋 [35]으로 VLA 사전 훈련
3. **Post-Train**: OXE 사전 훈련된 VLA를 Bridge 데이터셋으로 훈련

##### **성능 결과**

| 작업 | Bridge Finetune | OXE Pre-Train | Post-Train |
|------|----------------|---------------|------------|
| Put Spoon on Towel | 0.29 | 0.25 | 0.58 |
| Put Carrot on Plate | 0.31 | 0.21 | 0.25 |
| Stack Green Block on Yellow Block | 0.12 | 0.08 | 0.00 |
| Put Eggplant in Yellow Bucket | 0.14 | 0.46 | 0.21 |
| **Average** | **0.21** | **0.25** | **0.26** |

#### **Google Robot 환경**

##### **실험 설정**
1. **RT-Partial Finetune**: 테스트된 RT 작업만으로 VLA 파인튜닝
2. **RT Finetune**: 전체 RT 데이터셋으로 VLA 파인튜닝 (테스트 작업 포함)
3. **OXE Pre-Train**: OXE 데이터셋으로 사전 훈련
4. **Post-Train**: OXE 사전 훈련 후 RT 작업으로 훈련

##### **성능 결과**

| 작업 | RT-Partial Finetune | RT Finetune | OXE Pre-Train | Post-Train |
|------|-------------------|-------------|---------------|------------|
| Pick Coke Can | 0.21 | 0.29 | 0.42 | 0.00 |
| Move Near | 0.23 | 0.90 | 0.62 | 0.33 |
| Open/Close Drawer | 0.06 | 0.48 | 0.73 | 0.66 |
| Open Drawer & Place Apple | 0.27 | 0.36 | 0.50 | 0.77 |
| **Average** | **0.19** | **0.51** | **0.57** | **0.44** |

### **Figure 11: CALVIN Few-Shot에서의 Cross-Embodiment 사전 훈련 효과**

> **인용**: "Fig. 11: The effect of cross-embodiment pre-training on OXE datasets for few-shot learning." (논문 섹션)

#### **CALVIN Few-Shot 실험**
> **인용**: "To evaluate the impact of cross-embodiment datasets more comprehensively, we also perform experiments on CALVIN, which is not part of OXE. For CALVIN, we omit the co-train setting and mainly focus on whether cross-embodiment datasets benefit few-shot learning for robot manipulation on out-of-distribution tasks." (논문 섹션)

- **목적**: cross-embodiment 데이터셋의 영향을 더 포괄적으로 평가
- **데이터셋**: CALVIN (OXE에 포함되지 않음)
- **설정**: 작업당 10개 궤적만 사용 (CALVIN few-shot)
- **입력**: 정적 헤드 카메라 이미지만 사용

#### **성능 결과**

| 설정 | Task 1 | Task 2 | Avg. Len. |
|------|--------|--------|-----------|
| Few-shot w.o. OXE Pre-Train | 0.43 | 0.26 | 0.32 |
| Few-shot w. OXE Pre-Train | 0.60 | 0.43 | 0.57 |

**성능 향상**:
- **Task 1**: 0.43 → 0.60 (+17.2%)
- **Task 2**: 0.26 → 0.43 (+17.2%)
- **Avg. Len.**: 0.32 → 0.57 (+0.25 tasks)

## 🔍 **핵심 관찰사항 (Key Observations)**

### **1. Cross-Embodiment 데이터 사전 훈련의 제한적 효과**

> **인용**: "Pre-training with cross-embodiment data does not help significantly. Comparing OXE Pre-train and RT-Partial Finetune reveals that for both Google Robot and Bridge, co-training with cross-embodiment data does not lead to substantial performance improvements." (논문 섹션)

#### **Google Robot 환경**
- **OXE Pre-Train**: 0.57 (평균 성공률)
- **RT-Partial Finetune**: 0.19 (평균 성공률)
- **결과**: Cross-embodiment 데이터 사전 훈련이 상당한 성능 향상을 가져오지 않음

#### **Bridge 환경**
- **OXE Pre-Train**: 0.25 (평균 성공률)
- **Bridge Finetune**: 0.21 (평균 성공률)
- **결과**: 유사한 성능 수준

### **2. In-Domain 데이터의 우수성**

> **인용**: "In particular, for Google Robot, training with additional in-domain data (RT Finetune)—even from different tasks—achieves higher success rates (compared with RT-Partial Finetune). This indicates that in-domain data, even if task-agnostic, are more effective for improving model performance than cross-embodiment data." (논문 섹션)

#### **RT Finetune의 우수성**
- **RT Finetune**: 0.51 (평균 성공률)
- **RT-Partial Finetune**: 0.19 (평균 성공률)
- **결과**: In-domain 데이터가 cross-embodiment 데이터보다 더 효과적

#### **핵심 시사점**
- **In-domain 데이터**: 작업과 무관하더라도 모델 성능 향상에 더 효과적
- **Cross-embodiment 데이터**: 상대적으로 효과가 제한적

### **3. Post-Training의 잠재적 이점**

> **인용**: "Post-training after cross-embodiment pre-training shows potential benefits. The average performance of the post-trained model (52% on Google Robot and 38% on Bridge) exceeds that of the model fine-tuned exclusively on in-domain data (48% on Google Robot and 31% on Bridge). This suggests that cross-embodiment pre-training can provide a useful initialization that benefits subsequent fine-tuning." (논문 섹션)

#### **Post-Train 성능**
- **Google Robot**: 0.44 (평균 성공률)
- **Bridge**: 0.26 (평균 성공률)

#### **In-domain 전용 훈련과 비교**
- **Google Robot**: 0.48 (RT Finetune)
- **Bridge**: 0.31 (Bridge Finetune)

#### **핵심 시사점**
- **Post-training**: Cross-embodiment 사전 훈련 후 파인튜닝이 잠재적 이점 제공
- **초기화 효과**: Cross-embodiment 사전 훈련이 후속 파인튜닝에 유용한 초기화 제공

### **4. Few-Shot 학습에서의 사전 훈련 효과**

> **인용**: "Pre-training improves few-shot learning performance. In the few-shot setting for CALVIN, with a single-view on-head camera, pre-training significantly improves the performance by 17.2% in terms of single-task execution and 0.25 more executed tasks in each rollout." (논문 섹션)

#### **CALVIN Few-Shot 성능 향상**
- **Task 1**: 0.43 → 0.60 (+17.2%)
- **Task 2**: 0.26 → 0.43 (+17.2%)
- **Avg. Len.**: 0.32 → 0.57 (+0.25 tasks)

#### **핵심 시사점**
- **Few-shot 학습**: 사전 훈련이 few-shot 학습 성능을 크게 향상
- **표현 학습**: 대규모 cross-embodiment 데이터셋 사전 훈련이 로봇 조작을 위한 더 효과적인 표현 학습에 도움
- **적응성**: 새로운 조작 작업에 빠르게 적응 가능

## 📈 **성능 분석**

### **전체 성능 비교**

#### **Google Robot 환경**
1. **RT Finetune**: 0.51 (최고 성능)
2. **OXE Pre-Train**: 0.57
3. **Post-Train**: 0.44
4. **RT-Partial Finetune**: 0.19

#### **Bridge 환경**
1. **Post-Train**: 0.26 (최고 성능)
2. **OXE Pre-Train**: 0.25
3. **Bridge Finetune**: 0.21

### **데이터 유형별 효과성**

#### **In-Domain 데이터**
- **장점**: 작업과 무관하더라도 높은 효과성
- **사용 사례**: RT Finetune에서 최고 성능
- **특징**: 특정 도메인에 특화된 데이터

#### **Cross-Embodiment 데이터**
- **장점**: Few-shot 학습에서 효과적
- **제한점**: 일반적인 성능 향상에는 제한적
- **사용 사례**: Post-training에서 잠재적 이점

### **훈련 전략별 효과성**

#### **Pre-Training**
- **효과**: Few-shot 학습에서 효과적
- **제한점**: 일반적인 성능 향상에는 제한적
- **권장사항**: Few-shot 시나리오에서 활용

#### **Post-Training**
- **효과**: 잠재적 이점 제공
- **특징**: Cross-embodiment 사전 훈련 후 in-domain 파인튜닝
- **권장사항**: 균형잡힌 접근법

## ✅ **Finding 6: Cross-Embodiment 데이터의 효과**

> **인용**: "Finding 6: Extra in-domain data, even from different tasks, shows beneficial, and large-scale cross-embodiment pre-training further improves overall as well as few-shot performance." (논문 섹션)

### **핵심 발견사항**

#### **1. In-Domain 데이터의 우수성**
- **효과**: 작업과 무관한 in-domain 데이터도 유익함
- **성능**: RT Finetune에서 최고 성능 달성
- **시사점**: 도메인 특화 데이터의 중요성

#### **2. Cross-Embodiment 사전 훈련의 이점**
- **전체 성능**: Post-training에서 잠재적 이점
- **Few-shot 성능**: 17.2% 성능 향상
- **적응성**: 새로운 작업에 빠른 적응

### **실용적 시사점**

#### **데이터 전략**
1. **우선순위**: In-domain 데이터 우선 활용
2. **보완적 역할**: Cross-embodiment 데이터는 보완적 역할
3. **Few-shot 시나리오**: Cross-embodiment 사전 훈련 활용

#### **훈련 전략**
1. **일반적 성능**: In-domain 데이터 중심 훈련
2. **Few-shot 성능**: Cross-embodiment 사전 훈련 + Post-training
3. **균형**: 두 접근법의 균형잡힌 활용

## 🎯 **결론**

### **Cross-Embodiment 데이터 활용 가이드라인**

#### **언제 활용해야 하는가?**
1. **Few-shot 학습**: 새로운 작업에 빠른 적응이 필요한 경우
2. **Post-training**: Cross-embodiment 사전 훈련 후 in-domain 파인튜닝
3. **보완적 역할**: In-domain 데이터와 함께 활용

#### **어떻게 활용해야 하는가?**
1. **Pre-training**: Few-shot 시나리오에서 효과적
2. **Post-training**: 균형잡힌 접근법
3. **데이터 우선순위**: In-domain 데이터 우선, Cross-embodiment 데이터 보완

### **미래 연구 방향**
1. **효율적 활용**: Cross-embodiment 데이터의 더 효율적인 활용 방법
2. **데이터 품질**: 데이터 품질이 성능에 미치는 영향
3. **하이브리드 접근**: 두 유형 데이터의 최적 조합 방법

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
