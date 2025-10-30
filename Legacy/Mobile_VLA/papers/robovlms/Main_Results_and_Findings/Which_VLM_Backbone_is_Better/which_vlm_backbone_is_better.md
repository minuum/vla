# Which VLM backbone is better for VLAs?

> **인용**: 논문 "Which VLM backbone is better for VLAs?" 섹션

## 🎯 **핵심 질문 탐구**

### **연구 배경**
> **인용**: "Following such a finding, the choice of using a policy head for history fusion in VLA can be finalized. However, a critical question remains when selecting the most appropriate VLM to build our VLA:" (논문 섹션)

정책 헤드를 사용한 히스토리 융합 선택이 확정된 후에도, VLA를 구축하기 위한 가장 적절한 VLM을 선택할 때 중요한 질문이 남아있습니다.

### **Question 5: Which type of VLMs is most suitable for constructing VLAs?**

> **인용**: "Question 5: Which type of VLMs is most suitable for constructing VLAs?" (논문 섹션)

VLA 구축에 가장 적합한 VLM 유형을 찾는 것이 이 섹션의 핵심 질문입니다.

## 📊 **Table V: 다양한 VLM 백본 기반 VLA 성능**

### **Table V 개요**
> **인용**: "TABLE V: The performance of the built VLAs based on VLMs with different image token numbers and VL pre-train data scales. The first three rows are flamingo backbones with encoder-decoder structures, the rest backbones are decoder-only structures. Note that for VLMs with multi-stage training, the data scale refers to the data amount utilized for the final stage of fine-tuning. 'UNK' denotes unknown. Results are reported with the model checkpoints trained with 5 epochs on the ABCD training splits, all models are trained with a single side view image for fair comparison. We surprisingly found that both LLaVA and Qwen behave badly without an additional resampler to downsample the number of tokens." (논문 Table V)

#### **실험 설정**
- **훈련 데이터**: ABCD 분할에서 5 에포크 훈련
- **이미지 설정**: 공정한 비교를 위해 단일 사이드 뷰 이미지 사용
- **토큰 다운샘플링**: LLaVA와 Qwen은 추가 resampler 없이는 성능이 좋지 않음

### **성능 결과**

#### **Flamingo 백본 (Encoder-Decoder 구조)**

| Backbone | #Token | Data Scale | Model Size | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|----------|--------|------------|------------|---|---|---|---|---|-----------|
| Flamingo | 64 | 1B+ | 3B | 0.692 | 0.418 | 0.241 | 0.14 | 0.074 | 1.57 |
| Flamingo | 64 | 1B+ | 4B | 0.689 | 0.456 | 0.281 | 0.181 | 0.107 | 1.71 |
| Flamingo | 64 | 1B+ | 9B | 0.744 | 0.485 | 0.298 | 0.187 | 0.112 | 1.83 |

#### **Decoder-Only 구조 백본**

| Backbone | #Token | Data Scale | Model Size | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|----------|--------|------------|------------|---|---|---|---|---|-----------|
| Qwen-VL | 256 | 350K | 9B | 0.221 | 0.062 | 0.014 | 0.002 | 0.000 | 0.30 |
| MoonDream | 576 | UNK | 3B | 0.717 | 0.473 | 0.296 | 0.198 | 0.127 | 1.81 |
| Uform | 256 | 10M | 1.3B | 0.778 | 0.577 | 0.407 | 0.300 | 0.216 | 2.28 |
| **KosMos** | 64 | 90M | 2B | **0.922** | **0.807** | **0.701** | **0.615** | **0.549** | **3.59** |
| **Paligemma** | 256 | 10B | 3B | **0.931** | **0.836** | **0.752** | **0.683** | **0.616** | **3.82** |

## 🔬 **실험 설계 및 방법론**

### **실험의 도전과제**
> **인용**: "To thoroughly investigate this question, it would be ideal to conduct experiments in highly controlled settings. However, training VLMs on large-scale vision-language datasets is extremely resource-intensive." (논문 섹션)

#### **이상적인 실험 조건**
- **고도로 제어된 설정**: 이상적인 실험 조건
- **자원 집약적**: 대규모 비전-언어 데이터셋에서 VLM 훈련은 극도로 자원 집약적

#### **실용적 접근법**
> **인용**: "Therefore, we base our VLAs on a diverse selection of pre-trained large-scale vision-language backbones with varying architectures, training data scales, model sizes, and latent embeddings." (논문 섹션)

- **다양한 사전 훈련 백본**: 다양한 아키텍처, 훈련 데이터 규모, 모델 크기, 잠재 임베딩을 가진 백본 활용
- **실용적 접근**: 완전히 제어된 비교는 아니지만, 광범위한 실험을 통한 통찰 제공

### **선택된 VLM 백본들**

#### **Flamingo 모델 패밀리 (Encoder-Decoder)**
> **인용**: "These include Flamingo model family [1] (Encoder-Decoder), and a series of decoder-only VLMs, including LLaVA [28], Qwen-VL [2], MoonDream [44], UForm [41], Paligemma [3], and KosMos [36]." (논문 섹션)

- **구조**: Encoder-Decoder
- **특징**: 인코더-디코더 구조의 전통적인 VLM

#### **Decoder-Only VLMs**
- **LLaVA [28]**: 대표적인 디코더 전용 VLM
- **Qwen-VL [2]**: Alibaba의 VLM
- **MoonDream [44]**: 경량 VLM
- **UForm [41]**: 효율적인 VLM
- **Paligemma [3]**: Google의 VLM
- **KosMos [36]**: Microsoft의 VLM

### **실험 설정**
> **인용**: "Noticeably, in this section, for fair comparisons, all the models are trained with static images instead of both static and hand cameras. Although this approach may not offer a fully controlled comparison, our extensive experiments aim to provide insights into the impact of different VLM backbones on VLAs." (논문 섹션)

#### **공정한 비교를 위한 설정**
- **정적 이미지**: 정적 이미지만 사용 (정적 + 핸드 카메라 대신)
- **완전 제어된 비교는 아님**: 하지만 광범위한 실험을 통한 통찰 제공
- **목표**: 다양한 VLM 백본이 VLA에 미치는 영향에 대한 통찰

## 🔍 **핵심 관찰사항 (Key Observations)**

### **KosMos와 Paligemma의 뛰어난 성능**

> **인용**: "KosMos and Paligemma demonstrate the distinctively better performance: From Tab. V, we can see that these two backbones are much better than others with a significantly clear margin." (논문 섹션)

#### **성능 비교**
- **KosMos**: Avg. Len. 3.59 (최고 성능 중 하나)
- **Paligemma**: Avg. Len. 3.82 (최고 성능)
- **다른 백본들**: 상당한 성능 격차

#### **성능 우위의 원인**
> **인용**: "Their superior performance benefits from sufficient vision-language pre-trained on large vision-language datasets. This outcome is intuitive, as extensive pre-training facilitates stronger alignment between visual and linguistic features—an alignment critical for language-conditioned manipulation tasks." (논문 섹션)

##### **충분한 비전-언어 사전 훈련**
- **대규모 데이터셋**: 큰 비전-언어 데이터셋에서 충분한 사전 훈련
- **특징 정렬**: 시각적 특징과 언어적 특징 간의 강한 정렬
- **언어 조건 조작**: 언어 조건 조작 작업에 중요한 정렬

##### **직관적 설명**
- **광범위한 사전 훈련**: 더 강한 시각-언어 특징 정렬 촉진
- **언어 조건 작업**: 언어 조건 조작 작업에 중요한 정렬

### **다른 백본들의 성능**

#### **Flamingo 패밀리**
- **3B 모델**: Avg. Len. 1.57
- **4B 모델**: Avg. Len. 1.71  
- **9B 모델**: Avg. Len. 1.83
- **특징**: 모델 크기 증가에 따른 성능 향상

#### **Decoder-Only 백본들**
- **Qwen-VL**: Avg. Len. 0.30 (가장 낮은 성능)
- **MoonDream**: Avg. Len. 1.81
- **Uform**: Avg. Len. 2.28
- **KosMos**: Avg. Len. 3.59 (우수한 성능)
- **Paligemma**: Avg. Len. 3.82 (최고 성능)

### **토큰 수와 데이터 규모의 영향**

#### **토큰 수**
- **64 토큰**: Flamingo, KosMos
- **256 토큰**: Qwen-VL, Uform, Paligemma
- **576 토큰**: MoonDream

#### **데이터 규모**
- **1B+**: Flamingo 패밀리
- **350K**: Qwen-VL (상대적으로 적은 데이터)
- **10M**: Uform
- **90M**: KosMos
- **10B**: Paligemma (가장 큰 데이터 규모)

#### **모델 크기**
- **1.3B**: Uform
- **2B**: KosMos
- **3B**: Flamingo 3B, MoonDream, Paligemma
- **4B**: Flamingo 4B
- **9B**: Flamingo 9B, Qwen-VL

## 📈 **성능 분석**

### **최고 성능 백본들**

#### **Paligemma (최고 성능)**
- **Avg. Len.**: 3.82
- **특징**: 10B 데이터 규모, 256 토큰, 3B 모델 크기
- **장점**: 가장 큰 데이터 규모에서 훈련

#### **KosMos (두 번째 최고 성능)**
- **Avg. Len.**: 3.59
- **특징**: 90M 데이터 규모, 64 토큰, 2B 모델 크기
- **장점**: 상대적으로 작은 모델 크기로도 우수한 성능

### **성능 격차 분석**

#### **상위 그룹 (Avg. Len. > 3.0)**
- **Paligemma**: 3.82
- **KosMos**: 3.59
- **성능 격차**: 0.23

#### **중간 그룹 (Avg. Len. 1.5-3.0)**
- **Uform**: 2.28
- **Flamingo 9B**: 1.83
- **MoonDream**: 1.81
- **Flamingo 4B**: 1.71
- **Flamingo 3B**: 1.57

#### **하위 그룹 (Avg. Len. < 1.5)**
- **Qwen-VL**: 0.30
- **특징**: 가장 낮은 성능, 350K 데이터 규모

### **아키텍처별 성능 비교**

#### **Encoder-Decoder vs Decoder-Only**
- **Encoder-Decoder**: Flamingo 패밀리 (1.57-1.83)
- **Decoder-Only**: 다양한 성능 (0.30-3.82)
- **결론**: 아키텍처보다는 사전 훈련 데이터 규모와 품질이 더 중요

## 🔬 **추가 연구 방향**

### **Section III에서의 추가 논의**
> **인용**: "We discuss more influencing factors and interesting findings in Sec. III." (논문 섹션)

- **영향 요인**: 더 많은 영향 요인들에 대한 논의
- **흥미로운 발견**: 추가적인 흥미로운 발견사항들

## ✅ **Finding 5: VLA는 VLM 백본의 대규모 비전-언어 데이터셋에서 충분한 비전-언어 사전 훈련으로부터 이익을 얻습니다**

> **인용**: "Finding 5: VLAs benefit from the sufficient vision-language pre-training on large vision-language datasets of VLMs backbone." (논문 섹션)

### **핵심 시사점**

#### **1. 사전 훈련 데이터의 중요성**
- **데이터 규모**: 더 큰 데이터셋에서 훈련된 모델이 우수한 성능
- **데이터 품질**: 고품질 비전-언어 데이터의 중요성
- **충분한 훈련**: 충분한 사전 훈련이 필수적

#### **2. 모델 아키텍처보다 데이터가 중요**
- **아키텍처**: Encoder-Decoder vs Decoder-Only보다 데이터가 더 중요
- **성능 결정 요인**: 사전 훈련 데이터 규모와 품질이 성능을 결정

#### **3. 최적 백본 선택 가이드라인**
- **Paligemma**: 가장 큰 데이터 규모 (10B)로 최고 성능
- **KosMos**: 상대적으로 작은 모델 크기로도 우수한 성능
- **데이터 효율성**: KosMos가 더 효율적인 선택

#### **4. 실용적 시사점**
- **리소스 제약**: 제한된 리소스에서는 KosMos가 좋은 선택
- **최고 성능**: 충분한 리소스가 있다면 Paligemma가 최고 성능
- **균형**: 성능과 효율성의 균형 고려 필요

## 🎯 **결론**

### **VLM 백본 선택의 핵심 원칙**
1. **사전 훈련 데이터 규모**: 더 큰 데이터셋에서 훈련된 모델 우선
2. **데이터 품질**: 고품질 비전-언어 데이터의 중요성
3. **충분한 훈련**: 충분한 사전 훈련이 필수적
4. **리소스 고려**: 성능과 효율성의 균형

### **권장사항**
- **최고 성능**: Paligemma (10B 데이터, 3B 모델)
- **효율적 선택**: KosMos (90M 데이터, 2B 모델)
- **제한된 리소스**: KosMos가 더 실용적
- **충분한 리소스**: Paligemma가 최고 성능

### **미래 연구 방향**
- **데이터 효율성**: 더 적은 데이터로도 우수한 성능을 달성하는 방법
- **아키텍처 최적화**: 특정 VLA 작업에 최적화된 아키텍처
- **하이브리드 접근**: 다양한 백본의 장점을 결합한 방법

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
