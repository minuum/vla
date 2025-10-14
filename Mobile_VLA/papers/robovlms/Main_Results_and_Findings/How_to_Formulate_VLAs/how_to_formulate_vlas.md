# B. How to Formulate VLAs?

> **인용**: 논문 B. How to Formulate VLAs? 섹션

## 🎯 **VLA 설계 질문 탐구**

### **연구 목표**
> **인용**: "In this section, our study addresses questions regarding VLA formulations, including different design choices of VLA structures and various backbones." (논문 B 섹션)

이 섹션에서는 VLA 공식화에 대한 질문들을 다루며, 다양한 VLA 구조의 설계 선택과 다양한 백본을 포함합니다.

### **실험 방법론**
> **인용**: "To answer these questions, we conduct a series of controlled experimental studies to ablate various VLA formulations on the CALVIN benchmark for rapid evaluation." (논문 B 섹션)

이러한 질문에 답하기 위해 CALVIN 벤치마크에서 다양한 VLA 공식화를 제어된 실험 연구로 ablation하여 빠른 평가를 수행합니다.

## 🔬 **Question 3: What is the best-performing VLA structure?**

### **핵심 질문**
> **인용**: "More specifically, how should we model observations, states, and actions in robot manipulation tasks within the context of a VLA?" (논문 B 섹션)

로봇 조작 작업에서 관측, 상태, 액션을 VLA 맥락에서 어떻게 모델링해야 하는지에 대한 구체적인 질문입니다.

### **실험 설계**
> **인용**: "To explore this question, we implement several variants, leveraging various open-source VLM backbones such as OpenFlamingo [35], LLaVA [28], and KosMos [36]. These variants incorporate different historical information modeling strategies, and action spaces, as discussed and categorized in Sec.I." (논문 B 섹션)

- **VLM 백본**: OpenFlamingo, LLaVA, KosMos 등 다양한 오픈소스 VLM 백본 활용
- **변형 요소**: 다양한 히스토리 정보 모델링 전략과 액션 공간
- **분류**: Section I에서 논의되고 분류된 내용 기반

### **Table III: CALVIN 벤치마크에서의 VLA 구조 성능**

> **인용**: "The performance of various VLA structures in CALVIN is summarized in Tab. III." (논문 B 섹션)

#### **Table III: Ablation 연구 결과**

> **인용**: "TABLE III: The ablation study on CALVIN benchmark over the effect of action space, history integration, and history organizing format. All variants are trained on split ABCD and tested on split D. 'Disc.' is short for discrete and 'Cont.' represents continuous action space. Note that for VLAs with LLaVA backbone, we utilize a perceiver resampler to downsample its vision tokens to 64 for fair comparison. Results are reported with models trained maximally within 5 epochs on the ABCD training splits." (논문 Table III)

#### **LLaVA 백본 성능**

| Structure | Action Space | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|-----------|--------------|---|---|---|---|---|-----------|
| One-Step | Disc. | 0.809 | 0.484 | 0.278 | 0.175 | 0.103 | 1.85 |
| One-Step | Cont. | 0.793 | 0.592 | 0.420 | 0.329 | 0.235 | 2.37 |
| Interleaved | Cont. | 0.892 | 0.645 | 0.436 | 0.282 | 0.181 | 2.44 |
| Policy-Head | Cont. | 0.873 | 0.678 | 0.506 | 0.376 | 0.275 | 2.71 |

#### **Flamingo 백본 성능**

| Structure | Action Space | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|-----------|--------------|---|---|---|---|---|-----------|
| One-Step | Disc. | 0.681 | 0.318 | 0.133 | 0.062 | 0.029 | 1.22 |
| One-Step | Cont. | 0.681 | 0.354 | 0.158 | 0.076 | 0.035 | 1.30 |
| Policy-Head | Cont. | 0.964 | 0.896 | 0.824 | 0.740 | 0.662 | 4.09 |

#### **KosMos 백본 성능**

| Structure | Action Space | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|-----------|--------------|---|---|---|---|---|-----------|
| One-Step | Disc. | 0.424 | 0.097 | 0.023 | 0.005 | 0.002 | 0.55 |
| One-Step | Cont. | 0.881 | 0.599 | 0.364 | 0.221 | 0.124 | 2.19 |
| Interleaved | Cont. | 0.987 | 0.915 | 0.824 | 0.737 | 0.660 | 4.12 |
| Policy-Head | Cont. | 0.967 | 0.930 | 0.899 | 0.865 | 0.826 | 4.49 |

## 🔍 **핵심 관찰사항 (Key Observations)**

### **1. 연속 액션이 중요하다 (Continuous action matters)**

> **인용**: "Continuous action matters: By comparing two types of action spaces, continuous and discrete, as shown in Tab. III, we observe that under the single-frame formulation, continuous action spaces consistently outperform discrete ones, particularly as task horizons increase." (논문 B 섹션)

#### **성능 비교**
- **연속 액션**: 모든 백본에서 일관되게 우수한 성능
- **이산 액션**: 특히 작업 지평선이 증가할수록 성능 저하

#### **직관적 설명**
> **인용**: "This finding is intuitive: continuous actions can represent high-precision floating-point values, whereas discrete actions are limited to indexing action intervals. For long-horizon tasks, the accumulation of compounding errors significantly degrades the performance of discrete actions." (논문 B 섹션)

- **연속 액션**: 고정밀도 부동소수점 값 표현 가능
- **이산 액션**: 액션 간격 인덱싱에 제한
- **장기 작업**: 복합 오류 누적으로 인한 성능 저하

### **2. 히스토리 관측이 중요하다 (History observation matters)**

> **인용**: "History observation matters: As shown in Tab. III, under the same VLM structure (either Encoder-Decoder or Decoder-only), models incorporating history observations as input consistently outperform one-step models, achieving substantially higher success rates across all tasks." (논문 B 섹션)

#### **성능 향상**
- **동일한 VLM 구조**: 히스토리 관측을 포함한 모델이 일관되게 우수
- **모든 작업**: 상당히 높은 성공률 달성
- **히스토리 융합 전략**: 사용된 전략과 무관하게 개선

#### **히스토리 길이의 영향**
> **인용**: "Furthermore, increasing the length of an observable history can enhance performance, albeit at the cost of higher computational overhead." (논문 B 섹션)

- **성능 향상**: 관측 가능한 히스토리 길이 증가로 성능 향상
- **계산 비용**: 더 높은 계산 오버헤드 비용

### **3. 정책 헤드가 히스토리 융합을 개선한다 (Policy head improves history fusion)**

> **인용**: "Policy head improves history fusion: Among the formulations utilizing history, the interleaved history formulation performs worse than merging history via an additional policy head." (논문 B 섹션)

#### **성능 비교**
- **Policy Head**: 히스토리 융합에서 더 나은 성능
- **Interleaved**: 상대적으로 낮은 성능

#### **가설과 설명**
> **인용**: "We hypothesize that the policy head preserves the VLM's original vision-language fusion capabilities while effectively integrating historical information." (논문 B 섹션)

- **VLM 능력 보존**: 원래의 비전-언어 융합 능력 유지
- **히스토리 통합**: 히스토리 정보의 효과적 통합

#### **계산 효율성**
> **인용**: "Moreover, the interleaved formulation incurs significantly higher memory and FLOP costs during both training and inference. This suggests that incorporating history with an additional policy head is a more effective and efficient approach for VLAs." (논문 B 섹션)

- **메모리 비용**: Interleaved 방식이 훨씬 높은 메모리 비용
- **FLOP 비용**: 훈련과 추론 모두에서 높은 FLOP 비용
- **효율성**: Policy Head 방식이 더 효과적이고 효율적

## ✅ **Finding 3: VLA는 다단계 히스토리 관측을 입력으로 사용하고 연속 액션을 출력으로 사용할 때 최고 성능을 달성합니다**

> **인용**: "Finding 3: The VLA achieves its best performance when using multi-step historical observations as inputs and continuous actions as outputs. For integrating history with continuous action space, the policy head structure performs better." (논문 B 섹션)

### **핵심 요소**
1. **다단계 히스토리 관측**: 입력으로 사용
2. **연속 액션**: 출력으로 사용
3. **정책 헤드 구조**: 히스토리와 연속 액션 공간 통합에 최적

## 🌍 **Question 4: How do different formulations affect the generalization and data efficiency for VLAs?**

### **일반화와 데이터 효율성의 중요성**
> **인용**: "However, beyond the performance itself, one of the most important challenges for modern VLAs is achieving generalization to novel objects and environmental settings, which is critical for practical deployment across various robots and scenarios." (논문 B 섹션)

#### **현대 VLA의 주요 도전과제**
- **새로운 객체와 환경 설정에 대한 일반화**: 다양한 로봇과 시나리오에서의 실용적 배포에 중요
- **데이터 효율성**: 추가 도메인 내 훈련 샘플이 사용 가능할 때 높은 데이터 효율성 유지

#### **VLA의 요구사항**
> **인용**: "Conversely, when generalization is insufficient, fine-tuning the policy with a few new demonstrations becomes ideal. Thus, VLAs should inherit the generalization capabilities of VLMs in open-world settings while maintaining high data efficiency when additional in-domain training samples are available." (논문 B 섹션)

- **일반화 능력**: VLMs의 일반화 능력 상속
- **데이터 효율성**: 추가 도메인 내 훈련 샘플에서 높은 데이터 효율성 유지

### **실험 설계**
> **인용**: "To address the question, we empirically study and evaluate the generalization and data efficiency of various VLA formulations, aiming to provide practical insights for training high-performing VLAs. Specifically, we assess the generalization and data efficiency of different VLAs built with RoboVLMs by training models with different architectures and formulations on varying data scales using the CALVIN datasets." (논문 B 섹션)

#### **평가 방법**
- **일반화와 데이터 효율성**: 다양한 VLA 공식화 평가
- **실용적 통찰**: 고성능 VLA 훈련을 위한 실용적 통찰 제공
- **다양한 데이터 규모**: CALVIN 데이터셋을 사용한 다양한 데이터 규모에서 훈련

#### **비교 대상**
> **인용**: "As discussed earlier, we focus on comparing the interleaved and policy head formulations using the OpenFlamingo and KosMos backbones, which have shown strong potential among all configurations." (논문 B 섹션)

- **비교 방식**: Interleaved vs Policy Head 공식화
- **백본**: OpenFlamingo와 KosMos 백본
- **선택 이유**: 모든 구성 중 강한 잠재력을 보여줌

#### **제약사항**
> **인용**: "Note that the interleaved formulation can only be paired with a decoder-only structure." (논문 B 섹션)

- **Interleaved 방식**: 디코더 전용 구조와만 페어링 가능

## 📊 **Table IV: 다양한 공식화와 훈련 데이터 규모에서의 VLA 성능**

### **Table IV 개요**
> **인용**: "TABLE IV: The performance of VLAs implemented with different formulations and training data scales. The results for 0.1x and 1x data are the best-behaved model checkpoints within 5 epochs, and the results for 5x data are the model performance at 1st epoch. We name different implemented VLAs by their VLM backbones and the way of history modeling. Results are reported with models trained maximally within 5 epochs on the ABCD training splits." (논문 Table IV)

#### **0.1x 데이터 규모 성능**

| VLA Architecture | Data Scale | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------------------|------------|---|---|---|---|---|-----------|
| Flamingo P.H. 3B | 0.1x | 0.120 | 0.007 | 0.000 | 0.000 | 0.000 | 0.13 |
| Flamingo P.H. 4B | 0.1x | 0.448 | 0.084 | 0.014 | 0.003 | 0.001 | 0.55 |
| Flamingo P.H. 9B | 0.1x | 0.547 | 0.190 | 0.067 | 0.020 | 0.003 | 0.83 |
| KosMos Inter. | 0.1x | 0.938 | 0.701 | 0.445 | 0.270 | 0.140 | 2.49 |
| KosMos P.H. | 0.1x | 0.958 | 0.684 | 0.431 | 0.270 | 0.176 | 2.52 |

#### **1x 데이터 규모 성능**

| VLA Architecture | Data Scale | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------------------|------------|---|---|---|---|---|-----------|
| Flamingo P.H. 3B | 1x | 0.964 | 0.896 | 0.824 | 0.740 | 0.662 | 4.09 |
| Flamingo P.H. 4B | 1x | 0.936 | 0.847 | 0.750 | 0.667 | 0.586 | 3.79 |
| Flamingo P.H. 9B | 1x | 0.955 | 0.879 | 0.784 | 0.714 | 0.634 | 3.97 |
| KosMos Inter. | 1x | 0.987 | 0.915 | 0.824 | 0.737 | 0.660 | 4.12 |
| KosMos P.H. | 1x | 0.967 | 0.930 | 0.899 | 0.865 | 0.826 | 4.49 |

#### **5x 데이터 규모 성능**

| VLA Architecture | Data Scale | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------------------|------------|---|---|---|---|---|-----------|
| Flamingo P.H. 3B | 5x | 0.971 | 0.916 | 0.856 | 0.794 | 0.716 | 4.21 |
| KosMos Inter. | 5x | 0.989 | 0.940 | 0.892 | 0.842 | 0.795 | 4.46 |
| KosMos P.H. | 5x | 0.968 | 0.937 | 0.903 | 0.872 | 0.830 | 4.51 |

## 📈 **Figure 9: CALVIN 벤치마크 성능**

### **Figure 9 개요**
> **인용**: "Fig. 9: Performance on CALVIN benchmark, all models are trained on split ABCD/ABC, and evaluated on split D. We report the success rates of five consecutive tasks (left axis) and the averaged task length (right axis), using the model checkpoint at 5th epoch." (논문 Figure 9)

#### **9(a): ABCD 분할 훈련 결과**
- **모델**: Flamingo P.H. 3B, Flamingo P.H. 4B, Flamingo P.H. 9B, KosMos Inter., KosMos P.H.
- **평균 작업 길이**: 4.09, 3.79, 3.97, 4.12, 4.49

#### **9(b): ABC 분할 훈련 결과**
- **모델**: Flamingo P.H. 3B, Flamingo P.H. 9B, KosMos Inter., KosMos P.H.
- **평균 작업 길이**: 2.47, 2.20, 2.70, 4.25

## 🔍 **핵심 관찰사항**

### **1. 일반화 성능 (Generalization Performance)**

> **인용**: "For generalization performance (Fig. 9), our best model, based on the KosMos backbone and leveraging a policy head for history fusion, exhibits only a slight performance drop in zero-shot settings. In contrast, other formulations experience significant performance declines." (논문 B 섹션)

#### **KosMos P.H.의 우수성**
- **제로샷 설정**: 약간의 성능 저하만 경험
- **다른 공식화**: 상당한 성능 저하 경험

#### **모델 아키텍처의 영향**
> **인용**: "This finding highlights that the model architecture significantly impacts generalization. This conclusion is further supported by results in Fig. 5, where tasks in the evaluation set are paired with novel instructions, and Fig. 7, where our best model outperforms others by a large margin across all unseen tasks." (논문 B 섹션)

- **아키텍처 중요성**: 모델 아키텍처가 일반화에 상당한 영향
- **Figure 5**: 새로운 지시사항과 페어링된 작업에서의 결과
- **Figure 7**: 모든 Unseen 작업에서 큰 폭으로 우수한 성능

### **2. 데이터 효율성 (Data Efficiency)**

> **인용**: "For data efficiency, we observe trends similar to those for generalization. Our best model consistently achieves the highest performance when training data is scaled down, with a notably slower performance decline compared to other formulations." (논문 B 섹션)

#### **데이터 스케일링 효과**
- **일관된 최고 성능**: 훈련 데이터가 축소될 때도 최고 성능 달성
- **느린 성능 저하**: 다른 공식화 대비 현저히 느린 성능 저하

#### **모델 규모의 영향**
> **인용**: "Additionally, comparisons of encoder-decoder VLAs at different scales reveal that larger models tend to be more data efficient." (논문 B 섹션)

- **큰 모델**: 더 높은 데이터 효율성 경향
- **인코더-디코더 VLA**: 다양한 규모에서의 비교

## ✅ **Finding 4: 히스토리 융합을 위한 정책 헤드 활용이 일반화와 데이터 효율성 측면에서 최고입니다**

> **인용**: "Finding 4: Leveraging policy head for history fusion is the best in terms of generalization and data efficiency." (논문 B 섹션)

### **핵심 시사점**

#### **1. 아키텍처 선택의 중요성**
- **정책 헤드**: 히스토리 융합에 최적
- **일반화**: 제로샷 설정에서 우수한 성능 유지
- **데이터 효율성**: 제한된 데이터에서도 높은 성능

#### **2. 실용적 가치**
- **실제 배포**: 다양한 로봇과 시나리오에서의 실용적 배포 가능
- **적응성**: 새로운 객체와 환경에 대한 강건한 적응성
- **효율성**: 계산 비용과 성능의 균형

#### **3. VLA 설계 가이드라인**
- **다단계 히스토리**: 입력으로 활용
- **연속 액션**: 출력으로 사용
- **정책 헤드**: 히스토리 융합에 최적
- **백본 선택**: KosMos 백본의 우수성

## 🎯 **결론**

### **VLA 설계의 핵심 원칙**
1. **연속 액션 공간**: 고정밀도 표현과 장기 작업에 유리
2. **히스토리 관측**: 모든 작업에서 상당한 성능 향상
3. **정책 헤드**: 히스토리 융합에 가장 효과적이고 효율적
4. **백본 선택**: KosMos 백본의 우수한 일반화와 데이터 효율성

### **실용적 시사점**
- **모델 아키텍처**: 일반화와 데이터 효율성에 결정적 영향
- **계산 효율성**: 정책 헤드 방식이 메모리와 FLOP 비용 측면에서 효율적
- **확장성**: 다양한 데이터 규모에서 일관된 성능

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
