# A. Why do we prefer VLAs?

> **인용**: 논문 A. Why do we prefer VLAs? 섹션

## 🎯 **핵심 질문 탐구**

### **주요 연구 질문**
> **인용**: "This section investigates one of the key questions: Why do we want VLAs?" (논문 A 섹션)

이 섹션은 VLA를 선호하는 이유에 대한 핵심 질문을 탐구하며, 다음의 선행 질문에 답하는 것으로 시작합니다.

### **Question 1: Are VLAs a proper choice for building generalist robot policies?**

> **인용**: "Question 1: Are VLAs a proper choice for building generalist robot policies?" (논문 A 섹션)

이 질문에 답하기 위해, 논문은 SimplerEnv 시뮬레이션 벤치마크(WidowX+Bridge 및 Google Robot 환경 포함)에 대한 다양한 VLA 모델의 평가 결과를 제시합니다.

## 📊 **Figure 5: SimplerEnv 시뮬레이션 벤치마크 평가 결과**

### **벤치마크 개요**
> **인용**: "Fig. 5: Evaluation results on the SimplerEnv simulation benchmarks, including the WidowX+Bridge and Google Robot environments. Performance of baseline methods are referred from Li et al. [25]. The KosMos P.H. built by RoboVLMs is the best VLA structure investigated, which is trained over fixed training steps. Detailed numerical results can be further referred to Appendix H." (논문 Figure 5)

### **성능 비교 결과**

#### **WidowX+Bridge 환경에서의 성능**
- **KosMos P.H. (RoboVLMs)**: **0.38** (최고 성능)
- **Octo-Small**: 0.30
- **RT-1-X**: 0.16
- **Octo-Base**: 0.01
- **OpenVLA-7b**: 0.01
- **RT-2-X**: 중간 수준 성능

#### **Google Robot 환경에서의 성능**
- **KosMos P.H. (RoboVLMs)**: **0.62** (최고 성능)
- **RT-2-X**: 0.61
- **RT-1-X**: 0.49
- **OpenVLA-7b**: 0.33
- **Octo-Base**: 0.15
- **Octo-Small**: 중간 수준 성능

### **핵심 시사점**
- **KosMos P.H. (RoboVLMs)**는 두 벤치마크 환경 모두에서 일관되게 가장 높은 평균 성공률 달성
- 대부분의 모델에서 Google Robot 환경에서의 성능이 WidowX+Bridge 환경보다 전반적으로 더 높음
- VLA 구조 선택의 중요성 강조

## 🏆 **CALVIN 벤치마크 성능**

### **최고 성능 달성**
> **인용**: "On CALVIN benchmark, our best model achieves the highest performance in all metrics and demonstrates superior generalization ability when transferring from ABC to D (a novel scene unseen in the training splits) with an absolute improvement of 12.6% for the execution of a single task and a total improvement of 30.3% for 5 consecutive tasks." (논문 A 섹션)

- **모든 지표에서 최고 성능**: CALVIN 벤치마크의 모든 평가 지표에서 최고 성능 달성
- **일반화 능력**: ABC에서 D로 전이할 때 우수한 일반화 능력 입증
  - **단일 작업**: 12.6% 절대 개선
  - **5개 연속 작업**: 30.3% 총 개선

### **제로샷 성능**
> **인용**: "On average, under zero-shot settings, our model can finish 4.25 tasks out of 5 tasks for each single rollout, outperforming the previous SOTA model (GR-1) by 1.09 tasks." (논문 A 섹션)

- **평균 달성 작업**: 5개 작업 중 4.25개 작업 완료
- **SOTA 대비**: 이전 SOTA 모델(GR-1)보다 1.09개 작업 더 달성

## 🎯 **SimplerEnv 벤치마크 성능**

### **최고 평균 성능**
> **인용**: "On SimplerEnv, our model achieves the highest average performance on both WidowX + Bridge and Google Robot environments, demonstrating the general effectiveness and robustness against different settings and diverse manipulation tasks." (논문 A 섹션)

- **WidowX + Bridge 환경**: 최고 평균 성능 달성
- **Google Robot 환경**: 최고 평균 성능 달성
- **일반적 효과성**: 다양한 설정과 조작 작업에 대한 일반적 효과성 입증
- **강건성**: 다양한 설정에 대한 강건성 입증

## 🔬 **비전-언어 사전 훈련의 영향**

### **일반화 및 데이터 효율성**
> **인용**: "We also investigated the impact of vision-language pre-training on the generalization and data efficiently (Fig. 6 and Tab. IV), and the detailed result in shown in Appendix H." (논문 A 섹션)

### **Figure 6: 비전-언어 사전 훈련 Ablation 연구**

#### **실험 설정**
- **일반화 평가**: CALVIN의 공식 설정 채택 (ABC 분할에서 훈련, D 분할에서 성능 검증)
- **데이터 효율성 평가**: 다양한 모델 규모(3B~9B)와 데이터 규모
  - **10% 훈련 데이터** (0.1x ABCD)
  - **표준 설정** (ABCD)
  - **500% 훈련 데이터** (5x ABCD)

#### **주요 발견**
> **인용**: "We can see that vision-language pre-training is essential for both generalization and data efficiency." (논문 A 섹션)

- **일반화**: 비전-언어 사전 훈련이 일반화에 필수적
- **데이터 효율성**: 비전-언어 사전 훈련이 데이터 효율성에 필수적

#### **직관적 설명**
> **인용**: "This observation is intuitive, as an aligned vision-language representation provides a robust foundation for visual understanding, enabling the policy to focus on learning manipulation skills." (논문 A 섹션)

- **정렬된 비전-언어 표현**: 시각적 이해를 위한 강건한 기반 제공
- **정책 집중**: 조작 기술 학습에 집중할 수 있도록 함

## ✅ **Finding 1: VLA는 일반화된 로봇 정책을 구축하는 데 유망한 경로입니다**

> **인용**: "Finding 1: VLA is a promising path to generalist robot policies." (논문 A 섹션)

### **근거**
- **시뮬레이션 성능**: CALVIN과 SimplerEnv 벤치마크에서 최고 성능 달성
- **일반화 능력**: 새로운 장면과 작업에 대한 우수한 일반화
- **데이터 효율성**: 제한된 데이터에서도 우수한 성능
- **비전-언어 사전 훈련**: 강건한 기반 제공

## 🌍 **Question 2: How do VLAs perform in real-world scenarios?**

### **Sim-to-Real 갭 문제**
> **인용**: "However, although VLAs perform well in simulation, it is still an open problem whether VLAs are suitable for real-robot applications due to the sim-to-real gap [54]." (논문 A 섹션)

시뮬레이션에서 잘 작동하더라도, sim-to-real 갭으로 인해 실제 로봇 응용에 적합한지는 여전히 미해결 문제입니다.

### **실제 환경 배포**
> **인용**: "As discussed above, we deploy the best-performing RoboVLM model, that is, the one based on the decoder-only KosMos in real-world scenarios to validate its effectiveness." (논문 A 섹션)

디코더 전용 KosMos 기반의 최고 성능 RoboVLM 모델을 실제 시나리오에 배포하여 그 효과를 검증합니다.

### **실험 설정**
> **인용**: "As shown in Fig. 4, our experiment involves 20 tasks with multiple skills, including Open, Close, Press Button, Pick & Place, etc. For each task, we evaluate five rollouts, with the basic setting, novel skill description, unseen distractors, unseen target object, and unseen background." (논문 A 섹션)

- **작업 수**: 20개 작업
- **스킬 유형**: Open, Close, Press Button, Pick & Place 등
- **평가 롤아웃**: 각 작업당 5회 롤아웃
- **평가 설정**: Basic, Novel Skill Description, Unseen Distractors, Unseen Target Object, Unseen Background

### **로봇 시스템**
> **인용**: "Our robot system for real experiments is built on a 7-DoF Kinova Gen3 robot arm paired with a Robotiq 2F-85 gripper, please refer to Sec. IV for more details of the real robot." (논문 A 섹션)

- **로봇 팔**: 7-DoF Kinova Gen3
- **그리퍼**: Robotiq 2F-85

### **입력 시스템**
> **인용**: "For input, we take the RGB images for the two cameras equipped on the robot head and wrist separately. The head camera provides an overview of the workspace while the gripper camera offers a close observation of the interaction area between the end effector and the environment." (논문 A 섹션)

- **헤드 카메라**: 작업 공간의 전체적인 시야 제공
- **그리퍼 카메라**: 엔드 이펙터와 환경 간의 상호 작용 영역 근접 관찰

## 📊 **Figure 7: 실제 로봇 성능 비교**

### **성능 비교 결과**

#### **Simple Setting**
- **KosMos P.H. (RoboVLMs)**: 0.75
- **OpenVLA**: 0.45
- **Octo**: 0.55

#### **Novel Skill-Description Setting**
- **KosMos P.H. (RoboVLMs)**: 0.60
- **OpenVLA**: 0.40
- **Octo**: 0.50

#### **Unseen Distractor Setting**
- **KosMos P.H. (RoboVLMs)**: 0.50
- **OpenVLA**: 0.15
- **Octo**: 0.45

#### **Unseen Background Setting**
- **KosMos P.H. (RoboVLMs)**: 0.55
- **OpenVLA**: 0.25
- **Octo**: 0.20

#### **Unseen Object Setting**
- **KosMos P.H. (RoboVLMs)**: 0.33
- **OpenVLA**: 0.13
- **Octo**: 0.13

#### **Unseen Average Setting**
- **KosMos P.H. (RoboVLMs)**: 0.51
- **OpenVLA**: 0.24
- **Octo**: 0.33

### **핵심 관찰**
> **인용**: "We observe that the best VLA (KosMos P.H.) built by RoboVLMs achieves the best performance in all evaluation setups, extremely on Simple and Unseen Background, demonstrating their effectiveness and generalization ability, which is consistent with the results in SimplerEnv and CALVIN simulation." (논문 A 섹션)

- **모든 평가 설정에서 최고 성능**: 모든 평가 설정에서 최고 성능 달성
- **Simple과 Unseen Background에서 특히 우수**: 효과성과 일반화 능력 입증
- **시뮬레이션 결과와 일관성**: SimplerEnv와 CALVIN 시뮬레이션 결과와 일관성

## 🎯 **자기 수정 능력 (Self-Correction Ability)**

### **Figure 8: 자기 수정 능력 시각화**
> **인용**: "Fig. 8: Visualization for rollouts that the best setting VLA built by RoboVLMs emerges the ability of self-correction. For instance, in the Open The Oven task, the robots' first attempt does not reach the oven handle, and it adjusts the end-effector position to re-locate the handle at the second attempt. Note that the training dataset does not contain this kind of data." (논문 Figure 8)

### **자기 수정 능력의 특징**
> **인용**: "Furthermore, as shown in Fig. 8, KosMos P.H. emerges with self-correction ability, it can realize the incorrect positions of the end effector and correct its future trajectory to complete the task successfully. Note that this ability does not appear in the other tested baselines, and this kind of data is not contained in the training dataset." (논문 A 섹션)

- **오류 인식**: 엔드 이펙터의 부정확한 위치를 인식
- **궤적 수정**: 미래 궤적을 수정하여 작업 성공적으로 완료
- **독특한 능력**: 다른 테스트된 베이스라인에서는 나타나지 않는 능력
- **훈련 데이터에 없는 능력**: 훈련 데이터셋에 포함되지 않은 종류의 데이터

### **구체적 예시**
- **Open The Oven 작업**: 첫 시도에서 오븐 손잡이에 도달하지 못했을 때, 두 번째 시도에서 엔드 이펙터 위치를 조정하여 손잡이를 다시 찾아냄

## ✅ **Finding 2: RoboVLMs로 구축된 최적의 VLA 설정은 실제 시나리오에서 강력한 효과와 강건성을 보여줍니다**

> **인용**: "Finding 2: The best setup VLA built by RoboVLMs appears strong effectiveness and robustness in real scenarios." (논문 A 섹션)

### **근거**
- **모든 평가 설정에서 최고 성능**: 실제 로봇 벤치마크에서 모든 설정에서 최고 성능
- **Unseen 설정에서 특히 우수**: 새로운 객체, 방해물, 배경에 대한 우수한 성능
- **자기 수정 능력**: 훈련 데이터에 없는 상황에서도 스스로 오류를 수정하는 능력
- **시뮬레이션과 일관성**: 시뮬레이션 결과와 일관된 성능

## 🎯 **결론**

### **VLA의 우수성 입증**
1. **시뮬레이션 성능**: CALVIN과 SimplerEnv에서 최고 성능 달성
2. **실제 환경 성능**: 실제 로봇 벤치마크에서 모든 설정에서 최고 성능
3. **일반화 능력**: 새로운 장면, 작업, 객체에 대한 우수한 일반화
4. **자기 수정 능력**: 훈련 데이터에 없는 상황에서도 적응하는 능력
5. **비전-언어 사전 훈련**: 강건한 기반 제공

### **실용적 시사점**
- **VLA는 일반화된 로봇 정책을 구축하는 유망한 경로**
- **적절한 백본과 구조 선택의 중요성**
- **비전-언어 사전 훈련의 필수성**
- **실제 환경에서의 강건성과 효과성**

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
