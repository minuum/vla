# 📚 RoboVLMs 논문 Appendix C: IMPLEMENTATION DETAILS 섹션 분석

> **인용**: 논문 "APPENDIX C: IMPLEMENTATION DETAILS" 섹션

## 🎯 **1. 구현 세부사항 개요**

### **구현 세부사항의 중요성**
> **인용**: "With different formulations, the best setting of hyperparameters like batch size, weight decay, and learning rate could be varied." (논문 Appendix C 섹션)

#### **하이퍼파라미터의 중요성**
- **다양성**: 다양한 공식화에 따른 최적 하이퍼파라미터 설정의 차이
- **성능 영향**: 배치 크기, 가중치 감쇠, 학습률 등의 성능에 미치는 영향
- **최적화**: 각 실험에 최적화된 하이퍼파라미터 설정의 필요성

## ⚙️ **2. 하이퍼파라미터 및 훈련 세부사항**

### **하이퍼파라미터 설정 전략**
> **인용**: "Although OpenVLA suggests utilizing the same hyper-param as in the VLM pretrain phase, we find that a varied setting of the hyper-param could improve the performance." (논문 Appendix C 섹션)

#### **OpenVLA와의 차이점**
- **OpenVLA 제안**: VLM 사전 훈련 단계와 동일한 하이퍼파라미터 사용
- **RoboVLMs 발견**: 다양한 하이퍼파라미터 설정이 성능 향상에 도움
- **최적화**: 각 실험에 맞는 최적화된 하이퍼파라미터 설정

### **하이퍼파라미터 선택 방법**
> **인용**: "The hyperparameters for fine-tuning VLAs are mainly derived from the VLMs training setups, for example, we select the weight decay from [0, 1e − 1], and the learning rate as one of [1e − 4, 2e − 5, 1e − 5]. We conduct a grid search over and select the one with the best performance." (논문 Appendix C 섹션)

#### **가중치 감쇠 (Weight Decay)**
- **범위**: [0, 1e-1]
- **선택**: 그리드 서치를 통한 최적값 선택
- **목적**: 과적합 방지 및 일반화 성능 향상

#### **학습률 (Learning Rate)**
- **후보값**: [1e-4, 2e-5, 1e-5]
- **선택 방법**: 그리드 서치를 통한 최적값 선택
- **의의**: 모델 수렴 속도 및 최종 성능에 영향

#### **그리드 서치 (Grid Search)**
- **방법**: 모든 조합에 대한 체계적 탐색
- **목적**: 최적 성능을 달성하는 하이퍼파라미터 조합 발견
- **효율성**: 체계적 탐색을 통한 효율적 최적화

### **기본 설정**
> **인용**: "We set the global batch size as 128 and the warmup ratio is 0.25 epoch (5K steps for OpenX Embodiment pre-train)." (논문 Appendix C 섹션)

#### **배치 크기 (Batch Size)**
- **설정값**: 128
- **의의**: 메모리 효율성과 훈련 안정성의 균형
- **일관성**: 모든 실험에서 동일한 배치 크기 사용

#### **워밍업 (Warmup)**
- **일반 설정**: 0.25 epoch
- **OpenX Embodiment**: 5K steps
- **목적**: 훈련 초기 안정성 확보

### **하드웨어 설정**
> **인용**: "All models included in this paper are trained on a cluster of 4 x 8 A100 GPUs." (논문 Appendix C 섹션)

#### **GPU 클러스터**
- **구성**: 4 x 8 A100 GPUs
- **총 GPU 수**: 32개 A100 GPU
- **의의**: 대규모 모델 훈련을 위한 충분한 컴퓨팅 자원

## 📊 **3. Table VI: 하이퍼파라미터 설정**

### **실험별 하이퍼파라미터 설정**

#### **CALVIN Perform (Tab. II)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **SimplerEnv Perform (Fig. 14)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side
- **Batch Size**: 128
- **Warmup**: 5K Iters
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 50K Iters

#### **CALVIN VL Pre-train (Fig. 6)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **Real Perform (Fig. 7)**
- **Backbone**: All
- **Window Size**: 8
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **VLA Structure (Tab.III)**
**LLaVA 백본:**
- **Backbone**: LLaVA
- **Window Size**: 8
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Total Epochs/Iters**: 5 Ep

**기타 백본:**
- **Backbone**: Else
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **CALVIN Generalization (Fig. 9)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **CALVIN Data Efficiency (Tab. IV)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side+Wrist
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Total Epochs/Iters**: 5 Ep

#### **CALVIN Backbone (Tab V)**
- **Backbone**: All
- **Window Size**: 8
- **Chunk Size**: 10
- **Input View**: Side
- **Batch Size**: 128
- **Warmup**: 0.25 Ep
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Total Epochs/Iters**: 5 Ep

#### **Simpler Training Recipe (Fig 10)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side
- **Batch Size**: 128
- **Warmup**: 5K Iters
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Total Epochs/Iters**: 50K Iters

#### **CALVIN few-shot (Fig. 11)**
- **Backbone**: All
- **Window Size**: 16
- **Chunk Size**: 10
- **Input View**: Side
- **Batch Size**: 128
- **Warmup**: 0 Iter
- **Scheduler**: Constant
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Total Epochs/Iters**: 5K Iters

## 🔍 **4. 체크포인트 선택**

### **체크포인트 선택의 어려움**
> **인용**: "We find out that, normally, the performance of robot policies does not fully depend on offline evaluation metrics [16], such as the validation loss, due to the compounding error in long-horizon rollouts." (논문 Appendix C 섹션)

#### **오프라인 평가 지표의 한계**
- **검증 손실**: Validation loss의 한계
- **복합 오차**: Long-horizon rollouts에서의 복합 오차
- **성능 불일치**: 오프라인 지표와 실제 성능의 불일치

#### **체크포인트 선택의 도전과제**
- **복잡성**: 로봇 정책의 복잡한 성능 특성
- **장기 궤적**: Long-horizon rollouts의 복합 오차
- **평가 어려움**: 최적 체크포인트 선택의 어려움

### **공정한 비교를 위한 설정**
> **인용**: "Therefore, it is challenging to select the best checkpoint during training. For fair comparisons, we train all VLAs for a fixed number of epochs or timesteps." (논문 Appendix C 섹션)

#### **고정 훈련 설정**
- **목적**: 공정한 비교를 위한 일관된 설정
- **방법**: 고정된 에포크 수 또는 타임스텝 수
- **의의**: 실험 간 공정한 비교 가능

### **실험별 훈련 설정**

#### **CALVIN 실험**
> **인용**: "Concretely, on CALVIN, we train each model for 5 epochs with a batch size of 128 truncated trajectories and report the performance of the final model." (논문 Appendix C 섹션)

- **훈련 에포크**: 5 epochs
- **배치 크기**: 128 truncated trajectories
- **성능 보고**: 최종 모델의 성능
- **일관성**: 모든 모델에 동일한 설정 적용

#### **SimplerEnv 실험**
> **인용**: "For SimplerEnv, we train the model for 100K iterations with a batch size of 512 truncated trajectories and report the best-performing model with a 10K-iteration training interval." (논문 Appendix C 섹션)

- **훈련 반복**: 100K iterations
- **배치 크기**: 512 truncated trajectories
- **성능 보고**: 10K-iteration 간격으로 최고 성능 모델
- **최적화**: 최적 성능 모델 선택

#### **실제 환경 실험**
> **인용**: "In real-world experiments, we train the model for 5 epochs with a batch size of 512 truncated trajectories, and we only report the performance of the last model." (논문 Appendix C 섹션)

- **훈련 에포크**: 5 epochs
- **배치 크기**: 512 truncated trajectories
- **성능 보고**: 마지막 모델의 성능만 보고
- **실용성**: 실제 환경에서의 실용적 접근

## 📈 **5. 하이퍼파라미터 분석**

### **공통 설정**
- **Optimizer**: AdamW (모든 실험)
- **Scheduler**: Constant (모든 실험)
- **Batch Size**: 128 (대부분 실험)

### **실험별 차이점**

#### **Window Size**
- **CALVIN**: 16 (대부분), 8 (일부)
- **SimplerEnv**: 16
- **Real**: 8
- **의의**: 실험 목적에 따른 최적 윈도우 크기

#### **Input View**
- **Side+Wrist**: CALVIN, Real 실험
- **Side**: SimplerEnv, Backbone 실험
- **의의**: 실험 환경에 따른 최적 입력 뷰

#### **Learning Rate**
- **1e-4**: CALVIN, SimplerEnv 실험
- **2e-5**: LLaVA, Backbone, Training Recipe, Few-shot 실험
- **의의**: 모델 특성에 따른 최적 학습률

#### **Warmup**
- **0.25 Ep**: 대부분 실험
- **5K Iters**: SimplerEnv, Training Recipe
- **0 Iter**: Few-shot 실험
- **의의**: 실험 특성에 따른 최적 워밍업

## 🎯 **6. 구현 세부사항의 의의**

### **재현성 (Reproducibility)**
- **상세 설정**: 모든 하이퍼파라미터의 명확한 제시
- **일관성**: 실험 간 일관된 설정
- **검증 가능**: 다른 연구자들의 재현 가능

### **최적화 (Optimization)**
- **그리드 서치**: 체계적인 하이퍼파라미터 최적화
- **실험별 최적화**: 각 실험에 최적화된 설정
- **성능 향상**: 최적화를 통한 성능 향상

### **공정성 (Fairness)**
- **일관된 설정**: 공정한 비교를 위한 일관된 설정
- **고정 훈련**: 고정된 훈련 설정
- **객관적 평가**: 객관적인 성능 평가

## 🚀 **7. 결론**

### **구현 세부사항의 핵심**
1. **최적화**: 각 실험에 최적화된 하이퍼파라미터 설정
2. **일관성**: 실험 간 일관된 설정으로 공정한 비교
3. **재현성**: 상세한 설정 제시로 재현 가능한 실험

### **연구의 의의**
1. **체계적 접근**: 체계적인 하이퍼파라미터 최적화
2. **공정한 비교**: 일관된 설정을 통한 공정한 비교
3. **재현 가능성**: 상세한 구현 세부사항 제시

### **미래 연구 방향**
1. **자동화**: 자동화된 하이퍼파라미터 최적화
2. **효율성**: 더 효율적인 훈련 방법론
3. **일반화**: 다양한 도메인에 적용 가능한 설정

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
