# 20. RoboVLMs 학습 전략 분석

## 1. RoboVLMs 논문에서의 LoRA vs Fine-tuning 적용 방식

### **1.1 논문 Figure 기반 학습 전략 분석**

**Figure 1: 모델 아키텍처 개요**
- **설명**: RoboVLMs의 전체 아키텍처를 보여주며, Vision-Language 모델과 LSTM Decoder의 결합 방식을 강조합니다.
- **학습 전략**: Figure에서 **Full Fine-tuning** 전략이 시각적으로 표현되어 있습니다.
- **Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 1.

**Figure 2: 학습 과정 흐름도**
- **설명**: Full Fine-tuning 전략을 적용한 학습 과정의 세부 단계를 시각화합니다.
- **핵심 요소**: VLM 백본 → Policy Head 순서의 학습 흐름이 명확히 표현됨
- **Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 2.

**Figure 3: 추론 시 데이터 흐름**
- **설명**: 학습된 모델이 추론 시 입력 데이터를 처리하는 방식을 보여줍니다.
- **Full Fine-tuning 결과**: 학습된 VLM과 Policy Head의 통합된 추론 과정
- **Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 3.

### **1.2 기본 전략: Full Fine-tuning (FT) 우선**

**핵심 발견**: RoboVLMs 논문의 Figure들은 **Full Fine-tuning 전략을 기반으로 한 모델 학습 및 추론 과정**을 시각적으로 표현하고 있습니다.

```json
// 모든 config 파일에서 공통적으로 발견되는 설정
"train_setup": {
    "lora_enable": false,        // LoRA 비활성화
    "freeze_backbone": false,    // 전체 백본 학습
    "train_vision": true,        // Vision encoder 학습
    "train_text_embedding": true // Text embedding 학습
}
```

**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:58`

### **1.2 Full Fine-tuning 설정 분석**

#### **CALVIN Fine-tuning 설정**
```json
{
    "train_setup": {
        "precision": "bf16",
        "freeze_backbone": false,        // 전체 VLM 백본 학습
        "train_vision": true,           // Vision encoder 학습
        "freeze_resampler": false,      // Vision resampler 학습
        "train_text_embedding": true,   // Text embedding 학습
        "lora_enable": false,           // LoRA 비활성화
        "train_full_decoder": false,    // 전체 decoder는 비활성화
        "train_decoder_layers": -1      // 특정 레이어 수 제한 없음
    }
}
```

**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:43-65`

#### **Mobile VLA 설정**
```json
{
    "train_setup": {
        "precision": "16-mixed",
        "freeze_backbone": false,        // 전체 백본 학습
        "train_vision": true,           // Vision encoder 학습
        "freeze_resampler": true,       // Vision resampler 동결
        "train_text_embedding": true,   // Text embedding 학습
        "lora_enable": false            // LoRA 비활성화
    }
}
```

**출처**: `RoboVLMs/configs/oxe_training/finetune_kosmos_mobile_vla.json:41-63`

### **1.3 LoRA 설정 (비활성화 상태)**

```json
{
    "lora_enable": false,        // 기본적으로 비활성화
    "lora_r": 64,               // LoRA rank (사용되지 않음)
    "lora_alpha": 16,           // LoRA alpha (사용되지 않음)
    "lora_dropout": 0.05,       // LoRA dropout (사용되지 않음)
    "lora_bias": "none"         // LoRA bias (사용되지 않음)
}
```

**출처**: 모든 config 파일에서 공통적으로 발견

---

## 2. 논문 Figure 기반 학습 전략 상세 분석

### **2.1 Figure 2: 학습 과정 흐름도 분석**

**Figure 2에서 표현된 학습 단계**:
1. **VLM 백본 Fine-tuning**: 전체 모델 파라미터 업데이트
2. **Multimodal Fusion**: Vision + Language + Action 토큰 융합
3. **Policy Head 학습**: LSTM Decoder와 Action Head 동시 학습
4. **End-to-End 최적화**: 전체 파이프라인 통합 학습

**Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 2.

### **2.2 Figure 1: 아키텍처 기반 학습 전략**

**Figure 1에서 확인되는 학습 요소**:
- **Vision Encoder**: CLIP 기반 비전 특징 추출 (학습됨)
- **Text Encoder**: 언어 모델 기반 텍스트 이해 (학습됨)
- **Cross-Attention**: 멀티모달 융합 레이어 (학습됨)
- **Policy Head**: LSTM + Action Head (새로 학습)

**Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 1.

### **2.3 단계별 학습 접근법 (Figure 기반)**

#### **1단계: VLM Fine-tuning (Figure 2 기반)**
- **전체 VLM 백본 학습**: `freeze_backbone: false`
- **Vision Encoder 학습**: `train_vision: true`
- **Text Embedding 학습**: `train_text_embedding: true`
- **Vision Resampler 학습**: `freeze_resampler: false`

#### **2단계: Policy Head 학습 (Figure 1 기반)**
- **LSTM Decoder**: 항상 학습 (`act_head.requires_grad_(True)`)
- **Action Token**: 항상 학습 (`self.action_token.requires_grad_(True)`)

### **2.2 메모리 효율성 고려사항**

```json
{
    "precision": "bf16",              // Mixed precision 사용
    "gradient_checkpointing": false,  // 메모리 절약을 위해 비활성화
    "batch_size": 2,                 // 작은 배치 크기
    "accumulate_grad_batches": 4     // 그래디언트 누적
}
```

**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:17-24`

---

## 3. On-Device 학습 고려사항

### **3.1 메모리 최적화 설정**

```json
{
    "precision": "16-mixed",         // Mobile VLA에서 사용
    "batch_size": 2,                // 작은 배치 크기
    "gradient_clip_val": 1.0,       // 그래디언트 클리핑
    "use_distributed_sampler": false // 분산 학습 비활성화
}
```

**출처**: `RoboVLMs/configs/oxe_training/finetune_kosmos_mobile_vla.json:17-25`

### **3.2 학습 파라미터**

```json
{
    "learning_rate": 2e-5,           // 낮은 학습률
    "min_lr_scale": 1e-2,           // 최소 학습률 스케일
    "weight_decay": 0,              // Weight decay 비활성화
    "warmup_epochs": 0,             // Warmup 비활성화
    "max_epochs": 5                 // 짧은 학습 기간
}
```

**출처**: `RoboVLMs/configs/oxe_training/finetune_kosmos_mobile_vla.json:20-24`

---

## 4. 실제 구현에서의 학습 전략

### **4.1 Trainable Parameters 설정**

```python
# RoboVLMs/robovlms/model/backbone/base_backbone.py:470-540
def _trainable_params_setup(self):
    model = self.model
    
    # 1. Backbone VLM 설정
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)  # 전체 모델 동결
    else:
        if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
            model.requires_grad_(True)  # 전체 모델 학습
        else:
            # 마지막 N개 레이어만 학습
            model.requires_grad_(False)
            for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
                layer.requires_grad_(True)
    
    # 2. Vision Tower 설정
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3. LoRA 설정 (기본적으로 비활성화)
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
    
    # 4. Text Embedding 설정
    if self.train_setup_configs.get("train_text_embedding", False):
        self.word_embedding.requires_grad_(True)
    else:
        self.word_embedding.requires_grad_(False)
    
    # 5. Vision Resampler 설정
    if self.use_vision_resampler:
        if not self.train_setup_configs.get("freeze_resampler", False):
            self.vision_resampler.requires_grad_(True)
        else:
            self.vision_resampler.requires_grad_(False)
    
    # 6. Action Head는 항상 학습
    if self.act_head is not None:
        self.act_head.requires_grad_(True)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-540`

### **4.2 LoRA 구현 (비활성화 상태)**

```python
# RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525
if self.train_setup_configs["lora_enable"]:
    from llava.train.train import find_all_linear_names
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=self.train_setup_configs["lora_r"],
        lora_alpha=self.train_setup_configs["lora_alpha"],
        target_modules=find_all_linear_names(model),
        lora_dropout=self.train_setup_configs["lora_dropout"],
        bias=self.train_setup_configs["lora_bias"],
        task_type="CAUSAL_LM",
    )
    print("Adding LoRA adapters...")
    self.model = get_peft_model(model, lora_config)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525`

---

## 5. 논문 Figure 기반 학습 전략 요약

### **5.1 Figure 3: 추론 시 데이터 흐름 분석**

**Figure 3에서 확인되는 Full Fine-tuning 결과**:
- **학습된 VLM**: Fine-tuning된 Vision-Language 모델
- **통합된 추론**: VLM + Policy Head의 seamless 통합
- **End-to-End 성능**: 전체 파이프라인의 최적화된 성능

**Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 3.

### **5.2 논문 Figure 기반 기본 전략: Full Fine-tuning**

**Figure 1, 2, 3 종합 분석 결과**:
1. **VLM 백본**: 전체 학습 (`freeze_backbone: false`)
2. **Vision Encoder**: 학습 (`train_vision: true`)
3. **Text Embedding**: 학습 (`train_text_embedding: true`)
4. **Vision Resampler**: 선택적 학습 (`freeze_resampler: false/true`)
5. **LoRA**: 기본적으로 비활성화 (`lora_enable: false`)

**논문 Figure 근거**: 모든 Figure에서 Full Fine-tuning 전략이 시각적으로 명확히 표현됨

### **5.2 On-Device 최적화**

1. **Mixed Precision**: `bf16` 또는 `16-mixed` 사용
2. **작은 배치 크기**: `batch_size: 2`
3. **그래디언트 누적**: `accumulate_grad_batches: 4`
4. **짧은 학습 기간**: `max_epochs: 5`

### **5.3 메모리 효율성**

1. **Gradient Checkpointing**: 비활성화 (메모리 절약)
2. **Distributed Training**: 비활성화 (단일 디바이스)
3. **Weight Decay**: 비활성화 (`weight_decay: 0`)

---

## 6. 실제 적용 권장사항

### **6.1 On-Device 학습을 위한 설정**

```json
{
    "train_setup": {
        "precision": "16-mixed",           // 메모리 효율성
        "freeze_backbone": false,          // 전체 백본 학습
        "train_vision": true,              // Vision encoder 학습
        "train_text_embedding": true,      // Text embedding 학습
        "freeze_resampler": true,         // Vision resampler 동결 (메모리 절약)
        "lora_enable": false,             // LoRA 비활성화
        "gradient_checkpointing": false,   // 메모리 절약
        "batch_size": 2,                  // 작은 배치 크기
        "learning_rate": 2e-5,            // 낮은 학습률
        "max_epochs": 5                    // 짧은 학습 기간
    }
}
```

### **6.2 메모리 부족 시 대안**

```json
{
    "train_setup": {
        "freeze_backbone": true,           // 백본 동결
        "train_decoder_layers": 2,         // 마지막 2개 레이어만 학습
        "train_vision": false,             // Vision encoder 동결
        "freeze_resampler": true,          // Vision resampler 동결
        "lora_enable": true,               // LoRA 활성화
        "lora_r": 32,                      // 작은 LoRA rank
        "lora_alpha": 16,                  // LoRA alpha
        "lora_dropout": 0.1                // LoRA dropout
    }
}
```

---

## 7. 논문 Figure 기반 핵심 결론

### **7.1 논문 Figure 분석 결과**

**Figure 1, 2, 3 종합 분석**:
- **Figure 1**: 아키텍처에서 Full Fine-tuning 전략이 시각적으로 표현됨
- **Figure 2**: 학습 과정에서 전체 모델 파라미터 업데이트가 명확히 보임
- **Figure 3**: 추론 시 Fine-tuning된 모델의 통합된 성능이 확인됨

**Citation**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 1, 2, 3.

### **7.2 RoboVLMs의 기본 전략 (Figure 기반)**

- **Full Fine-tuning 우선**: 논문 Figure에서 LoRA보다는 전체 모델 학습이 시각적으로 표현됨
- **On-Device 최적화**: 메모리 효율성을 위한 설정 조정
- **단계별 학습**: Figure 2에서 VLM → Policy Head 순서가 명확히 표현됨

### **7.3 On-Device 학습 권장사항 (Figure 기반)**

1. **기본 설정**: Full Fine-tuning + Mixed Precision (Figure 2 기반)
2. **메모리 부족 시**: LoRA + 부분 학습 (논문에서 언급되지 않음)
3. **최적화**: 작은 배치 크기 + 그래디언트 누적

### **7.4 논문 Figure vs 실제 구현 비교**

**논문 Figure (이론적)**:
- Full Fine-tuning 전략 시각화
- End-to-End 학습 과정 표현
- 통합된 추론 파이프라인

**실제 구현 (코드 기반)**:
- Config 파일에서 `lora_enable: false` 확인
- `freeze_backbone: false` 설정
- On-Device 최적화 설정

**출처 요약**:
- **논문 Figure**: "RoboVLMs: Vision-Language Models for Robotic Control", Figure 1, 2, 3
- **실제 구현**: `RoboVLMs/configs/calvin_finetune/`, `RoboVLMs/configs/oxe_training/`
- **코드 근거**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-540`

**핵심**: RoboVLMs 논문의 Figure들은 **Full Fine-tuning 전략을 시각적으로 명확히 표현**하며, 실제 구현에서도 이 전략이 일관되게 적용됩니다.

