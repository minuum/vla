# 🔬 Pretrained VLM Instruction Grounding Test 결과

**테스트 일시**: 2026-01-11 09:39  
**Checkpoint**: Epoch 3 (Best, val_loss=0.093)  
**파일**: `runs/mobile_vla_pretrained/.../epoch_epoch=03-val_loss=val_loss=0.093.ckpt`

---

## ❌ 결과: Instruction Grounding 실패

### 실험 1: 파라미터 상태 확인

| 구성 요소 | Trainable | Frozen | 상태 |
|-----------|-----------|--------|------|
| **VLM** | 1 | 885 | ✅ Frozen (99.9%) |
| **Action Head** | 24 | 0 | ✅ Trainable (100%) |

### 실험 2: 다른 이미지, 동일 Instruction

- **Instruction**: "Navigate around the obstacle on the left side"
- **Image 1**: seed=42
- **Image 2**: seed=123 (다른 이미지)

**결과**: Action 차이 = **0.009378**

⚠️ **문제 발견**: 다른 이미지인데 거의 동일한 Action 출력  
→ Vision 처리가 제대로 작동하지 않음

### 실험 3: 동일 이미지, 다른 Instruction

- **Image**: seed=42 (동일)
- **Instruction 1**: "Navigate around the obstacle on the **left** side"
- **Instruction 2**: "Navigate around the obstacle on the **right** side"

**결과**: Action 차이 = **0.000000** ❌

❌ **Instruction Grounding 완전 실패**  
→ LEFT/RIGHT를 전혀 구분하지 못함

---

## 📊 비교 분석

| 모델 | VLM Source | Epoch | Val Loss | Image 반응 | Instruction 반응 | Grounding |
|------|-----------|-------|----------|-----------|-----------------|-----------|
| **Chunk5** | Kosmos-2 scratch | 6 | 0.067 | 0.073 | 0.000 | ❌ |
| **Pretrained VLM** | **Google Robot** | **3** | **0.093** | **0.009** | **0.000** | ❌ |

### 핵심 발견

1. **Pretrained VLM도 Instruction Grounding 실패**
   - LEFT/RIGHT 차이: 0.000000 (완전히 동일)
   - 기존 Chunk5와 동일한 문제

2. **Vision 처리도 문제**
   - 다른 이미지 차이: 0.009 (매우 작음)
   - 기존 0.073보다 **더 악화됨**

3. **Val Loss는 낮지만 Grounding은 실패**
   - Val Loss 0.093: 수치상으로는 양호
   - 실제 Grounding: 완전 실패
   - **Loss ≠ Grounding 능력**

---

## 🤔 원인 분석

### 1. Frozen VLM의 근본적 한계

```
RoboVLMs Pretrained VLM
  ↓ [Frozen - no fine-tuning]
모든 instruction이 동일한 embedding으로 고정됨
  ↓
Action Head가 instruction 구분 불가능
```

**문제**: VLM이 Frozen 상태이므로  
- Text embedding이 학습되지 않음
- LEFT ≈ RIGHT ≈ 모든 instruction
- Action Head는 embedding 차이가 없어서 구분 불가

### 2. Pretrained의 한계

RoboVLMs Google Robot checkpoint는:
- ✅ Robot instruction 이해 능력 **있음**
- ❌ 하지만 **Frozen 시키면** 그 능력 활용 못함

**핵심**: Pretrained VLM을 freeze하면  
→ Pretrained의 장점이 사라짐!

---

## 💡 해결 방안

### 옵션 1: LoRA Fine-tuning (권장) ⭐

```json
{
  "train_setup": {
    "freeze_backbone": false,  // ← Unfreeze!
    "lora_enable": true,       // ← LoRA 활성화
    "lora_r": 16,
    "lora_alpha": 32
  }
}
```

**예상 효과**:
- ✅ VLM이 instruction 차이 학습 가능
- ✅ 메모리 효율적 (LoRA adapters만 학습)
- ✅ Pretrained 지식 유지하면서 fine-tune

### 옵션 2: VLM Unfreezing

```json
{
  "train_setup": {
    "freeze_backbone": false,
    "train_decoder_layers": 4  // 마지막 4개 layer만
  }
}
```

**단점**:
- ❌ 메모리 많이 필요
- ❌ Overfitting 위험

### 옵션 3: PaliGemma 계속

기존 PaliGemma + LoRA 전략 계속 진행
- Epoch 0에서 이미 0.05 차이 나타남
- 추가 학습으로 개선 가능성

---

## 🎯 결론

### Pretrained VLM 실험 결과

| 항목 | 결과 |
|------|------|
| **학습** | ✅ 성공 (10 epochs, val_loss=0.093) |
| **VLM Frozen** | ✅ 확인 (885/886 frozen) |
| **Instruction Grounding** | ❌ **완전 실패** (diff=0.000) |
| **Vision 처리** | ⚠️ **악화** (diff=0.009) |

### 핵심 교훈

> **Frozen VLM은 Instruction Grounding 불가능**

1. VLM을 freeze하면 text embedding이 고정됨
2. Pretrained 여부와 무관하게 instruction 구분 못함
3. **LoRA fine-tuning이 필수**

### 다음 단계

**옵션 A**: Pretrained VLM + LoRA fine-tuning  
**옵션 B**: PaliGemma + LoRA 계속 학습

**권장**: **옵션 B** (PaliGemma LoRA)
- 이미 0.05 차이 확인됨
- 추가 epochs로 개선 가능성 높음
- 메모리 효율적

---

**테스트 완료 시각**: 2026-01-11 09:42
