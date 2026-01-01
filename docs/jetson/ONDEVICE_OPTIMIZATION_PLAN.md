# Jetson On-Device VLA 메모리 최적화 계획

**날짜**: 2026-01-01  
**목표**: 16GB 통합 메모리 내에서 Mobile VLA On-Device 실행

---

## 🎯 목표 메모리 예산: 8-11GB

### Jetson Orin 메모리 특성

#### 통합 메모리 (Unified Memory)
```
총 메모리: 15.29GB
├─ CPU + GPU 공유
├─ Zero-copy 메모리 공유
└─ 동적 할당 (필요에 따라 CPU↔GPU)
```

**중요**: RAM과 GPU 메모리가 **동일한 물리 메모리**를 사용!
- FP16 측정값: RAM 7.28GB + GPU 3.11GB ≠ 10.39GB
- **실제 사용**: 약 7.28GB (GPU 3.11GB는 RAM에서 할당됨)

---

## 📊 현재 상태 (FP16 Kosmos-2)

### 측정 결과
```
베이스라인:  RAM 4.66GB
로딩 후:     RAM 7.28GB, GPU 3.11GB
순수 증가:   +2.62GB (모델 가중치)
총 사용:     7.28GB
여유:        8.01GB (52%)
```

### 구성 요소별 추정
```
모델 파일 (FP16):  6.3GB (디스크)
메모리 로드:       ~2.6GB (압축/공유)
GPU Tensor:        3.11GB
OS + ROS2:         ~2GB
여유:              8GB
```

**좋은 소식**: 현재도 여유 8GB 확보!

---

## 🎯 목표 구성: INT8 Vision + INT4 LLM

### 이상적인 메모리 예산

| 구성 요소 | FP16 (현재) | INT8/INT4 (목표) | 절감 |
|-----------|-------------|------------------|------|
| **Vision Encoder** | ~1.2GB | ~0.6GB (INT8) | -50% |
| **LLM** | ~1.5GB | ~0.4GB (INT4) | -73% |
| **Action Head** | ~0.2GB | 0.2GB (FP16) | 0% |
| **Activation** | ~1.5GB | ~1.0GB | -33% |
| **KV Cache** | ~1.0GB | ~0.8GB | -20% |
| **OS + ROS2** | ~2.0GB | ~2.0GB | 0% |
| **TensorRT/CUDA** | ~0.6GB | ~1.0GB | +67% |
| **총합** | **~8GB** | **~6GB** | **-25%** |

---

## 🔍 RoboVLMs 원본 메모리 측정 필요

### 측정 계획

#### 1. RoboVLMs 원본 (FP16)
```bash
# Vision Encoder만
python measure_vision_encoder.py --model kosmos-2

# LLM만  
python measure_llm.py --model kosmos-2

# 전체 파이프라인
python measure_full_pipeline.py
```

**측정 항목**:
- 모델 로딩 후 메모리
- 추론 시 Peak 메모리
- 배치별 메모리 증가량

#### 2. INT8 Vision Encoder
```python
# BitsAndBytes 대신 PyTorch 네이티브 quantization
import torch.quantization as quantization

# Post-Training Quantization (PTQ)
model_int8 = quantization.quantize_dynamic(
    vision_encoder,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
```

#### 3. INT4 LLM
```python
# bitsandbytes 대신 PyTorch 2.0+ native
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

---

## 🚀 단계별 실행 계획

### Phase 1: 측정 (1-2일)

**목표**: 현재 메모리 사용량 정확히 파악

```bash
# 1-1. 베이스라인
python scripts/measure_baseline_memory.py

# 1-2. Vision Encoder만 (FP16)
python scripts/measure_vision_encoder_fp16.py

# 1-3. LLM만 (FP16)
python scripts/measure_llm_fp16.py

# 1-4. Full Pipeline (FP16)
python scripts/measure_full_pipeline_fp16.py
```

**예상 결과**:
- Vision: 1-1.5GB
- LLM: 1.5-2GB
- Total: 3-4GB (모델 가중치만)

---

### Phase 2: INT8 Vision Encoder (2-3일)

#### 2-1. PyTorch Native Quantization (권장)
```python
# vision_encoder.py 수정
import torch.quantization as tq

def load_vision_encoder_int8(model_path):
    # FP16 로딩
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    
    # Freeze
    for p in model.parameters():
        p.requires_grad = False
    
    # Dynamic Quantization (PTQ)
    model_int8 = tq.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Linear만 INT8
        dtype=torch.qint8
    )
    
    return model_int8
```

**장점**:
- BitsAndBytes 의존성 없음
- PyTorch 네이티브 (Jetson 호환)
- 정확도 손실 최소 (~1-2%)

**예상 메모리**: ~0.6GB (50% 절감)

#### 2-2. TensorRT (선택적, 더 빠름)
```python
import torch_tensorrt

# TensorRT 변환
model_trt = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.int8}
)
```

**장점**:
- 더 빠른 추론 (~2-3x)
- 메모리 효율적

**단점**:
- 복잡한 설정
- 디버깅 어려움

---

### Phase 3: INT4 LLM (3-4일)

#### 3-1. GPTQ (권장, BitsAndBytes 대신)
```python
from transformers import GPTQConfig

# GPTQ 4bit
config = GPTQConfig(
    bits=4,
    dataset="c4",  # calibration dataset
    desc_act=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=config,
    device_map="auto"
)
```

**장점**:
- BitsAndBytes보다 안정적
- Jetson 호환 가능성 높음
- INT4로 75% 메모리 절감

**예상 메모리**: ~0.4GB

#### 3-2. AWQ (대안)
```python
from awq import AutoAWQForCausalLM

# AWQ 4bit
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True
)
```

---

### Phase 4: 통합 및 벤치마크 (2-3일)

```python
# 최종 구성
class MobileVLAOptimized:
    def __init__(self):
        # Vision: INT8
        self.vision = load_vision_encoder_int8()
        
        # LLM: INT4
        self.llm = load_llm_int4()
        
        # Action Head: FP16
        self.action_head = ActionHead()
    
    def forward(self, image, text):
        # Vision encoding (INT8)
        vis_feat = self.vision(image)
        
        # LLM (INT4)
        llm_out = self.llm(vis_feat, text)
        
        # Action prediction (FP16)
        action = self.action_head(llm_out)
        
        return action
```

**측정**:
- 총 메모리
- 추론 속도
- 정확도 변화

---

## 📝 예상 결과

### 메모리 사용량 비교

| 구성 | 메모리 | 여유 |
|------|--------|------|
| **FP16 (현재)** | 7-8GB | 7-8GB |
| **INT8 Vision + FP16 LLM** | 6-7GB | 8-9GB |
| **INT8 Vision + INT4 LLM** | **5-6GB** | **9-10GB** |

### 성능 Trade-off

| 구성 | 메모리 | 속도 | 정확도 |
|------|--------|------|--------|
| FP16 | 100% | 100% | 100% |
| INT8 Vision | 75% | 120% | 98% |
| INT8+INT4 | **60%** | **150%** | **95%** |

---

## 🛠️ 즉시 실행 가능

### 1. Vision Encoder INT8 테스트
```bash
# 스크립트 생성
cat > test_vision_int8.py << 'EOF'
import torch
import torch.quantization as tq
from transformers import AutoModel

# FP16 로딩
model = AutoModel.from_pretrained(
    ".vlms/kosmos-2-patch14-224",
    torch_dtype=torch.float16
).vision_model

# INT8 변환
model.eval()
model_int8 = tq.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print(f"FP16 size: {sum(p.numel()*2 for p in model.parameters())/1e9:.2f}GB")
print(f"INT8 size: {sum(p.numel() for p in model_int8.parameters())/1e9:.2f}GB")
EOF

python test_vision_int8.py
```

### 2. 메모리 측정 스크립트
```bash
# 구성 요소별 메모리 측정
python scripts/measure_model_memory.py --breakdown
```

---

## 🎯 다음 액션

### 즉시 (오늘)
1. ✅ FP16 Kosmos-2 메모리 정확히 측정
2. ⏳ Vision Encoder INT8 변환 테스트
3. ⏳ 메모리 절감 효과 확인

### 단기 (이번 주)
1. INT8 Vision + FP16 LLM 통합
2. 추론 속도 벤치마크
3. 정확도 평가

### 중기 (다음 주)
1. INT4 LLM 적용
2. 최종 메모리 6GB 이하 달성
3. 논문 데이터 수집

---

## 💡 핵심 포인트

### Jetson 통합 메모리
```
⚠️ RAM + GPU ≠ 총 메모리
✅ RAM = 총 메모리 (GPU는 RAM에서 할당)

FP16: RAM 7.28GB (실제 사용)
      GPU 3.11GB (RAM 내 Tensor)
```

### 메모리 절감 우선순위
1. **LLM INT4**: 75% 절감 (가장 큼)
2. **Vision INT8**: 50% 절감
3. **Activation 최적화**: 30% 절감

### 현실적 목표
- **현재**: 7-8GB
- **목표**: 5-6GB
- **절감**: 2-3GB (25-37%)
- **여유**: 9-10GB (충분!)

---

**결론**: Jetson 16GB에서 **충분히 가능**하며, 현재도 여유가 있음!

**다음**: Vision Encoder INT8 변환 테스트 진행
