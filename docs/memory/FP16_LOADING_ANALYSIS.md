# FP16 모델 로딩 가능성 분석

## 🔍 현재 상황

### 관찰된 사실
- **시간**: 10분+ 소요 (정상 1-2분)
- **RAM**: 2.71GB → 4.49GB (+1.78GB)
- **상태**: RUNNING이지만 출력 없음
- **여유 RAM**: 10.39GB (충분)

---

## 💡 가능성 분석

### 1. 모델 파일 크기 확인 필요

**가설**: Kosmos-2 모델이 예상보다 클 수 있음

**확인 방법**:
```bash
# 모델 디렉토리 크기
du -sh .vlms/kosmos-2-patch14-224

# 개별 파일 크기
ls -lh .vlms/kosmos-2-patch14-224/
```

**예상**:
- Kosmos-2 2B 모델: FP32 ~8GB, FP16 ~4GB
- Safetensors 파일들의 총 크기

---

### 2. 디스크 I/O 병목

**가설**: 느린 스토리지에서 대용량 파일 읽기

**가능성**: 🟡 중간 (40%)

**증거**:
- RAM 증가가 점진적 (+1.78GB)
- 10분 지속 = 매우 느린 I/O

**해결 방안**:
```bash
# 스토리지 속도 확인
hdparm -t /dev/nvme0n1  # 또는 해당 디스크

# 모델을 tmpfs로 이동 (RAM에 로딩)
mkdir /tmp/model_cache
cp -r .vlms/kosmos-2-patch14-224 /tmp/model_cache/
```

---

### 3. Transformers 캐시 생성

**가설**: Transformers가 모델 캐시 생성 중

**가능성**: 🟢 높음 (60%)

**증거**:
- 첫 로딩은 항상 느림
- 캐시 디렉토리: `~/.cache/huggingface/`

**확인**:
```bash
# 캐시 디렉토리 확인
ls -lh ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/
```

**해결 방안**:
```bash
# 캐시 위치 변경 (빠른 디스크로)
export HF_HOME=/tmp/huggingface_cache
```

---

### 4. GPU 메모리 할당 지연

**가설**: CUDA가 GPU 메모리 할당에 시간이 걸림

**가능성**: 🔴 낮음 (10%)

**증거**:
- PyTorch CUDA는 이미 작동 확인됨
- 보통 수 초 내 완료

---

### 5. device_map="auto" 계산 시간

**가설**: Accelerate가 최적 디바이스 맵 계산 중

**가능성**: 🟡 중간 (30%)

**증거**:
- `device_map="auto"`는 모델 크기 분석 필요
- 큰 모델일수록 시간 소요

**해결 방안**:
```python
# device_map 직접 지정
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0},  # 모두 GPU 0에
    trust_remote_code=True
)
```

---

### 6. 교착 상태 (Deadlock)

**가설**: 어딘가에서 대기/멈춤

**가능성**: 🟡 중간 (20%)

**증거**:
- 10분+ 무반응
- 프로세스는 살아있음

**가능한 원인**:
- Lock 대기
- 네트워크 요청 (인터넷 연결 필요?)
- 파일 lock

---

## 🎯 추천 순서

### 1단계: 모델 파일 확인 (즉시)
```bash
du -sh .vlms/kosmos-2-patch14-224
ls -lh .vlms/kosmos-2-patch14-224/*.safetensors
```

### 2단계: 캐시 확인 (즉시)
```bash
du -sh ~/.cache/huggingface/
export HF_HOME=/tmp/hf_cache
```

### 3단계: 간단한 설정으로 재시도
```python
# Timeout 설정
import signal
signal.alarm(60)  # 60초 제한

# 간단한 device_map
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True
)
```

### 4단계: 디버그 모드
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 종합 평가

| 원인 | 가능성 | 해결 난이도 |
|------|--------|------------|
| 디스크 I/O | 40% | 쉬움 |
| 캐시 생성 | 60% | 쉬움 |
| DeviceMap 계산 | 30% | 쉬움 |
| 교착 상태 | 20% | 중간 |
| GPU 할당 | 10% | 어려움 |

**결론**: **캐시 생성이 가장 유력**하며, 해결 방법은 간단함

---

## 🚀 즉시 시도 가능

```bash
# 1. 모델 파일 크기 확인
du -sh .vlms/kosmos-2-patch14-224

# 2. 캐시 경로 변경 및 재시도
export HF_HOME=/tmp/hf_cache
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# 3. 타임아웃과 함께 재시도
timeout 120 .venv/bin/python -c "
from transformers import AutoModelForVision2Seq
import torch

model = AutoModelForVision2Seq.from_pretrained(
    '.vlms/kosmos-2-patch14-224',
    torch_dtype=torch.float16,
    device_map={'': 0},
    low_cpu_mem_usage=True
)
print('✅ 성공!')
"
```

---

**가장 유력한 원인**: Transformers 캐시 생성 (60%)  
**권장 조치**: 캐시 경로 확인 및 device_map 직접 지정
