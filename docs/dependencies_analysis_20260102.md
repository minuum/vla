# PyTorch 환경 의존성 비교 및 호환성 분석

**일시**: 2026-01-02  
**목적**: Jesson Orin INT8 quantization을 위한 올바른 의존성 조합 찾기

---

## 📦 현재 설치된 버전 (Jetson 실제)

```
Python: 3.10.12 ✅
PyTorch: 2.3.0 (CUDA 12.2) ✅
torchvision: 0.18.0a0+6043bc2 (설치 안 됨 - 필요)
torchaudio: 2.3.0+952ea74 (설치 안 됨 - 선택)
transformers: 4.41.2 ✅
accelerate: 1.12.0 (업그레이드됨, 원래 1.7.0)
bitsandbytes: 0.43.1 (소스 빌드) ✅
CUDA: 12.2.140 ✅
```

---

## 📝 Poetry pyproject.toml 버전

### Mobile_VLA (Robo+/Mobile_VLA/pyproject.toml)
```toml
torch = "2.3.0"
torchvision = "0.18.0"
torchaudio = "2.3.0"
transformers = "4.41.2"
accelerate = "^1.7.0"  # ← 1.7.0 이상
```

### RoboVLMs (RoboVLMs/pyproject.toml)
```toml
torch = { path = "wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl" }
torchvision = { path = "wheels/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl" }
torchaudio = { path = "wheels/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl" }
transformers = "4.41.2"
accelerate = "^1.7.0"  # ← 1.7.0 이상
```

**핵심**: RoboVLMs가 **로컬 wheel 파일**을 사용!

---

## 🔍 문제 분석

### INT8 quantization 실패 원인

1. **accelerate 버전**: 1.7.0, 1.12.0 모두 `dispatch_model`에 BitsAndBytes 예외 처리 없음
2. **transformers**: 4.41.2가 `dispatch_model` 무조건 호출
3. **호환성 문제**: accelerate + transformers + bitsandbytes 조합 문제

### 검증된 조합 찾기 (HuggingFace 예제 기반)

**Llama 2 INT8 예제** (성공 사례):
```
transformers: 4.33.0
accelerate: 0.23.0
bitsandbytes: 0.41.1
```

**BLOOM INT8 예제** (성공 사례):
```
transformers: 4.31.0
accelerate: 0.21.0
bitsandbytes: 0.40.0
```

**공통점**:
- transformers < 4.35
- accelerate < 1.0
- bitsandbytes < 0.42

---

## 💡 해결 방안

### Option 1: 구버전 다운그레이드 (⚠️ 위험)

```bash
pip install transformers==4.33.0 accelerate==0.23.0
```

**문제점**:
- Kosmos-2 지원 여부 불확실
- Poetry 환경 깨질 수 있음
- 다른 패키지 의존성 충돌

### Option 2: Poetry 환경 재구성 (✅ 권장)

```bash
cd /home/soda/vla/RoboVLMs
poetry install --no-root
poetry shell
```

**장점**:
- 격리된 환경
- wheel 파일 사용 (Jetson 최적화)
- 의존성 lock

### Option 3: FP16 사용 (✅ 즉시 가능)

```python
model = Kosmos2ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```

**메모리**: 3GB (INT8 2GB vs FP16 3GB)

### Option 4: Transformers 패치 (🔧 고급)

`transformers/modeling_utils.py` 수정하여 BitsAndBytes 예외 처리 추가

---

## 🎯 권장 진행 순서

1. **RoboVLMs Poetry 환경 확인**
   ```bash
   cd /home/soda/vla/RoboVLMs
   ls -la wheels/  # wheel 파일 존재 확인
   ```

2. **Poetry 환경에서 테스트**
   ```bash
   poetry install
   poetry run python test_int8.py
   ```

3. **실패 시 FP16으로 진행**
   - 안정적
   - 3GB 메모리 (충분히 절감)

---

## ❓ 확인 필요 사항

1. `RoboVLMs/wheels/` 디렉토리에 wheel 파일 존재 여부
2. Poetry lock 파일의 실제 버전
3. 다른 프로젝트와의 의존성 공유 필요 여부

---

**다음 단계**: RoboVLMs wheels 디렉토리 확인 →
