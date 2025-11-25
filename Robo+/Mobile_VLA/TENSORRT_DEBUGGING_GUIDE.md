# 🔧 TensorRT 디버깅 가이드

## 🚨 현재 발생한 오류 분석

### ❌ **CUDA 초기화 오류**
```
[TRT] [E] createInferRuntime: Error Code 6: API Usage Error 
(CUDA initialization failure with error: 35)
```

### 🔍 **오류 원인 분석**
1. **CUDA 드라이버 버전 불일치**
2. **TensorRT와 CUDA 버전 호환성 문제**
3. **GPU 메모리 할당 실패**
4. **환경 변수 설정 문제**

## 🛠️ TensorRT 디버깅 해결 방법

### 1. **시스템 환경 확인**

#### ✅ **현재 환경 상태**
- **GPU**: NVIDIA RTX A5000
- **CUDA Driver**: 560.35.05
- **CUDA Runtime**: 12.6
- **CUDA Compiler**: 12.1.105
- **TensorRT**: 10.13.2.6

#### ⚠️ **버전 호환성 문제**
- **TensorRT 10.13.2.6**는 **CUDA 12.1**과 호환
- **시스템 CUDA**: 12.6 (불일치)

### 2. **해결 방법**

#### 🔧 **방법 1: CUDA 12.1 설치**
```bash
# CUDA 12.1 설치
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

#### 🔧 **방법 2: TensorRT 버전 다운그레이드**
```bash
# TensorRT 8.6.1 설치 (CUDA 12.6 호환)
pip uninstall tensorrt
pip install tensorrt==8.6.1
```

#### 🔧 **방법 3: 환경 변수 설정**
```bash
# CUDA 경로 설정
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### 3. **TensorRT 최적화 기법**

#### 🚀 **주요 최적화 기법**
1. **그래프 최적화**: 불필요한 레이어 제거 및 융합
2. **정밀도 보정**: FP16/INT8 양자화
3. **커널 자동 튜닝**: GPU 최적화
4. **텐서 메모리 최적화**: 메모리 재사용

#### 📊 **예상 성능 향상**
- **FP16 양자화**: 2-3배 성능 향상
- **INT8 양자화**: 3-5배 성능 향상
- **그래프 최적화**: 10-30% 성능 향상

### 4. **대안 해결책**

#### 🔄 **ONNX Runtime 최적화**
```python
# ONNX Runtime 최적화 설정
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_use_max_workspace': '1',
        'do_copy_in_default_stream': '1',
    }),
    'CPUExecutionProvider'
]
```

#### 🔄 **PyTorch 최적화**
```python
# PyTorch 최적화
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
model = torch.jit.script(model)  # TorchScript 최적화
```

### 5. **디버깅 스크립트**

#### 🔍 **TensorRT 상태 확인**
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def check_tensorrt_status():
    """TensorRT 상태 확인"""
    try:
        # TensorRT 버전 확인
        print(f"TensorRT Version: {trt.__version__}")
        
        # CUDA 컨텍스트 확인
        cuda.init()
        device = cuda.Device(0)
        print(f"GPU: {device.name()}")
        print(f"Compute Capability: {device.compute_capability()}")
        
        # 메모리 확인
        free, total = cuda.mem_get_info()
        print(f"GPU Memory: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
        
        return True
    except Exception as e:
        print(f"TensorRT check failed: {e}")
        return False
```

#### 🔍 **간단한 TensorRT 테스트**
```python
def simple_tensorrt_test():
    """간단한 TensorRT 테스트"""
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        if builder.platform_has_fast_fp16:
            print("✅ FP16 지원됨")
        else:
            print("❌ FP16 지원 안됨")
            
        if builder.platform_has_fast_int8:
            print("✅ INT8 지원됨")
        else:
            print("❌ INT8 지원 안됨")
            
        return True
    except Exception as e:
        print(f"TensorRT test failed: {e}")
        return False
```

### 6. **권장 해결 순서**

#### 🎯 **1단계: 환경 확인**
```bash
# 1. CUDA 환경 확인
nvidia-smi
nvcc --version

# 2. TensorRT 설치 확인
python -c "import tensorrt as trt; print(trt.__version__)"
```

#### 🎯 **2단계: 버전 호환성 해결**
```bash
# TensorRT 8.6.1 설치 (CUDA 12.6 호환)
pip uninstall tensorrt
pip install tensorrt==8.6.1
```

#### 🎯 **3단계: 테스트 실행**
```bash
# 간단한 TensorRT 테스트
python -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print('TensorRT 초기화 성공!')
"
```

### 7. **성능 비교 예상**

#### 📊 **TensorRT 적용 시 예상 성능**
| 프레임워크 | 현재 성능 | TensorRT 적용 후 | 향상률 |
|------------|-----------|------------------|--------|
| **PyTorch** | 0.377ms | 0.1-0.2ms | **2-4배** |
| **ONNX Runtime** | 4.852ms | 1-2ms | **3-5배** |

#### 🤖 **로봇 제어에서의 의미**
- **0.1ms 추론**: 거의 즉시 반응
- **20ms 제어 주기**: 0.5% 사용 (매우 여유)
- **10ms 제어 주기**: 1% 사용 (완벽한 실시간)

### 8. **참고 자료**

#### 📚 **공식 문서**
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/tensorrt/)
- [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

#### 🔧 **디버깅 도구**
- **NVIDIA Nsight Systems**: 성능 프로파일링
- **TensorRT Inspector**: 모델 분석
- **CUDA-GDB**: GPU 디버깅

---

**현재 상태**: TensorRT 설치 완료, CUDA 버전 호환성 문제  
**다음 단계**: TensorRT 8.6.1 다운그레이드로 해결 시도  
**예상 결과**: 2-4배 성능 향상으로 0.1ms 추론 가능
