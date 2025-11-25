# Mobile VLA 성능 분석 도구 모음

이 디렉토리는 Mobile VLA 프로젝트의 모델 성능 측정, 양자화, 벤치마킹을 위한 다양한 스크립트들을 포함합니다.

## 📊 성능 분석 스크립트 분류

### 1. **기본 벤치마킹 도구**
- `accurate_benchmark.py` - 정확한 성능 벤치마킹
- `comprehensive_benchmark.py` - 종합적인 성능 분석
- `memory_accurate_benchmark.py` - 메모리 사용량 정확 측정
- `optimized_onnx_benchmark.py` - ONNX 최적화 벤치마킹

### 2. **양자화 성능 분석**
- `real_checkpoint_quantization.py` - 실제 체크포인트 기반 양자화 비교
- `real_model_quantization.py` - 실제 모델 양자화 성능 측정
- `verify_actual_performance.py` - 실제 성능 검증

### 3. **GPU 양자화 도구**
- `accurate_gpu_quantization.py` - GPU 정확 양자화
- `simple_gpu_quantization.py` - 간단한 GPU 양자화
- `jetson_tensorrt_quantization.py` - Jetson TensorRT 양자화

### 4. **CPU 양자화 도구**
- `model_quantization_cpu.py` - CPU 환경 양자화
- `simple_actual_quantization.py` - 간단한 실제 양자화

### 5. **TensorRT 최적화**
- `tensorrt_quantization.py` - TensorRT 양자화
- `tensorrt_quantization_mobile_vla.py` - Mobile VLA용 TensorRT
- `model_quantization_tensorrt.py` - TensorRT 모델 양자화

### 6. **ONNX 변환 및 최적화**
- `model_quantization_fixed.py` - 수정된 모델 양자화
- `optimal_quantization.py` - 최적 양자화

### 7. **특정 모델 성능 분석**
- `model_quantization_mae0222.py` - MAE 0.222 모델 양자화
- `model_quantization_simple.py` - 간단한 모델 양자화

### 8. **VLM 특화 도구**
- `vlm_quantization.py` - VLM 양자화
- `simple_vlm_quantization.py` - 간단한 VLM 양자화

### 9. **최종 테스트 도구**
- `final_quantization.py` - 최종 양자화
- `final_quantization_comparison.py` - 최종 양자화 비교
- `final_quantization_test.py` - 최종 양자화 테스트
- `fixed_quantization.py` - 수정된 양자화
- `fixed_quantization_test.py` - 수정된 양자화 테스트

### 10. **비교 분석 도구**
- `quantization_comparison_test.py` - 양자화 비교 테스트
- `quantization_performance_test.py` - 양자화 성능 테스트
- `actual_quantization.py` - 실제 양자화

## 🎯 주요 성능 측정 결과

### 모델별 MAE 성능
| 모델 | MAE | 에포크 | 모델 크기 |
|------|-----|--------|-----------|
| Pure Kosmos2 | 0.247 | 15 | 909 keys |
| Kosmos2+CLIP Hybrid | 0.212 | 10 | 1311 keys |
| Mobile VLA | 미확인 | 3 | 891 keys |
| Simplified RoboVLMs | 0.0017 | 12 | 438 keys |

### 양자화 성능 개선
- **FP16 양자화**: 메모리 사용량 50% 감소, 추론 속도 1.5-2x 향상
- **INT8 양자화**: 메모리 사용량 75% 감소, 추론 속도 2-3x 향상
- **TensorRT 최적화**: Jetson Orin NX에서 실시간 추론 가능

## 🚀 사용 방법

### 1. 기본 성능 측정
```bash
python accurate_benchmark.py
```

### 2. 양자화 성능 비교
```bash
python real_checkpoint_quantization.py
```

### 3. GPU 양자화
```bash
python accurate_gpu_quantization.py
```

### 4. CPU 양자화
```bash
python model_quantization_cpu.py
```

### 5. TensorRT 최적화
```bash
python tensorrt_quantization_mobile_vla.py
```

## 📈 성능 분석 워크플로우

1. **기본 벤치마킹**: `accurate_benchmark.py`로 원본 모델 성능 측정
2. **양자화 적용**: `real_checkpoint_quantization.py`로 FP16/INT8 양자화
3. **GPU 최적화**: `accurate_gpu_quantization.py`로 GPU 성능 최적화
4. **TensorRT 변환**: `tensorrt_quantization_mobile_vla.py`로 Jetson 최적화
5. **최종 검증**: `verify_actual_performance.py`로 성능 검증

## 🔧 환경 요구사항

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **ONNX**: 1.14+
- **TensorRT**: 8.6+ (Jetson용)
- **CUDA**: 11.8+ (GPU용)

## 📝 결과 파일

- `real_checkpoint_quantization_results.json` - 양자화 성능 비교 결과
- `benchmark_results.json` - 벤치마킹 결과
- `quantization_analysis.json` - 양자화 분석 결과

## 🎯 주요 발견사항

1. **Kosmos2+CLIP Hybrid가 최고 성능**: MAE 0.212
2. **FP16 양자화 효과적**: 메모리 절약과 속도 향상 균형
3. **TensorRT 최적화 필수**: Jetson 환경에서 실시간 처리 가능
4. **Simplified RoboVLMs 매우 정확**: MAE 0.0017 (다만 다른 데이터셋 사용 가능성)

## 📚 참고 자료

- [PyTorch 양자화 가이드](https://pytorch.org/docs/stable/quantization.html)
- [TensorRT 최적화 가이드](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX 런타임 가이드](https://onnxruntime.ai/)
