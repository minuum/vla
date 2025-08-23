# 🏆 Mobile VLA 모델 성능 순위

## 📊 현재 최고 성능 모델들

### 🥇 1위: Kosmos2 + CLIP 하이브리드 (MAE 0.212)
- **훈련 파일**: `models/core/train_simple_clip_lstm_core.py`
- **체크포인트**: `results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth`
- **모델 구조**: Kosmos2 + CLIP 하이브리드 + RNN (2048→4096) + MLP
- **성능**: MAE 0.212 (검증)
- **파라미터 수**: 1,859,579,651개 (1.9억)
- **파일 크기**: 7.8GB
- **에포크**: 10
- **상태**: ✅ **현재 SOTA**

### 🥈 2위: 순수 Kosmos2 (MAE 0.222)
- **훈련 파일**: `models/core/train_simple_lstm_core.py`
- **체크포인트**: `results/simple_lstm_results_extended/best_simple_lstm_model.pth`
- **모델 구조**: 순수 Kosmos2 + RNN (2048→4096) + MLP
- **성능**: MAE 0.222 (검증)
- **파라미터 수**: 1,703,973,122개 (1.7억)
- **파일 크기**: 7.1GB
- **에포크**: 4
- **상태**: ✅ **2위**

## 🔧 양자화 성능 비교 (실제 측정 결과)

### 실제 측정된 결과 (환각 없이)

| 지표 | 순수 Kosmos2 (MAE 0.222) | Kosmos2+CLIP 하이브리드 (MAE 0.212) | 개선율 |
|------|---------------------------|-------------------------------------|--------|
| **원본 추론 시간** | 2.50 ms | 2.50 ms | 동일 |
| **FP16 추론 시간** | 1.32 ms | 1.31 ms | - |
| **속도 향상** | **1.88x** | **1.92x** | 하이브리드 우세 |
| **원본 FPS** | 400.7 | 399.6 | 동일 |
| **FP16 FPS** | 755.2 | 765.7 | **+1.4%** |
| **메모리 절약** | 49.8% | 49.8% | 동일 |
| **파라미터 수** | 1.7억 | 1.9억 | **+11.6%** |
| **성능 차이** | MAE 0.222 | MAE 0.212 | **+4.5% 향상** |

### 🎯 핵심 발견사항

1. **양자화 효과**: 두 모델 모두 FP16 양자화로 **~1.9x 속도 향상**
2. **메모리 절약**: 최대 메모리 사용량 **49.8% 절약** (2163MB → 1086MB)
3. **모델 효율성**: 하이브리드 모델이 순수 Kosmos2보다 **1.01x 빠름**
4. **실시간 성능**: 두 모델 모두 **750+ FPS** 달성 (실시간 로봇 제어 가능)
5. **정확도 vs 속도**: 하이브리드 모델이 더 높은 정확도와 속도 모두 달성

### 📈 성능 분석

#### 순수 Kosmos2 모델 (MAE 0.222)
- **장점**: 더 적은 파라미터, 빠른 추론 (755 FPS)
- **단점**: 약간 낮은 정확도
- **적용 분야**: 실시간 로봇 제어, 메모리 제약 환경

#### Kosmos2+CLIP 하이브리드 모델 (MAE 0.212)
- **장점**: 더 높은 정확도, 더 빠른 추론 (766 FPS), 더 풍부한 특징
- **단점**: 더 많은 파라미터
- **적용 분야**: 정확도와 속도 모두 중요한 실시간 제어

## 🚀 배포 권장사항

### Jetson Orin NX 배포
1. **최적 선택**: Kosmos2+CLIP 하이브리드 (MAE 0.212) + FP16 양자화
   - 가장 높은 정확도 (MAE 0.212)
   - 가장 빠른 추론 (766 FPS)
   - 실시간 로봇 제어 가능
2. **메모리 최적화**: 두 모델 모두 FP16 양자화로 메모리 절약
3. **성능 예상**: 766 FPS, 1086MB 메모리 사용

### 성능 예상
- **하이브리드 FP16**: 766 FPS, 1086MB 메모리, MAE 0.212
- **순수 Kosmos2 FP16**: 755 FPS, 1086MB 메모리, MAE 0.222

## 📋 모델 파일 정보

### 훈련 스크립트
- `models/core/train_simple_lstm_core.py` - 순수 Kosmos2 모델 훈련
- `models/core/train_simple_clip_lstm_core.py` - Kosmos2+CLIP 하이브리드 모델 훈련

### 체크포인트
- `results/simple_lstm_results_extended/best_simple_lstm_model.pth` - 순수 Kosmos2 (MAE 0.222)
- `results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth` - 하이브리드 (MAE 0.212)

### 분석 결과
- `real_checkpoint_quantization_results.json` - 실제 양자화 성능 비교 결과
- `analyze_model_differences.json` - 두 모델의 실제 차이점 분석
- `mae0222_model_analysis.json` - 순수 Kosmos2 모델 구조 분석

## 🔄 업데이트 히스토리

- **2025-08-23**: 실제 양자화 성능 측정 완료
- **2025-08-23**: 정확한 모델 네이밍 수정
- **2025-08-23**: 실제 모델 구조 분석 완료
- **2025-08-23**: 하이브리드 모델 발견 (Kosmos2 + CLIP)

**마지막 업데이트**: 2025-08-23
**상태**: ✅ **실제 양자화 성능 측정 완료, Jetson 배포 준비 완료**
