# 📁 Models Directory Structure

Mobile VLA 프로젝트의 다양한 모델 구현들을 체계적으로 정리합니다.

## 🗂️ 폴더 구조

```
models/
├── basic/                    # 기본 모델들
│   ├── 2d_optimized/        # 2D 액션 최적화 모델
│   └── 3d_original/         # 원본 3D 액션 모델
├── enhanced/                # 향상된 모델들
│   ├── with_resampler/      # Vision Resampler 포함
│   ├── with_clip_norm/      # CLIP 정규화 포함
│   └── with_state/          # 상태 임베딩 포함
└── experimental/            # 실험적 모델들
    ├── full_features/       # 모든 기능 포함
    └── ablations/           # 기능 제거 실험
```

## 🚀 현재 구현된 모델

### ✅ **Enhanced 2D Model with Vision Resampler**
**위치**: `models/enhanced/with_resampler/`

**주요 특징**:
- ✅ Vision Resampler (PerceiverResampler)
- ✅ 2D 액션 예측 (Z축 제외)
- ✅ Kosmos2 백본 모델

**사용법**:
```bash
cd models/enhanced/with_resampler/
python train_enhanced_model.py --data_path /path/to/h5/data
```

## 📊 모델 비교

| 모델 | 액션 차원 | Vision Resampler | 상태 |
|------|-----------|------------------|------|
| Basic 2D | 2D | ❌ | ✅ 구현됨 |
| Enhanced 2D | 2D | ✅ | ✅ 구현됨 |

## 🎯 다음 단계

1. **CLIP Normalization** 추가
2. **State Embedding** 추가
3. **Full Features** 모델 구현
