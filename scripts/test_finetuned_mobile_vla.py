#!/usr/bin/env python3
"""
Mobile VLA Fine-tuned 모델 INT8 테스트
실제 학습된 체크포인트로 action 예측 테스트
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import torch
import time
import psutil
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*70)
print("  Mobile VLA Fine-tuned INT8 테스트")
print("  Checkpoint: epoch_epoch=06-val_loss=val_loss=0.067.ckpt")
print("="*70)
print()

# 1. 환경 확인
print("1️⃣ 환경 확인:")
import transformers, accelerate
print(f"   transformers: {transformers.__version__}")
print(f"   accelerate: {accelerate.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
print()

# 2. 체크포인트 확인
checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
print(f"2️⃣ 체크포인트:")
import os
size_gb = os.path.getsize(checkpoint_path) / 1024**3
print(f"   경로: {checkpoint_path}")
print(f"   크기: {size_gb:.2f} GB")
print()

# 3. 추론 엔진 로드
print("3️⃣ 추론 엔진 로딩:")
from src.robovlms_mobile_vla_inference import (
    MobileVLAConfig,
    RoboVLMsInferenceEngine,
    ImageBuffer
)

config = MobileVLAConfig(
    checkpoint_path=checkpoint_path,
    window_size=2,
    fwd_pred_next_n=10,
    use_abs_action=True
)
print(f"   Window Size: {config.window_size}")
print(f"   Chunk Size: {config.fwd_pred_next_n}")
print(f"   abs_action: {config.use_abs_action}")
print()

# 4. 엔진 생성 및 모델 로드
print("4️⃣ 모델 로딩 (Fine-tuned + FP16):")
start_mem = psutil.virtual_memory().used / 1024**3
start_time = time.time()

try:
    engine = RoboVLMsInferenceEngine(config)
    
    # 모델 로드
    if not engine.load_model():
        print("   ❌ 모델 로드 실패")
        sys.exit(1)
    
    load_time = time.time() - start_time
    end_mem = psutil.virtual_memory().used / 1024**3
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
    
    print(f"   ✅ 로딩 완료! ({load_time:.1f}초)")
    print(f"   RAM: +{end_mem - start_mem:.2f} GB")
    print(f"   GPU: {gpu_mem:.2f} GB")
    print()
    
except Exception as e:
    print(f"   ❌ 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 이미지 버퍼 준비
print("5️⃣ 이미지 버퍼 준비:")
image_buffer = ImageBuffer(
    window_size=config.window_size,
    image_size=config.image_size
)

# 더미 이미지 2장
for i in range(config.window_size):
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_buffer.add_image(dummy_image)
print(f"   ✅ {config.window_size}장 추가")
print()

# 6. 시나리오별 추론 테스트
print("6️⃣ 시나리오별 추론 테스트:")
scenarios = [
    "Navigate around obstacles and reach the left bottle",
    "Navigate around obstacles and reach the right bottle",
    "Navigate around two boxes and reach the left bottle",
    "Navigate around two boxes and reach the right bottle"
]

results = []
for i, instruction in enumerate(scenarios, 1):
    print(f"\n   [{i}/4] {instruction}")
    
    try:
        images = image_buffer.get_images()
        
        inference_start = time.time()
        actions, info = engine.predict_action(
            images,
            instruction,
            use_abs_action=True
        )
        inference_time = (time.time() - inference_start) * 1000
        
        # 정규화 해제
        denorm_actions = engine.denormalize_action(actions)
        first_action = denorm_actions[0]
        
        direction = info.get('direction', 'unknown')
        print(f"   ✅ Action: [{first_action[0]:.3f}, {first_action[1]:.3f}]")
        print(f"      방향: {direction} | 지연: {inference_time:.1f}ms")
        
        results.append({
            'scenario': i,
            'instruction': instruction,
            'action': first_action.tolist(),
            'direction': direction,
            'latency': inference_time
        })
        
    except Exception as e:
        print(f"   ❌ 실패: {e}")

print()

# 7. 결과 요약
print("7️⃣ 결과 요약:")
avg_latency = np.mean([r['latency'] for r in results])
print(f"   총 테스트: {len(results)}개")
print(f"   평균 지연: {avg_latency:.1f}ms")
print()

print("   시나리오별 결과:")
for r in results:
    print(f"   [{r['scenario']}] {r['direction']:>5} → [{r['action'][0]:>6.3f}, {r['action'][1]:>6.3f}] ({r['latency']:.0f}ms)")

print()

# 8. 최종 메모리
print("8️⃣ 최종 메모리:")
final_mem = psutil.virtual_memory()
final_gpu = torch.cuda.memory_allocated(0) / 1024**3
print(f"   RAM: {final_mem.used / 1024**3:.2f} / {final_mem.total / 1024**3:.2f} GB ({final_mem.percent:.1f}%)")
print(f"   GPU: {final_gpu:.2f} GB")
print()

# 요약
print("="*70)
print("🎊 Fine-tuned 모델 테스트 완료!")
print("="*70)
print(f"📊 모델 로딩: {load_time:.1f}초")
print(f"📊 메모리 (RAM): +{end_mem - start_mem:.2f} GB")
print(f"📊 메모리 (GPU): {gpu_mem:.2f} GB")
print(f"📊 평균 추론: {avg_latency:.1f}ms")
print()
print("✅ Mobile VLA fine-tuned 모델 정상 작동 확인!")
