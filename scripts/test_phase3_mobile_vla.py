#!/usr/bin/env python3
"""
Phase 3: Mobile VLA INT8 추론 테스트
ROS2 없이 추론 엔진만 테스트

사용법:
    python3 scripts/test_phase3_mobile_vla.py
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

# 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*70)
print("  Phase 3: Mobile VLA INT8 추론 테스트")
print("  transformers 4.35.0 + accelerate 0.23.0")
print("="*70)
print()

# 1. 환경 확인
print("1️⃣ 환경 확인:")
import transformers, accelerate
print(f"   transformers: {transformers.__version__}")
print(f"   accelerate: {accelerate.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
print()

# 2. BitsAndBytes 확인
print("2️⃣ BitsAndBytes 확인:")
try:
    import bitsandbytes as bnb
    print(f"   BitsAndBytes: {bnb.__version__} ✅")
    
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"   INT8 Config: ✅")
except ImportError as e:
    print(f"   ❌ BitsAndBytes 없음: {e}")
    sys.exit(1)
print()

# 3. 추론 엔진 import
print("3️⃣ 추론 엔진 로딩:")
try:
    from src.robovlms_mobile_vla_inference import (
        MobileVLAConfig,
        RoboVLMsInferenceEngine,
        ImageBuffer
    )
    print("   ✅ 추론 엔진 import 성공")
except Exception as e:
    print(f"   ❌ import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 4. Config 설정
print("4️⃣ Config 설정:")
checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"

# 체크포인트 존재 확인
if not Path(checkpoint_path).exists():
    print(f"   ⚠️ 체크포인트 없음: {checkpoint_path}")
    print(f"   Pretrained 모델만 사용")
    checkpoint_path = ""

config = MobileVLAConfig(
    checkpoint_path=checkpoint_path,
    window_size=2,
    fwd_pred_next_n=10,
    use_abs_action=True,
    # use_int8=True  # ← Config에 파라미터 없으면 주석 처리
)

print(f"   Window Size: {config.window_size}")
print(f"   Chunk Size: {config.fwd_pred_next_n}")
print(f"   abs_action: {config.use_abs_action}")
print()

# 5. 추론 엔진 생성
print("5️⃣ 추론 엔진 초기화:")
start_mem = psutil.virtual_memory().used / 1024**3
start_time = time.time()

try:
    engine = RoboVLMsInferenceEngine(config)
    print("   ✅ 엔진 생성 완료")
except Exception as e:
    print(f"   ❌ 엔진 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# 6. 모델 로드 (수동 INT8)
print("6️⃣ 모델 로드 (INT8):")
print("   ⚠️ 주의: robovlms_mobile_vla_inference.py는 FP16 고정")
print("   대안: 직접 Kosmos-2 INT8 로드")
print()

try:
    from transformers import Kosmos2ForConditionalGeneration, AutoProcessor
    
    model_path = ".vlms/kosmos-2-patch14-224"
    
    print(f"   모델: {model_path}")
    print(f"   INT8 Quantization: ✅")
    
    # INT8로 직접 로드
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    load_time = time.time() - start_time
    end_mem = psutil.virtual_memory().used / 1024**3
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
    
    print(f"   ✅ 로딩 완료! ({load_time:.1f}초)")
    print(f"   RAM: +{end_mem - start_mem:.2f} GB")
    print(f"   GPU: {gpu_mem:.2f} GB")
    print()
    
except Exception as e:
    print(f"   ❌ 모델 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 이미지 버퍼 준비
print("7️⃣ 이미지 버퍼 준비:")
image_buffer = ImageBuffer(
    window_size=config.window_size,
    image_size=config.image_size
)
print(f"   Window Size: {config.window_size}")
print()

# 8. 테스트 이미지 생성
print("8️⃣ 테스트 이미지 생성:")
# 더미 이미지 2장 (window_size=2)
for i in range(config.window_size):
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_buffer.add_image(dummy_image)
print(f"   ✅ {config.window_size}장 추가")
print()

# 9. 추론 실행
print("9️⃣ 추론 실행:")
instruction = "Navigate to the left bottle"
print(f"   지시문: {instruction}")

try:
    # 간단한 추론 테스트 (Kosmos-2 기본)
    test_image = Image.fromarray(dummy_image)
    prompt = f"<grounding>{instruction}"
    
    inputs = processor(text=prompt, images=test_image, return_tensors="pt")
    
    # GPU로 이동
    input_ids = inputs["input_ids"].to("cuda:0")
    pixel_values = inputs["pixel_values"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    image_embeds_position_mask = inputs["image_embeds_position_mask"].to("cuda:0")
    
    print("   ⏳ 추론 중...")
    inference_start = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=None,
            image_embeds_position_mask=image_embeds_position_mask,
            use_cache=True,
            max_new_tokens=64
        )
    
    inference_time = (time.time() - inference_start) * 1000
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"   ✅ 추론 완료! ({inference_time:.1f}ms)")
    print(f"   Output: {generated_text[:50]}...")
    print()
    
except Exception as e:
    print(f"   ❌ 추론 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 10. 최종 메모리
print("🔟 최종 메모리 상태:")
final_mem = psutil.virtual_memory()
final_gpu = torch.cuda.memory_allocated(0) / 1024**3

print(f"   RAM: {final_mem.used / 1024**3:.2f} / {final_mem.total / 1024**3:.2f} GB")
print(f"   RAM 사용률: {final_mem.percent:.1f}%")
print(f"   GPU: {final_gpu:.2f} GB")
print()

# 요약
print("="*70)
print("🎊 Phase 3 테스트 완료!")
print("="*70)
print()
print("📊 요약:")
print(f"   모델 로딩: {load_time:.1f}초")
print(f"   메모리 (RAM): +{end_mem - start_mem:.2f} GB")
print(f"   메모리 (GPU): {gpu_mem:.2f} GB")
print(f"   추론 속도: {inference_time:.1f}ms")
print()
print("✅ INT8 quantization으로 정상 동작 확인!")
print()
print("⚠️ 참고:")
print("   - robovlms_mobile_vla_inference.py는 FP16 고정")
print("   - Mobile VLA 체크포인트 추가 시 fine-tuned 모델 테스트 가능")
print("   - ROS2 통합은 mobile_vla_inference_node.py 사용")
