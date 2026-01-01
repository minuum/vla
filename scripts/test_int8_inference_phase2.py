#!/usr/bin/env python3
"""
Phase 2: INT8 Quantization 실제 추론 테스트
Mobile VLA 모델을 INT8로 로드하고 추론 벤치마크
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import time
import psutil
from pathlib import Path
from transformers import BitsAndBytesConfig, Kosmos2ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

print("="*70)
print("  Phase 2: INT8 Mobile VLA 추론 테스트")
print("  transformers 4.35.0 + accelerate 0.23.0")
print("="*70)
print()

# 1. 환경 확인
print("1️⃣ 환경 확인:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()} (version {torch.version.cuda})")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
print()

# 2. INT8 설정
print("2️⃣ INT8 Quantization 설정:")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
print(f"   load_in_8bit: {bnb_config.load_in_8bit}")
print()

# 3. 모델 로딩
model_path = ".vlms/kosmos-2-patch14-224"
print(f"3️⃣ 모델 로딩: {model_path}")

start_mem = psutil.virtual_memory().used / 1024**3
start_time = time.time()

try:
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
    print(f"   Device: {model.device}")
    print()
    
except Exception as e:
    print(f"   ❌ 모델 로딩 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 테스트 이미지 생성
print("4️⃣ 테스트 이미지 생성:")
# 간단한 더미 이미지 (224x224 RGB)
test_image = Image.new('RGB', (224, 224), color=(73, 109, 137))
print("   ✅ 224x224 RGB 테스트 이미지")
print()

# 5. 추론 테스트
print("5️⃣ 추론 테스트:")
prompt = "<grounding>An image of"

try:
    # Input 처리
    inputs = processor(text=prompt, images=test_image, return_tensors="pt")
    
    # GPU로 명시적 이동
    input_ids = inputs["input_ids"].to("cuda:0")
    pixel_values = inputs["pixel_values"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    image_embeds_position_mask = inputs["image_embeds_position_mask"].to("cuda:0")
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Pixel values: {pixel_values.shape}")
    print()
    
    # 추론 시작
    print("   ⏳ 추론 중...")
    inference_times = []
    
    for i in range(3):  # 3번 반복 측정
        start = time.time()
        
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
        
        inference_time = time.time() - start
        inference_times.append(inference_time)
        
        if i == 0:  # 첫 번째 결과만 출력
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"   Output (#{i+1}): {generated_text[:50]}...")
    
    avg_time = np.mean(inference_times)
    print()
    print(f"   ✅ 추론 완료!")
    print(f"   평균 시간: {avg_time:.3f}초 (3회 평균)")
    print(f"   최소 시간: {min(inference_times):.3f}초")
    print(f"   최대 시간: {max(inference_times):.3f}초")
    print()
    
except Exception as e:
    print(f"   ❌ 추론 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. 최종 메모리 상태
print("6️⃣ 최종 메모리 상태:")
final_mem = psutil.virtual_memory()
final_gpu = torch.cuda.memory_allocated(0) / 1024**3

print(f"   RAM: {final_mem.used / 1024**3:.2f} / {final_mem.total / 1024**3:.2f} GB")
print(f"   RAM 사용률: {final_mem.percent:.1f}%")
print(f"   GPU: {final_gpu:.2f} GB")
print()

# 7. 요약
print("="*70)
print("🎊 Phase 2 추론 테스트 완료!")
print("="*70)
print()
print("📊 요약:")
print(f"   모델 로딩: {load_time:.1f}초")
print(f"   메모리 (RAM): +{end_mem - start_mem:.2f} GB")
print(f"   메모리 (GPU): {gpu_mem:.2f} GB")
print(f"   추론 속도: {avg_time:.3f}초 (평균)")
print()
print("✅ INT8 quantization으로 정상 동작 확인!")
