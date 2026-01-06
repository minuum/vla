import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time

# 경로 설정
sys.path.append("/home/billy/25-1kp/vla/RoboVLMs_upstream")
sys.path.insert(0, "/home/billy/25-1kp/vla")  # Mobile_VLA 패키지 접근용

from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline

def test_on_server():
    print("🚀 [Billy Server] Starting Inference Test...")
    
    # 1. 설정
    ckpt_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    
    if not Path(ckpt_path).exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    # 2. 파이프라인 초기화 (모델 로딩)
    print("Step 1: Loading Model...")
    start_load = time.time()
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        device="cuda"
    )
    load_time = time.time() - start_load
    print(f"✅ Model Loaded in {load_time:.2f}s")
    
    # 메모리 측정 (Fact)
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"📊 Current GPU Memory Usage: {mem_used:.2f} GB (Note: Pytorch Context + Model)")

    # 3. 추론 테스트
    print("\nStep 2: Running Inference (Dummy Input)...")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
    
    # Warmup
    pipeline.predict(dummy_image, instruction)
    
    # Measure
    start_infer = time.time()
    result = pipeline.predict(dummy_image, instruction)
    infer_time = (time.time() - start_infer) * 1000  # ms
    
    action = result['action']
    raw_norm = result['action_normalized']
    
    print(f"✅ Inference Done in {infer_time:.1f} ms")
    print(f"📝 Result:")
    print(f"   - Instruction: {instruction}")
    print(f"   - Raw Normalized Output (Model): {raw_norm} (Expect: Near 1.0 or -1.0)")
    print(f"   - Final Action (Corrected):      {action}   (Expect: Near 1.15 or -1.15)")
    
    # 검증: Gain 적용 여부
    # Raw가 1.0 근처일 때 Final이 1.15 근처여야 함
    if abs(action[0]) > 1.1 or abs(action[1]) > 1.1:
        print("🎉 SUCCESS: Gain correction logic is working! (Output reachs ~1.15)")
    else:
        print("⚠️ CHECK: Output magnitude is small. It might be correct for this input, or logic issue.")

if __name__ == "__main__":
    test_on_server()
