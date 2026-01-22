#!/usr/bin/env python3
"""
PaliGemma-3B Resource Usage Test on Jetson
- Loads base model (google/paligemma-3b-pt-224)
- Measures loading time and memory usage
- Checks for OOM risks
"""

import time
import psutil
import torch
import os
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from jtop import jtop

def get_memory_usage():
    """Get current memory usage (System RAM)"""
    vm = psutil.virtual_memory()
    return vm.used / (1024**3), vm.total / (1024**3)

def monitor_jetson_memory(interval=1.0, stop_event=None):
    """Monitor Jetson memory using jtop (Shared VRAM)"""
    # Note: Using psutil for simplicity if jtop is complex to thread
    # On Jetson, System RAM is Shared VRAM.
    pass

def main():
    print("=" * 60)
    print("🚀 PaliGemma-3B Resource Test on Jetson")
    print("=" * 60)
    
    model_id = "google/paligemma-3b-pt-224"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    
    # 1. Initial Memory
    start_mem, total_mem = get_memory_usage()
    print(f"Memory before load: {start_mem:.2f} GB / {total_mem:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

    # 2. Load Options
    print("\n📦 Loading Model (float16)...")
    # Using float16 for efficiency (standard for inference)
    dtype = torch.float16
    
    try:
        start_time = time.time()
        
        # Load Processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Load Model
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" # This handles moving to GPU automatically
        ).to(device)
        
        load_time = time.time() - start_time
        
        # 3. Post-Load Memory
        end_mem, _ = get_memory_usage()
        
        print("\n✅ Load Successful!")
        print(f"Time taken: {load_time:.2f}s")
        print(f"Memory after load: {end_mem:.2f} GB")
        print(f"Memory Delta (Model Size): {end_mem - start_mem:.2f} GB")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # 4. Dummy Inference Test (To check activation memory)
        print("\n🧠 Running Dummy Inference...")
        
        # Create dummy input
        from PIL import Image
        import numpy as np
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        text = "detect: cup <loc0000><loc0000><loc0000><loc0000>" # Dummy prompt
        
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
            )
        inference_time = time.time() - start_time
        
        print(f"✅ Inference Successful!")
        print(f"Inference Time: {inference_time:.4f}s")
        
        final_mem, _ = get_memory_usage()
        print(f"Peak Memory during inference: ~{final_mem:.2f} GB")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

        current_mem, _ = get_memory_usage()
        print(f"Memory at failure: {current_mem:.2f} GB")
        
        if "out of memory" in str(e).lower():
            print("\n⚠️  OOM DETECTED!")
            print("Action Plan:")
            print("1. Use INT8 quantization (bitsandbytes)")
            print("2. Reduce batch size (already 1)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
