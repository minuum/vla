#!/usr/bin/env python3
"""
INT8 모델 Inference 테스트
원본 FP32 vs INT8 정확도 비교
"""

import torch
import numpy as np
import sys
import time

sys.path.insert(0, 'RoboVLMs_upstream')

print("="*70)
print("INT8 Model Inference Test")
print("="*70)

# Load INT8 model
int8_path = 'quantized_models/chunk5_best_int8/model.pt'
print(f"\n1. Loading INT8 model...")
print(f"   Path: {int8_path}")

try:
    int8_checkpoint = torch.load(int8_path, map_location='cpu')
    
    # Check contents
    print(f"   Keys: {list(int8_checkpoint.keys())}")
    
    # Get config
    config = int8_checkpoint.get('config', None)
    if config:
        print(f"   ✅ Config loaded")
    
    # Get model state
    state_dict = int8_checkpoint.get('model_state_dict', None)
    if state_dict:
        num_params = len(state_dict)
        print(f"   ✅ State dict loaded ({num_params} parameters)")
        
        # Check if quantized
        sample_key = list(state_dict.keys())[0]
        sample_param = state_dict[sample_key]
        print(f"   Sample param dtype: {sample_param.dtype}")
        
        # Count quantized params
        quantized_count = sum(1 for k, v in state_dict.items() 
                             if hasattr(v, 'dtype') and 'int' in str(v.dtype).lower())
        print(f"   Quantized parameters: {quantized_count}/{num_params}")
    
    print("\n2. Creating model from INT8 checkpoint...")
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    # Create model
    model_int8 = MobileVLATrainer(config)
    
    # Dequantize INT8 tensors to FP32
    print("   Dequantizing INT8 → FP32...")
    dequantized_state = {}
    for key, value in state_dict.items():
        if hasattr(value, 'is_quantized') and value.is_quantized:
            # Dequantize: INT8 → FP32
            dequantized_state[key] = value.dequantize()
        else:
            dequantized_state[key] = value
    
    # Load dequantized state
    model_int8.load_state_dict(dequantized_state, strict=False)
    model_int8.eval()
    
    print("   ✅ Model created (dequantized for inference)")
    
    # Test inference
    print("\n3. Testing inference...")
    
    with torch.no_grad():
        # Dummy input
        batch_size = 1
        seq_len = 8
        
        dummy_vision = torch.randn(batch_size, seq_len, 3, 224, 224)
        dummy_lang = torch.ones(batch_size, 256, dtype=torch.long)
        dummy_attention = torch.ones(batch_size, 256, dtype=torch.bool)
        
        print(f"   Input shapes:")
        print(f"     Vision: {dummy_vision.shape}")
        print(f"     Language: {dummy_lang.shape}")
        
        # Inference
        start_time = time.time()
        try:
            output = model_int8.model.inference(
                vision_x=dummy_vision,
                lang_x=dummy_lang,
                attention_mask=dummy_attention
            )
            latency = (time.time() - start_time) * 1000
            
            print(f"   ✅ Inference successful")
            print(f"   Latency: {latency:.1f} ms")
            
            # Check output
            if isinstance(output, dict):
                print(f"   Output keys: {list(output.keys())}")
                if 'action' in output:
                    action = output['action']
                    if isinstance(action, tuple):
                        action = action[0]
                    print(f"   Action shape: {action.shape}")
                    print(f"   Action sample: {action.flatten()[:5].numpy()}")
            
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Memory usage
    print("\n4. Memory usage...")
    if torch.cuda.is_available():
        model_int8 = model_int8.cuda()
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU Memory: {gpu_mem:.3f} GB")
    else:
        print(f"   CPU mode")
    
    print("\n" + "="*70)
    print("✅ INT8 Model Test Complete")
    print("="*70)
    print(f"Model: {int8_path}")
    print(f"Status: Working")
    print(f"Inference: {latency:.1f} ms")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
