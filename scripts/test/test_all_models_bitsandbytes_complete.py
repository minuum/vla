#!/usr/bin/env python3
"""
BitsAndBytes INT8 - All Models Complete Test
"""

import torch
import json
import sys
import time
import os
from transformers import BitsAndBytesConfig

sys.path.insert(0, 'RoboVLMs_upstream')

print("="*70)
print("BitsAndBytes INT8 - All Models Complete Test")
print("="*70)

# BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Test models with correct paths
models = [
    {
        'name': 'Chunk5 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt',
        'val_loss': 0.067
    },
    {
        'name': 'Left Chunk10 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt',
        'val_loss': 0.010
    },
    {
        'name': 'Right Chunk10 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_right_chunk10_20251218.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_right_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.013.ckpt',
        'val_loss': 0.013
    }
]

results = []

for model_info in models:
    print(f"\n{'='*70}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*70}")
    
    try:
        # Check files
        if not os.path.exists(model_info['config']):
            print(f"   ❌ Config not found: {model_info['config']}")
            results.append({'name': model_info['name'], 'status': 'CONFIG_NOT_FOUND'})
            continue
            
        if not os.path.exists(model_info['checkpoint']):
            print(f"   ❌ Checkpoint not found: {model_info['checkpoint']}")
            results.append({'name': model_info['name'], 'status': 'CHECKPOINT_NOT_FOUND'})
            continue
        
        # Load config
        with open(model_info['config'], 'r') as f:
            config = json.load(f)
        
        # Load checkpoint
        print(f"\n1. Loading checkpoint...")
        checkpoint = torch.load(model_info['checkpoint'], map_location='cpu')
        
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
        if not state_dict:
            print(f"   ❌ No state_dict found")
            results.append({'name': model_info['name'], 'status': 'NO_STATE_DICT'})
            continue
        
        print(f"   ✅ Loaded {len(state_dict)} parameters")
        
        # Create model
        print(f"\n2. Loading with BitsAndBytes INT8...")
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated() / 1024**3
        
        model = MobileVLATrainer(config, quantization_config=bnb_config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.cuda()
        
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        model_size = gpu_mem - baseline
        
        print(f"   ✅ GPU Memory: {model_size:.3f} GB")
        
        # Test inference
        print(f"\n3. Testing inference...")
        
        with torch.no_grad():
            dummy_vision = torch.randn(1, 8, 3, 224, 224).cuda()
            dummy_lang = torch.ones(1, 256, dtype=torch.long).cuda()
            dummy_attention = torch.ones(1, 256, dtype=torch.bool).cuda()
            
            start = time.time()
            output = model.model.inference(
                vision_x=dummy_vision,
                lang_x=dummy_lang,
                attention_mask=dummy_attention
            )
            latency = (time.time() - start) * 1000
        
        print(f"   ✅ Latency: {latency:.1f} ms")
        
        # Verify output
        if isinstance(output, dict) and 'action' in output:
            action = output['action']
            if isinstance(action, tuple):
                action = action[0]
            print(f"   ✅ Action shape: {action.shape}")
        
        results.append({
            'name': model_info['name'],
            'val_loss': model_info['val_loss'],
            'gpu_memory': model_size,
            'latency': latency,
            'status': 'SUCCESS'
        })
        
        print(f"\n✅ {model_info['name']}: SUCCESS")
        print(f"   Memory: {model_size:.2f} GB, Latency: {latency:.0f} ms")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ {model_info['name']}: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        
        results.append({
            'name': model_info['name'],
            'val_loss': model_info['val_loss'],
            'status': 'FAILED',
            'error': str(e)
        })

# Final Report
print(f"\n{'='*70}")
print("FINAL RESULTS - BitsAndBytes INT8")
print(f"{'='*70}\n")

print(f"{'Model':<25} {'Val Loss':<12} {'GPU Mem':<12} {'Latency':<12} {'Status':<10}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

for r in results:
    if r['status'] == 'SUCCESS':
        print(f"{r['name']:<25} {r['val_loss']:<12.3f} {r['gpu_memory']:<11.2f}G {r['latency']:<11.0f}ms {r['status']:<10}")
    else:
        print(f"{r['name']:<25} {r['val_loss']:<12.3f} {'N/A':<12} {'N/A':<12} {r['status']:<10}")

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"\n{'='*70}")
print(f"✅ Success: {success_count}/{len(results)} models")
print(f"{'='*70}")

if success_count > 0:
    avg_memory = sum(r['gpu_memory'] for r in results if r['status'] == 'SUCCESS') / success_count
    avg_latency = sum(r['latency'] for r in results if r['status'] == 'SUCCESS') / success_count
    print(f"\n📊 Average Performance:")
    print(f"   GPU Memory: {avg_memory:.2f} GB (vs FP32: 6.3GB)")
    print(f"   Latency: {avg_latency:.0f} ms (vs FP32: 15000ms)")
    print(f"   Memory Reduction: {(1 - avg_memory/6.3)*100:.1f}%")
    print(f"   Speed Improvement: {15000/avg_latency:.1f}x")
