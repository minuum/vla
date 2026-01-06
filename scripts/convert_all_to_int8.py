#!/usr/bin/env python3
"""
모든 최근 모델을 Static INT8로 변환
PyTorch 공식 Static Quantization 사용
"""

import torch
import json
import sys
import os

sys.path.insert(0, 'RoboVLMs_upstream')

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from torch.quantization import (
    get_default_qconfig,
    float_qparams_weight_only_qconfig,
    prepare,
    convert
)

def set_qconfig_recursive(module):
    """각 layer 타입별로 적절한 qconfig 설정"""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Embedding):
            child.qconfig = float_qparams_weight_only_qconfig
        elif isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
            child.qconfig = get_default_qconfig('fbgemm')
        else:
            set_qconfig_recursive(child)

def quantize_model(checkpoint_path, config_path, output_path):
    """모델을 Static INT8로 변환"""
    
    print("="*70)
    print(f"Converting: {os.path.basename(checkpoint_path)}")
    print("="*70)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    print("1. Loading FP32 model...")
    model = MobileVLATrainer(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError("No state_dict found")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Set qconfigs
    print("2. Setting qconfigs...")
    set_qconfig_recursive(model)
    model.qconfig = get_default_qconfig('fbgemm')
    
    # Prepare
    print("3. Preparing for quantization...")
    model_prepared = prepare(model, inplace=False)
    
    # Calibrate
    print("4. Calibrating (10 steps)...")
    with torch.no_grad():
        for i in range(10):
            try:
                dummy_vision = torch.randn(1, 8, 3, 224, 224)
                dummy_lang = torch.ones(1, 256, dtype=torch.long)
                dummy_attention = torch.ones(1, 256, dtype=torch.bool)
                
                _ = model_prepared.model.forward(
                    dummy_vision,
                    dummy_lang,
                    attention_mask=dummy_attention
                )
            except Exception as e:
                pass
    
    # Convert to INT8
    print("5. Converting to INT8...")
    model_int8 = convert(model_prepared, inplace=False)
    
    # Save
    print("6. Saving...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': model_int8.state_dict(),
        'config': config
    }, output_path)
    
    # Report
    size_mb = os.path.getsize(output_path) / 1024**2
    size_gb = size_mb / 1024
    
    print(f"✅ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    print()
    
    return size_gb

# 변환할 모델 목록
models = [
    {
        'name': 'Chunk5 Best (Val Loss 0.067)',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt',
        'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json',
        'output': 'quantized_models/chunk5_best_int8/model.pt'
    },
    {
        'name': 'Chunk10 Best (Val Loss 0.284)',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=05-val_loss=val_loss=0.284.ckpt',
        'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json',
        'output': 'quantized_models/chunk10_best_int8/model.pt'
    },
    {
        'name': 'Left Chunk10 Best (Val Loss 0.010)',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt',
        'config': 'Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json',
        'output': 'quantized_models/left_chunk10_best_int8/model.pt'
    },
    {
        'name': 'Right Chunk10 Best (Val Loss 0.013)',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_right_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.013.ckpt',
        'config': 'Mobile_VLA/configs/mobile_vla_right_chunk10_20251218.json',
        'output': 'quantized_models/right_chunk10_best_int8/model.pt'
    }
]

print("\n" + "="*70)
print("모든 최근 모델 → Static INT8 변환")
print("="*70)
print(f"Total models: {len(models)}")
print()

results = []

for model_info in models:
    try:
        size = quantize_model(
            model_info['checkpoint'],
            model_info['config'],
            model_info['output']
        )
        results.append({
            'name': model_info['name'],
            'size_gb': size,
            'status': '✅ Success'
        })
    except Exception as e:
        print(f"❌ Failed: {e}\n")
        results.append({
            'name': model_info['name'],
            'size_gb': 0,
            'status': f'❌ Error: {str(e)[:50]}'
        })

# Summary
print("\n" + "="*70)
print("변환 결과 요약")
print("="*70)

for r in results:
    print(f"{r['status']}")
    print(f"  {r['name']}")
    if r['size_gb'] > 0:
        print(f"  크기: {r['size_gb']:.2f} GB")
    print()

print("="*70)
print(f"완료: {sum(1 for r in results if '✅' in r['status'])}/{len(results)}")
print("="*70)
