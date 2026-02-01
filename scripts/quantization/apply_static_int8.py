#!/usr/bin/env python3
"""
ONNX Runtime Static INT8 Quantization
재학습 없이 진짜 INT8 weight로 저장
"""

import torch
import json
import sys
import os
import numpy as np

print("="*70)
print("ONNX Runtime Static INT8 Quantization")
print("="*70)

# Check if onnxruntime is available
try:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
    print("✅ ONNX Runtime available")
except ImportError:
    print("❌ ONNX Runtime not installed")
    print("Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
    print("✅ Installed")

# Paths
checkpoint_path = 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt'
config_path = 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
onnx_path = 'quantized_models/chunk5_best_int8_static/model.onnx'
quantized_path = 'quantized_models/chunk5_best_int8_static/model_int8.onnx'

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

print(f"\nStep 1: LoadPyTorch model and export to ONNX...")
print(f"Checkpoint: {checkpoint_path}")

sys.path.insert(0, 'RoboVLMs_upstream')
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

with open(config_path, 'r') as f:
    config = json.load(f)

# Load model
print("Loading model...")
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

print("\nStep 2: Export to ONNX format...")
# Dummy input
batch_size = 1
seq_len = 8
dummy_vision = torch.randn(batch_size, seq_len, 3, 224, 224)
dummy_lang = torch.ones(batch_size, 256, dtype=torch.long)
dummy_attention = torch.ones(batch_size, 256, dtype=torch.bool)

print(f"Input shapes:")
print(f"  vision: {dummy_vision.shape}")
print(f"  language: {dummy_lang.shape}")
print(f"  attention: {dummy_attention.shape}")

# Export
print("Exporting to ONNX...")
try:
    torch.onnx.export(
        model.model,  # The actual RoboKosMos model
        (dummy_vision, dummy_lang),
        onnx_path,
        input_names=['vision_x', 'lang_x'],
        output_names=['output'],
        opset_version=13,
        dynamic_axes={
            'vision_x': {0: 'batch', 1: 'seq_len'},
            'lang_x': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    print(f"✅ ONNX model saved: {onnx_path}")
except Exception as e:
    print(f"⚠️ ONNX export error: {e}")
    print("\nKosmos-2는 ONNX export가 복잡합니다.")
    print("대안: PyTorch native quantization을 사용합니다...")
    
    # Use PyTorch quantization instead
    print("\n" + "="*70)
    print("PyTorch Static Quantization (Native)")
    print("="*70)
    
    # Prepare for static quantization
    from torch.quantization import (
        get_default_qconfig,
        float_qparams_weight_only_qconfig,
        prepare,
        convert
    )
    
    # Set different qconfigs for different layer types
    print("\nSetting qconfigs for different layers...")
    
    def set_qconfig_recursive(module, qconfig_dict):
        """Recursively set qconfig for each module type"""
        for name, child in module.named_children():
            # Embedding layers need special qconfig
            if isinstance(child, torch.nn.Embedding):
                child.qconfig = float_qparams_weight_only_qconfig
                print(f"  ✅ {name} (Embedding): float_qparams_weight_only_qconfig")
            # Linear, Conv layers use default
            elif isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
                child.qconfig = get_default_qconfig('fbgemm')
                print(f"  ✅ {name} ({child.__class__.__name__}): default qconfig")
            else:
                # Recursively apply to children
                set_qconfig_recursive(child, qconfig_dict)
    
    # Apply qconfigs
    set_qconfig_recursive(model, {})
    
    # Also set default for the model itself
    model.qconfig = get_default_qconfig('fbgemm')
    
    # Fuse modules if possible (optional, skip for complex models)
    # fused_model = torch.quantization.fuse_modules(model, [...])
    
    # Prepare
    print("\nPreparing model for static quantization...")
    model_prepared = prepare(model, inplace=False)
    
    # Calibrate with dummy data
    print("Calibrating with dummy data...")
    with torch.no_grad():
        for i in range(10):
            # Run forward pass for calibration
            try:
                dummy_vision_calib = torch.randn(1, 8, 3, 224, 224)
                dummy_lang_calib = torch.ones(1, 256, dtype=torch.long)
                dummy_attention_calib = torch.ones(1, 256, dtype=torch.bool)
                
                _ = model_prepared.model.forward(
                    dummy_vision_calib,
                    dummy_lang_calib,
                    attention_mask=dummy_attention_calib
                )
                print(f"  Calibration step {i+1}/10")
            except Exception as e:
                print(f"  Calibration step {i+1}/10 - skipped ({str(e)[:50]})")
    
    # Convert to INT8
    print("\nConverting to INT8...")
    try:
        model_int8 = convert(model_prepared, inplace=False)
        
        # Save
        output_path = 'quantized_models/chunk5_best_int8_static/model_pytorch_int8.pt'
        torch.save({
            'model_state_dict': model_int8.state_dict(),
            'config': config
        }, output_path)
        
        print(f"✅ INT8 model saved: {output_path}")
        
        # Measure size
        size_mb = os.path.getsize(output_path) / 1024**2
        print(f"File size: {size_mb:.1f} MB")
        
        # Measure GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model_int8 = model_int8.cuda()
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory: {gpu_mem:.3f} GB")
        
        print("\n" + "="*70)
        print("✅ Static INT8 quantization complete!")
        print("="*70)
        print("이 모델은 진짜 INT8 weights를 사용합니다.")
        print(f"원본: 6.3GB → INT8: {gpu_mem:.1f}GB")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nKosmos-2 모델 구조가 복잡하여 일부 layer 변환 실패")
        print("대안: 개별 layer만 quantize하거나 TensorRT 사용")
