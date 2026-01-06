#!/usr/bin/env python3
"""
Quick sanity check for QAT dtype conversion
Tests if the modified QuantizedVisionWrapper handles FP16 input correctly
"""

import torch
import sys
sys.path.insert(0, 'RoboVLMs_upstream')

from robovlms.train.mobile_vla_qat_trainer import QuantizedVisionWrapper
from torch.quantization import get_default_qat_qconfig

print("="*60)
print("Testing QAT Dtype Conversion")
print("="*60)

# Create a simple mock vision model
class MockVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
    
    def forward(self, pixel_values=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values is required")
        x = self.conv(pixel_values)
        # Return a simple output object with last_hidden_state
        class Output:
            def __init__(self, hs):
                self.last_hidden_state = hs
        return Output(x)

# Create wrapper
print("\n1. Creating QuantizedVisionWrapper...")
vision_model = MockVisionModel()
wrapper = QuantizedVisionWrapper(vision_model)
wrapper.qconfig = get_default_qat_qconfig('fbgemm')

# Prepare for QAT
from torch.quantization import prepare_qat
wrapper = prepare_qat(wrapper, inplace=True)

print("✅ Wrapper created and prepared for QAT")

# Test with FP16 input (simulating mixed precision)
print("\n2. Testing with FP16 input (simulating AMP)...")
try:
    # Create FP16 input
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float16)
    print(f"   Input dtype: {pixel_values.dtype}")
    
    # Forward pass
    output = wrapper(pixel_values=pixel_values)
    
    print(f"   Output dtype: {output.last_hidden_state.dtype}")
    print("✅ Forward pass successful with FP16 input!")
    
except RuntimeError as e:
    print(f"❌ Error: {e}")
    print("\nThis means dtype conversion didn't work.")
    sys.exit(1)

# Test with FP32 input
print("\n3. Testing with FP32 input (normal case)...")
try:
    pixel_values_fp32 = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"   Input dtype: {pixel_values_fp32.dtype}")
    
    output = wrapper(pixel_values=pixel_values_fp32)
    
    print(f"   Output dtype: {output.last_hidden_state.dtype}")
    print("✅ Forward pass successful with FP32 input!")
    
except RuntimeError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! Dtype conversion working correctly.")
print("="*60)
print("\nNext step: Run 1-epoch test with full training pipeline")
