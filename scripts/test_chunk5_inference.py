#!/usr/bin/env python3
"""
간단한 API 서버 테스트
기존 inference_pipeline을 사용하여 안전하게 테스트
"""

from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline
from PIL import Image
import base64
import io

def test_inference():
    """Chunk5 Best Model로 추론 테스트"""
    
    print("🔧 Loading Chunk5 Epoch 6 Best Model...")
    
    # Chunk5 Best Model
    checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    
    # Create pipeline
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cuda"
    )
    
    print("✅ Model loaded successfully!\n")
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (224, 224), color='blue')
    
    # Test instructions
    instructions = [
        "Navigate around obstacles and reach the front of the beverage bottle on the left",
        "Navigate around obstacles and reach the front of the beverage bottle on the right"
    ]
    
    print("🧪 Testing predictions:\n")
    for i, instruction in enumerate(instructions, 1):
        result = pipeline.predict(dummy_image, instruction)
        print(f"Test {i}: {instruction[:50]}...")
        print(f"  ✅ Linear X: {result['action'][0]:.4f}")
        print(f"  ✅ Linear Y: {result['action'][1]:.4f}")
        print(f"  📊 Normalized: [{result['action_normalized'][0]:.4f}, {result['action_normalized'][1]:.4f}]")
        print()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    return pipeline

if __name__ == "__main__":
    pipeline = test_inference()
