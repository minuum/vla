#!/usr/bin/env python3
"""
PaliGemma Checkpoint Test
Loads trained checkpoint and tests LEFT vs RIGHT instruction grounding
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.utils.config_utils import load_config

# Config and checkpoint paths
CONFIG_PATH = "Mobile_VLA/configs/mobile_vla_paligemma_lora.json"
CHECKPOINT_PATH = "runs/mobile_vla_paligemma/paligemma/mobile_vla_paligemma_finetune/2026-01-07/mobile_vla_paligemma_lora/epoch_epoch=00-val_loss=val_loss=0.040.ckpt"

# Test instructions
INSTRUCTIONS = {
    "LEFT": "Navigate around the obstacle on the left side and reach the cup",
    "RIGHT": "Navigate around the obstacle on the right side and reach the cup",
}

def create_dummy_image():
    """Create a dummy RGB image"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def main():
    print("=" * 70)
    print("PALIGEMMA ABLATION TEST")
    print("=" * 70)
    print()
    
    # Check files exist
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    print(f"✓ Config: {CONFIG_PATH}")
    print(f"✓ Checkpoint: {CHECKPOINT_PATH}")
    print()
    
    # Load config
    print("Loading config...")
    configs = load_config(CONFIG_PATH)
    
    # Load model from checkpoint
    print("Loading model from checkpoint (to CPU first)...")
    print("(This may take 1-2 minutes...)")
    
    model = MobileVLATrainer.load_from_checkpoint(
        CHECKPOINT_PATH,
        configs=configs,
        strict=False,
        map_location='cpu'
    )
    print("✓ Model loaded to CPU")
    print("Moving model to GPU...")
    model.eval()
    model = model.cuda()
    
    print("✓ Model loaded successfully")
    print()
    
    # Create test image
    test_image = create_dummy_image()
    
    # Convert to tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(test_image).unsqueeze(0).cuda()  # [1, 3, 224, 224]
    
    # Prepare vision_x: [batch, seq_len, C, H, W]
    vision_x = image_tensor.unsqueeze(1).repeat(1, 8, 1, 1, 1)  # [1, 8, 3, 224, 224]
    
    results = {}
    
    print("Testing instructions...")
    print()
    
    # Test each instruction
    for label, instruction in INSTRUCTIONS.items():
        print(f"[{label}] {instruction}")
        
        # Tokenize instruction
        # Access from model.model (the actual backbone)
        if hasattr(model.model, 'processor'):
            processor = model.model.processor
        elif hasattr(model, 'processor'):
            processor = model.processor
        else:
            raise AttributeError("Cannot find processor/tokenizer")
            
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor
            
        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        lang_x = encoded['input_ids'].cuda()  # [1, seq_len]
        attention_mask = encoded.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        
        # Inference
        with torch.no_grad():
            outputs = model.model.inference(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask
            )
        
        # Extract action
        if isinstance(outputs, tuple):
            action = outputs[0]  # (velocity, gripper)
        else:
            action = outputs
        
        # Action shape: [batch, window_size, fwd_pred_next_n, action_dim]
        # We want the first step: action[0, 0, 0, :]
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
            print(f"  Action shape: {action_np.shape}")
            
            # Extract first prediction
            if len(action_np.shape) == 4:  # [batch, window, chunk, dim]
                linear_x = float(action_np[0, 0, 0, 0])
                linear_y = float(action_np[0, 0, 0, 1])
            elif len(action_np.shape) == 3:  # [batch, step, dim]
                linear_x = float(action_np[0, 0, 0])
                linear_y = float(action_np[0, 0, 1])
            else:
                linear_x = float(action_np[0, 0])
                linear_y = float(action_np[0, 1])
            
            print(f"  → linear_x: {linear_x:.4f}")
            print(f"  → linear_y: {linear_y:.4f}")
            print()
            
            results[label] = linear_y
    
    # Analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"LEFT  → linear_y: {results['LEFT']:.4f}")
    print(f"RIGHT → linear_y: {results['RIGHT']:.4f}")
    print(f"Difference: {abs(results['LEFT'] - results['RIGHT']):.4f}")
    print()
    
    # Evaluation
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    if results['LEFT'] > 0.1 and results['RIGHT'] < -0.1:
        print("✅ SUCCESS: Model correctly distinguishes LEFT vs RIGHT!")
        print("   LEFT  → positive (turn left)")
        print("   RIGHT → negative (turn right)")
    elif abs(results['LEFT'] - results['RIGHT']) < 0.05:
        print("❌ FAIL: Model IGNORES instructions")
        print("   Both outputs are nearly identical")
    else:
        print("⚠️  PARTIAL: Outputs differ but not clearly separated")
        print(f"   Expected: LEFT > 0, RIGHT < 0")
        print(f"   Got: LEFT = {results['LEFT']:.4f}, RIGHT = {results['RIGHT']:.4f}")

if __name__ == "__main__":
    main()
