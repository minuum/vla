#!/usr/bin/env python3
"""
Basket Navigation 모델 추론 테스트 (Simple Version)
신규 학습된 모델의 성능 검증
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
import sys

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def test_basket_model():
    """Basket Navigation 모델로 실제 데이터 추론 테스트"""
    
    print("="*70)
    print("🚀 Basket Navigation 모델 추론 테스트")
    print("="*70)
    print()
    
    # Best Basket Model (Epoch 4)
    checkpoint_path = "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=04-val_loss=val_loss=0.020.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_basket.json"
    
    print(f"📁 Checkpoint: {Path(checkpoint_path).name}")
    print(f"📁 Config: {Path(config_path).name}")
    print()
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    print("🔧 Loading model...")
    trainer = MobileVLATrainer.load_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        map_location="cuda"
    )
    trainer.model.to('cuda')
    trainer.model.eval()
    print("✅ Model loaded successfully!\n")
    
    # Load actual test image from dataset
    test_file = "/home/billy/25-1kp/vla/ROS_action/basket_dataset/episode_20260129_010041_basket_1box_hori_left_core_medium.h5"
    
    print(f"📸 Loading test image from: {Path(test_file).name}")
    with h5py.File(test_file, 'r') as f:
        # Test multiple frames
        test_frames = [0, 5, 9, 14, 17]  # Beginning, middle, end
        
        print(f"\n{'='*70}")
        print("🎬 Frame-by-Frame Analysis")
        print(f"{'='*70}\n")
        
        for frame_idx in test_frames:
            img_np = f['images'][frame_idx]
            true_action = f['actions'][frame_idx][:2]  # Only linear_x, linear_y
            
            # Prepare image for model
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained('.vlms/kosmos-2-patch14-224')
            
            pil_img = Image.fromarray(img_np)
            instruction_raw = "Navigate to the brown pot on the left"
            instruction = f"<grounding>An image of a robot {instruction_raw}"
            
            # Process inputs
            inputs = processor(
                images=pil_img,
                text=instruction,
                return_tensors="pt"
            )
            
            # Move to device and add temporal dimension
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].cuda()
            
            if inputs['pixel_values'].dim() == 4:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(1)  # Add temporal dim
            
            # Inference using trainer's inference method
            with torch.no_grad():
                # Use the trainer's model forward (it handles VLM + action head)
                batch = {
                    'vision_x': inputs['pixel_values'],
                    'lang_x': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
                
                pred_output = trainer.model.inference(
                    batch['vision_x'],
                    batch['lang_x'],
                    batch['attention_mask'],
                    None,  # action_labels
                    None,  # action_mask
                    None,  # caption_labels
                    None,  # caption_mask
                    None   # vision_gripper
                )
            
            # Extract action
            action_output = pred_output['action']
            if isinstance(action_output, tuple):
                action_output = action_output[0]
            
            if action_output.dim() >= 3:
                pred_action = action_output[0, 0, 0].cpu().numpy()  # First action in chunk
            else:
                pred_action = action_output[0, 0].cpu().numpy()
            
            # Snap-to-grid
            def snap_to_grid(val, threshold=0.1):
                if abs(val) < threshold:
                    return 0.0
                return round(val * 2) / 2.0
            
            snapped_action = np.array([
                snap_to_grid(pred_action[0]),
                snap_to_grid(pred_action[1])
            ])
            
            # Calculate errors
            error_raw = np.abs(pred_action - true_action)
            error_snapped = np.abs(snapped_action - true_action)
            
            print(f"📸 Frame {frame_idx:02d}:")
            print(f"  Ground Truth:     [{true_action[0]:7.4f}, {true_action[1]:7.4f}]")
            print(f"  Raw Prediction:   [{pred_action[0]:7.4f}, {pred_action[1]:7.4f}]  (Error: {error_raw[0]:.4f}, {error_raw[1]:.4f})")
            print(f"  After Snap-Grid:  [{snapped_action[0]:7.4f}, {snapped_action[1]:7.4f}]  (Error: {error_snapped[0]:.4f}, {error_snapped[1]:.4f})")
            
            # Check perfect match
            if error_snapped[0] < 0.01 and error_snapped[1] < 0.01:
                print(f"  ✅ PERFECT MATCH!")
            elif error_snapped[0] < 0.1 and error_snapped[1] < 0.1:
                print(f"  ✅ Very close!")
            else:
                print(f"  ⚠️  Mismatch detected")
            print()
    
    print(f"{'='*70}")
    print("✅ Inference test completed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_basket_model()
