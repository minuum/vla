#!/usr/bin/env python3
"""
Basket Navigation 모델 추론 테스트 (With Proper Window Size)
Window Size=8을 적용한 정확한 성능 측정
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

def test_basket_model_windowed():
    """Basket Navigation 모델로 Window=8 방식 추론 테스트"""
    
    print("="*70)
    print("🚀 Basket Navigation 모델 추론 테스트 (Window Size = 8)")
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
    
    window_size = config['window_size']
    print(f"📐 Window Size: {window_size}")
    print()
    
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
    
    print(f"📸 Loading test data from: {Path(test_file).name}")
    
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('.vlms/kosmos-2-patch14-224')
    
    with h5py.File(test_file, 'r') as f:
        total_frames = len(f['images'])
        print("  Total frames in episode: " + str(total_frames))
        print()
        
        # Test at different positions (where we have enough history)
        test_positions = [9, 12, 15]  # Middle and later frames
        
        print("="*70)
        print("🎬 Frame-by-Frame Analysis (with 8-frame window)")
        print("="*70)
        print()
        
        for target_frame in test_positions:
            # Extract window of 8 frames ending at target_frame
            start_frame = max(0, target_frame - window_size + 1)
            end_frame = start_frame + window_size
            
            if end_frame > total_frames:
                continue
            
            # Load window of images
            window_images = []
            for i in range(start_frame, end_frame):
                img_np = f['images'][i]
                pil_img = Image.fromarray(img_np)
                window_images.append(pil_img)
            
            # Get ground truth for target frame
            true_action = f['actions'][target_frame][:2]  # Only linear_x, linear_y
            
            instruction = "Navigate to the brown pot on the left"
            
            # Process all images in window
            # Stack them as a temporal sequence
            all_pixel_values = []
            
            for img in window_images:
                inputs = processor(
                    images=img,
                    text=instruction,
                    return_tensors="pt"
                )
                all_pixel_values.append(inputs['pixel_values'])
            
            # Stack along temporal dimension: (1, T, C, H, W)
            pixel_values = torch.cat(all_pixel_values, dim=0).unsqueeze(0).cuda()  # (1, 8, 3, 224, 224)
            
            # Use the last frame's text encoding (instruction is same for all)
            lang_x = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            
            # Inference with full window
            with torch.no_grad():
                pred_output = trainer.model.inference(
                    pixel_values,
                    lang_x,
                    attention_mask,
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
            
            # Get the last timestep prediction (corresponding to target_frame)
            if action_output.dim() >= 3:
                # (B, T, chunk_size, action_dim) -> take last timestep, first chunk
                pred_action = action_output[0, -1, 0].cpu().numpy()
            else:
                pred_action = action_output[0, 0].cpu().numpy()
            
            # Snap-to-grid
            def snap_to_grid(val):
                targets = np.array([-1.15, 0.0, 1.15])
                closest_idx = np.argmin(np.abs(targets - val))
                return targets[closest_idx]
            
            snapped_action = np.array([
                snap_to_grid(pred_action[0]),
                snap_to_grid(pred_action[1])
            ])
            
            # Calculate errors
            error_raw = np.abs(pred_action - true_action)
            error_snapped = np.abs(snapped_action - true_action)
            
            print(f"📸 Frame {target_frame:02d} (window: {start_frame:02d}-{end_frame-1:02d}):")
            print(f"  Ground Truth:     [{true_action[0]:7.4f}, {true_action[1]:7.4f}]")
            print(f"  Raw Prediction:   [{pred_action[0]:7.4f}, {pred_action[1]:7.4f}]  (Error: {error_raw[0]:.4f}, {error_raw[1]:.4f})")
            print(f"  After Snap-Grid:  [{snapped_action[0]:7.4f}, {snapped_action[1]:7.4f}]  (Error: {error_snapped[0]:.4f}, {error_snapped[1]:.4f})")
            
            # Check perfect match
            if error_snapped[0] < 0.01 and error_snapped[1] < 0.01:
                print(f"  ✅ PERFECT MATCH!")
            elif error_snapped[0] < 0.3 and error_snapped[1] < 0.3:
                print(f"  ✅ Very close!")
            else:
                print(f"  ⚠️  Mismatch detected")
            
            # Show what action this corresponds to
            action_desc = ""
            if abs(snapped_action[0]) < 0.1 and abs(snapped_action[1]) < 0.1:
                action_desc = "STOP"
            elif snapped_action[0] > 0.5 and abs(snapped_action[1]) < 0.1:
                action_desc = "Forward (W)"
            elif snapped_action[0] > 0.5 and snapped_action[1] > 0.5:
                action_desc = "Forward-Left (Q)"
            elif snapped_action[0] > 0.5 and snapped_action[1] < -0.5:
                action_desc = "Forward-Right (E)"
            elif abs(snapped_action[0]) < 0.1 and snapped_action[1] > 0.5:
                action_desc = "Left (A)"
            elif abs(snapped_action[0]) < 0.1 and snapped_action[1] < -0.5:
                action_desc = "Right (D)"
            
            print("  🎮 Predicted Action: " + action_desc)
            print()
    
    print("="*70)
    print("✅ Windowed inference test completed!")
    print("="*70)

if __name__ == "__main__":
    test_basket_model_windowed()
