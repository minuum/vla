#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

# Add RoboVLMs to path
sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset

def test_v2_model(checkpoint_path, config_path, model_name):
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load trainer (which loads the model)
    try:
        trainer = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path,
            config_path=config_path,
            map_location="cuda"
        )
        model = trainer.model.to('cuda')
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load tokenizer
    from robovlms.utils.model_utils import build_tokenizer
    tokenizer_config = config.get('tokenizer', None)
    tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
    
    # Kosmos-2 fix: MobileVLAH5Dataset expects a tokenizer, not a processor
    # If build_tokenizer returned an AutoProcessor (for Kosmos-2), we should pass the inner tokenizer
    dataset_tokenizer = tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        dataset_tokenizer = tokenizer.tokenizer
    
    # Load validation dataset
    val_cfg = config['val_dataset']
    val_dataset = MobileVLAH5Dataset(
        data_dir=val_cfg['data_dir'],
        episode_pattern=val_cfg['episode_pattern'],
        model_name=val_cfg['model_name'],
        train_split=val_cfg['train_split'],
        is_validation=True,
        window_size=config['window_size'],
        fwd_pred_next_n=config['fwd_pred_next_n'],
        tokenizer=dataset_tokenizer,
        tokenizer_config=tokenizer_config
    )
    
    print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
    
    results = []
    with torch.no_grad():
        # Pick 20 samples across the dataset
        num_samples = min(20, len(val_dataset))
        indices = np.linspace(0, len(val_dataset)-1, num_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            sample = val_dataset[idx]
            batch = val_dataset.collater([sample])
            
            # Move to gpu
            gpu_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    gpu_batch[k] = v.to('cuda')
                else:
                    gpu_batch[k] = v
            
            # Forward call using inference mode to avoid loss calculation
            prediction = model.inference(
                gpu_batch['rgb'],
                gpu_batch['text'],
                attention_mask=gpu_batch['text_mask'],
                vision_gripper=gpu_batch['hand_rgb'],
                raw_text=gpu_batch['raw_text'],
                data_source=gpu_batch['data_source']
            )
            
            # Extract prediction and GT
            # For continuous action models, prediction['action'] is the action_logits
            # Shape for MobileVLA: (B, window_size, chunk, 2)
            pred_logits = prediction['action']
            if isinstance(pred_logits, tuple):
                pred_logits = pred_logits[0]
                
            # Take the prediction from the last window step
            # pred_logits shape: (B, L, chunk, 2)
            pred_chunk = pred_logits.cpu().numpy()[0, -1] # (chunk, 2)
            pred_action = pred_chunk[0] # First step of the predicted chunk
            
            # GT from batch['action_chunck']
            gt_chunk = gpu_batch['action_chunck'].cpu().numpy()[0, -1]
            gt_action = gt_chunk[0]
            
            results.append({
                'idx': idx,
                'instruction': gpu_batch['raw_text'][0],
                'pred_x': pred_action[0],
                'pred_y': pred_action[1],
                'true_x': gt_action[0],
                'true_y': gt_action[1]
            })
            
            if i % 5 == 0:
                print(f"[{i}/{num_samples}] Sample {idx}: Pred={pred_action}, GT={gt_action}")

    # Create visualization
    plt.figure(figsize=(12, 10))
    
    true_actions = np.array([[r['true_x'], r['true_y']] for r in results])
    pred_actions = np.array([[r['pred_x'], r['pred_y']] for r in results])
    
    plt.scatter(true_actions[:, 0], true_actions[:, 1], color='blue', label='Ground Truth', s=100, alpha=0.6)
    plt.scatter(pred_actions[:, 0], pred_actions[:, 1], color='red', label='Prediction', s=100, alpha=0.6, marker='x')
    
    # Draw error lines
    for i in range(len(results)):
        plt.plot([true_actions[i, 0], pred_actions[i, 0]], 
                 [true_actions[i, 1], pred_actions[i, 1]], 
                 color='gray', linestyle='--', alpha=0.3)
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel('Linear X (Forward)')
    plt.ylabel('Linear Y (Side)')
    plt.title(f'V2 Model Inference Test: {model_name}\n(Predictions vs Ground Truth on Val Set)')
    plt.legend()
    
    # Add metrics
    mae_x = np.mean(np.abs(true_actions[:, 0] - pred_actions[:, 0]))
    mae_y = np.mean(np.abs(true_actions[:, 1] - pred_actions[:, 1]))
    plt.text(0.05, 0.95, f"MAE X: {mae_x:.4f}\nMAE Y: {mae_y:.4f}", 
             transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    save_dir = Path("docs/model_comparison/v2_tests")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}_inference.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ Results saved to {save_path}")
    plt.close()
    
    return {
        'model_name': model_name,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'samples': results
    }

def main():
    v2_17_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_17/2026-02-15/exp-v2-17/epoch_epoch=09-val_loss=val_loss=0.001.ckpt"
    v2_17_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_17.json"
    
    v2_12_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_12/2026-02-16/exp-v2-12/epoch_epoch=07-val_loss=val_loss=0.001.ckpt"
    v2_12_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_12.json"
    
    results = []
    
    if os.path.exists(v2_17_ckpt):
        res17 = test_v2_model(v2_17_ckpt, v2_17_config, "exp-v2-17")
        if res17: results.append(res17)
    else:
        print(f"⚠️ Checkpoint not found: {v2_17_ckpt}")
        
    if os.path.exists(v2_12_ckpt):
        res12 = test_v2_model(v2_12_ckpt, v2_12_config, "exp-v2-12")
        if res12: results.append(res12)
    else:
        print(f"⚠️ Checkpoint not found: {v2_12_ckpt}")

    # Print summary table
    if results:
        print("\n" + "="*40)
        print(f"{'Model':<15} | {'MAE X':<10} | {'MAE Y':<10}")
        print("-" * 40)
        for r in results:
            print(f"{r['model_name']:<15} | {r['mae_x']:<10.4f} | {r['mae_y']:<10.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
