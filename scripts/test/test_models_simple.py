#!/usr/bin/env python3
"""
간단한 모델 인퍼런스 테스트 - validation 데이터로 테스트
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_dataset import MobileVLAH5Dataset


def test_model_simple(checkpoint_path, config_path, save_name):
    """간단한 모델 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(checkpoint_path).name}")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    try:
        trainer = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path,
            config_path=config_path,
            map_location="cuda"
        )
        trainer.model.to('cuda')
        trainer.model.eval()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed loading: {e}")
        return None
    
    # Load validation dataset
    val_dataset = MobileVLAH5Dataset(
        data_dir=config['val_dataset']['data_dir'],
        episode_pattern=config['val_dataset']['episode_pattern'],
        model_name=config['val_dataset']['model_name'],
        train_split=config['val_dataset']['train_split'],
        is_validation=True,
        window_size=config['window_size'],
        fwd_pred_next_n=config['fwd_pred_next_n'],
        norm_action=config['norm_action'],
        norm_min=config['norm_min'],
        norm_max=config['norm_max']
    )
    
    print(f"✅ Dataset loaded: {len(val_dataset)} samples")
    
    # Test on a few samples
    results = []
    
    with torch.no_grad():
        for i in range(min(10, len(val_dataset))):
            sample = val_dataset[i]
            
            # Prepare batch
            batch = {
                'images': sample['images'].unsqueeze(0).cuda(),  # (1, window, 3, 224, 224)
                'input_ids': sample['input_ids'].unsqueeze(0).cuda(),
                'attention_mask': sample['attention_mask'].unsqueeze(0).cuda(),
                'actions': sample['actions'].unsqueeze(0).cuda(),
                'instruction': sample['instruction']
            }
            
            # Forward
            try:
                output = trainer.model(batch)
                pred_action = output['action_pred'].cpu().numpy()[0, 0]  # (2,)
                true_action = sample['actions'].cpu().numpy()[0]  # (2,)
                
                results.append({
                    'instruction': sample['instruction'],
                    'pred_x': pred_action[0],
                    'pred_y': pred_action[1],
                    'true_x': true_action[0],
                    'true_y': true_action[1],
                })
                
                print(f"\nSample {i+1}:")
                print(f"  Instruction: {sample['instruction']}")
                print(f"  Pred: [{pred_action[0]:.3f}, {pred_action[1]:.3f}]")
                print(f"  True: [{true_action[0]:.3f}, {true_action[1]:.3f}]")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    if not results:
        print("❌ No successful predictions")
        return None
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pred_x = [r['pred_x'] for r in results]
    pred_y = [r['pred_y'] for r in results]
    true_x = [r['true_x'] for r in results]
    true_y = [r['true_y'] for r in results]
    
    # Scatter plot
    axes[0].scatter(true_x, true_y, label='Ground Truth', marker='o', s=100, alpha=0.6)
    axes[0].scatter(pred_x, pred_y, label='Predictions', marker='^', s=100, alpha=0.6)
    axes[0].set_xlabel('Linear X')
    axes[0].set_ylabel('Linear Y')
    axes[0].set_title(f'{save_name} - Predictions vs Ground Truth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Error
    error_x = [p - t for p, t in zip(pred_x, true_x)]
    error_y = [p - t for p, t in zip(pred_y, true_y)]
    
    axes[1].bar(range(len(results)), error_y, alpha=0.7)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Linear Y Error (Pred - True)')
    axes[1].set_title(f'{save_name} - Linear Y Prediction Error')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_dir = Path("docs/model_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{save_name}_validation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {save_dir / f'{save_name}_validation_test.png'}")
    plt.close()
    
    return results


def main():
    """메인"""
    print("="*60)
    print("🚀 간단한 모델 인퍼런스 테스트")
    print("="*60)
    
    models = [
        {
            'name': 'chunk10_epoch07',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=07-val_loss=val_loss=0.317.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'chunk10_epoch08',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'chunk10_last',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/last.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
    ]
    
    all_results = {}
    for model_info in models:
        if not Path(model_info['checkpoint']).exists():
            print(f"⚠️  Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        results = test_model_simple(
            model_info['checkpoint'],
            model_info['config'],
            model_info['name']
        )
        
        if results:
            all_results[model_info['name']] = results
        
        # Clean up
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
