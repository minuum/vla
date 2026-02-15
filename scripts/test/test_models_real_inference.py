#!/usr/bin/env python3
"""
각 모델별 실제 추론 테스트
Validation 데이터셋으로 성능 비교
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset


def test_model(checkpoint_path, config_path, model_name, num_samples=20):
    """단일 모델 테스트"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {model_name}")
    print(f"{'='*70}")
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Load model
    try:
        print("Loading model...")
        trainer = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path,
            config_path=config_path,
            map_location="cuda"
        )
        trainer.model.to('cuda')
        trainer.model.eval()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load validation dataset
    try:
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
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    # Test samples
    results = []
    predictions = []
    ground_truths = []
    
    num_test = min(num_samples, len(val_dataset))
    
    with torch.no_grad():
        for i in tqdm(range(num_test), desc="Testing"):
            try:
                sample = val_dataset[i]
                
                # Prepare batch
                batch = {
                    'rgb': sample['rgb'].unsqueeze(0).cuda(),
                    'hand_rgb': sample['hand_rgb'].unsqueeze(0).cuda(),
                    'text': sample['text'].unsqueeze(0).cuda(),
                    'text_mask': sample['text_mask'].unsqueeze(0).cuda(),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).cuda(),
                    'action_mask': sample['action_mask'].unsqueeze(0).cuda(),
                }
                
                # Forward
                output = trainer.model(batch)
                pred_action = output['action_pred'].cpu().numpy()[0, 0]  # (2,)
                true_action = sample['actions'].cpu().numpy()[0]  # (2,)
                
                predictions.append(pred_action)
                ground_truths.append(true_action)
                
                # Calculate error
                error_x = pred_action[0] - true_action[0]
                error_y = pred_action[1] - true_action[1]
                
                results.append({
                    'pred_x': pred_action[0],
                    'pred_y': pred_action[1],
                    'true_x': true_action[0],
                    'true_y': true_action[1],
                    'error_x': error_x,
                    'error_y': error_y,
                    'instruction': sample['lang']
                })
                
            except Exception as e:
                print(f"\n❌ Error at sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if not results:
        print("❌ No successful predictions")
        return None
    
    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    errors = predictions - ground_truths
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt((errors**2).mean(axis=0))
    
    metrics = {
        'model_name': model_name,
        'num_samples': len(results),
        'mae_x': mae[0],
        'mae_y': mae[1],
        'rmse_x': rmse[0],
        'rmse_y': rmse[1],
        'mean_error_x': errors[:, 0].mean(),
        'mean_error_y': errors[:, 1].mean(),
        'std_error_x': errors[:, 0].std(),
        'std_error_y': errors[:, 1].std(),
    }
    
    print(f"\n📊 Results:")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  MAE X: {metrics['mae_x']:.4f}")
    print(f"  MAE Y: {metrics['mae_y']:.4f}")
    print(f"  RMSE X: {metrics['rmse_x']:.4f}")
    print(f"  RMSE Y: {metrics['rmse_y']:.4f}")
    
    # Clean up
    del trainer
    torch.cuda.empty_cache()
    
    return {
        'metrics': metrics,
        'results': results,
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def main():
    """메인 실행"""
    print("="*70)
    print("🚀 모델별 실제 추론 테스트")
    print("="*70)
    
    # 테스트할 모델들
    models = [
        {
            'name': 'BasketNav_Chunk5_Epoch04_BEST',
            'checkpoint': 'runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=04-val_loss=val_loss=0.020.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk5_basket.json'
        },
        {
            'name': 'BasketNav_Chunk5_Epoch09',
            'checkpoint': 'runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=09-val_loss=val_loss=0.021.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk5_basket.json'
        },
        {
            'name': 'Chunk5_Epoch06',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
        },
        {
            'name': 'Chunk5_Epoch08',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=08-val_loss=val_loss=0.086.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
        },
        {
            'name': 'Chunk5_Epoch09',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=09-val_loss=val_loss=0.083.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
        },
        {
            'name': 'Chunk10_Epoch07',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=07-val_loss=val_loss=0.317.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'Chunk10_Epoch08',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
    ]
    
    # 각 모델 테스트
    all_results = {}
    
    for model_info in models:
        if not Path(model_info['checkpoint']).exists():
            print(f"\n⚠️  Checkpoint not found: {model_info['name']}")
            continue
        
        result = test_model(
            model_info['checkpoint'],
            model_info['config'],
            model_info['name'],
            num_samples=20
        )
        
        if result:
            all_results[model_info['name']] = result
    
    # 결과 비교
    if all_results:
        print("\n" + "="*70)
        print("📊 모델 비교 결과")
        print("="*70)
        print()
        
        # 메트릭 표
        metrics_list = []
        for name, result in all_results.items():
            metrics_list.append(result['metrics'])
        
        df = pd.DataFrame(metrics_list)
        df = df.sort_values('rmse_y')
        
        print(df.to_string(index=False))
        
        print("\n" + "="*70)
        print("🏆 Best Model")
        print("="*70)
        best_model = df.iloc[0]
        print(f"\nModel: {best_model['model_name']}")
        print(f"RMSE Y: {best_model['rmse_y']:.4f}")
        print(f"MAE Y: {best_model['mae_y']:.4f}")
        
        # CSV 저장
        output_dir = Path("docs/model_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "inference_test_results.csv", index=False)
        print(f"\n✅ Results saved: {output_dir / 'inference_test_results.csv'}")
    
    print("\n" + "="*70)
    print("✅ 모든 테스트 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
