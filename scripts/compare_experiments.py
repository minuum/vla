#!/usr/bin/env python3
"""
VLA 모델 비교 분석 스크립트

여러 학습 케이스의 결과를 비교 분석합니다.

케이스:
1. Baseline (기존)
2. action_token Xavier init
3. abs_action (방향 제거)
4. OpenVLA style (LR 2e-5, 27 epochs)
5. No chunking
6. Hybrid (Classification + Regression)

작성일: 2025-12-09
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, 'RoboVLMs_upstream')


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    name: str
    checkpoint_path: str
    mae: float
    direction_accuracy: float
    left_mean: float
    right_mean: float
    left_right_diff: float
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """모델 로드"""
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    model = MobileVLATrainer.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    model.to(device)
    return model


def evaluate_model(
    model,
    data_files: List[Path],
    device: str = 'cuda',
    abs_action: bool = False
) -> Dict:
    """모델 평가"""
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    
    results = []
    directions = []
    
    for h5_file in data_files:
        with h5py.File(h5_file, 'r') as f:
            images = []
            for t in range(8):
                img = Image.fromarray(f['images'][t].astype(np.uint8))
                images.append(transform(img))
            vision_x = torch.stack(images).unsqueeze(0).to(device)
            
            gt = f['actions'][:8, 1].mean()  # linear_y GT
            
            # 언어 명령에서 방향 추출
            lang = f['language_instruction'][()]
            if isinstance(lang, bytes):
                lang = lang.decode('utf-8')
            elif hasattr(lang, '__iter__') and len(lang) > 0:
                lang = lang[0].decode('utf-8') if isinstance(lang[0], bytes) else str(lang[0])
            
            direction = 1.0 if 'left' in lang.lower() else -1.0
            directions.append(direction)
        
        with torch.no_grad():
            image_features = model.model.encode_images(vision_x)
            bs, seq_len, n_tok, hidden = image_features.shape
            action_hs = image_features[:, :, -1:, :].squeeze(2)
            
            action_mask = torch.ones(bs, seq_len, dtype=torch.bool).to(device)
            actions = model.model.act_head(action_hs, actions=None, action_masks=action_mask)
            if isinstance(actions, tuple):
                actions = actions[0]
            
            pred = actions[0].mean(dim=(0, 1)).cpu().numpy()[1]
            
            # abs_action 모델인 경우 방향 적용
            if abs_action:
                pred = abs(pred) * direction
            
            results.append({
                'pred': pred,
                'gt': gt,
                'direction': direction,
                'error': abs(pred - gt),
                'correct_direction': np.sign(pred) == np.sign(gt)
            })
    
    # 통계 계산
    mae = np.mean([r['error'] for r in results])
    direction_accuracy = np.mean([r['correct_direction'] for r in results])
    
    left_preds = [r['pred'] for r in results if r['direction'] == 1.0]
    right_preds = [r['pred'] for r in results if r['direction'] == -1.0]
    
    left_mean = np.mean(left_preds) if left_preds else 0
    right_mean = np.mean(right_preds) if right_preds else 0
    
    return {
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'left_mean': left_mean,
        'right_mean': right_mean,
        'left_right_diff': left_mean - right_mean,
        'n_samples': len(results)
    }


def find_best_checkpoint(run_dir: str) -> Optional[str]:
    """가장 낮은 val_loss 체크포인트 찾기"""
    run_path = Path(run_dir)
    
    checkpoints = list(run_path.glob("**/epoch_*.ckpt"))
    if not checkpoints:
        return None
    
    # val_loss 파싱
    best_ckpt = None
    best_loss = float('inf')
    
    for ckpt in checkpoints:
        name = ckpt.name
        if 'val_loss=' in name:
            try:
                loss = float(name.split('val_loss=')[-1].replace('.ckpt', ''))
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = str(ckpt)
            except:
                pass
    
    return best_ckpt


def main():
    print('='*70)
    print('📊 VLA 모델 비교 분석')
    print('='*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 로드
    data_dir = Path('ROS_action/mobile_vla_dataset')
    all_files = sorted(list(data_dir.glob('episode_*.h5')))[:100]  # 100개 샘플
    print(f'평가 데이터: {len(all_files)}개 에피소드')
    
    # 케이스 정의
    cases = [
        {
            'name': 'Baseline (기존)',
            'run_dir': 'runs/mobile_vla_kosmos2_frozen_lora_leftright_20251204',
            'abs_action': False
        },
        {
            'name': 'action_token Xavier',
            'run_dir': 'runs/mobile_vla_kosmos2_fixed_20251209',
            'abs_action': False
        },
        {
            'name': 'abs_action (방향 제거)',
            'run_dir': 'runs/mobile_vla_kosmos2_abs_action_20251209',
            'abs_action': True
        },
    ]
    
    results = []
    
    for case in cases:
        print(f'\n=== {case["name"]} ===')
        
        ckpt = find_best_checkpoint(case['run_dir'])
        if ckpt is None:
            print(f'  ❌ 체크포인트 없음: {case["run_dir"]}')
            continue
        
        print(f'  체크포인트: {Path(ckpt).name}')
        
        try:
            model = load_model(ckpt, device)
            metrics = evaluate_model(model, all_files, device, case['abs_action'])
            
            result = EvaluationResult(
                name=case['name'],
                checkpoint_path=ckpt,
                mae=metrics['mae'],
                direction_accuracy=metrics['direction_accuracy'],
                left_mean=metrics['left_mean'],
                right_mean=metrics['right_mean'],
                left_right_diff=metrics['left_right_diff']
            )
            results.append(result)
            
            print(f'  MAE: {metrics["mae"]:.4f}')
            print(f'  방향 정확도: {metrics["direction_accuracy"]*100:.1f}%')
            print(f'  LEFT mean: {metrics["left_mean"]:+.4f}')
            print(f'  RIGHT mean: {metrics["right_mean"]:+.4f}')
            print(f'  Left-Right 차이: {metrics["left_right_diff"]:+.4f}')
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f'  ❌ 오류: {e}')
    
    # 결과 요약
    print('\n' + '='*70)
    print('📋 결과 요약')
    print('='*70)
    print(f'{"케이스":<25} {"MAE":<10} {"방향정확도":<12} {"L-R차이":<10}')
    print('-'*60)
    
    for r in results:
        print(f'{r.name:<25} {r.mae:<10.4f} {r.direction_accuracy*100:<12.1f}% {r.left_right_diff:<+10.4f}')
    
    # 최고 성능 케이스
    if results:
        best = max(results, key=lambda x: x.direction_accuracy)
        print(f'\n🏆 최고 방향 정확도: {best.name} ({best.direction_accuracy*100:.1f}%)')
        
        best_mae = min(results, key=lambda x: x.mae)
        print(f'🏆 최저 MAE: {best_mae.name} ({best_mae.mae:.4f})')


if __name__ == '__main__':
    main()
