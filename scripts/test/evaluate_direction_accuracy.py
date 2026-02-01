#!/usr/bin/env python3
"""
방향 정확도 평가 스크립트
No Chunk 모델(Epoch 4)의 Left/Right 방향 구분 정확도를 측정합니다.
"""

import torch
import numpy as np
import h5py
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json

# Add RoboVLMs to path
sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def get_direction_from_text(instruction):
    """
    텍스트에서 방향 추출
    Left: +1.0, Right: -1.0, Straight: 0.0
    """
    instr_lower = instruction.lower()
    if 'left' in instr_lower:
        return 1.0
    elif 'right' in instr_lower:
        return -1.0
    else:
        return 0.0

def evaluate_direction_accuracy(checkpoint_path, data_dir, num_episodes=50):
    """
    체크포인트와 데이터셋으로 방향 정확도 평가
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        data_dir: H5 데이터셋 디렉토리
        num_episodes: 평가할 에피소드 수
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔄 모델 로딩 중... ({checkpoint_path})")
    
    # 모델 로드
    model = MobileVLATrainer.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    model.to(device)
    
    print(f"✅ 모델 로드 완료! (디바이스: {device})")
    
    # 데이터 파일 찾기
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob('episode_20251*.h5'))[:num_episodes]
    
    print(f"📂 데이터셋: {len(episode_files)}개 에피소드")
    
    results = {
        'total': 0,
        'correct': 0,
        'left_correct': 0,
        'left_total': 0,
        'right_correct': 0,
        'right_total': 0,
        'straight_correct': 0,
        'straight_total': 0,
        'errors': []
    }
    
    with torch.no_grad():
        for ep_file in tqdm(episode_files, desc="평가 중"):
            try:
                with h5py.File(ep_file, 'r') as f:
                    # 데이터 추출
                    images = f['images'][:]  # (T, H, W, C)
                    actions = f['actions'][:]  # (T, 3) - [linear_x, linear_y, angular_z]
                    instruction_raw = f['language_instruction'][0] if 'language_instruction' in f else ''
                    instruction = instruction_raw.decode('utf-8') if isinstance(instruction_raw, bytes) else instruction_raw
                    
                    # Ground truth 방향
                    gt_direction = get_direction_from_text(instruction)
                    
                    # 샘플 선택 (중간 프레임)
                    mid_idx = len(images) // 2
                    img = images[mid_idx]  # (H, W, C)
                    gt_action = actions[mid_idx]  # (2,)
                    
                    # 이미지 전처리
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
                    img_tensor = torch.nn.functional.interpolate(
                        img_tensor.unsqueeze(0), 
                        size=(224, 224), 
                        mode='bilinear'
                    )  # (1, C, H, W)
                    
                    # Sequence 차원 추가: (1, C, H, W) -> (B=1, Seq=1, C, H, W)
                    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 1, C, H, W)
                    
                    # 추론
                    bs, seq_len, c, h, w = img_tensor.shape
                    encoded = model.model.vision_encoder(img_tensor.view(-1, c, h, w))
                    encoded = encoded.view(bs, seq_len, -1, encoded.shape[-1])
                    
                    action_hs = encoded[:, :, -1:, :].squeeze(2)
                    action_mask = torch.ones(bs, seq_len, dtype=torch.bool).to(device)
                    
                    pred_actions = model.model.act_head(action_hs, actions=None, action_masks=action_mask)
                    if isinstance(pred_actions, tuple):
                        pred_actions = pred_actions[0]
                    
                    pred_action = pred_actions[0, -1, 0].cpu().numpy()  # [linear_x, linear_y]
                    
                    # 예측 방향 (linear_y의 부호로 판단)
                    pred_direction = np.sign(pred_action[1])
                    
                    # 정확도 계산
                    results['total'] += 1
                    
                    if gt_direction == 1.0:  # Left
                        results['left_total'] += 1
                        if pred_direction > 0:
                            results['left_correct'] += 1
                            results['correct'] += 1
                        else:
                            results['errors'].append({
                                'file': ep_file.name,
                                'instruction': instruction,
                                'gt_direction': 'LEFT',
                                'pred_direction': 'RIGHT' if pred_direction < 0 else 'STRAIGHT',
                                'gt_linear_y': float(gt_action[1]),
                                'pred_linear_y': float(pred_action[1])
                            })
                    
                    elif gt_direction == -1.0:  # Right
                        results['right_total'] += 1
                        if pred_direction < 0:
                            results['right_correct'] += 1
                            results['correct'] += 1
                        else:
                            results['errors'].append({
                                'file': ep_file.name,
                                'instruction': instruction,
                                'gt_direction': 'RIGHT',
                                'pred_direction': 'LEFT' if pred_direction > 0 else 'STRAIGHT',
                                'gt_linear_y': float(gt_action[1]),
                                'pred_linear_y': float(pred_action[1])
                            })
                    
                    else:  # Straight (0.0)
                        results['straight_total'] += 1
                        if abs(pred_direction) < 0.1:
                            results['straight_correct'] += 1
                            results['correct'] += 1
                        else:
                            results['errors'].append({
                                'file': ep_file.name,
                                'instruction': instruction,
                                'gt_direction': 'STRAIGHT',
                                'pred_direction': 'LEFT' if pred_direction > 0 else 'RIGHT',
                                'gt_linear_y': float(gt_action[1]),
                                'pred_linear_y': float(pred_action[1])
                            })
            
            except Exception as e:
                print(f"⚠️ 에러 발생: {ep_file.name} - {e}")
                continue
    
    # 정확도 계산
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0.0
    results['left_accuracy'] = results['left_correct'] / results['left_total'] if results['left_total'] > 0 else 0.0
    results['right_accuracy'] = results['right_correct'] / results['right_total'] if results['right_total'] > 0 else 0.0
    results['straight_accuracy'] = results['straight_correct'] / results['straight_total'] if results['straight_total'] > 0 else 0.0
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="방향 정확도 평가")
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--data-dir', type=str, default='ROS_action/mobile_vla_dataset', help='데이터셋 디렉토리')
    parser.add_argument('--num-episodes', type=int, default=50, help='평가할 에피소드 수')
    parser.add_argument('--output', type=str, default='direction_accuracy_results.json', help='결과 저장 파일')
    args = parser.parse_args()
    
    # 평가 실행
    results = evaluate_direction_accuracy(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        num_episodes=args.num_episodes
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 방향 정확도 평가 결과")
    print("="*60)
    print(f"전체 정확도: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"\n방향별 정확도:")
    print(f"  LEFT:     {results['left_accuracy']*100:.2f}% ({results['left_correct']}/{results['left_total']})")
    print(f"  RIGHT:    {results['right_accuracy']*100:.2f}% ({results['right_correct']}/{results['right_total']})")
    print(f"  STRAIGHT: {results['straight_accuracy']*100:.2f}% ({results['straight_correct']}/{results['straight_total']})")
    
    if results['errors']:
        print(f"\n⚠️ 오류 케이스: {len(results['errors'])}개")
        for i, error in enumerate(results['errors'][:5], 1):
            print(f"\n{i}. {error['file']}")
            print(f"   명령: {error['instruction']}")
            print(f"   GT: {error['gt_direction']} ({error['gt_linear_y']:.3f})")
            print(f"   예측: {error['pred_direction']} ({error['pred_linear_y']:.3f})")
    
    # 결과 저장
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장됨: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
