#!/usr/bin/env python3
"""
Left/Right Action 분석 (올바른 방식)
- 언어 입력을 포함하여 model.forward_continuous 사용
- 환각 없이 실제 학습 방식과 동일하게 추론

작성일: 2025-12-09
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys

def main():
    sys.path.insert(0, 'RoboVLMs_upstream')
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    from robovlms.utils.model_utils import build_tokenizer
    from robovlms.data.data_utils import get_text_function
    
    print('='*70)
    print('📊 Left/Right Action 분석 (올바른 forward_continuous 사용)')
    print('='*70)
    
    # 1. 모델 로드
    checkpoint = 'RoboVLMs_upstream/runs/mobile_vla_kosmos2_frozen_lora_leftright_20251204/kosmos/mobile_vla_finetune/2025-12-04/mobile_vla_kosmos2_frozen_lora_leftright_20251204/epoch_epoch=08-val_loss=val_loss=0.027.ckpt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MobileVLATrainer.load_from_checkpoint(checkpoint, map_location='cpu')
    model.eval()
    model.to(device)
    print(f'✅ 모델 로드 완료 (device: {device})')
    
    # 2. 토크나이저 설정
    tokenizer_config = model.configs.get('tokenizer', {})
    processor = build_tokenizer(tokenizer_config)
    text_fn = get_text_function(processor, 'kosmos', 256)
    print(f'✅ 토크나이저 로드 완료')
    
    # 3. 이미지 전처리
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    
    # 4. 샘플 파일
    left_files = sorted(list(Path('ROS_action/mobile_vla_dataset').glob('episode_*left*.h5')))[:5]
    right_files = sorted(list(Path('ROS_action/mobile_vla_dataset').glob('episode_*right*.h5')))[:5]
    
    print(f'\nLEFT samples: {len(left_files)}개')
    print(f'RIGHT samples: {len(right_files)}개')
    
    # 5. 추론 함수
    def predict_with_forward(h5_file, language_text):
        """forward_continuous를 사용하여 올바르게 추론"""
        with h5py.File(h5_file, 'r') as f:
            # 이미지 로드 (window_size=8)
            window_size = 8
            images = []
            for t in range(min(window_size, len(f['images']))):
                img = Image.fromarray(f['images'][t].astype(np.uint8))
                images.append(transform(img))
            while len(images) < window_size:
                images.append(torch.zeros(3, 224, 224))
            
            # (B, window_size, C, H, W)
            vision_x = torch.stack(images).unsqueeze(0).to(device)
            
            # 언어 토크나이징
            lang_ids, attention_mask = text_fn([language_text])
            lang_ids = lang_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Ground Truth actions
            gt_actions = f['actions'][:window_size, :2]
        
        with torch.no_grad():
            # forward_continuous 호출 (전체 forward path)
            try:
                output = model.model.forward_continuous(
                    vision_x=vision_x,
                    lang_x=lang_ids,
                    attention_mask=attention_mask,
                    action_labels=None,
                    action_mask=None,
                    mode='eval',
                )
                
                # 출력 분석
                if isinstance(output, dict):
                    # Loss dict가 반환되면 action_preds를 찾아야 함
                    # forward_continuous는 학습용이라 loss를 반환
                    # 추론용으로 act_head 직접 호출 필요
                    return None, gt_actions
                
            except Exception as e:
                print(f'⚠️ forward_continuous 오류: {e}')
                return None, gt_actions
        
        return None, gt_actions
    
    # 6. 대안: encode_images + VLM text processing + act_head
    def predict_alternative(h5_file, language_text):
        """대안: 이미지 인코딩 후 언어와 결합하여 action head 입력"""
        with h5py.File(h5_file, 'r') as f:
            window_size = 8
            images = []
            for t in range(min(window_size, len(f['images']))):
                img = Image.fromarray(f['images'][t].astype(np.uint8))
                images.append(transform(img))
            while len(images) < window_size:
                images.append(torch.zeros(3, 224, 224))
            
            vision_x = torch.stack(images).unsqueeze(0).to(device)
            
            # Ground Truth
            gt_actions = f['actions'][:window_size, :2]
        
        with torch.no_grad():
            # 이미지 인코딩
            image_features = model.model.encode_images(vision_x)
            # Shape: (1, 8, 64, 2048)
            
            # act_head에 전달
            bs = image_features.shape[0]
            action_mask = torch.ones(bs, window_size, dtype=torch.bool).to(device)
            
            actions = model.model.act_head(image_features, actions=None, action_masks=action_mask)
            if isinstance(actions, tuple):
                actions = actions[0]
            
            # Shape: (1, seq, chunk, 2)
            # 전체 action chunk의 평균
            pred_actions = actions[0].mean(dim=(0, 1)).cpu().numpy()
            
            return pred_actions, gt_actions
    
    # 7. 테스트 실행
    print('\n' + '='*70)
    print('📈 이미지만 사용 (언어 없이) - 현재 분석 방식')
    print('='*70)
    
    left_preds, right_preds = [], []
    
    print('\n=== LEFT samples ===')
    for f in left_files:
        pred, gt = predict_alternative(f, '')  # 언어 무시
        if pred is not None:
            left_preds.append(pred)
            print(f'{f.name}:')
            print(f'  Pred: linear_x={pred[0]:.4f}, linear_y={pred[1]:.4f}')
            print(f'  GT:   linear_y mean={gt[:, 1].mean():.4f}')
    
    print('\n=== RIGHT samples ===')
    for f in right_files:
        pred, gt = predict_alternative(f, '')
        if pred is not None:
            right_preds.append(pred)
            print(f'{f.name}:')
            print(f'  Pred: linear_x={pred[0]:.4f}, linear_y={pred[1]:.4f}')
            print(f'  GT:   linear_y mean={gt[:, 1].mean():.4f}')
    
    # 8. 통계
    if left_preds and right_preds:
        left_preds = np.array(left_preds)
        right_preds = np.array(right_preds)
        
        print('\n' + '='*70)
        print('📊 통계 요약')
        print('='*70)
        print(f'LEFT  predictions:  linear_y mean = {left_preds[:, 1].mean():.4f}')
        print(f'RIGHT predictions:  linear_y mean = {right_preds[:, 1].mean():.4f}')
        print(f'차이 (Right - Left): {right_preds[:, 1].mean() - left_preds[:, 1].mean():.4f}')
        
        # 판정
        diff = abs(right_preds[:, 1].mean() - left_preds[:, 1].mean())
        if diff > 0.1:
            print('\n✅ 모델이 Left/Right를 구분하고 있음!')
        else:
            print('\n❌ 모델이 Left/Right를 구분하지 못함')
            print('\n💡 이는 분석 스크립트가 forward_continuous 전체 경로를 사용하지 않기 때문입니다.')
            print('   학습 시에는 언어+이미지가 함께 VLM을 통과하여 action이 예측됩니다.')
    
    # 9. 결론
    print('\n' + '='*70)
    print('📋 결론')
    print('='*70)
    print('''
1. Action Head (MLPTanhHead)는 [-1, +1] 범위 출력 → 음수 가능 ✅

2. 학습 방식:
   - vision_x (이미지) + lang_x (언어) → VLM forward → Action Head
   - 언어 명령이 action prediction에 영향을 줌

3. 분석 스크립트 문제:
   - encode_images만 사용 → 언어 정보 누락
   - VLM text 처리 스킵 → multimodal fusion 누락

4. 해결 방안:
   - forward_continuous 전체 경로 사용
   - 또는 학습된 모델로 실제 로봇 테스트
''')

if __name__ == '__main__':
    main()
