#!/usr/bin/env python3
"""
빠른 해결책: 언어 특징 직접 주입 테스트
- VLM output에서 언어 토큰 부분을 직접 추출하여 action_hs에 concat
- 재학습 없이 즉시 효과 테스트

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
    
    print('='*70)
    print('🔧 빠른 해결책: 언어 특징 직접 주입 테스트')
    print('='*70)
    
    checkpoint = 'RoboVLMs_upstream/runs/mobile_vla_kosmos2_frozen_lora_leftright_20251204/kosmos/mobile_vla_finetune/2025-12-04/mobile_vla_kosmos2_frozen_lora_leftright_20251204/epoch_epoch=08-val_loss=val_loss=0.027.ckpt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MobileVLATrainer.load_from_checkpoint(checkpoint, map_location='cpu')
    model.eval()
    model.to(device)
    print(f'✅ 모델 로드 (device: {device})')
    
    tokenizer = model.model.tokenizer
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    
    # 데이터 로드
    left_h5 = sorted(list(Path('ROS_action/mobile_vla_dataset').glob('episode_*left*.h5')))[0]
    with h5py.File(left_h5, 'r') as f:
        left_images = []
        for t in range(8):
            img = Image.fromarray(f['images'][t].astype(np.uint8))
            left_images.append(transform(img))
        left_vision = torch.stack(left_images).unsqueeze(0).to(device)
    
    # 언어 명령
    left_text = '<grounding>An image of a robot Navigate around obstacles and reach the front of the beverage bottle on the left'
    right_text = '<grounding>An image of a robot Navigate around obstacles and reach the front of the beverage bottle on the right'
    
    def tokenize(text):
        tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
        return tokens['input_ids'].to(device), tokens['attention_mask'].to(device).bool()
    
    left_ids, left_mask = tokenize(left_text)
    right_ids, right_mask = tokenize(right_text)
    
    # 언어 임베딩 직접 추출
    print('\n=== 언어 임베딩 분석 ===')
    with torch.no_grad():
        left_embed = model.model.word_embedding(left_ids)  # (1, 256, 2048)
        right_embed = model.model.word_embedding(right_ids)
        
        # 유효 토큰만 평균
        left_valid = left_embed[0, left_ids[0] != 0].mean(dim=0)  # (2048,)
        right_valid = right_embed[0, right_ids[0] != 0].mean(dim=0)
        
        print(f'Left lang embedding norm: {left_valid.norm().item():.4f}')
        print(f'Right lang embedding norm: {right_valid.norm().item():.4f}')
        print(f'Difference norm: {(left_valid - right_valid).norm().item():.4f}')
    
    # 언어 조건부 action 예측 (수동)
    print('\n=== 언어 조건부 예측 (이미지+언어 직접 결합) ===')
    
    def predict_with_lang_injection(vision_x, lang_embed):
        """이미지 특징에 언어 특징을 직접 더하여 예측"""
        with torch.no_grad():
            # 이미지 인코딩
            image_features = model.model.encode_images(vision_x)  # (1, 8, 64, 2048)
            
            # 마지막 토큰 추출 (기존 방식)
            bs, seq_len, n_tok, hidden = image_features.shape
            latent_num = 1
            action_hs = image_features[:, :, -latent_num:, :]  # (1, 8, 1, 2048)
            
            # 언어 특징 추가 (직접 더하기)
            lang_expanded = lang_embed.view(1, 1, 1, -1).expand_as(action_hs)
            action_hs_with_lang = action_hs + lang_expanded * 0.1  # 스케일링
            
            # Action Head
            action_hs_input = action_hs_with_lang.squeeze(2)  # (1, 8, 2048)
            action_mask = torch.ones(bs, seq_len, dtype=torch.bool).to(device)
            
            actions = model.model.act_head(action_hs_input, actions=None, action_masks=action_mask)
            if isinstance(actions, tuple):
                actions = actions[0]
            
            return actions[0].mean(dim=(0, 1)).cpu().numpy()
    
    # 테스트
    print('\n=== 테스트 결과 ===')
    
    # LEFT 이미지 + LEFT 언어
    pred_ll = predict_with_lang_injection(left_vision, left_valid)
    print(f'LEFT 이미지 + LEFT 언어:  linear_y = {pred_ll[1]:.4f}')
    
    # LEFT 이미지 + RIGHT 언어
    pred_lr = predict_with_lang_injection(left_vision, right_valid)
    print(f'LEFT 이미지 + RIGHT 언어: linear_y = {pred_lr[1]:.4f}')
    
    diff = abs(pred_ll[1] - pred_lr[1])
    print(f'\n차이: {diff:.4f}')
    
    if diff > 0.05:
        print('✅ 언어 특징 주입으로 차이 발생!')
        print('   → 언어 정보가 action에 영향을 줄 수 있음 확인')
        print('   → 이 방식으로 모델 수정 후 재학습 권장')
    else:
        print('❌ 여전히 차이 없음')
        print('   → Action Head가 언어 정보를 활용하도록 재학습 필요')
    
    # 다양한 스케일 테스트
    print('\n=== 언어 스케일 파라미터 테스트 ===')
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0]:
        with torch.no_grad():
            image_features = model.model.encode_images(left_vision)
            bs, seq_len, n_tok, hidden = image_features.shape
            action_hs = image_features[:, :, -1:, :]
            
            # LEFT 언어
            lang_l = left_valid.view(1, 1, 1, -1).expand_as(action_hs)
            hs_l = action_hs + lang_l * scale
            actions_l = model.model.act_head(hs_l.squeeze(2), actions=None, 
                           action_masks=torch.ones(bs, seq_len, dtype=torch.bool).to(device))
            pred_l = actions_l[0][0].mean(dim=(0, 1)).cpu().numpy()[1]
            
            # RIGHT 언어
            lang_r = right_valid.view(1, 1, 1, -1).expand_as(action_hs)
            hs_r = action_hs + lang_r * scale
            actions_r = model.model.act_head(hs_r.squeeze(2), actions=None,
                           action_masks=torch.ones(bs, seq_len, dtype=torch.bool).to(device))
            pred_r = actions_r[0][0].mean(dim=(0, 1)).cpu().numpy()[1]
            
            print(f'Scale {scale:.2f}: LEFT={pred_l:.4f}, RIGHT={pred_r:.4f}, diff={abs(pred_l-pred_r):.4f}')

if __name__ == '__main__':
    main()
