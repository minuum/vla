#!/usr/bin/env python3
"""
Mobile VLA 추론 스크립트 (Left/Right 방향 수정 버전)

핵심 수정:
- 모델이 linear_y의 크기만 예측 (방향은 학습되지 않음)
- 언어 명령에서 "left"/"right"를 감지하여 부호 결정

작성일: 2025-12-09
Author: Claude (Antigravity)
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys
from typing import Tuple, Optional

# RoboVLMs 경로 추가
sys.path.insert(0, 'RoboVLMs_upstream')
from robovlms.train.mobile_vla_trainer import MobileVLATrainer


class MobileVLAInference:
    """Mobile VLA 추론 클래스 (방향 수정 버전)"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f'Loading model from {checkpoint_path}...')
        self.model = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path, map_location='cpu'
        )
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        print(f'✅ Model loaded (device: {self.device})')
    
    def get_direction_from_text(self, text: str) -> float:
        """언어 명령에서 방향 추출
        
        Args:
            text: 언어 명령 (예: "Navigate to the left")
            
        Returns:
            1.0 (left), -1.0 (right), 0.0 (unknown)
        """
        text = text.lower()
        if 'left' in text:
            return 1.0  # 왼쪽 → linear_y 양수
        elif 'right' in text:
            return -1.0  # 오른쪽 → linear_y 음수
        return 0.0
    
    def preprocess_images(self, images: list) -> torch.Tensor:
        """이미지 전처리
        
        Args:
            images: PIL Image 리스트 (8개)
            
        Returns:
            (1, 8, 3, 224, 224) 텐서
        """
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            processed.append(self.transform(img))
        return torch.stack(processed).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def predict(
        self, 
        images: list, 
        language_instruction: str,
        apply_direction_fix: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Action 예측
        
        Args:
            images: PIL Image 리스트 (8개)
            language_instruction: 언어 명령
            apply_direction_fix: 방향 수정 적용 여부
            
        Returns:
            (actions, direction)
            - actions: (2,) numpy array [linear_x, linear_y]
            - direction: 방향 (-1, 0, 1)
        """
        # 이미지 전처리
        vision_x = self.preprocess_images(images)
        
        # 방향 추출
        direction = self.get_direction_from_text(language_instruction)
        
        # 이미지 인코딩
        image_features = self.model.model.encode_images(vision_x)
        bs, seq_len, n_tok, hidden = image_features.shape
        
        # 마지막 토큰만 사용 (latent=1)
        action_hs = image_features[:, :, -1:, :].squeeze(2)  # (1, 8, 2048)
        
        # Action Head
        action_mask = torch.ones(bs, seq_len, dtype=torch.bool).to(self.device)
        actions = self.model.model.act_head(
            action_hs, actions=None, action_masks=action_mask
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        
        # 평균 prediction
        pred = actions[0].mean(dim=(0, 1)).cpu().numpy()  # (2,)
        
        # 방향 수정 적용
        if apply_direction_fix and direction != 0:
            pred[1] = abs(pred[1]) * direction
        
        return pred, direction


def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5_file', type=str, required=True)
    args = parser.parse_args()
    
    # 모델 로드
    model = MobileVLAInference(args.checkpoint)
    
    # 데이터 로드
    with h5py.File(args.h5_file, 'r') as f:
        images = [Image.fromarray(f['images'][t].astype(np.uint8)) for t in range(8)]
        
        lang = f['language_instruction'][()]
        if isinstance(lang, bytes):
            lang = lang.decode('utf-8')
        elif hasattr(lang, '__iter__') and len(lang) > 0:
            lang = lang[0].decode('utf-8') if isinstance(lang[0], bytes) else str(lang[0])
        
        gt = f['actions'][:8, :2]
    
    # 예측
    pred, direction = model.predict(images, lang)
    
    print(f'Language: {lang}')
    print(f'Direction: {direction}')
    print(f'Prediction: linear_x={pred[0]:.4f}, linear_y={pred[1]:.4f}')
    print(f'GT mean: linear_x={gt[:, 0].mean():.4f}, linear_y={gt[:, 1].mean():.4f}')


if __name__ == '__main__':
    main()
