#!/usr/bin/env python3
"""
최근 학습된 Basket Navigation 모델의 추론 성능 검증
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
import sys
import os

# RoboVLMs 경로 추가
sys.path.insert(0, os.path.abspath("RoboVLMs_upstream"))
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def verify_model():
    print("="*70)
    print("🚀 가용한 최신 모델 추론 검증 시작")
    print("="*70)
    print()

    # 최신 체크포인트 및 설정 경로
    checkpoint_path = "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-02-01/mobile_vla_chunk5_basket_20260129/epoch_epoch=06-val_loss=val_loss=0.020.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_basket.json"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        # last.ckpt 시도
        checkpoint_path = "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-02-01/mobile_vla_chunk5_basket_20260129/last.ckpt"
        print(f"🔄 대신 last.ckpt 로드 시도: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print("❌ 로드 가능한 체크포인트가 없습니다.")
            return

    # 모델 로드
    print("🔧 모델 로딩 중 (CUDA)...")
    try:
        trainer = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path,
            config_path=config_path,
            map_location="cuda"
        )
        trainer.model.to('cuda')
        trainer.model.eval()
        print("✅ 모델 로드 완료!")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    # 프로세서 로드
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('.vlms/kosmos-2-patch14-224')
    
    # 테스트용 H5 파일 (임의 선택)
    test_dir = "ROS_action/basket_dataset"
    h5_files = [f for f in os.listdir(test_dir) if f.endswith(".h5")]
    if not h5_files:
        print("❌ 테스트할 H5 파일이 없습니다.")
        return
    
    test_file = os.path.join(test_dir, h5_files[0])
    print(f"📊 테스트 파일: {Path(test_file).name}")

    with h5py.File(test_file, 'r') as f:
        total_frames = len(f['images'])
        window_size = 8
        
        # 중간 지점 테스트
        target_frame = min(20, total_frames - 1)
        start_frame = max(0, target_frame - window_size + 1)
        end_frame = start_frame + window_size
        
        print(f"🔎 프레임 {target_frame} 분석 (Window: {start_frame}~{end_frame-1})")
        
        window_images = []
        for i in range(start_frame, end_frame):
            img_np = f['images'][i]
            window_images.append(Image.fromarray(img_np))
        
        true_action = f['actions'][target_frame][:2]
        instruction = "Navigate to the brown pot" # 기본 지시어
        
        # 데이터 처리
        all_pixel_values = []
        for img in window_images:
            inputs = processor(images=img, text=instruction, return_tensors="pt")
            all_pixel_values.append(inputs['pixel_values'])
        
        pixel_values = torch.cat(all_pixel_values, dim=0).unsqueeze(0).cuda()
        lang_x = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        
        # 추론
        with torch.no_grad():
            pred_output = trainer.model.inference(
                pixel_values, lang_x, attention_mask,
                None, None, None, None, None
            )
        
        action_output = pred_output['action']
        if isinstance(action_output, tuple): action_output = action_output[0]
        
        # 마지막 타임스텝의 첫 번째 청크 예측값 추출
        pred_action = action_output[0, -1, 0].cpu().numpy()
        
        print(f"📍 결과:")
        print(f"  Ground Truth:   {true_action}")
        print(f"  Raw Prediction: {pred_action}")
        
        # 오차 계산
        error = np.linalg.norm(pred_action - true_action)
        print(f"  L2 Error:       {error:.4f}")
        
        if error < 0.5:
            print("✅ 추론 결과가 타당한 범위 내에 있습니다.")
        else:
            print("⚠️ 오차가 다소 큽니다. 데이터 분포 확인이 필요할 수 있습니다.")

    print("\n" + "="*70)
    print("✅ 검증 완료")
    print("="*70)

if __name__ == "__main__":
    verify_model()
