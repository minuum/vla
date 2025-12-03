#!/usr/bin/env python3
"""
Mobile VLA Inference Test Script
학습된 모델 체크포인트를 로드하여 실제 데이터에 대한 추론(Inference)을 수행하고 결과를 시각화합니다.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import json
from pathlib import Path

# RoboVLMs 모듈 임포트 경로 설정
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "RoboVLMs_upstream"))

from robovlms.model.backbone.base_backbone import BaseRoboVLM
from robovlms.data.data_utils import get_text_function

def load_model_from_checkpoint(checkpoint_path, config_path):
    """체크포인트에서 모델 로드"""
    print(f"🔄 Loading model from {checkpoint_path}...")
    
    # 1. Config 로드
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    # 2. 모델 초기화에 필요한 인자 구성
    train_setup_configs = configs.get('train_setup', {})
    train_setup_configs['lora_enable'] = True
    
    # Mobile VLA에 맞는 파라미터 설정
    window_size = configs.get('window_size', 8)
    fwd_pred_next_n = configs.get('fwd_pred_next_n', 10)
    
    print("🏗️ Building Mobile VLA Model...")
    try:
        model = BaseRoboVLM(
            configs=configs,
            train_setup_configs=train_setup_configs,
            act_head_configs=configs.get('act_head', None),
            fwd_head_configs=configs.get('fwd_head', None),
            window_size=window_size,
            fwd_pred_next_n=fwd_pred_next_n,
        )
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        raise e

    # 3. 체크포인트 로드
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # LightningModule의 'model.' prefix 제거 및 키 매핑
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                # 'model.' 제거 (LightningModule 래퍼 제거)
                new_key = k[6:]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # strict=False로 로드 (LoRA 등 유연하게 처리)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"⚠️ Load results: {msg}")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        raise e
    
    model.eval()
    model.cuda()
    return model

def preprocess_image(image_array, image_size=224):
    """MobileVLAH5Dataset과 동일한 전처리 적용"""
    # (H, W, C) -> PIL Image
    img = Image.fromarray(image_array.astype(np.uint8))
    # Resize (Bilinear)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    # PIL -> numpy -> tensor (0-1 range)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    # (H, W, C) -> (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    return img_tensor

def process_input_data(h5_path, model, index=20):
    """H5 파일에서 입력 데이터 추출 및 전처리"""
    print(f"📂 Loading data from {h5_path} (Index: {index})...")
    
    with h5py.File(h5_path, 'r') as f:
        if 'images' not in f:
             # try 'observations/images' (calvin format)
             if 'observations' in f and 'images' in f['observations']:
                 images = f['observations']['images'][:]
             else:
                 raise ValueError(f"Cannot find images dataset in {h5_path}. Keys: {list(f.keys())}")
        else:
            images = f['images'][:] # (Total_Frames, H, W, C)
            
        print(f"DEBUG: Total images: {len(images)}")
        
        # 임의의 네비게이션 명령어
        text_str = "Navigate to the target location" 
        
        # Window Size (8) 만큼 가져오기
        window_size = model.window_size
        start_idx = max(0, index - window_size + 1)
        end_idx = index + 1
        
        # Ensure valid range
        if start_idx >= len(images):
            start_idx = max(0, len(images) - window_size)
            end_idx = len(images)
            print(f"WARNING: Index {index} out of bounds. Adjusted to last window.")
            
        img_seq_raw = images[start_idx:end_idx]
        print(f"DEBUG: img_seq_raw length: {len(img_seq_raw)}")
        
        # 패딩 처리 (앞부분이 부족할 경우 첫 프레임 복사)
        if len(img_seq_raw) > 0 and len(img_seq_raw) < window_size:
            pad_len = window_size - len(img_seq_raw)
            padding = np.tile(img_seq_raw[0:1], (pad_len, 1, 1, 1))
            img_seq_raw = np.concatenate([padding, img_seq_raw], axis=0)
            
    # 이미지 전처리
    if len(img_seq_raw) == 0:
         raise ValueError("img_seq_raw is empty after processing!")
         
    img_tensors = []
    for img in img_seq_raw:
        img_tensors.append(preprocess_image(img))
    
    # (Window, C, H, W) -> (1, Window, C, H, W) [Batch Dim 추가]
    # img_tensors는 list of tensors (C, H, W)
    vision_x = torch.stack(img_tensors).unsqueeze(0).cuda()
    
    # 텍스트 전처리 (Tokenizer 사용)
    tokenizer = model.tokenizer
    # 간단한 토크나이징 (RoboVLMs 방식)
    tokens = tokenizer(
        text_str, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    
    lang_x = tokens["input_ids"].cuda()
    attention_mask = tokens["attention_mask"].cuda()
    
    return vision_x, lang_x, attention_mask, img_seq_raw

def run_inference(model, vision_x, lang_x, attention_mask):
    """모델 추론 실행"""
    print("🚀 Running Inference...")
    
    with torch.no_grad():
        # MobileVLA Trainer/Model 구조에 맞게 호출
        # mode='inference'로 호출하면 logits(actions) 반환
        prediction = model.inference(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask
        )
        
    # prediction['action']은 (velocities, None) 튜플일 수 있음 (MobileVLALSTMDecoder)
    actions = prediction['action']
    if isinstance(actions, tuple) or isinstance(actions, list):
        actions = actions[0] # velocities
    
    # Actions shape: (Batch, Seq_Len, Fwd_Pred, Action_Dim)
    # 우리는 마지막 타임스텝의 예측값(Chunk)이 필요함
    # actions: (1, 8, 10, 2)
    
    # 마지막 윈도우 시점의 예측 결과 가져오기
    # (Batch=0, Last_Seq=-1, :)
    pred_chunk = actions[0, -1, :, :].cpu().numpy() # (10, 2)
    
    return pred_chunk

def visualize_results(img_seq_raw, pred_actions, save_path="inference_result.png"):
    """결과 시각화"""
    print("📊 Visualizing results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 마지막 관찰 이미지
    last_img = img_seq_raw[-1]
    # BGR to RGB if needed (H5 might be RGB, cv2 uses BGR. PIL uses RGB)
    # Assuming H5 saved as RGB via PIL in collector
    axes[0].imshow(last_img)
    axes[0].set_title("Last Observation (RGB)")
    axes[0].axis('off')
    
    # 2. 예측 궤적 (2D Velocity -> Trajectory)
    # linear_x (전진), linear_y (좌우)
    # 간단한 적분으로 경로 시각화 (dt=0.4s 가정)
    dt = 0.4
    x, y = 0, 0
    traj_x = [0]
    traj_y = [0]
    
    # 로봇 좌표계: X가 전진(Up), Y가 좌측(Left)
    for vx, vy in pred_actions:
        # 정규화된 액션(-1~1)을 실제 물리량으로 복원해야 할 수 있음
        # 여기서는 경향성 확인을 위해 그대로 사용
        dx = vx * dt
        dy = vy * dt
        
        # 전역 좌표계로 변환 (누적)
        x += dx
        y += dy
        traj_x.append(x)
        traj_y.append(y)
        
    axes[1].plot(traj_y, traj_x, 'b-o', linewidth=2, label='Predicted Path')
    axes[1].plot(0, 0, 'rs', markersize=10, label='Start (Robot)')
    
    # 그래프 데코레이션
    axes[1].set_title("Predicted 2D Trajectory (Top-Down View)")
    axes[1].set_xlabel("Lateral (Y) - Left(+)/Right(-)")
    axes[1].set_ylabel("Longitudinal (X) - Fwd(+)/Bwd(-)")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].axis('equal')
    
    # 방향 표시 (화살표)
    if len(traj_x) > 1:
        axes[1].arrow(traj_y[-2], traj_x[-2], traj_y[-1]-traj_y[-2], traj_x[-1]-traj_x[-2], 
                     head_width=0.05, head_length=0.1, fc='b', ec='b')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"💾 Saved visualization to {save_path}")

if __name__ == "__main__":
    # 설정
    ckpt_path = "RoboVLMs_upstream/runs/mobile_vla_lora_20251114/kosmos/mobile_vla_finetune/2025-11-20/mobile_vla_lora_20251114/epoch_epoch=05-val_loss=val_loss=0.280.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_20251114_lora.json"
    # 테스트할 데이터 파일 (존재하는 파일 중 하나 선택)
    data_path = "ROS_action/mobile_vla_dataset/episode_20251119_170441_1box_hori_left_core_medium.h5"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        # 대체 파일 찾기
        import glob
        files = glob.glob("ROS_action/mobile_vla_dataset/*.h5")
        if files:
            data_path = files[0]
            print(f"⚠️ Using alternative data file: {data_path}")
        else:
            print("❌ No H5 files found.")
            sys.exit(1)

    # 1. 모델 로드
    model = load_model_from_checkpoint(ckpt_path, config_path)
    
    # 2. 데이터 처리
    # Index 50: 에피소드 중간 쯤에서 테스트
    vision_x, lang_x, attention_mask, img_seq_raw = process_input_data(data_path, model, index=50)
    
    # 3. 추론 실행
    pred_actions = run_inference(model, vision_x, lang_x, attention_mask)
    
    print("\n📊 Predicted Actions (First 5 steps):")
    print("   Linear_X (Fwd) | Linear_Y (Left)")
    print("-" * 35)
    for i, (vx, vy) in enumerate(pred_actions[:5]):
        print(f"t+{i}: {vx: .4f}      | {vy: .4f}")
        
    # 4. 시각화
    visualize_results(img_seq_raw, pred_actions)
