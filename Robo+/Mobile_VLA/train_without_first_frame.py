"""
🚀 첫 프레임 제외 학습 스크립트
시작 프레임이 0으로 고정이라는 점을 고려하여 첫 프레임을 제외하고 학습
실제 의미있는 프레임들만 사용하여 모델 성능 개선
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json

from fixed_robovlms_model import FixedRoboVLMStyleSingleImageModel

class NoFirstFrameDataset(Dataset):
    """첫 프레임을 제외한 데이터셋 (시작 프레임 고정 고려)"""
    
    def __init__(self, data_path, processor, split='train', frame_selection='random'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.frame_selection = frame_selection  # 'random', 'middle', 'all'
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        print(f"   - 프레임 선택 방식: {frame_selection}")
        print(f"   - 첫 프레임 제외: True")
    
    def _load_episodes(self):
        """에피소드 로드 (첫 프레임 제외)"""
        if os.path.isdir(self.data_path):
            h5_files = list(Path(self.data_path).glob("*.h5"))
        else:
            h5_files = [self.data_path]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]  # [18, H, W, 3]
                        actions = f['actions'][:]  # [18, 3]
                        
                        # 첫 프레임 제외 (프레임 1-17만 사용)
                        valid_frames = list(range(1, 18))  # 1, 2, 3, ..., 17
                        
                        if self.frame_selection == 'random':
                            # 랜덤하게 프레임 선택
                            frame_idx = np.random.choice(valid_frames)
                        elif self.frame_selection == 'middle':
                            # 중간 프레임 선택
                            frame_idx = valid_frames[len(valid_frames)//2]  # 9
                        elif self.frame_selection == 'all':
                            # 모든 유효한 프레임을 개별 에피소드로 생성
                            for frame_idx in valid_frames:
                                single_image = images[frame_idx]  # [H, W, 3]
                                single_action = actions[frame_idx]  # [3]
                                
                                self.episodes.append({
                                    'image': single_image,
                                    'action': single_action,
                                    'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                                    'frame_idx': frame_idx,
                                    'original_file': h5_file.name
                                })
                            continue  # 다음 파일로
                        
                        # 단일 프레임 선택
                        single_image = images[frame_idx]  # [H, W, 3]
                        single_action = actions[frame_idx]  # [3]
                        
                        self.episodes.append({
                            'image': single_image,
                            'action': single_action,
                            'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                            'frame_idx': frame_idx,
                            'original_file': h5_file.name
                        })
                        
            except Exception as e:
                print(f"❌ {h5_file} 로드 실패: {e}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지: [H, W, 3] → [3, H, W] (PyTorch 형식)
        image = episode['image']  # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        # 0-1 범위로 정규화
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # 액션: [3]
        action = episode['action']  # [3]
        
        return {
            'image': image,  # [3, H, W]
            'action': action,  # [3]
            'episode_id': episode['episode_id'],
            'frame_idx': episode['frame_idx']
        }

def create_no_first_frame_loaders(data_path, processor, batch_size=4, train_split=0.8, frame_selection='random'):
    """첫 프레임 제외 데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = NoFirstFrameDataset(data_path, processor, 'full', frame_selection)
    
    # 훈련/검증 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"📊 첫 프레임 제외 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 프레임 선택: {frame_selection}")
    
    return train_loader, val_loader

def train_without_first_frame(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda'
):
    """첫 프레임 제외 훈련"""
    
    model = model.to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 손실 함수
    def compute_loss(predicted_actions, target_actions):
        # Z축 가중치 적용
        z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
        weighted_target = target_actions * z_weight.unsqueeze(0)
        weighted_pred = predicted_actions * z_weight.unsqueeze(0)
        
        return nn.functional.mse_loss(weighted_pred, weighted_target)
    
    # 조기 종료
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 첫 프레임 제외 훈련 시작!")
    print(f"📊 설정: {num_epochs} 에포크, 학습률: {learning_rate}")
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 실제 액션이 있는 프레임들만 사용
            images = batch['image']  # [batch, 3, H, W]
            actions = batch['action']  # [batch, 3] - 실제 액션 (0이 아님)
            
            images = images.float().to(device)
            actions = actions.float().to(device)
            
            optimizer.zero_grad()
            
            try:
                # 예측 (실제 액션 예측)
                predicted_actions = model(images, "Navigate to target")
                
                # 손실 계산
                loss = compute_loss(predicted_actions, actions)
                
                # 역전파
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].float().to(device)
                    actions = batch['action'].float().to(device)
                    
                    predicted_actions = model(images, "Navigate to target")
                    loss = compute_loss(predicted_actions, actions)
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"❌ 검증 배치 처리 중 오류: {e}")
                    continue
        
        # 평균 손실
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        # 학습률 조정
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} 완료:")
        print(f"   - 훈련 손실: {avg_train_loss:.4f}")
        print(f"   - 검증 손실: {avg_val_loss:.4f}")
        print(f"   - 학습률: {scheduler.get_last_lr()[0]:.6f}")
        
        # 조기 종료 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 최고 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'dropout': model.dropout,
                    'z_axis_weight': model.z_axis_weight,
                    'use_claw_matrix': model.use_claw_matrix,
                    'use_hierarchical': model.use_hierarchical,
                    'use_advanced_attention': model.use_advanced_attention,
                    'training_type': 'without_first_frame'
                }
            }, 'no_first_frame_model_best.pth')
            print(f"   ✅ 최고 모델 저장! (검증 손실: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"   ⏰ 조기 종료 (Patience: {early_stopping_patience})")
                break
    
    return model

def main():
    """메인 훈련 함수"""
    
    # 설정
    config = {
        'data_path': '../../ROS_action/mobile_vla_dataset',
        'batch_size': 4,
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'dropout': 0.2,
        'z_axis_weight': 0.05,
        'use_claw_matrix': True,
        'use_hierarchical': True,
        'use_advanced_attention': True,
        'early_stopping_patience': 5,
        'frame_selection': 'all',  # 'all', 'random', 'middle'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🚀 첫 프레임 제외 학습 시작!")
    print(f"📊 설정: {json.dumps(config, indent=2)}")
    
    # 프로세서 로드
    print("🔧 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 로더 생성
    print("📊 데이터 로더 생성 중...")
    train_loader, val_loader = create_no_first_frame_loaders(
        config['data_path'],
        processor,
        batch_size=config['batch_size'],
        frame_selection=config['frame_selection']
    )
    
    # 모델 초기화
    print("🤖 모델 초기화 중...")
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=config['dropout'],
        use_claw_matrix=config['use_claw_matrix'],
        use_hierarchical=config['use_hierarchical'],
        use_advanced_attention=config['use_advanced_attention'],
        z_axis_weight=config['z_axis_weight']
    )
    
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 활성화된 고급 기능:")
    print(f"   - Fixed Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Fixed Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Fixed Advanced Attention: {model.use_advanced_attention}")
    print(f"   - 훈련 방식: 첫 프레임 제외")
    
    # 훈련 실행
    print("🎯 훈련 시작!")
    try:
        trained_model = train_without_first_frame(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            early_stopping_patience=config['early_stopping_patience'],
            device=config['device']
        )
        
        print("✅ 첫 프레임 제외 훈련 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 결과 저장
    results = {
        'model_type': 'Fixed_RoboVLMs_Style_Without_First_Frame',
        'training_type': 'without_first_frame',
        'frame_selection': config['frame_selection'],
        'data_size': len(train_loader.dataset) + len(val_loader.dataset),
        'config': config,
        'advanced_features': {
            'fixed_claw_matrix': config['use_claw_matrix'],
            'fixed_hierarchical_planning': config['use_hierarchical'],
            'fixed_advanced_attention': config['use_advanced_attention']
        },
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_status': 'completed'
    }
    
    with open('no_first_frame_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 결과 저장 완료: no_first_frame_training_results.json")
    
    # 모델 상태 확인
    if os.path.exists('no_first_frame_model_best.pth'):
        checkpoint = torch.load('no_first_frame_model_best.pth', map_location='cpu')
        print(f"📊 최고 모델 성능:")
        print(f"   - 에포크: {checkpoint['epoch']}")
        print(f"   - 훈련 손실: {checkpoint['train_loss']:.4f}")
        print(f"   - 검증 손실: {checkpoint['val_loss']:.4f}")

if __name__ == "__main__":
    main()
