#!/usr/bin/env python3
"""
🚀 Advanced Mobile VLA Model Training Script
Claw Matrix + Hierarchical Planning + Advanced Attention 훈련
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import json
import os
import random
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# RoboVLMs 모듈 추가
sys.path.append(str(Path(__file__).parent / "robovlms" / "models"))
from advanced_mobile_vla_model import AdvancedMobileVLAModel, test_advanced_model

class RoboticsDataAugmentation:
    """로봇 데이터 증강 클래스"""
    def __init__(self):
        self.action_noise_std = 0.005
        self.image_noise_std = 0.01
        
    def augment_episode(self, episode, augment_prob=0.8):
        """에피소드 증강"""
        images = episode['images']
        actions = episode['actions'].clone() if hasattr(episode['actions'], 'clone') else episode['actions'].copy()
        
        # 액션 노이즈 추가
        if random.random() < augment_prob:
            noise = np.random.normal(0, self.action_noise_std, actions.shape)
            actions = actions + noise
            actions = np.clip(actions, -1.15, 1.15)
        
        return {
            'images': images,
            'actions': actions,
            'episode_id': episode['episode_id']
        }

class MobileVLADataset(Dataset):
    """Mobile VLA 데이터셋 로더 - 여러 데이터셋 통합"""
    def __init__(self, data_paths, max_episodes=None):
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.episodes = []
        
        print(f"📁 데이터셋 로딩: {len(self.data_paths)}개 경로")
        
        for data_path in self.data_paths:
            print(f"   📂 {data_path} 처리 중...")
            
            # H5 파일들 찾기
            h5_files = list(Path(data_path).glob("*.h5"))
            
            # 폴더 구조의 에피소드들 찾기
            episode_dirs = [d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('episode_')]
            
            print(f"      발견된 H5 파일: {len(h5_files)}개")
            print(f"      발견된 에피소드 폴더: {len(episode_dirs)}개")
            
            # H5 파일 처리
            for h5_file in tqdm(h5_files, desc=f"로딩 H5 {Path(data_path).name}"):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        # 데이터 구조 확인 - images와 actions 사용
                        if 'images' in f and 'actions' in f:
                            images = f['images'][:]
                            actions = f['actions'][:]
                            
                            # 데이터 검증
                            if images.shape[0] > 0 and actions.shape[0] > 0:
                                self.episodes.append({
                                    'images': images,
                                    'actions': actions,
                                    'episode_id': len(self.episodes)
                                })
                        else:
                            print(f"⚠️ {h5_file.name}: images 또는 actions 키가 없습니다")
                            
                except Exception as e:
                    print(f"❌ {h5_file.name} 로딩 실패: {e}")
                    continue
            
            # 폴더 구조 처리
            for episode_dir in episode_dirs:
                try:
                    # 이미지 파일들 찾기
                    image_files = sorted([f for f in episode_dir.glob("*.jpg")])
                    if not image_files:
                        continue
                    
                    # 액션 파일 로드
                    actions_file = episode_dir / "actions.npy"
                    if not actions_file.exists():
                        continue
                    
                    actions = np.load(actions_file)
                    
                    # 이미지들을 numpy 배열로 로드
                    images = []
                    for img_file in image_files:
                        try:
                            from PIL import Image
                            img = Image.open(img_file).convert('RGB')
                            img_array = np.array(img)
                            images.append(img_array)
                        except Exception as e:
                            print(f"⚠️ 이미지 로드 오류 {img_file}: {e}")
                            continue
                    
                    if len(images) == 0:
                        continue
                    
                    images = np.array(images)  # [num_frames, height, width, 3]
                    
                    # 데이터 검증
                    if images.shape[0] > 0 and actions.shape[0] > 0:
                        self.episodes.append({
                            'images': images,
                            'actions': actions,
                            'episode_id': episode_dir.name
                        })
                        
                except Exception as e:
                    print(f"⚠️ 에피소드 폴더 처리 오류 {episode_dir}: {e}")
                    continue
        
        print(f"✅ 로딩 완료: {len(self.episodes)}개 유효한 에피소드")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return {
            'images': torch.tensor(episode['images'], dtype=torch.float32),
            'actions': torch.tensor(episode['actions'], dtype=torch.float32),
            'episode_id': idx  # episode_id 대신 idx 사용
        }

def collate_fn(batch):
    """배치 데이터 정리"""
    images = [item['images'] for item in batch]
    actions = [item['actions'] for item in batch]
    episode_ids = [item['episode_id'] for item in batch]
    
    # 패딩으로 배치 크기 맞추기
    max_images = max(f.shape[0] for f in images)
    max_actions = max(a.shape[0] for a in actions)
    
    padded_images = []
    padded_actions = []
    
    for image, action in zip(images, actions):
        # 이미지 패딩
        if image.shape[0] < max_images:
            pad_size = max_images - image.shape[0]
            padded_image = torch.cat([image, torch.zeros(pad_size, *image.shape[1:])], dim=0)
        else:
            padded_image = image[:max_images]
        padded_images.append(padded_image)
        
        # 액션 패딩
        if action.shape[0] < max_actions:
            pad_size = max_actions - action.shape[0]
            padded_action = torch.cat([action, torch.zeros(pad_size, *action.shape[1:])], dim=0)
        else:
            padded_action = action[:max_actions]
        padded_actions.append(padded_action)
    
    return {
        'images': torch.stack(padded_images),
        'actions': torch.stack(padded_actions),
        'episode_ids': episode_ids
    }

def train_advanced_mobile_vla():
    """고급 Mobile VLA 모델 훈련"""
    print("🚀 Advanced Mobile VLA Model 훈련 시작")
    
    # 설정
    config = {
        'data_paths': [
            '../../ROS_action/mobile_vla_dataset',  # 원본 데이터
            'augmented_dataset',  # 증강된 데이터 (721개)
            'distance_aware_augmented_dataset'  # 거리 인식 증강 데이터 (481개)
        ],
        'batch_size': 1,  # 이전 코드와 동일
        'learning_rate': 1e-4,
        'num_epochs': 20,  # kosmos2_optimized와 동일
        'save_interval': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_episodes': None,  # 제한 없음 - 전체 데이터셋 사용
        'vision_dim': 768,
        'language_dim': 768,
        'action_dim': 3,
        'fusion_dim': 512,
        'plan_dim': 256,
        'num_claw_layers': 3,
        'num_subgoals': 6,
        'frames_per_subgoal': 3,
        'use_claw_matrix': True,
        'use_hierarchical': True,
        'use_advanced_attention': True
    }
    
    print(f"�� 설정: {config}")
    print(f"💻 디바이스: {config['device']}")
    
    # 데이터셋 로드
    print("📊 데이터셋 로딩 중...")
    dataset = MobileVLADataset(config['data_paths'], config['max_episodes'])
    
    if len(dataset) == 0:
        print("❌ 유효한 데이터가 없습니다!")
        return
    
    print(f"📈 총 {len(dataset)}개 에피소드 로드 완료")
    print(f"   - 원본: 72개")
    print(f"   - 증강: 721개") 
    print(f"   - 거리인식 증강: 481개")
    print(f"   - 총합: {len(dataset)}개")
    
    # 데이터 로더 생성
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # 검증 데이터셋 (전체의 10%)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # 모델 초기화
    print("🤖 Advanced Mobile VLA 모델 초기화 중...")
    
    # Kosmos2 processor 로드
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model = AdvancedMobileVLAModel(
        processor=processor,
        vision_dim=config['vision_dim'],
        language_dim=config['language_dim'],
        action_dim=config['action_dim'],
        fusion_dim=config['fusion_dim'],
        plan_dim=config['plan_dim'],
        num_claw_layers=config['num_claw_layers'],
        num_subgoals=config['num_subgoals'],
        frames_per_subgoal=config['frames_per_subgoal'],
        use_claw_matrix=config['use_claw_matrix'],
        use_hierarchical=config['use_hierarchical'],
        use_advanced_attention=config['use_advanced_attention']
    ).to(config['device']).float()  # float32로 명시적 설정
    
    # 손실 함수와 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("🎯 훈련 시작!")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['images'].to(config['device']).float()
                actions = batch['actions'].to(config['device']).float()
                
                # 모델 순전파
                optimizer.zero_grad()
                
                # 거리 라벨 생성 (Long 타입으로 설정)
                batch_size = images.shape[0]
                distance_labels = torch.randint(0, 3, (batch_size,), device=config['device']).long()
                
                # 모델 호출 (distance_labels 포함)
                predicted_actions = model(
                    images=images,
                    distance_labels=distance_labels
                )
                
                # 손실 계산 (액션 형태 맞추기)
                target_actions = actions[:, :predicted_actions.shape[1], :predicted_actions.shape[2]]
                loss = criterion(predicted_actions, target_actions)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 진행률 업데이트
                progress_bar.set_postfix({'Loss': f'{loss.item():.3f}'})
                
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 처리 중 오류: {str(e)}")
                continue
        
        # 에포크 완료
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['images'].to(config['device']).float()
                    actions = batch['actions'].to(config['device']).float()
                    
                    # 거리 라벨 생성 (Long 타입으로 설정)
                    batch_size = images.shape[0]
                    distance_labels = torch.randint(0, 3, (batch_size,), device=config['device']).long()
                    
                    # 모델 호출
                    predicted_actions = model(
                        images=images,
                        distance_labels=distance_labels
                    )
                    
                    # 손실 계산
                    target_actions = actions[:, :predicted_actions.shape[1], :predicted_actions.shape[2]]
                    loss = criterion(predicted_actions, target_actions)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"⚠️ 검증 배치 처리 중 오류: {str(e)}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"📊 Epoch {epoch+1} 완료 - 훈련 손실: {avg_train_loss:.4f}, 검증 손실: {avg_val_loss:.4f}")
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"advanced_mobile_vla_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            print(f"🏆 새로운 최고 모델 저장: {best_model_path}")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 모델 저장
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = f"advanced_mobile_vla_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, save_path)
            print(f"💾 모델 저장: {save_path}")
    
    # 최종 모델 저장
    final_save_path = "advanced_mobile_vla_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses
    }, final_save_path)
    
    # 훈련 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Advanced Mobile VLA Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('advanced_mobile_vla_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 저장
    results = {
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('advanced_mobile_vla_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ 훈련 완료!")
    print(f"📈 최종 훈련 손실: {results['final_train_loss']:.4f}")
    print(f"🎯 최종 검증 손실: {results['final_val_loss']:.4f}")
    print(f"🏆 최고 검증 손실: {results['best_val_loss']:.4f}")
    print(f"💾 모델 저장: {final_save_path}")
    print(f"🏆 최고 모델 저장: advanced_mobile_vla_best.pth")
    print(f"📊 결과 저장: advanced_mobile_vla_training_results.json")

if __name__ == "__main__":
    train_advanced_mobile_vla()
