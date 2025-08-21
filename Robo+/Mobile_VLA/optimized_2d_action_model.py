"""
🚀 2D 액션 최적화 모델
실제 데이터 분석 결과를 바탕으로 Z축을 제외하고 2D 액션에 최적화된 모델
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

from transformers import AutoModel
from PIL import Image

class Optimized2DActionDataset(Dataset):
    """2D 액션에 최적화된 데이터셋 (Z축 제외)"""
    
    def __init__(self, data_path, processor, split='train', frame_selection='random'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.frame_selection = frame_selection
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 2D 액션 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        print(f"   - 프레임 선택: {frame_selection}")
        print(f"   - Z축 제외: True")
    
    def _load_episodes(self):
        """에피소드 로드 (Z축 제외, 2D 액션만)"""
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
                            frame_idx = np.random.choice(valid_frames)
                        elif self.frame_selection == 'middle':
                            frame_idx = valid_frames[len(valid_frames)//2]  # 9
                        elif self.frame_selection == 'all':
                            for frame_idx in valid_frames:
                                single_image = images[frame_idx]  # [H, W, 3]
                                single_action = actions[frame_idx]  # [3]
                                
                                # 2D 액션으로 변환 (Z축 제외)
                                action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                                
                                self.episodes.append({
                                    'image': single_image,
                                    'action': action_2d,  # 2D 액션
                                    'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                                    'frame_idx': frame_idx,
                                    'original_file': h5_file.name
                                })
                            continue
                        
                        # 단일 프레임 선택
                        single_image = images[frame_idx]  # [H, W, 3]
                        single_action = actions[frame_idx]  # [3]
                        
                        # 2D 액션으로 변환 (Z축 제외)
                        action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                        
                        self.episodes.append({
                            'image': single_image,
                            'action': action_2d,  # 2D 액션
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
        
        # 액션: 2D [linear_x, linear_y]
        action = episode['action']  # [2]
        
        return {
            'image': image,  # [3, H, W]
            'action': action,  # [2] - 2D 액션
            'episode_id': episode['episode_id'],
            'frame_idx': episode['frame_idx']
        }

class Optimized2DActionModel(nn.Module):
    """2D 액션에 최적화된 모델 (Z축 제외)"""
    
    def __init__(self, processor, vision_dim=1024, language_dim=1024, action_dim=2, hidden_dim=512, dropout=0.2, use_claw_matrix=True, use_hierarchical=True, use_advanced_attention=True):
        super().__init__()
        
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim  # 2D 액션
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 고급 기능 플래그
        self.use_claw_matrix = use_claw_matrix
        self.use_hierarchical = use_hierarchical
        self.use_advanced_attention = use_advanced_attention
        
        # Kosmos2 백본
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # Kosmos2 모델을 평가 모드로 설정
        self.kosmos.eval()
        
        # 차원 어댑터들 (동적 생성)
        self.feature_adapter = nn.Linear(1024, vision_dim)
        self.language_adapter = None  # 동적 생성
        
        # 레이어 정규화
        self.layer_norm_vision = nn.LayerNorm(vision_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 드롭아웃
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # 고급 기능들 (2D 액션에 맞게 조정)
        if use_claw_matrix:
            self.claw_matrix = OptimizedClawMatrixFusion(vision_dim, language_dim, action_dim, hidden_dim, dropout)
        if use_hierarchical:
            self.hierarchical_planner = OptimizedHierarchicalPlanner(hidden_dim, action_dim, dropout)
        if use_advanced_attention:
            self.advanced_attention = OptimizedAdvancedAttention(hidden_dim, dropout)
        
        # 2D 액션 헤드 (Z축 제외)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)  # 2D 액션 출력
        )
        
        print(f"🤖 2D 액션 최적화 모델 초기화 완료:")
        print(f"   - 액션 차원: {action_dim}D (Z축 제외)")
        print(f"   - Claw Matrix: {use_claw_matrix}")
        print(f"   - Hierarchical Planning: {use_hierarchical}")
        print(f"   - Advanced Attention: {use_advanced_attention}")
    
    def to(self, device):
        """모델을 디바이스로 이동하고 Kosmos2 모델도 함께 이동"""
        super().to(device)
        self.kosmos = self.kosmos.to(device)
        return self
    
    def extract_vision_features(self, single_image: torch.Tensor) -> torch.Tensor:
        """시각 특징 추출 (2D 최적화)"""
        batch_size = single_image.shape[0]
        
        # Kosmos2 프로세서를 위한 형식 변환
        if single_image.max() > 1:
            single_image = single_image / 255.0
        
        # 이미지를 PIL 형식으로 변환
        images = []
        for i in range(single_image.shape[0]):
            img = single_image[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            images.append(pil_img)
        
        # Kosmos2 프로세서로 입력 준비
        inputs = self.kosmos_processor(
            images=images, 
            return_tensors="pt",
            padding=True
        )
        
        # 모든 입력을 모델 디바이스로 이동
        inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Kosmos2 vision 모델 사용 (이전 성공 코드 방식)
        with torch.no_grad():
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output  # [batch_size, 1024]
            else:
                # fallback
                vision_features = torch.zeros(batch_size, 1024).to(self.kosmos.device)
        
        # 차원 조정
        vision_features = self.feature_adapter(vision_features)
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str, batch_size: int = 1) -> torch.Tensor:
        """언어 특징 추출 (2D 최적화)"""
        # 텍스트 처리 (이전 성공 코드 방식)
        with torch.no_grad():
            inputs = self.kosmos_processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Kosmos2 텍스트 모델 사용
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
        
        # 배치 차원 확장
        language_features = language_features.expand(batch_size, -1)
        
        # 차원 조정 (동적 어댑터 생성)
        if language_features.shape[-1] != self.language_dim:
            if self.language_adapter is None:
                self.language_adapter = nn.Linear(
                    language_features.shape[-1], 
                    self.language_dim
                ).to(language_features.device)
            language_features = self.language_adapter(language_features)
        
        # 강화된 정규화 및 드롭아웃
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, single_image: torch.Tensor, text: str) -> torch.Tensor:
        """2D 액션 예측 (Z축 제외)"""
        batch_size = single_image.shape[0]
        
        # 특징 추출
        vision_features = self.extract_vision_features(single_image)
        language_features = self.extract_language_features(text, batch_size)
        
        # 기본 융합
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # 고급 기능 적용
        if self.use_claw_matrix and hasattr(self, 'claw_matrix'):
            # 2D 액션용 더미 액션 생성
            dummy_actions = torch.zeros(batch_size, self.hidden_dim).to(vision_features.device)
            fused_features = self.claw_matrix(vision_features, language_features, dummy_actions)
        else:
            # Claw Matrix를 사용하지 않는 경우 기본 융합
            if fused_features.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'fusion_adapter'):
                    self.fusion_adapter = nn.Linear(fused_features.shape[-1], self.hidden_dim).to(fused_features.device)
                fused_features = self.fusion_adapter(fused_features)
        
        # 정규화 및 드롭아웃
        fused_features = self.layer_norm_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)
        
        # Advanced Attention 적용
        if self.use_advanced_attention and hasattr(self, 'advanced_attention'):
            fused_features = self.advanced_attention(fused_features)
        
        # Hierarchical Planning 적용
        if self.use_hierarchical and hasattr(self, 'hierarchical_planner'):
            fused_features = self.hierarchical_planner(fused_features)
        
        # 2D 액션 예측 (Z축 제외)
        actions_2d = self.action_head(fused_features)  # [batch_size, 2]
        
        return actions_2d

class OptimizedClawMatrixFusion(nn.Module):
    """2D 액션에 최적화된 Claw Matrix"""
    
    def __init__(self, vision_dim, language_dim, action_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 2D 액션에 맞게 조정
        self.vl_cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.la_cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.av_cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # 프로젝션 레이어들
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)  # 2D 액션용
        
        # 출력 프로젝션
        self.vision_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.language_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 피드포워드 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm4 = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_features, language_features, dummy_actions):
        """2D 액션에 최적화된 Claw Matrix 융합"""
        # 프로젝션
        v_proj = self.vision_proj(vision_features)
        l_proj = self.language_proj(language_features)
        a_proj = self.action_proj(dummy_actions)
        
        # Vision-Language 크로스 어텐션
        vl_out, _ = self.vl_cross_attention(v_proj, l_proj, l_proj)
        vl_out = self.vision_out_proj(vl_out)
        v_proj = self.norm1(v_proj + vl_out)
        
        # Language-Action 크로스 어텐션
        la_out, _ = self.la_cross_attention(l_proj, a_proj, a_proj)
        la_out = self.language_out_proj(la_out)
        l_proj = self.norm2(l_proj + la_out)
        
        # Action-Vision 크로스 어텐션
        av_out, _ = self.av_cross_attention(a_proj, v_proj, v_proj)
        av_out = self.action_out_proj(av_out)
        a_proj = self.norm3(a_proj + av_out)
        
        # 융합
        fused = v_proj + l_proj + a_proj
        
        # 피드포워드
        ffn_out = self.ffn(fused)
        fused = self.norm4(fused + ffn_out)
        
        return fused

class OptimizedHierarchicalPlanner(nn.Module):
    """2D 액션에 최적화된 계층적 계획"""
    
    def __init__(self, hidden_dim, action_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 2D 액션에 맞게 조정
        self.goal_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.subgoal_generator = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, features):
        """2D 액션 계층적 계획"""
        # 목표 분해
        goal = self.goal_decomposer(features)
        
        # 서브골 생성
        goal_features = torch.cat([features, goal], dim=-1)
        subgoals = self.subgoal_generator(goal_features)
        
        return subgoals

class OptimizedAdvancedAttention(nn.Module):
    """2D 액션에 최적화된 고급 어텐션"""
    
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 2D 액션에 맞게 조정
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm4 = nn.LayerNorm(hidden_dim)
    
    def forward(self, features):
        """2D 액션 고급 어텐션"""
        # Self Attention
        attn_out, _ = self.self_attention(features, features, features)
        features = self.norm1(features + attn_out)
        
        # Temporal Attention (시퀀스가 있는 경우)
        if features.dim() == 3:
            temp_out, _ = self.temporal_attention(features, features, features)
            features = self.norm2(features + temp_out)
        
        # Spatial Attention (공간 정보가 있는 경우)
        if features.dim() == 3:
            spatial_out, _ = self.spatial_attention(features, features, features)
            features = self.norm3(features + spatial_out)
        
        # Feedforward
        ffn_out = self.ffn(features)
        features = self.norm4(features + ffn_out)
        
        return features

def create_2d_data_loaders(data_path, processor, batch_size=4, train_split=0.8, frame_selection='random'):
    """2D 액션 데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = Optimized2DActionDataset(data_path, processor, 'full', frame_selection)
    
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
    
    print(f"📊 2D 액션 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 액션 차원: 2D (Z축 제외)")
    
    return train_loader, val_loader

def train_2d_optimized_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda'
):
    """2D 액션 최적화 모델 훈련"""
    
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
    
    # 2D 액션 손실 함수 (Z축 제외)
    def compute_2d_loss(predicted_actions, target_actions):
        # 2D 액션만 사용 (linear_x, linear_y)
        return nn.functional.mse_loss(predicted_actions, target_actions)
    
    # 조기 종료
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 2D 액션 최적화 모델 훈련 시작!")
    print(f"📊 설정: {num_epochs} 에포크, 학습률: {learning_rate}")
    print(f"🎯 액션 차원: 2D (Z축 제외)")
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = batch['image']  # [batch, 3, H, W]
            actions = batch['action']  # [batch, 2] - 2D 액션
            
            images = images.float().to(device)
            actions = actions.float().to(device)
            
            optimizer.zero_grad()
            
            try:
                # 2D 액션 예측
                predicted_actions = model(images, "Navigate to target")
                
                # 2D 손실 계산
                loss = compute_2d_loss(predicted_actions, actions)
                
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
                    loss = compute_2d_loss(predicted_actions, actions)
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
                    'action_dim': model.action_dim,
                    'use_claw_matrix': model.use_claw_matrix,
                    'use_hierarchical': model.use_hierarchical,
                    'use_advanced_attention': model.use_advanced_attention,
                    'training_type': '2d_optimized'
                }
            }, 'optimized_2d_action_model_best.pth')
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
        'action_dim': 2,  # 2D 액션
        'use_claw_matrix': True,
        'use_hierarchical': True,
        'use_advanced_attention': True,
        'early_stopping_patience': 5,
        'frame_selection': 'all',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🚀 2D 액션 최적화 모델 훈련 시작!")
    print(f"📊 설정: {json.dumps(config, indent=2)}")
    
    # 프로세서 로드
    print("🔧 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 데이터 로더 생성
    print("📊 2D 액션 데이터 로더 생성 중...")
    train_loader, val_loader = create_2d_data_loaders(
        config['data_path'],
        processor,
        batch_size=config['batch_size'],
        frame_selection=config['frame_selection']
    )
    
    # 모델 초기화
    print("🤖 2D 액션 최적화 모델 초기화 중...")
    model = Optimized2DActionModel(
        processor=processor,
        action_dim=config['action_dim'],
        dropout=config['dropout'],
        use_claw_matrix=config['use_claw_matrix'],
        use_hierarchical=config['use_hierarchical'],
        use_advanced_attention=config['use_advanced_attention']
    )
    
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 2D 액션 최적화:")
    print(f"   - 액션 차원: {model.action_dim}D (Z축 제외)")
    print(f"   - Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Advanced Attention: {model.use_advanced_attention}")
    
    # 훈련 실행
    print("🎯 2D 액션 최적화 훈련 시작!")
    try:
        trained_model = train_2d_optimized_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            early_stopping_patience=config['early_stopping_patience'],
            device=config['device']
        )
        
        print("✅ 2D 액션 최적화 훈련 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 결과 저장
    results = {
        'model_type': 'Optimized_2D_Action_Model',
        'training_type': '2d_optimized',
        'action_dimension': 2,
        'z_axis_excluded': True,
        'data_size': len(train_loader.dataset) + len(val_loader.dataset),
        'config': config,
        'advanced_features': {
            'optimized_claw_matrix': config['use_claw_matrix'],
            'optimized_hierarchical_planning': config['use_hierarchical'],
            'optimized_advanced_attention': config['use_advanced_attention']
        },
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_status': 'completed'
    }
    
    with open('optimized_2d_action_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 결과 저장 완료: optimized_2d_action_training_results.json")
    
    # 모델 상태 확인
    if os.path.exists('optimized_2d_action_model_best.pth'):
        checkpoint = torch.load('optimized_2d_action_model_best.pth', map_location='cpu')
        print(f"📊 최고 모델 성능:")
        print(f"   - 에포크: {checkpoint['epoch']}")
        print(f"   - 훈련 손실: {checkpoint['train_loss']:.4f}")
        print(f"   - 검증 손실: {checkpoint['val_loss']:.4f}")
        print(f"   - 액션 차원: {checkpoint['config']['action_dim']}D")

if __name__ == "__main__":
    main()
