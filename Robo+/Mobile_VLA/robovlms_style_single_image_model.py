"""
🚀 RoboVLMs Style Single Image VLA Model
단일 이미지 입력 → 단일 액션 출력 (RoboVLMs 스타일)
Claw Matrix, Hierarchical Planning, Advanced Attention 사용
원본 72개 데이터셋으로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from typing import Optional, Tuple, Dict, Any
import numpy as np

class RoboVLMStyleSingleImageModel(nn.Module):
    """
    🎯 RoboVLMs 스타일 모델
    - 입력: 단일 이미지 1장
    - 출력: 단일 액션 (3D)
    - 고급 기능: Claw Matrix, Hierarchical Planning, Advanced Attention
    """
    
    def __init__(
        self,
        processor,
        vision_dim: int = 1024,
        language_dim: int = 1024,
        action_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_claw_matrix: bool = True,
        use_hierarchical: bool = True,
        use_advanced_attention: bool = True,
        z_axis_weight: float = 0.05
    ):
        super().__init__()
        
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.z_axis_weight = z_axis_weight
        
        # 기능 사용 플래그
        self.use_claw_matrix = use_claw_matrix
        self.use_hierarchical = use_hierarchical
        self.use_advanced_attention = use_advanced_attention
        
        # Kosmos2 모델 (RoboVLMs 스타일)
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        for param in self.kosmos.parameters():
            param.requires_grad = True
        
        # 동적 feature adapter
        self.feature_adapter = nn.Linear(1024, vision_dim)
        
        # 강화된 정규화
        self.layer_norm_vision = nn.LayerNorm(vision_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 드롭아웃
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # Claw Matrix (RoboVLMs 고급 기능)
        if use_claw_matrix:
            self.claw_matrix = ClawMatrixFusion(
                vision_dim, language_dim, action_dim, hidden_dim, dropout
            )
        
        # Hierarchical Planning (RoboVLMs 고급 기능)
        if use_hierarchical:
            self.hierarchical_planner = HierarchicalPlanner(
                hidden_dim, action_dim, dropout
            )
        
        # Advanced Attention (RoboVLMs 고급 기능)
        if use_advanced_attention:
            self.advanced_attention = AdvancedAttention(
                hidden_dim, dropout
            )
        
        # 최종 액션 헤드 (RoboVLMs 스타일)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Z축 가중치 (Final Fixed 스타일 유지)
        self.z_axis_weight_layer = nn.Parameter(torch.tensor([1.0, 1.0, z_axis_weight]))
        
    def extract_vision_features(self, single_image: torch.Tensor) -> torch.Tensor:
        """단일 이미지 특징 추출 (RoboVLMs 스타일)"""
        # single_image: [batch_size, 3, H, W] - 단일 이미지
        
        # Kosmos2 처리
        with torch.no_grad():
            inputs = self.processor(images=single_image, return_tensors="pt")
            inputs = {k: v.to(single_image.device) for k, v in inputs.items()}
            
            # pixel_values 사용
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output
            else:
                vision_outputs = self.kosmos.vision_model(inputs)
                vision_features = vision_outputs.pooler_output
        
        # 특징 차원 조정
        vision_features = self.feature_adapter(vision_features)
        
        # 강화된 정규화 및 드롭아웃
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str, batch_size: int = 1) -> torch.Tensor:
        """언어 특징 추출 (RoboVLMs 스타일)"""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # 배치 차원 확장
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].expand(batch_size, -1)
        
        with torch.no_grad():
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)
        
        # 차원 조정 (Kosmos2의 실제 출력 차원에 맞춤)
        if language_features.shape[-1] != self.language_dim:
            if not hasattr(self, 'language_adapter'):
                self.language_adapter = nn.Linear(language_features.shape[-1], self.language_dim).to(language_features.device)
            language_features = self.language_adapter(language_features)
        
        # 강화된 정규화 및 드롭아웃
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, single_image: torch.Tensor, text: str) -> torch.Tensor:
        """순전파 (RoboVLMs 스타일)"""
        # single_image: [batch_size, 3, H, W] - 단일 이미지
        # text: 문자열
        
        batch_size = single_image.shape[0]
        
        # 특징 추출
        vision_features = self.extract_vision_features(single_image)
        language_features = self.extract_language_features(text, batch_size)
        
        # 기본 융합
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # Claw Matrix 적용 (RoboVLMs 고급 기능)
        if self.use_claw_matrix and hasattr(self, 'claw_matrix'):
            claw_output = self.claw_matrix(fused_features)
            # Claw Matrix 출력을 hidden_dim으로 조정
            if claw_output.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'claw_adapter'):
                    self.claw_adapter = nn.Linear(claw_output.shape[-1], self.hidden_dim).to(claw_output.device)
                fused_features = self.claw_adapter(claw_output)
            else:
                fused_features = claw_output
        else:
            # Claw Matrix를 사용하지 않는 경우 기본 융합 특징을 hidden_dim으로 조정
            if fused_features.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'fusion_adapter'):
                    self.fusion_adapter = nn.Linear(fused_features.shape[-1], self.hidden_dim).to(fused_features.device)
                fused_features = self.fusion_adapter(fused_features)
        
        # 정규화 및 드롭아웃
        fused_features = self.layer_norm_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)
        
        # Advanced Attention 적용 (RoboVLMs 고급 기능)
        if self.use_advanced_attention and hasattr(self, 'advanced_attention'):
            fused_features = self.advanced_attention(fused_features)
        
        # Hierarchical Planning 적용 (RoboVLMs 고급 기능)
        if self.use_hierarchical and hasattr(self, 'hierarchical_planner'):
            fused_features = self.hierarchical_planner(fused_features)
        
        # 최종 액션 예측
        actions = self.action_head(fused_features)
        
        # Z축 가중치 적용 (Final Fixed 스타일)
        actions = actions * self.z_axis_weight_layer.unsqueeze(0)
        
        return actions

class ClawMatrixFusion(nn.Module):
    """Claw Matrix Fusion (RoboVLMs 고급 기능)"""
    
    def __init__(self, vision_dim, language_dim, action_dim, hidden_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Vision-Language Cross Attention
        self.vl_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Language-Action Cross Attention
        self.la_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Action-Vision Cross Attention
        self.av_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 프로젝션 레이어들
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)  # hidden_dim → hidden_dim
        
        # 출력 프로젝션
        self.vision_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.language_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ffn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, features):
        # features: [batch_size, vision_dim + language_dim]
        batch_size = features.shape[0]
        
        # 특징 분리 (실제 차원에 맞춤)
        total_dim = features.shape[-1]
        vision_dim = total_dim // 2  # 절반을 vision으로
        language_dim = total_dim - vision_dim  # 나머지를 language로
        
        vision_features = features[:, :vision_dim]
        language_features = features[:, vision_dim:]
        
        # 더미 액션 토큰 생성 (학습 시에는 실제 액션 사용)
        dummy_actions = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        
        # 프로젝션
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [batch, 1, hidden]
        language_proj = self.language_proj(language_features).unsqueeze(1)  # [batch, 1, hidden]
        action_proj = self.action_proj(dummy_actions).unsqueeze(1)  # [batch, 1, hidden]
        
        # Vision-Language Cross Attention
        vl_attended, _ = self.vl_cross_attention(
            query=vision_proj,
            key=language_proj,
            value=language_proj
        )
        vl_out = self.vision_out_proj(vl_attended.squeeze(1))
        vl_out = self.norm1(vision_features + vl_out)
        vl_out = vl_out + self.ffn1(vl_out)
        
        # Language-Action Cross Attention
        la_attended, _ = self.la_cross_attention(
            query=language_proj,
            key=action_proj,
            value=action_proj
        )
        la_out = self.language_out_proj(la_attended.squeeze(1))
        la_out = self.norm2(language_features + la_out)
        la_out = la_out + self.ffn2(la_out)
        
        # Action-Vision Cross Attention
        av_attended, _ = self.av_cross_attention(
            query=action_proj,
            key=vision_proj,
            value=vision_proj
        )
        av_out = self.action_out_proj(av_attended.squeeze(1))
        av_out = self.norm3(dummy_actions + av_out)
        av_out = av_out + self.ffn3(av_out)
        
        # 최종 융합
        fused = torch.cat([vl_out, la_out, av_out], dim=-1)
        
        return fused

class HierarchicalPlanner(nn.Module):
    """Hierarchical Planning (RoboVLMs 고급 기능)"""
    
    def __init__(self, hidden_dim, action_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout = dropout
        
        # 목표 분해
        self.goal_decomposition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 서브골 생성
        self.subgoal_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 액션 생성
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
    def forward(self, features):
        # 목표 분해
        goals = self.goal_decomposition(features)
        
        # 서브골 생성
        subgoals = self.subgoal_generator(goals)
        
        # 액션 생성
        actions = self.action_generator(subgoals)
        
        # 특징 업데이트
        updated_features = features + goals
        
        return updated_features

class AdvancedAttention(nn.Module):
    """Advanced Attention (RoboVLMs 고급 기능)"""
    
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Multi-head Self Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, features):
        # Self Attention
        attended_features, _ = self.self_attention(
            query=features.unsqueeze(1),
            key=features.unsqueeze(1),
            value=features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Residual + Norm
        features = self.norm1(features + attended_features)
        
        # FFN
        ffn_out = self.ffn(features)
        
        # Residual + Norm
        features = self.norm2(features + ffn_out)
        
        return features

def train_robovlms_style_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda'
):
    """RoboVLMs 스타일 모델 훈련"""
    
    model = model.to(device)
    
    # 옵티마이저 (RoboVLMs 스타일)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 학습률 스케줄러 (RoboVLMs 스타일)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 손실 함수 (RoboVLMs 스타일)
    def compute_loss(predicted_actions, target_actions):
        # Z축 가중치 적용
        z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
        weighted_target = target_actions * z_weight.unsqueeze(0)
        weighted_pred = predicted_actions * z_weight.unsqueeze(0)
        
        return F.mse_loss(weighted_pred, weighted_target)
    
    # 조기 종료
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # 단일 이미지 입력
            images = batch['image']  # [batch, 3, H, W] - 이미 단일 이미지
            actions = batch['action']  # [batch, 3] - 이미 단일 액션
            
            images = images.float().to(device)
            actions = actions.float().to(device)
            
            optimizer.zero_grad()
            
            # 예측 (단일 이미지 → 단일 액션)
            predicted_actions = model(images, "Navigate to target")
            
            # 손실 계산
            loss = compute_loss(predicted_actions, actions)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # 단일 이미지 입력
                images = batch['image']  # [batch, 3, H, W]
                actions = batch['action']  # [batch, 3]
                
                images = images.float().to(device)
                actions = actions.float().to(device)
                
                predicted_actions = model(images, "Navigate to target")
                loss = compute_loss(predicted_actions, actions)
                val_loss += loss.item()
        
        # 평균 손실
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 학습률 조정
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
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
                    'use_advanced_attention': model.use_advanced_attention
                }
            }, 'robovlms_style_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

# 🚀 사용 예시
if __name__ == "__main__":
    # 모델 초기화
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model = RoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        z_axis_weight=0.05
    )
    
    print("🚀 RoboVLMs Style Single Image VLA Model 초기화 완료!")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 사용 기능:")
    print(f"   - Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Advanced Attention: {model.use_advanced_attention}")
    print(f"   - Z축 가중치: {model.z_axis_weight}")
    print(f"   - 드롭아웃: {model.dropout}")
    print(f"🎯 입력: 단일 이미지 [batch, 3, H, W]")
    print(f"🎯 출력: 단일 액션 [batch, 3]")
    print(f"🎯 용도: 실시간 로봇 제어 (RoboVLMs 스타일)")
