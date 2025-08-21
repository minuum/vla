"""
🚀 Optimized Hybrid Mobile VLA Model
Final Fixed와 Advanced Mobile VLA의 장점을 결합한 하이브리드 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from typing import Optional, Tuple, Dict, Any
import numpy as np

class OptimizedHybridMobileVLAModel(nn.Module):
    """
    🎯 최적화된 하이브리드 모델
    - Final Fixed의 Z축 가중치 조정 적용
    - Advanced Mobile VLA의 고급 기능 선택적 사용
    - 과적합 방지를 위한 강화된 정규화
    """
    
    def __init__(
        self,
        processor,
        vision_dim: int = 1024,
        language_dim: int = 1024,
        action_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,  # 강화된 드롭아웃
        use_claw_matrix: bool = True,
        use_hierarchical: bool = False,  # 선택적 사용
        use_advanced_attention: bool = True,
        z_axis_weight: float = 0.05,  # Final Fixed의 Z축 가중치
        feature_fusion_type: str = "adaptive"  # adaptive, weighted, attention
    ):
        super().__init__()
        
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.z_axis_weight = z_axis_weight
        self.feature_fusion_type = feature_fusion_type
        
        # 기능 사용 플래그
        self.use_claw_matrix = use_claw_matrix
        self.use_hierarchical = use_hierarchical
        self.use_advanced_attention = use_advanced_attention
        
        # Kosmos2 모델 초기화
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # 파라미터 requires_grad 설정
        for param in self.kosmos.parameters():
            param.requires_grad = True
        
        # 동적 feature adapter
        self.feature_adapter = nn.Linear(1024, vision_dim)
        
        # 강화된 정규화 레이어
        self.layer_norm_vision = nn.LayerNorm(vision_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 드롭아웃 레이어들
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # 선택적 Claw Matrix
        if use_claw_matrix:
            self.claw_matrix = OptimizedClawMatrix(
                vision_dim, language_dim, action_dim, hidden_dim, dropout
            )
        
        # 선택적 Hierarchical Planning
        if use_hierarchical:
            self.hierarchical_planner = OptimizedHierarchicalPlanner(
                hidden_dim, action_dim, dropout
            )
        
        # 선택적 Advanced Attention
        if use_advanced_attention:
            self.advanced_attention = OptimizedAdvancedAttention(
                hidden_dim, dropout
            )
        
        # 적응형 특징 융합
        if feature_fusion_type == "adaptive":
            self.adaptive_fusion = AdaptiveFeatureFusion(
                vision_dim, language_dim, hidden_dim, dropout
            )
        elif feature_fusion_type == "weighted":
            self.weighted_fusion = WeightedFeatureFusion(
                vision_dim, language_dim, hidden_dim
            )
        
        # 최종 액션 헤드 (Final Fixed 스타일)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Z축 가중치 적용을 위한 가중치 레이어
        self.z_axis_weight_layer = nn.Parameter(torch.tensor([1.0, 1.0, z_axis_weight]))
        
        # 앙상블 가중치 (Final Fixed와 Advanced 모델 조합)
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
        
    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """시각 특징 추출 (최적화된 버전)"""
        batch_size, num_frames, channels, height, width = images.shape
        
        # 이미지를 2D로 변환
        images_2d = images.view(-1, channels, height, width)
        
        # Kosmos2 처리
        with torch.no_grad():
            inputs = self.processor(images=images_2d, return_tensors="pt")
            inputs = {k: v.to(images.device) for k, v in inputs.items()}
            
            # pixel_values 사용
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output
            else:
                vision_outputs = self.kosmos.vision_model(inputs)
                vision_features = vision_outputs.pooler_output
        
        # 특징 차원 조정
        vision_features = self.feature_adapter(vision_features)
        
        # 배치 차원 복원
        vision_features = vision_features.view(batch_size, num_frames, -1)
        
        # 강화된 정규화 및 드롭아웃
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str) -> torch.Tensor:
        """언어 특징 추출 (최적화된 버전)"""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)
        
        # 강화된 정규화 및 드롭아웃
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(
        self,
        images: torch.Tensor,
        text: str,
        distance_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """순전파 (최적화된 버전)"""
        
        # 특징 추출
        vision_features = self.extract_vision_features(images)
        language_features = self.extract_language_features(text)
        
        # 적응형 특징 융합
        if hasattr(self, 'adaptive_fusion'):
            fused_features = self.adaptive_fusion(vision_features, language_features)
        elif hasattr(self, 'weighted_fusion'):
            fused_features = self.weighted_fusion(vision_features, language_features)
        else:
            # 기본 융합
            fused_features = torch.cat([vision_features.mean(dim=1), language_features], dim=-1)
            fused_features = self.layer_norm_fusion(fused_features)
            fused_features = self.dropout_fusion(fused_features)
        
        # 선택적 고급 기능 적용
        if self.use_claw_matrix and hasattr(self, 'claw_matrix'):
            fused_features = self.claw_matrix(fused_features)
        
        if self.use_advanced_attention and hasattr(self, 'advanced_attention'):
            fused_features = self.advanced_attention(fused_features)
        
        # 최종 액션 예측
        actions = self.action_head(fused_features)
        
        # Z축 가중치 적용 (Final Fixed 스타일)
        actions = actions * self.z_axis_weight_layer.unsqueeze(0)
        
        return actions

class OptimizedClawMatrix(nn.Module):
    """최적화된 Claw Matrix (과적합 방지)"""
    
    def __init__(self, vision_dim, language_dim, action_dim, hidden_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 경량화된 Claw Matrix
        self.claw_fusion = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 정규화
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features):
        return self.layer_norm(self.claw_fusion(features))

class OptimizedHierarchicalPlanner(nn.Module):
    """최적화된 Hierarchical Planning (선택적 사용)"""
    
    def __init__(self, hidden_dim, action_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout = dropout
        
        # 경량화된 계층적 계획
        self.goal_decomposition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, features):
        goals = self.goal_decomposition(features)
        actions = self.action_generator(goals)
        return actions

class OptimizedAdvancedAttention(nn.Module):
    """최적화된 Advanced Attention"""
    
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 경량화된 어텐션
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features):
        # 자기 어텐션 적용
        attended_features, _ = self.attention(features, features, features)
        return self.layer_norm(features + attended_features)

class AdaptiveFeatureFusion(nn.Module):
    """적응형 특징 융합"""
    
    def __init__(self, vision_dim, language_dim, hidden_dim, dropout):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 적응형 가중치
        self.adaptive_weights = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # vision, language 가중치
            nn.Softmax(dim=-1)
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, vision_features, language_features):
        # 시각 특징 평균
        vision_avg = vision_features.mean(dim=1)
        
        # 적응형 가중치 계산
        combined = torch.cat([vision_avg, language_features], dim=-1)
        weights = self.adaptive_weights(combined)
        
        # 가중 융합
        weighted_vision = vision_avg * weights[:, 0:1]
        weighted_language = language_features * weights[:, 1:2]
        
        # 최종 융합
        fused = torch.cat([weighted_vision, weighted_language], dim=-1)
        return self.fusion_layer(fused)

class WeightedFeatureFusion(nn.Module):
    """가중 특징 융합"""
    
    def __init__(self, vision_dim, language_dim, hidden_dim):
        super().__init__()
        
        self.vision_weight = nn.Parameter(torch.tensor(0.6))
        self.language_weight = nn.Parameter(torch.tensor(0.4))
        
        self.fusion_layer = nn.Linear(vision_dim + language_dim, hidden_dim)
        
    def forward(self, vision_features, language_features):
        vision_avg = vision_features.mean(dim=1)
        
        # 가중 융합
        weighted_vision = vision_avg * self.vision_weight
        weighted_language = language_features * self.language_weight
        
        fused = torch.cat([weighted_vision, weighted_language], dim=-1)
        return self.fusion_layer(fused)

class HybridEnsembleModel(nn.Module):
    """Final Fixed와 Advanced 모델의 앙상블"""
    
    def __init__(self, final_fixed_model, advanced_model, ensemble_weight=0.5):
        super().__init__()
        
        self.final_fixed_model = final_fixed_model
        self.advanced_model = advanced_model
        self.ensemble_weight = nn.Parameter(torch.tensor(ensemble_weight))
        
    def forward(self, images, text, distance_labels=None):
        # 각 모델의 예측
        final_fixed_pred = self.final_fixed_model(images, text, distance_labels)
        advanced_pred = self.advanced_model(images, text, distance_labels)
        
        # 앙상블 가중치 적용
        ensemble_pred = (
            self.ensemble_weight * final_fixed_pred + 
            (1 - self.ensemble_weight) * advanced_pred
        )
        
        return ensemble_pred

# 🎯 최적화된 훈련 함수
def train_optimized_hybrid_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,  # 낮은 학습률
    weight_decay=1e-4,   # 강화된 정규화
    early_stopping_patience=5,
    device='cuda'
):
    """최적화된 하이브리드 모델 훈련"""
    
    model = model.to(device)
    
    # 옵티마이저 (AdamW with 강화된 정규화)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 학습률 스케줄러 (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 손실 함수 (Final Fixed 스타일)
    def compute_loss(predicted_actions, target_actions):
        # Z축 가중치 적용
        z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
        weighted_target = target_actions * z_weight.unsqueeze(0).unsqueeze(0)
        weighted_pred = predicted_actions * z_weight.unsqueeze(0).unsqueeze(0)
        
        return F.mse_loss(weighted_pred, weighted_target)
    
    # 조기 종료
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images, actions, distance_labels = batch
            images = images.float().to(device)
            actions = actions.long().to(device)
            
            optimizer.zero_grad()
            
            # 예측
            predicted_actions = model(images, "Navigate to target", distance_labels)
            
            # 손실 계산
            loss = compute_loss(predicted_actions, actions.float())
            
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
                images, actions, distance_labels = batch
                images = images.float().to(device)
                actions = actions.long().to(device)
                
                predicted_actions = model(images, "Navigate to target", distance_labels)
                loss = compute_loss(predicted_actions, actions.float())
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
                    'z_axis_weight': model.z_axis_weight
                }
            }, 'optimized_hybrid_model_best.pth')
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
    
    model = OptimizedHybridMobileVLAModel(
        processor=processor,
        dropout=0.3,  # 강화된 드롭아웃
        use_claw_matrix=True,
        use_hierarchical=False,  # 선택적 사용
        use_advanced_attention=True,
        z_axis_weight=0.05,  # Final Fixed 스타일
        feature_fusion_type="adaptive"
    )
    
    print("🚀 Optimized Hybrid Mobile VLA Model 초기화 완료!")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 사용 기능:")
    print(f"   - Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Advanced Attention: {model.use_advanced_attention}")
    print(f"   - Z축 가중치: {model.z_axis_weight}")
    print(f"   - 드롭아웃: {model.dropout}")
