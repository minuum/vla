"""
🎯 Hybrid Optimization Strategy for Mobile VLA
Final Fixed와 Advanced Mobile VLA의 장점을 결합한 최적화 전략
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import numpy as np
from typing import Optional, Dict, Any

class OptimizedHybridMobileVLAModel(nn.Module):
    """최적화된 하이브리드 모델"""
    
    def __init__(
        self,
        processor,
        vision_dim: int = 1024,
        language_dim: int = 1024,
        action_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        z_axis_weight: float = 0.05,  # Final Fixed 스타일
        use_advanced_features: bool = True
    ):
        super().__init__()
        
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.z_axis_weight = z_axis_weight
        self.use_advanced_features = use_advanced_features
        
        # Kosmos2 모델
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
        
        # 선택적 고급 기능
        if use_advanced_features:
            self.advanced_fusion = AdvancedFeatureFusion(
                vision_dim, language_dim, hidden_dim, dropout
            )
        
        # 최종 액션 헤드 (Final Fixed 스타일)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Z축 가중치
        self.z_axis_weight_layer = nn.Parameter(torch.tensor([1.0, 1.0, z_axis_weight]))
        
    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """시각 특징 추출"""
        batch_size, num_frames, channels, height, width = images.shape
        images_2d = images.view(-1, channels, height, width)
        
        with torch.no_grad():
            inputs = self.processor(images=images_2d, return_tensors="pt")
            inputs = {k: v.to(images.device) for k, v in inputs.items()}
            
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output
            else:
                vision_outputs = self.kosmos.vision_model(inputs)
                vision_features = vision_outputs.pooler_output
        
        vision_features = self.feature_adapter(vision_features)
        vision_features = vision_features.view(batch_size, num_frames, -1)
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str) -> torch.Tensor:
        """언어 특징 추출"""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)
        
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, images: torch.Tensor, text: str, distance_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파"""
        vision_features = self.extract_vision_features(images)
        language_features = self.extract_language_features(text)
        
        if self.use_advanced_features and hasattr(self, 'advanced_fusion'):
            fused_features = self.advanced_fusion(vision_features, language_features)
        else:
            # 기본 융합 (Final Fixed 스타일)
            vision_avg = vision_features.mean(dim=1)
            fused_features = torch.cat([vision_avg, language_features], dim=-1)
            fused_features = self.layer_norm_fusion(fused_features)
            fused_features = self.dropout_fusion(fused_features)
        
        actions = self.action_head(fused_features)
        actions = actions * self.z_axis_weight_layer.unsqueeze(0)
        
        return actions

class AdvancedFeatureFusion(nn.Module):
    """고급 특징 융합 (경량화)"""
    
    def __init__(self, vision_dim, language_dim, hidden_dim, dropout):
        super().__init__()
        
        self.adaptive_weights = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, vision_features, language_features):
        vision_avg = vision_features.mean(dim=1)
        combined = torch.cat([vision_avg, language_features], dim=-1)
        weights = self.adaptive_weights(combined)
        
        weighted_vision = vision_avg * weights[:, 0:1]
        weighted_language = language_features * weights[:, 1:2]
        
        fused = torch.cat([weighted_vision, weighted_language], dim=-1)
        return self.fusion_layer(fused)

class HybridEnsembleModel(nn.Module):
    """앙상블 모델"""
    
    def __init__(self, final_fixed_model, advanced_model, ensemble_weight=0.5):
        super().__init__()
        self.final_fixed_model = final_fixed_model
        self.advanced_model = advanced_model
        self.ensemble_weight = nn.Parameter(torch.tensor(ensemble_weight))
        
    def forward(self, images, text, distance_labels=None):
        final_fixed_pred = self.final_fixed_model(images, text, distance_labels)
        advanced_pred = self.advanced_model(images, text, distance_labels)
        
        ensemble_pred = (
            self.ensemble_weight * final_fixed_pred + 
            (1 - self.ensemble_weight) * advanced_pred
        )
        
        return ensemble_pred

def train_optimized_hybrid_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda'
):
    """최적화된 훈련"""
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    def compute_loss(predicted_actions, target_actions):
        z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
        weighted_target = target_actions * z_weight.unsqueeze(0).unsqueeze(0)
        weighted_pred = predicted_actions * z_weight.unsqueeze(0).unsqueeze(0)
        return F.mse_loss(weighted_pred, weighted_target)
    
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
            predicted_actions = model(images, "Navigate to target", distance_labels)
            loss = compute_loss(predicted_actions, actions.float())
            loss.backward()
            
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
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

# 🎯 최적화 전략들
class OptimizationStrategies:
    """다양한 최적화 전략"""
    
    @staticmethod
    def strategy_1_final_fixed_style():
        """전략 1: Final Fixed 스타일 최적화"""
        return {
            'dropout': 0.2,
            'z_axis_weight': 0.05,
            'use_advanced_features': False,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'num_epochs': 6
        }
    
    @staticmethod
    def strategy_2_balanced_hybrid():
        """전략 2: 균형잡힌 하이브리드"""
        return {
            'dropout': 0.3,
            'z_axis_weight': 0.05,
            'use_advanced_features': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 15
        }
    
    @staticmethod
    def strategy_3_advanced_optimized():
        """전략 3: 고급 기능 최적화"""
        return {
            'dropout': 0.4,
            'z_axis_weight': 0.05,
            'use_advanced_features': True,
            'learning_rate': 5e-5,
            'weight_decay': 1e-3,
            'num_epochs': 20
        }
    
    @staticmethod
    def strategy_4_ensemble_approach():
        """전략 4: 앙상블 접근"""
        return {
            'ensemble_weight': 0.6,  # Final Fixed에 더 높은 가중치
            'dropout': 0.3,
            'z_axis_weight': 0.05,
            'use_advanced_features': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 15
        }

# 🚀 사용 예시
if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 전략 2: 균형잡힌 하이브리드
    config = OptimizationStrategies.strategy_2_balanced_hybrid()
    
    model = OptimizedHybridMobileVLAModel(
        processor=processor,
        dropout=config['dropout'],
        z_axis_weight=config['z_axis_weight'],
        use_advanced_features=config['use_advanced_features']
    )
    
    print("🚀 Optimized Hybrid Mobile VLA Model 초기화 완료!")
    print(f"📊 설정:")
    print(f"   - 드롭아웃: {config['dropout']}")
    print(f"   - Z축 가중치: {config['z_axis_weight']}")
    print(f"   - 고급 기능: {config['use_advanced_features']}")
    print(f"   - 학습률: {config['learning_rate']}")
    print(f"   - 가중치 감쇠: {config['weight_decay']}")
    print(f"   - 에포크: {config['num_epochs']}")
