"""
🚀 완전히 수정된 RoboVLMs Style Single Image VLA Model
차원과 데이터타입 문제를 모두 해결한 완전한 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from typing import Optional, Tuple, Dict, Any
import numpy as np
import PIL.Image

# 수정된 컴포넌트들 임포트
from fixed_claw_matrix import (
    FixedClawMatrixFusion, 
    FixedHierarchicalPlanner, 
    FixedAdvancedAttention
)

class FixedRoboVLMStyleSingleImageModel(nn.Module):
    """
    🎯 완전히 수정된 RoboVLMs 스타일 모델
    - 입력: 단일 이미지 1장
    - 출력: 단일 액션 (3D)
    - 고급 기능: Fixed Claw Matrix, Hierarchical Planning, Advanced Attention
    - 모든 차원 문제 해결됨
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
        
        # 동적 feature adapter들
        self.vision_adapter = nn.Linear(1024, vision_dim)  # Kosmos2 vision 출력 -> vision_dim
        self.language_adapter = None  # 동적 생성
        
        # 강화된 정규화
        self.layer_norm_vision = nn.LayerNorm(vision_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 드롭아웃
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # 기본 융합 어댑터
        self.fusion_adapter = nn.Linear(vision_dim + language_dim, hidden_dim)
        
        # Fixed Claw Matrix (수정된 버전)
        if use_claw_matrix:
            self.claw_matrix = FixedClawMatrixFusion(
                input_dim=vision_dim + language_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # Fixed Hierarchical Planning (수정된 버전)
        if use_hierarchical:
            self.hierarchical_planner = FixedHierarchicalPlanner(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                dropout=dropout
            )
        
        # Fixed Advanced Attention (수정된 버전)
        if use_advanced_attention:
            self.advanced_attention = FixedAdvancedAttention(
                hidden_dim=hidden_dim,
                dropout=dropout
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
        
    def _preprocess_image(self, images: torch.Tensor) -> list:
        """이미지를 PIL 형태로 전처리"""
        # images: [batch_size, 3, H, W] 형태의 텐서
        
        # 0-1 범위로 정규화 (필요한 경우)
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min())
        
        # 텐서를 PIL 이미지로 변환
        pil_images = []
        for i in range(images.shape[0]):
            # [3, H, W] -> [H, W, 3]
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            # [0, 1] -> [0, 255]
            img_np = (img_np * 255).astype(np.uint8)
            # PIL 이미지로 변환
            pil_img = PIL.Image.fromarray(img_np)
            pil_images.append(pil_img)
        
        return pil_images
        
    def extract_vision_features(self, single_image: torch.Tensor) -> torch.Tensor:
        """단일 이미지 특징 추출 (RoboVLMs 스타일)"""
        # single_image: [batch_size, 3, H, W] - 단일 이미지
        
        batch_size = single_image.shape[0]
        
        # 이미지 전처리
        pil_images = self._preprocess_image(single_image)
        
        # 배치 처리를 위해 개별적으로 처리
        vision_features_list = []
        
        for pil_img in pil_images:
            with torch.no_grad():
                inputs = self.processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(single_image.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                
                # Kosmos2 vision 모델 사용
                if 'pixel_values' in inputs:
                    vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                    vision_feat = vision_outputs.pooler_output  # [1, 1024]
                else:
                    # fallback
                    vision_feat = torch.zeros(1, 1024).to(single_image.device)
                
                vision_features_list.append(vision_feat)
        
        # 배치로 결합
        vision_features = torch.cat(vision_features_list, dim=0)  # [batch_size, 1024]
        
        # 특징 차원 조정
        vision_features = self.vision_adapter(vision_features)  # [batch_size, vision_dim]
        
        # 강화된 정규화 및 드롭아웃
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str, batch_size: int = 1) -> torch.Tensor:
        """언어 특징 추출 (RoboVLMs 스타일)"""
        
        # 텍스트 처리
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
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
        """순전파 (RoboVLMs 스타일)"""
        # single_image: [batch_size, 3, H, W] - 단일 이미지
        # text: 문자열
        
        batch_size = single_image.shape[0]
        
        # 특징 추출
        vision_features = self.extract_vision_features(single_image)
        language_features = self.extract_language_features(text, batch_size)
        
        # 기본 융합
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # Claw Matrix 적용 (수정된 버전)
        if self.use_claw_matrix and hasattr(self, 'claw_matrix'):
            fused_features = self.claw_matrix(fused_features)  # input_dim -> hidden_dim
        else:
            # Claw Matrix를 사용하지 않는 경우 기본 융합
            fused_features = self.fusion_adapter(fused_features)  # input_dim -> hidden_dim
        
        # 정규화 및 드롭아웃
        fused_features = self.layer_norm_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)
        
        # Advanced Attention 적용 (수정된 버전)
        if self.use_advanced_attention and hasattr(self, 'advanced_attention'):
            fused_features = self.advanced_attention(fused_features)  # hidden_dim -> hidden_dim
        
        # Hierarchical Planning 적용 (수정된 버전)
        if self.use_hierarchical and hasattr(self, 'hierarchical_planner'):
            fused_features = self.hierarchical_planner(fused_features)  # hidden_dim -> hidden_dim
        
        # 최종 액션 예측
        actions = self.action_head(fused_features)  # hidden_dim -> action_dim
        
        # Z축 가중치 적용 (Final Fixed 스타일)
        actions = actions * self.z_axis_weight_layer.unsqueeze(0)
        
        return actions

def train_fixed_robovlms_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda'
):
    """수정된 RoboVLMs 스타일 모델 훈련"""
    
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
        
        for batch_idx, batch in enumerate(train_loader):
            # 단일 이미지 입력
            images = batch['image']  # [batch, 3, H, W] - 이미 단일 이미지
            actions = batch['action']  # [batch, 3] - 이미 단일 액션
            
            images = images.float().to(device)
            actions = actions.float().to(device)
            
            optimizer.zero_grad()
            
            try:
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
                
                # 진행상황 출력
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
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
                    # 단일 이미지 입력
                    images = batch['image']  # [batch, 3, H, W]
                    actions = batch['action']  # [batch, 3]
                    
                    images = images.float().to(device)
                    actions = actions.float().to(device)
                    
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
                    'use_advanced_attention': model.use_advanced_attention
                }
            }, 'fixed_robovlms_model_best.pth')
            print(f"   ✅ 최고 모델 저장! (검증 손실: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"   ⏰ 조기 종료 (Patience: {early_stopping_patience})")
                break
    
    return model

# 🚀 사용 예시
if __name__ == "__main__":
    # 모델 초기화
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        z_axis_weight=0.05
    )
    
    print("🚀 Fixed RoboVLMs Style Single Image VLA Model 초기화 완료!")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 사용 기능:")
    print(f"   - Fixed Claw Matrix: {model.use_claw_matrix}")
    print(f"   - Fixed Hierarchical Planning: {model.use_hierarchical}")
    print(f"   - Fixed Advanced Attention: {model.use_advanced_attention}")
    print(f"   - Z축 가중치: {model.z_axis_weight}")
    print(f"   - 드롭아웃: {model.dropout}")
    print(f"🎯 입력: 단일 이미지 [batch, 3, H, W]")
    print(f"🎯 출력: 단일 액션 [batch, 3]")
    print(f"🎯 용도: 실시간 로봇 제어 (RoboVLMs 스타일)")
    
    # 간단한 테스트
    print(f"\n🧪 간단한 테스트:")
    test_image = torch.randn(2, 3, 224, 224)
    test_text = "Navigate to target"
    
    try:
        with torch.no_grad():
            output = model(test_image, test_text)
        print(f"   ✅ 테스트 성공! 출력: {output.shape}")
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
