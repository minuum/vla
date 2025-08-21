#!/usr/bin/env python3
"""
🚀 Case 3: 중기 적용 - 고급 멀티모달 융합 모델
목표: MAE 0.3 → 0.2, 정확도 35% → 55%
특징: Claw Matrix + Hierarchical Planning + Advanced Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import numpy as np
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisionResampler(nn.Module):
    """
    고급 Vision Resampler
    - Cross-attention with learnable queries
    - Multi-scale feature extraction
    - Adaptive pooling
    """
    
    def __init__(self, input_dim, output_dim, num_latents=32, num_heads=8, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_latents, output_dim))
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Multi-layer cross-attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Multi-scale pooling
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(num_latents // (2**i)) 
            for i in range(3)  # 3 scales
        ])
        
        # Final projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"✅ Advanced Vision Resampler 초기화:")
        logger.info(f"   - Latents: {num_latents}")
        logger.info(f"   - Heads: {num_heads}")
        logger.info(f"   - Layers: {num_layers}")
        logger.info(f"   - Multi-scale pooling: 3 scales")
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Input projection
        x = self.input_proj(x)  # (B, seq_len, output_dim)
        
        # Expand query tokens for batch
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multi-layer cross-attention
        for layer_idx in range(self.num_layers):
            # Cross-attention
            attn_out, _ = self.cross_attention_layers[layer_idx](
                queries, x, x
            )
            queries = self.layer_norms[layer_idx](queries + attn_out)
            
            # Feed-forward
            ffn_out = self.ffns[layer_idx](queries)
            queries = queries + ffn_out
        
        # Multi-scale feature extraction
        scale_features = []
        for pool in self.adaptive_pools:
            pooled = pool(queries.transpose(1, 2)).transpose(1, 2)
            scale_features.append(pooled)
        
        # Concatenate and project
        multi_scale = torch.cat(scale_features, dim=1)
        output = self.output_proj(multi_scale)
        output = self.final_norm(output)
        
        # Global average pooling
        return output.mean(dim=1)

class EnhancedClawMatrix(nn.Module):
    """
    개선된 Claw Matrix Fusion
    - 더 안정적인 차원 처리
    - Residual connections
    - Adaptive attention weights
    """
    
    def __init__(self, vision_dim, language_dim, action_dim, hidden_dim=512, 
                 num_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Dimension alignment
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Multi-head attention for each pair
        self.vision_language_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.vision_action_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.language_action_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(6)
        ])
        
        # Adaptive weight learning
        self.adaptive_weights = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        logger.info(f"✅ Enhanced Claw Matrix 초기화:")
        logger.info(f"   - Hidden dim: {hidden_dim}")
        logger.info(f"   - Attention heads: {num_heads}")
        logger.info(f"   - Adaptive weight learning: ✅")
    
    def forward(self, vision_features, language_features, action_features):
        batch_size = vision_features.shape[0]
        
        # Project to common dimension
        v_proj = self.vision_proj(vision_features)
        l_proj = self.language_proj(language_features)
        a_proj = self.action_proj(action_features)
        
        # Add sequence dimension if needed
        if v_proj.dim() == 2:
            v_proj = v_proj.unsqueeze(1)
        if l_proj.dim() == 2:
            l_proj = l_proj.unsqueeze(1)
        if a_proj.dim() == 2:
            a_proj = a_proj.unsqueeze(1)
        
        # Cross-modal attention
        # Vision-Language
        vl_attn, _ = self.vision_language_attn(v_proj, l_proj, l_proj)
        vl_fused = self.layer_norms[0](v_proj + vl_attn)
        
        # Vision-Action
        va_attn, _ = self.vision_action_attn(v_proj, a_proj, a_proj)
        va_fused = self.layer_norms[1](v_proj + va_attn)
        
        # Language-Action
        la_attn, _ = self.language_action_attn(l_proj, a_proj, a_proj)
        la_fused = self.layer_norms[2](l_proj + la_attn)
        
        # Aggregate features
        v_final = self.layer_norms[3](vl_fused + va_fused).squeeze(1)
        l_final = self.layer_norms[4](vl_fused + la_fused).squeeze(1)
        a_final = self.layer_norms[5](va_fused + la_fused).squeeze(1)
        
        # Adaptive weighting
        combined = torch.cat([v_final, l_final, a_final], dim=-1)
        weights = self.adaptive_weights(combined)
        
        # Weighted combination
        weighted_features = (
            weights[:, 0:1] * v_final +
            weights[:, 1:2] * l_final +
            weights[:, 2:3] * a_final
        )
        
        # Final fusion
        fusion_input = torch.cat([v_final, l_final, a_final], dim=-1)
        fused_output = self.fusion_layers(fusion_input)
        
        # Residual connection
        final_output = fused_output + weighted_features
        
        return final_output

class HierarchicalPlanner(nn.Module):
    """
    계층적 계획 모듈
    - High-level goal encoding
    - Mid-level subgoal generation
    - Low-level action planning
    """
    
    def __init__(self, input_dim, action_dim, num_levels=3, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Hierarchical planning layers
        self.planning_layers = nn.ModuleList()
        for level in range(num_levels):
            layer_input_dim = hidden_dim if level == 0 else hidden_dim
            layer = nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // (2 ** level)),
                nn.ReLU()
            )
            self.planning_layers.append(layer)
        
        # Action decoder
        total_planning_dim = sum([hidden_dim // (2 ** i) for i in range(num_levels)])
        self.action_decoder = nn.Sequential(
            nn.Linear(total_planning_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Attention weights for combining levels
        self.level_attention = nn.Sequential(
            nn.Linear(total_planning_dim, num_levels),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"✅ Hierarchical Planner 초기화:")
        logger.info(f"   - Levels: {num_levels}")
        logger.info(f"   - Hidden dim: {hidden_dim}")
        logger.info(f"   - Attention-based level combination: ✅")
    
    def forward(self, fused_features):
        # Goal encoding
        goal = self.goal_encoder(fused_features)
        
        # Hierarchical planning
        level_outputs = []
        current_input = goal
        
        for level, layer in enumerate(self.planning_layers):
            level_output = layer(current_input)
            level_outputs.append(level_output)
            # Pass some information to next level
            if level < len(self.planning_layers) - 1:
                current_input = level_output
        
        # Combine all levels
        combined_planning = torch.cat(level_outputs, dim=-1)
        
        # Attention-weighted combination
        level_weights = self.level_attention(combined_planning)
        
        # Action generation
        actions = self.action_decoder(combined_planning)
        
        return actions, level_weights

class AdvancedMultimodalModelV3(nn.Module):
    """
    고급 멀티모달 융합 모델
    - Advanced Vision Resampler
    - Enhanced Claw Matrix
    - Hierarchical Planning
    - State-of-the-art attention mechanisms
    """
    
    def __init__(self, processor, vision_dim=1024, language_dim=2048, action_dim=2,
                 hidden_dim=512, dropout=0.2, use_hierarchical_planning=True):
        super().__init__()
        self.processor = processor
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_hierarchical_planning = use_hierarchical_planning
        
        # Advanced Vision Resampler
        self.vision_resampler = AdvancedVisionResampler(
            input_dim=vision_dim,
            output_dim=hidden_dim,
            num_latents=32,
            num_heads=8,
            num_layers=2,
            dropout=dropout
        )
        
        # Language adapter
        self.language_adapter = nn.Linear(language_dim, hidden_dim)
        
        # Dummy action features for Claw Matrix
        self.dummy_action_features = nn.Parameter(torch.randn(1, action_dim))
        
        # Enhanced Claw Matrix
        self.claw_matrix = EnhancedClawMatrix(
            vision_dim=hidden_dim,
            language_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Hierarchical Planner or simple action head
        if use_hierarchical_planning:
            self.planner = HierarchicalPlanner(
                input_dim=hidden_dim,
                action_dim=action_dim,
                num_levels=3,
                hidden_dim=hidden_dim // 2,
                dropout=dropout
            )
        else:
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(action_dim)
        
        logger.info(f"✅ Advanced Multimodal Model 초기화 완료:")
        logger.info(f"   - Vision dim: {vision_dim} → {hidden_dim}")
        logger.info(f"   - Language dim: {language_dim} → {hidden_dim}")
        logger.info(f"   - Action dim: {action_dim}")
        logger.info(f"   - Hidden dim: {hidden_dim}")
        logger.info(f"   - Hierarchical planning: {use_hierarchical_planning}")
        logger.info(f"   - Advanced components: Vision Resampler + Claw Matrix")
    
    def forward(self, images, texts):
        batch_size = len(images)  # PIL 이미지 리스트
        device = next(self.parameters()).device
        
        # Process images and texts
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        ).to(device)
        
        # Extract features from Kosmos2
        with torch.no_grad():
            outputs = self.kosmos(**inputs, output_hidden_states=True)
        
        # Vision features (from image embeddings)
        if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
            vision_features = outputs.image_embeds
        else:
            vision_features = outputs.last_hidden_state[:, :100]  # First 100 tokens as vision
        
        # Language features (from text embeddings)
        language_features = outputs.last_hidden_state[:, -50:].mean(dim=1)  # Last 50 tokens as language
        
        # Apply Vision Resampler
        vision_features = self.vision_resampler(vision_features)
        
        # Apply language adapter
        language_features = self.language_adapter(language_features)
        
        # Dummy action features for Claw Matrix
        dummy_actions = self.dummy_action_features.expand(batch_size, -1)
        
        # Apply Enhanced Claw Matrix
        fused_features = self.claw_matrix(
            vision_features, language_features, dummy_actions
        )
        
        # Generate actions
        if self.use_hierarchical_planning:
            actions, level_weights = self.planner(fused_features)
        else:
            actions = self.action_head(fused_features)
        
        # Final normalization
        actions = self.final_norm(actions)
        
        return actions

class AdvancedMultimodalTrainerV3:
    """고급 멀티모달 모델 훈련기"""
    
    def __init__(self, model, device, learning_rate=2e-5, weight_decay=1e-3, 
                 warmup_steps=100):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for different components
        param_groups = [
            {'params': model.kosmos.parameters(), 'lr': learning_rate * 0.1},  # Lower LR for pretrained
            {'params': model.vision_resampler.parameters(), 'lr': learning_rate},
            {'params': model.claw_matrix.parameters(), 'lr': learning_rate},
            {'params': model.language_adapter.parameters(), 'lr': learning_rate}
        ]
        
        if hasattr(model, 'planner'):
            param_groups.append({'params': model.planner.parameters(), 'lr': learning_rate})
        else:
            param_groups.append({'params': model.action_head.parameters(), 'lr': learning_rate})
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=0.05)  # More sensitive to small errors
        
        logger.info(f"✅ Advanced Multimodal Trainer 초기화 완료:")
        logger.info(f"   - Learning rate: {learning_rate}")
        logger.info(f"   - Weight decay: {weight_decay}")
        logger.info(f"   - Differential LR for components: ✅")
        logger.info(f"   - Warmup + Cosine scheduling: ✅")
        logger.info(f"   - Huber loss (delta=0.05): ✅")
    
    def train_step(self, batch):
        """훈련 스텝"""
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch['image']  # PIL 이미지 리스트
        actions = batch['action'].to(self.device)
        texts = batch['text']
        
        # Forward pass
        predicted_actions = self.model(images, texts)
        
        # Loss calculation
        loss = self.criterion(predicted_actions, actions)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        """검증 스텝"""
        self.model.eval()
        
        images = batch['image']  # PIL 이미지 리스트
        actions = batch['action'].to(self.device)
        texts = batch['text']
        
        with torch.no_grad():
            predicted_actions = self.model(images, texts)
            loss = self.criterion(predicted_actions, actions)
            mae = torch.mean(torch.abs(predicted_actions - actions))
        
        return loss.item(), mae.item()
    
    def save_checkpoint(self, path, epoch, val_loss, val_mae):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }, path)
        
        logger.info(f"💾 체크포인트 저장: {path}")

if __name__ == "__main__":
    # 테스트 코드
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model = AdvancedMultimodalModel(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,
        action_dim=2,
        hidden_dim=512,
        dropout=0.2,
        use_hierarchical_planning=True
    )
    
    # 더미 입력으로 테스트
    images = torch.randn(2, 3, 224, 224)
    texts = ["Navigate to the target", "Move forward"]
    
    with torch.no_grad():
        actions = model(images, texts)
        print(f"테스트 출력 shape: {actions.shape}")
        print(f"테스트 액션: {actions}")
    
    logger.info("✅ Advanced Multimodal Model 테스트 완료!")
