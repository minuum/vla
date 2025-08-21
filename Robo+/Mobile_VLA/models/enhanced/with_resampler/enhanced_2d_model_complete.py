"""
🚀 Enhanced 2D Action Model with Vision Resampler
Vision Resampler를 포함한 향상된 2D 액션 모델
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVisionResampler(nn.Module):
    """Simple Vision Resampler for reducing image token count"""
    def __init__(self, input_dim, output_dim, num_latents=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        
        # Learnable latents
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        
        # Projection layers
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # Project input to output dimension
        x = self.input_proj(x)
        
        # Expand latents to batch size
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention between latents and input
        attn_out, _ = self.attention(latents, x, x)
        latents = self.norm1(latents + attn_out)
        
        # Self-attention on latents
        self_attn_out, _ = self.attention(latents, latents, latents)
        latents = self.norm2(latents + self_attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(latents)
        latents = latents + ffn_out
        
        # Final projection
        output = self.output_proj(latents)
        
        # Return mean across latents
        return output.mean(dim=1)

class Enhanced2DActionModel(nn.Module):
    """Enhanced 2D Action Model with Vision Resampler"""
    
    def __init__(self, processor, vision_dim=1024, language_dim=1024, action_dim=2, 
                 hidden_dim=512, dropout=0.2, use_vision_resampler=True):
        super().__init__()
        
        # Basic settings
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Feature flags
        self.use_vision_resampler = use_vision_resampler
        
        # Kosmos2 model
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos_processor = processor
        
        # Device setup
        self.device = next(self.kosmos.parameters()).device
        self.kosmos = self.kosmos.to(self.device)
        self.kosmos.eval()
        
        # Dynamic adapters
        self.language_adapter = None
        self.fusion_adapter = None
        
        # Feature extractor
        self.feature_adapter = nn.Linear(vision_dim, hidden_dim)
        
        # Normalization and dropout
        self.layer_norm_vision = nn.LayerNorm(hidden_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # Vision Resampler
        if self.use_vision_resampler:
            self.vision_resampler = SimpleVisionResampler(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_latents=64
            )
            logger.info("✅ Simple Vision Resampler initialized")
        else:
            self.vision_resampler = None
        
        # 2D Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        logger.info(f"✅ Enhanced 2D Action Model 초기화 완료")
        logger.info(f"   - 액션 차원: {action_dim}D")
        logger.info(f"   - 비전 리샘플러: {use_vision_resampler}")
    
    def extract_vision_features(self, images):
        """Vision feature extraction with resampler"""
        batch_size = images.shape[0]
        
        # 이미지 정규화: [-1, 1] → [0, 1]
        images_normalized = (images + 1.0) / 2.0
        images_normalized = torch.clamp(images_normalized, 0.0, 1.0)
        
        # Convert to PIL images
        pil_images = []
        for i in range(batch_size):
            img = images_normalized[i].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        
        # Kosmos2 processor for input preparation
        inputs = self.kosmos_processor(
            images=pil_images, 
            return_tensors="pt",
            padding=True
        )
        
        # Move all inputs to model device
        inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Use Kosmos2 vision model
        with torch.no_grad():
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output
            else:
                vision_features = torch.zeros(batch_size, 1024).to(self.kosmos.device)
        
        # Dimension adjustment
        vision_features = self.feature_adapter(vision_features)
        
        # Apply Vision Resampler
        if self.use_vision_resampler and self.vision_resampler is not None:
            # Convert to sequence format: [batch_size, 1, hidden_dim]
            vision_features = vision_features.unsqueeze(1)
            vision_features = self.vision_resampler(vision_features)
        
        # Normalization and dropout
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str, batch_size: int = 1):
        """Language feature extraction"""
        with torch.no_grad():
            inputs = self.kosmos_processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Use Kosmos2 text model
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)
        
        # Expand batch dimension
        language_features = language_features.expand(batch_size, -1)
        
        # Dimension adjustment (dynamic adapter creation)
        if language_features.shape[-1] != self.language_dim:
            if self.language_adapter is None:
                self.language_adapter = nn.Linear(
                    language_features.shape[-1], 
                    self.language_dim
                ).to(language_features.device)
            language_features = self.language_adapter(language_features)
        
        # Normalization and dropout
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, single_image: torch.Tensor, text: str):
        """Enhanced 2D action prediction with vision resampler"""
        batch_size = single_image.shape[0]
        
        # Feature extraction
        vision_features = self.extract_vision_features(single_image)
        language_features = self.extract_language_features(text, batch_size)
        
        # Basic fusion
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # Dimension adjustment
        if fused_features.shape[-1] != self.hidden_dim:
            if self.fusion_adapter is None:
                self.fusion_adapter = nn.Linear(fused_features.shape[-1], self.hidden_dim).to(fused_features.device)
            fused_features = self.fusion_adapter(fused_features)
        
        # Normalization and dropout
        fused_features = self.layer_norm_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)
        
        # 2D action prediction (Z-axis excluded)
        actions_2d = self.action_head(fused_features)
        
        return actions_2d
