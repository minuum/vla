#!/usr/bin/env python3
"""
Enhanced Kosmos2+CLIP Hybrid Model with CLIP Normalization
Vision Resampler와 CLIP Normalization을 통합한 향상된 하이브리드 모델

주요 기능:
1. Vision Resampler 통합
2. CLIP Normalization 적용
3. 모바일 최적화
4. 성능 향상
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, Kosmos2ForConditionalGeneration, Kosmos2Processor
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision_resampler_implementation import MobileOptimizedVisionResampler
from clip_normalization_implementation import MobileOptimizedCLIPNormalization, AdaptiveCLIPNormalization

logger = logging.getLogger(__name__)

class EnhancedKosmos2CLIPHybridWithNormalization(nn.Module):
    """
    Enhanced Kosmos2+CLIP Hybrid Model with CLIP Normalization
    
    Vision Resampler와 CLIP Normalization을 통합한 최고 성능 모델
    """
    
    def __init__(
        self,
        action_dim: int = 2,  # 2D 액션 (linear_x, linear_y) - Z값은 항상 0
        vision_resampler_tokens: int = 64,
        hidden_dim: int = 768,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        mobile_optimized: bool = True,
        use_clip_normalization: bool = True,
        normalization_type: str = "mobile"  # "mobile", "adaptive", "ensemble"
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.vision_resampler_tokens = vision_resampler_tokens
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.mobile_optimized = mobile_optimized
        self.use_clip_normalization = use_clip_normalization
        self.normalization_type = normalization_type
        
        # Initialize CLIP model
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Kosmos2 model
        logger.info("Loading Kosmos2 model...")
        self.kosmos2_model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", use_safetensors=True)
        self.kosmos2_processor = Kosmos2Processor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # Vision Resampler
        logger.info("Initializing Vision Resampler...")
        self.vision_resampler = MobileOptimizedVisionResampler(
            input_dim=768,  # CLIP ViT-B/32 output dimension
            output_dim=768,
            num_tokens=vision_resampler_tokens
        )
        
        # CLIP Normalization
        if use_clip_normalization:
            logger.info(f"Initializing CLIP Normalization ({normalization_type})...")
            if normalization_type == "mobile":
                self.clip_normalization = MobileOptimizedCLIPNormalization(
                    feature_dim=768,
                    use_half_precision=False,  # Disable to avoid dtype mismatch
                    use_gradient_checkpointing=True
                )
            elif normalization_type == "adaptive":
                self.clip_normalization = AdaptiveCLIPNormalization(
                    feature_dim=768,
                    num_heads=8,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown normalization type: {normalization_type}")
        else:
            self.clip_normalization = None
        
        # Feature Fusion Layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 256, hidden_dim),  # 768 + 256 = 1024
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM projection layer - handle dynamic input size
        self.lstm_projection = None  # Will be created dynamically
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,  # Match projection output
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Enhanced Kosmos2+CLIP Hybrid Model with Normalization initialized:")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Vision resampler tokens: {vision_resampler_tokens}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - LSTM hidden dim: {lstm_hidden_dim}")
        logger.info(f"  - Mobile optimized: {mobile_optimized}")
        logger.info(f"  - CLIP normalization: {use_clip_normalization}")
        logger.info(f"  - Normalization type: {normalization_type}")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def extract_clip_features(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Extract CLIP features"""
        # Process images and texts
        inputs = self.clip_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # Get CLIP features
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds  # [batch_size, 768]
            text_features = outputs.text_embeds    # [batch_size, 768]
        
        # Combine features
        combined_features = torch.cat([image_features, text_features], dim=-1)  # [batch_size, 1536]
        
        return combined_features
    
    def extract_kosmos2_features(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Extract Kosmos2 features"""
        batch_size = len(images)
        kosmos2_features = []
        
        for i in range(batch_size):
            # Process individual image-text pair
            inputs = self.kosmos2_processor(
                text=[texts[i]],
                images=[images[i]],
                return_tensors="pt"
            )
            
            # Move to same device as model
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            
            # Get Kosmos2 features
            with torch.no_grad():
                outputs = self.kosmos2_model(**inputs)
                # Use last hidden state (different attribute name)
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
                elif hasattr(outputs, 'encoder_last_hidden_state'):
                    features = outputs.encoder_last_hidden_state.mean(dim=1)  # [1, 768]
                else:
                    # Fallback: use logits and project to feature dimension
                    features = outputs.logits.mean(dim=1)  # [1, vocab_size]
                    if features.size(-1) != 768:
                        # Project to 768 dimensions if needed
                        if not hasattr(self, 'kosmos2_projection'):
                            self.kosmos2_projection = nn.Linear(features.size(-1), 768).to(features.device)
                        features = self.kosmos2_projection(features)
                kosmos2_features.append(features)
        
        # Stack features
        kosmos2_features = torch.cat(kosmos2_features, dim=0)  # [batch_size, 768]
        
        return kosmos2_features
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: List[str],
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: [batch_size, channels, height, width] Input images
            texts: List of text commands
            return_features: Whether to return intermediate features
            
        Returns:
            actions: [batch_size, action_dim] Predicted actions
        """
        batch_size = images.size(0)
        
        # Extract CLIP features
        clip_features = self.extract_clip_features(images, texts)  # [batch_size, 1536]
        
        # Extract Kosmos2 features
        kosmos2_features = self.extract_kosmos2_features(images, texts)  # [batch_size, 768]
        
        # Reshape CLIP features for vision resampler
        clip_image_features = clip_features[:, :768].unsqueeze(1)  # [batch_size, 1, 768]
        clip_text_features = clip_features[:, 768:].unsqueeze(1)   # [batch_size, 1, 768]
        
        # Apply Vision Resampler to CLIP image features
        resampled_image_features = self.vision_resampler(clip_image_features)  # [batch_size, num_tokens, 768]
        
        # Apply CLIP Normalization if enabled
        if self.use_clip_normalization and self.clip_normalization is not None:
            resampled_image_features = self.clip_normalization(resampled_image_features)
            kosmos2_features = self.clip_normalization(kosmos2_features.unsqueeze(1)).squeeze(1)
        
        # Combine resampled image features and text features
        combined_clip_features = torch.cat([
            resampled_image_features.mean(dim=1),  # [batch_size, 768]
            clip_text_features.squeeze(1)          # [batch_size, 768]
        ], dim=-1)  # [batch_size, 1536]
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_clip_features)  # [batch_size, 768]
        
        # Combine with Kosmos2 features
        final_features = torch.cat([fused_features, kosmos2_features], dim=-1)  # [batch_size, 1536]
        
        # Project to LSTM input size - handle dynamic dimensions
        if self.lstm_projection is None or final_features.size(-1) != self.lstm_projection.in_features:
            # Create dynamic projection layer if needed
            self.lstm_projection = nn.Linear(final_features.size(-1), self.lstm_hidden_dim).to(final_features.device)
        
        lstm_input = self.lstm_projection(final_features).unsqueeze(1)  # [batch_size, 1, lstm_hidden_dim]
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(lstm_input)  # [batch_size, 1, lstm_hidden_dim]
        
        # Get final hidden state
        final_hidden = lstm_output[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Action prediction
        actions = self.action_head(final_hidden)  # [batch_size, action_dim]
        
        if return_features:
            return actions, {
                'clip_features': clip_features,
                'kosmos2_features': kosmos2_features,
                'resampled_features': resampled_image_features,
                'fused_features': fused_features,
                'lstm_output': lstm_output
            }
        
        return actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'vision_resampler_tokens': self.vision_resampler_tokens,
            'action_dim': self.action_dim,
            'mobile_optimized': self.mobile_optimized,
            'use_clip_normalization': self.use_clip_normalization,
            'normalization_type': self.normalization_type
        }

class EnhancedKosmos2CLIPHybridWithNormalizationTrainer:
    """
    Trainer for Enhanced Kosmos2+CLIP Hybrid Model with Normalization
    """
    
    def __init__(
        self,
        model: EnhancedKosmos2CLIPHybridWithNormalization,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        
        logger.info(f"Enhanced trainer with normalization initialized on {device}")
    
    def train_step(
        self, 
        images: torch.Tensor, 
        texts: List[str], 
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        images = images.to(self.device)
        actions = actions.to(self.device)
        
        # Forward pass
        predicted_actions = self.model(images, texts)
        
        # Compute loss
        loss = self.criterion(predicted_actions, actions)
        
        # Compute MAE
        mae = F.l1_loss(predicted_actions, actions).item()
        
        return {
            'loss': loss.item(),
            'mae': mae
        }
    
    def validate_step(
        self, 
        images: torch.Tensor, 
        texts: List[str], 
        actions: torch.Tensor
    ) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            images = images.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            predicted_actions = self.model(images, texts)
            
            # Compute loss
            loss = self.criterion(predicted_actions, actions)
            
            # Compute MAE
            mae = F.l1_loss(predicted_actions, actions).item()
            
            return {
                'loss': loss.item(),
                'mae': mae
            }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_info': self.model.get_model_info()
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

def test_enhanced_model_with_normalization():
    """Test the enhanced model with normalization"""
    print("Testing Enhanced Kosmos2+CLIP Hybrid Model with Normalization...")
    
    # Test data
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)  # Use rand instead of randn for [0,1] range
    texts = ["go forward", "turn left"]
    actions = torch.randn(batch_size, 3)
    
    # Test different normalization types
    normalization_types = ["mobile", "adaptive"]
    
    for norm_type in normalization_types:
        print(f"\nTesting with {norm_type} normalization...")
        
        # Create model
        model = EnhancedKosmos2CLIPHybridWithNormalization(
            action_dim=2,  # 2D 액션 (linear_x, linear_y) - Z값은 항상 0
            vision_resampler_tokens=64,
            mobile_optimized=True,
            use_clip_normalization=True,
            normalization_type=norm_type
        )
        
        # Get model info
        model_info = model.get_model_info()
        print(f"  Model info: {model_info}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                predicted_actions = model(images, texts)
                print(f"  Input images shape: {images.shape}")
                print(f"  Predicted actions shape: {predicted_actions.shape}")
                print(f"  Predicted actions mean: {predicted_actions.mean().item():.4f}")
                print(f"  Predicted actions std: {predicted_actions.std().item():.4f}")
                print(f"  ✅ {norm_type} normalization test passed!")
        except Exception as e:
            print(f"  ❌ {norm_type} normalization test failed: {e}")
    
    print("\n✅ All Enhanced Model with Normalization tests completed!")

if __name__ == "__main__":
    test_enhanced_model_with_normalization()
