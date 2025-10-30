#!/usr/bin/env python3
"""
Enhanced Kosmos2+CLIP Hybrid Model with Simple Claw Matrix
Vision Resampler, CLIP Normalization, Simple Claw Matrix를 통합한 최고 성능 모델

주요 기능:
1. Vision Resampler 통합
2. CLIP Normalization 적용
3. Simple Claw Matrix 적용
4. 모바일 최적화
5. 성능 향상
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
from simple_claw_matrix_implementation import SimpleClawMatrix, MobileOptimizedSimpleClawMatrix

logger = logging.getLogger(__name__)

class EnhancedKosmos2CLIPHybridWithSimpleClawMatrix(nn.Module):
    """
    Enhanced Kosmos2+CLIP Hybrid Model with Simple Claw Matrix
    
    Vision Resampler, CLIP Normalization, Simple Claw Matrix를 통합한 최고 성능 모델
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
        use_claw_matrix: bool = True,
        normalization_type: str = "mobile",  # "mobile", "adaptive"
        claw_matrix_type: str = "mobile"  # "mobile", "simple"
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
        self.use_claw_matrix = use_claw_matrix
        self.normalization_type = normalization_type
        self.claw_matrix_type = claw_matrix_type
        
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
        
        # Simple Claw Matrix
        if use_claw_matrix:
            logger.info(f"Initializing Simple Claw Matrix ({claw_matrix_type})...")
            if claw_matrix_type == "mobile":
                self.claw_matrix = MobileOptimizedSimpleClawMatrix(
                    feature_dim=768,
                    use_half_precision=False,
                    use_gradient_checkpointing=True
                )
            elif claw_matrix_type == "simple":
                self.claw_matrix = SimpleClawMatrix(
                    feature_dim=768
                )
            else:
                raise ValueError(f"Unknown claw matrix type: {claw_matrix_type}")
        else:
            self.claw_matrix = None
        
        # Feature Fusion Layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 768 * 2 = 1536
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
        
        logger.info(f"Enhanced Kosmos2+CLIP Hybrid Model with Simple Claw Matrix initialized:")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Vision resampler tokens: {vision_resampler_tokens}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - LSTM hidden dim: {lstm_hidden_dim}")
        logger.info(f"  - Mobile optimized: {mobile_optimized}")
        logger.info(f"  - CLIP normalization: {use_clip_normalization}")
        logger.info(f"  - Normalization type: {normalization_type}")
        logger.info(f"  - Simple claw matrix: {use_claw_matrix}")
        logger.info(f"  - Claw matrix type: {claw_matrix_type}")
    
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
        Forward pass with Simple Claw Matrix
        
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
        
        # Apply Simple Claw Matrix if enabled
        if self.use_claw_matrix and self.claw_matrix is not None:
            # Prepare features for claw matrix
            vision_features = resampled_image_features  # [batch_size, num_tokens, 768]
            language_features = clip_text_features.expand(-1, resampled_image_features.size(1), -1)  # [batch_size, num_tokens, 768]
            
            # Apply simple claw matrix
            aligned_features = self.claw_matrix(vision_features, language_features)  # [batch_size, num_tokens, 768]
            
            # Use aligned features
            resampled_image_features = aligned_features
        
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
            'normalization_type': self.normalization_type,
            'use_claw_matrix': self.use_claw_matrix,
            'claw_matrix_type': self.claw_matrix_type
        }

def test_enhanced_model_with_simple_claw_matrix():
    """Test the enhanced model with simple claw matrix"""
    print("Testing Enhanced Kosmos2+CLIP Hybrid Model with Simple Claw Matrix...")
    
    # Test data
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)  # Use rand instead of randn for [0,1] range
    texts = ["go forward", "turn left"]
    actions = torch.randn(batch_size, 3)
    
    # Test different combinations
    test_configs = [
        {"normalization_type": "mobile", "claw_matrix_type": "mobile"},
        {"normalization_type": "adaptive", "claw_matrix_type": "mobile"},
        {"normalization_type": "mobile", "claw_matrix_type": "simple"},
        {"normalization_type": "adaptive", "claw_matrix_type": "simple"}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTesting configuration {i+1}: {config}")
        
        # Create model
        model = EnhancedKosmos2CLIPHybridWithSimpleClawMatrix(
            action_dim=3,
            vision_resampler_tokens=64,
            mobile_optimized=True,
            use_clip_normalization=True,
            use_claw_matrix=True,
            normalization_type=config["normalization_type"],
            claw_matrix_type=config["claw_matrix_type"]
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
                print(f"  ✅ Configuration {i+1} test passed!")
        except Exception as e:
            print(f"  ❌ Configuration {i+1} test failed: {e}")
    
    print("\n✅ All Enhanced Model with Simple Claw Matrix tests completed!")

if __name__ == "__main__":
    test_enhanced_model_with_simple_claw_matrix()
