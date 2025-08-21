#!/usr/bin/env python3
"""
Case 2: 단기 적용 (Short-term Optimization) - V2
CLIP Normalization이 적용된 2D 액션 모델 (완전히 새로운 파일)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
import logging
import open_clip

logger = logging.getLogger(__name__)

class CLIPNormalization(nn.Module):
    """CLIP Normalization 레이어"""
    
    def __init__(self, input_dim, clip_dim=512, temperature=0.07):
        super().__init__()
        self.input_dim = input_dim
        self.clip_dim = clip_dim
        self.temperature = temperature
        
        # CLIP 공간으로의 프로젝션
        self.clip_proj = nn.Linear(input_dim, clip_dim)
        self.inverse_proj = nn.Linear(clip_dim, input_dim)
        
        # 정규화 레이어
        self.norm = nn.LayerNorm(input_dim)
        
        logger.info(f"✅ CLIP Normalization 초기화 완료:")
        logger.info(f"   - input_dim: {input_dim}")
        logger.info(f"   - clip_dim: {clip_dim}")
        logger.info(f"   - temperature: {temperature}")
    
    def forward(self, features, clip_features=None):
        """
        CLIP Normalization 적용
        Args:
            features: 입력 특징 [batch_size, feature_dim]
            clip_features: CLIP 특징 (선택사항) [batch_size, clip_dim]
        Returns:
            normalized_features: 정규화된 특징 [batch_size, feature_dim]
        """
        batch_size = features.shape[0]
        
        # CLIP 공간으로 프로젝션
        clip_projected = self.clip_proj(features)  # [batch_size, clip_dim]
        
        if clip_features is not None:
            # CLIP 특징과 정렬
            clip_projected = F.normalize(clip_projected, dim=-1)
            clip_features = F.normalize(clip_features, dim=-1)
            
            # Cosine similarity 계산
            similarity = torch.matmul(clip_projected, clip_features.T) / self.temperature
            
            # Attention 가중치 계산
            attention_weights = F.softmax(similarity, dim=-1)
            
            # CLIP 특징과의 가중 평균
            aligned_features = torch.matmul(attention_weights, clip_features)
        else:
            # CLIP 특징이 없는 경우 단순 정규화
            aligned_features = F.normalize(clip_projected, dim=-1)
        
        # 원래 공간으로 역프로젝션
        normalized_features = self.inverse_proj(aligned_features)
        
        # Residual connection + 정규화
        normalized_features = self.norm(features + normalized_features)
        
        return normalized_features

class CLIPNormalized2DActionModelV2(nn.Module):
    """
    CLIP Normalization이 적용된 2D 액션 모델 V2
    - CLIP 특징과 정렬
    - 더 나은 feature alignment
    """
    
    def __init__(self, processor, vision_dim=1024, language_dim=2048, action_dim=2, 
                 hidden_dim=256, dropout=0.4, use_vision_resampler=True, 
                 use_clip_normalization=True):
        super().__init__()
        
        self.processor = processor
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout
        self.use_vision_resampler = use_vision_resampler
        self.use_clip_normalization = use_clip_normalization
        
        # CLIP 모델 로드 (RoboVLMs 방식)
        if use_clip_normalization:
            try:
                self.clip_model, self.clip_preprocess, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                # CLIP 모델을 GPU로 이동
                if torch.cuda.is_available():
                    self.clip_model = self.clip_model.cuda()
                logger.info("✅ CLIP 모델 로드 완료 (open_clip)")
            except Exception as e:
                logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}")
                self.clip_model = None
                self.clip_preprocess = None
                self.use_clip_normalization = False
        else:
            self.clip_model = None
            self.clip_preprocess = None
            self.use_clip_normalization = False
        
        # 특징 어댑터
        self.feature_adapter = nn.Linear(vision_dim, hidden_dim)
        self.language_adapter = nn.Linear(language_dim, hidden_dim)  # 2048 → 256
        
        # CLIP Normalization 레이어
        if use_clip_normalization:
            self.clip_norm_vision = CLIPNormalization(hidden_dim, clip_dim=512)
            self.clip_norm_language = CLIPNormalization(hidden_dim, clip_dim=512)
        
        # 정규화 레이어
        self.layer_norm_vision = nn.LayerNorm(hidden_dim)
        self.layer_norm_language = nn.LayerNorm(hidden_dim)
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        
        # Vision Resampler (Case 2에서 최적화된 버전 사용)
        if use_vision_resampler:
            from optimized_vision_resampler import OptimizedVisionResampler
            self.vision_resampler = OptimizedVisionResampler(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_latents=16,
                num_heads=4,
                dropout=dropout
            )
        else:
            self.vision_resampler = None
        
        # 액션 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 최종 정규화
        self.final_norm = nn.LayerNorm(action_dim)
        
        logger.info(f"✅ CLIP Normalized 2D Model V2 초기화 완료:")
        logger.info(f"   - hidden_dim: {hidden_dim}")
        logger.info(f"   - action_dim: {action_dim}")
        logger.info(f"   - use_clip_normalization: {use_clip_normalization}")
        logger.info(f"   - use_vision_resampler: {use_vision_resampler}")
    
    def extract_clip_features_from_pil(self, images):
        """PIL 이미지에서 CLIP 특징 추출"""
        if self.clip_model is None:
            return None
        
        batch_size = len(images)
        clip_features = []
        
        for i in range(batch_size):
            # PIL 이미지를 직접 사용
            pil_img = images[i]
            
            # CLIP 전처리 및 특징 추출
            with torch.no_grad():
                clip_input = self.clip_preprocess(pil_img).unsqueeze(0)
                # CLIP 입력을 GPU로 이동
                if torch.cuda.is_available():
                    clip_input = clip_input.cuda()
                clip_output = self.clip_model.encode_image(clip_input)
                clip_features.append(clip_output)
        
        return torch.cat(clip_features, dim=0)
    
    def extract_clip_features(self, images):
        """텐서 이미지에서 CLIP 특징 추출 (기존 메서드)"""
        if self.clip_model is None:
            return None
        
        batch_size = images.shape[0]
        clip_features = []
        
        for i in range(batch_size):
            # 이미지를 PIL로 변환
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img + 1) / 2.0  # [-1,1] → [0,1]
            img = (img * 255).astype('uint8')
            pil_img = Image.fromarray(img)
            
            # CLIP 전처리 및 특징 추출
            with torch.no_grad():
                clip_input = self.clip_preprocess(pil_img).unsqueeze(0)
                clip_output = self.clip_model.encode_image(clip_input)
                clip_features.append(clip_output)
        
        return torch.cat(clip_features, dim=0)
    
    def extract_vision_features(self, images):
        """시각 특징 추출 (CLIP Normalization 포함)"""
        batch_size = len(images)  # PIL 이미지 리스트
        
        # PIL 이미지를 직접 처리
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items()}
        
        # Kosmos2 vision 모델 사용
        with torch.no_grad():
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output
            else:
                vision_features = torch.zeros(batch_size, 1024).to(self.kosmos.device)
        
        # 특징 어댑터
        vision_features = self.feature_adapter(vision_features)
        
        # CLIP Normalization 적용 (PIL 이미지에서 CLIP 특징 추출)
        if self.use_clip_normalization:
            clip_features = self.extract_clip_features_from_pil(images)
            vision_features = self.clip_norm_vision(vision_features, clip_features)
        
        # Vision Resampler 적용
        if self.use_vision_resampler and self.vision_resampler is not None:
            vision_features = vision_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            vision_features = self.vision_resampler(vision_features)
        
        # 정규화 및 드롭아웃
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, texts):
        """언어 특징 추출 (CLIP Normalization 포함)"""
        batch_size = len(texts)
        
        # 텍스트 처리
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items()}
        
        # Kosmos2 text 모델 사용
        with torch.no_grad():
            if 'input_ids' in inputs:
                text_outputs = self.kosmos.text_model(inputs['input_ids'])
                # pooler_output이 없으므로 마지막 hidden state의 평균 사용
                language_features = text_outputs.last_hidden_state.mean(dim=1)
            else:
                language_features = torch.zeros(batch_size, 2048).to(self.kosmos.device)
        
        # 특징 어댑터
        language_features = self.language_adapter(language_features)
        
        # CLIP Normalization 적용 (텍스트의 경우 단순 정규화)
        if self.use_clip_normalization:
            language_features = self.clip_norm_language(language_features)
        
        # 정규화 및 드롭아웃
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, images, texts):
        """순전파"""
        # 특징 추출
        vision_features = self.extract_vision_features(images)
        language_features = self.extract_language_features(texts)
        
        # 특징 결합
        combined_features = torch.cat([vision_features, language_features], dim=1)
        
        # 액션 예측
        actions = self.action_head(combined_features)
        actions = self.final_norm(actions)
        
        return actions

class CLIPNormalizedTrainerV2:
    """CLIP Normalization 훈련기 V2"""
    
    def __init__(self, model, device, learning_rate=3e-5, weight_decay=1e-3):
        self.model = model.to(device)
        self.device = device
        
        # 최적화기
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 손실 함수
        self.criterion = nn.HuberLoss(delta=0.1)
        
        logger.info(f"✅ CLIP Normalized Trainer V2 초기화 완료:")
        logger.info(f"   - learning_rate: {learning_rate}")
        logger.info(f"   - weight_decay: {weight_decay}")
    
    def train_step(self, batch):
        """훈련 스텝"""
        self.model.train()
        
        images = batch['image']  # PIL 이미지 리스트
        actions = batch['action'].to(self.device)
        texts = batch['text']
        
        # 순전파
        predicted_actions = self.model(images, texts)
        
        # 손실 계산
        loss = self.criterion(predicted_actions, actions)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        """검증 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image']  # PIL 이미지 리스트
            actions = batch['action'].to(self.device)
            texts = batch['text']
            
            predicted_actions = self.model(images, texts)
            loss = self.criterion(predicted_actions, actions)
            
            # MAE 계산
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            return loss.item(), mae.item()
    
    def save_checkpoint(self, path, epoch, loss, mae):
        """체크포인트 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'mae': mae,
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'action_dim': self.model.action_dim,
                'dropout': self.model.dropout_rate,
                'use_clip_normalization': self.model.use_clip_normalization,
                'use_vision_resampler': self.model.use_vision_resampler
            }
        }, path)
        
        logger.info(f"💾 체크포인트 저장: {path}")

def create_clip_normalized_model_v2(processor, device, use_clip_normalization=True, use_vision_resampler=True):
    """CLIP Normalization 모델 생성 V2"""
    
    model = CLIPNormalized2DActionModelV2(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,  # Kosmos2 text 모델 출력 차원
        action_dim=2,
        hidden_dim=256,
        dropout=0.4,
        use_vision_resampler=use_vision_resampler,
        use_clip_normalization=use_clip_normalization
    )
    
    trainer = CLIPNormalizedTrainerV2(
        model=model,
        device=device,
        learning_rate=3e-5,
        weight_decay=1e-3
    )
    
    return model, trainer

if __name__ == "__main__":
    # 테스트 코드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model, trainer = create_clip_normalized_model_v2(processor, device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 모델 정보:")
    logger.info(f"   - 총 파라미터: {total_params:,}")
    logger.info(f"   - 훈련 가능 파라미터: {trainable_params:,}")
    logger.info(f"   - 모델 크기: {total_params * 4 / 1024 / 1024:.2f} MB")
