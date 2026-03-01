#!/usr/bin/env python3
"""
Case 3: 중기 적용 - 간단한 고급 모델 (Case 1 기반)
목표: MAE 0.3 → 0.2, 정확도 35% → 55%
특징: Case 1의 안정적인 구조 사용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimpleCase3Model(nn.Module):
    """
    Case 3 모델 - Case 1 기반
    - Case 1의 안정적인 구조 사용
    - 2D 액션 출력
    """
    
    def __init__(self, processor, vision_dim=1024, language_dim=2048, action_dim=2, 
                 hidden_dim=256, dropout=0.4, use_vision_resampler=False):
        super().__init__()
        
        self.processor = processor
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout_rate = dropout
        self.use_vision_resampler = use_vision_resampler
        
        # 단순화된 특징 어댑터 (hidden_dim 256으로 감소)
        self.feature_adapter = nn.Linear(vision_dim, hidden_dim)
        self.language_adapter = nn.Linear(language_dim, hidden_dim)  # 2048 → 256
        
        # 정규화 강화 (dropout 0.4)
        self.layer_norm_vision = nn.LayerNorm(hidden_dim)
        self.layer_norm_language = nn.LayerNorm(hidden_dim)
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        
        # 개선된 액션 헤드 (더 깊은 네트워크)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 추가 정규화 레이어
        self.final_norm = nn.LayerNorm(action_dim)
        
        logger.info(f"✅ Case 3 Model 초기화 완료:")
        logger.info(f"   - hidden_dim: {hidden_dim}")
        logger.info(f"   - action_dim: {action_dim}")
        logger.info(f"   - dropout: {dropout}")
        logger.info(f"   - action_head layers: 4")
    
    def extract_vision_features(self, images):
        """시각 특징 추출 (Case 1과 동일)"""
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
        
        # 단순화된 특징 처리
        vision_features = self.feature_adapter(vision_features)
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, texts):
        """언어 특징 추출 (Case 1과 동일)"""
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
        
        # 단순화된 특징 처리
        language_features = self.language_adapter(language_features)
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, images, texts):
        """순전파 (Case 1과 동일)"""
        # 특징 추출
        vision_features = self.extract_vision_features(images)
        language_features = self.extract_language_features(texts)
        
        # 특징 결합
        combined_features = torch.cat([vision_features, language_features], dim=1)
        
        # 단순화된 액션 예측 (4층)
        actions = self.action_head(combined_features)
        actions = self.final_norm(actions)
        
        return actions

class SimpleCase3Trainer:
    """Case 3 훈련기 - Case 1과 동일"""
    
    def __init__(self, model, device, learning_rate=5e-5, weight_decay=1e-3):
        self.model = model.to(device)
        self.device = device
        
        # 최적화된 하이퍼파라미터
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 코사인 어닐링 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Huber Loss (이상치에 강함)
        self.criterion = nn.HuberLoss(delta=0.1)
        
        logger.info(f"✅ Case 3 Trainer 초기화 완료:")
        logger.info(f"   - learning_rate: {learning_rate}")
        logger.info(f"   - weight_decay: {weight_decay}")
        logger.info(f"   - scheduler: CosineAnnealingLR")
        logger.info(f"   - criterion: HuberLoss")
    
    def train_step(self, batch):
        """단일 훈련 스텝"""
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
        """단일 검증 스텝"""
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
            'mae': mae
        }, path)
        logger.info(f"💾 체크포인트 저장: {path}")
    
    def load_checkpoint(self, path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"📂 체크포인트 로드: {path}")
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['mae']

def create_simple_case3_model(processor, device=None):
    """Case 3 모델 생성"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCase3Model(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,
        action_dim=2,
        hidden_dim=256,
        dropout=0.4
    )
    
    trainer = SimpleCase3Trainer(
        model=model,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-3
    )
    
    return model, trainer
