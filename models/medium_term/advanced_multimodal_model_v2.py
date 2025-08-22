import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor
import open_clip
import numpy as np
from typing import List, Tuple, Optional

class AdvancedVisionResampler(nn.Module):
    """고급 비전 리샘플러 - 더 정교한 토큰 압축"""
    
    def __init__(self, input_dim=1024, output_dim=256, num_tokens=64, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        # 학습 가능한 쿼리 토큰들
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim))
        
        # 멀티레이어 크로스 어텐션
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 피드포워드 네트워크
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 4, output_dim)
            ) for _ in range(num_layers)
        ])
        
        # 레이어 정규화
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers * 2)
        ])
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, vision_features):
        """
        Args:
            vision_features: (batch_size, seq_len, input_dim)
        Returns:
            resampled_features: (batch_size, num_tokens, output_dim)
        """
        batch_size = vision_features.shape[0]
        
        # 입력을 출력 차원으로 프로젝션
        projected_features = self.input_projection(vision_features)  # (B, seq_len, output_dim)
        
        # 쿼리 토큰을 배치 크기에 맞게 확장
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_tokens, output_dim)
        
        # 멀티레이어 크로스 어텐션
        for i in range(len(self.cross_attention_layers)):
            # 크로스 어텐션
            attn_output, _ = self.cross_attention_layers[i](
                query_tokens, projected_features, projected_features
            )
            query_tokens = self.norm_layers[i*2](query_tokens + attn_output)
            
            # 피드포워드 네트워크
            ffn_output = self.ffn_layers[i](query_tokens)
            query_tokens = self.norm_layers[i*2+1](query_tokens + ffn_output)
        
        return query_tokens

class EnhancedClawMatrix(nn.Module):
    """향상된 Claw Matrix - 더 정교한 멀티모달 융합"""
    
    def __init__(self, vision_dim=256, language_dim=256, action_dim=2, hidden_dim=512):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 비전-언어 어텐션
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        
        # 언어-액션 어텐션
        self.language_action_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        
        # 액션-비전 어텐션
        self.action_vision_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        
        # 프로젝션 레이어들
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.language_projection = nn.Linear(language_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features, language_features, action_features):
        """
        Args:
            vision_features: (batch_size, vision_seq_len, vision_dim)
            language_features: (batch_size, language_seq_len, language_dim)
            action_features: (batch_size, action_seq_len, action_dim)
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        # 프로젝션
        vision_proj = self.vision_projection(vision_features)
        language_proj = self.language_projection(language_features)
        action_proj = self.action_projection(action_features)
        
        # 비전-언어 어텐션
        vl_output, _ = self.vision_language_attention(
            vision_proj, language_proj, language_proj
        )
        vl_output = self.norm1(vision_proj + vl_output)
        
        # 언어-액션 어텐션
        la_output, _ = self.language_action_attention(
            language_proj, action_proj, action_proj
        )
        la_output = self.norm2(language_proj + la_output)
        
        # 액션-비전 어텐션
        av_output, _ = self.action_vision_attention(
            action_proj, vision_proj, vision_proj
        )
        av_output = self.norm3(action_proj + av_output)
        
        # 평균 풀링
        vl_pooled = vl_output.mean(dim=1)  # (batch_size, hidden_dim)
        la_pooled = la_output.mean(dim=1)  # (batch_size, hidden_dim)
        av_pooled = av_output.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 융합
        fused = torch.cat([vl_pooled, la_pooled, av_pooled], dim=1)
        output = self.output_projection(fused)
        
        return output

class HierarchicalPlanner(nn.Module):
    """계층적 계획 모듈 - 고수준 목표를 하위 목표로 분해"""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_subgoals=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_subgoals = num_subgoals
        
        # 목표 분해 네트워크
        self.goal_decomposer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * num_subgoals)
        )
        
        # 하위 목표 처리 네트워크
        self.subgoal_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 최종 액션 생성기
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_dim * num_subgoals, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, 2)  # 2D 액션
        )
        
    def forward(self, fused_features):
        """
        Args:
            fused_features: (batch_size, input_dim)
        Returns:
            actions: (batch_size, 2)
        """
        # 목표 분해
        subgoals = self.goal_decomposer(fused_features)  # (B, hidden_dim * num_subgoals)
        subgoals = subgoals.view(-1, self.num_subgoals, self.hidden_dim)  # (B, num_subgoals, hidden_dim)
        
        # 하위 목표 처리
        processed_subgoals = []
        for i in range(self.num_subgoals):
            subgoal = subgoals[:, i, :]  # (B, hidden_dim)
            processed = self.subgoal_processor(subgoal)  # (B, hidden_dim)
            processed_subgoals.append(processed)
        
        # 하위 목표 결합
        combined_subgoals = torch.cat(processed_subgoals, dim=1)  # (B, hidden_dim * num_subgoals)
        
        # 액션 생성
        actions = self.action_generator(combined_subgoals)  # (B, 2)
        
        return actions

class AdvancedMultimodalModelV2(nn.Module):
    """고급 멀티모달 모델 V2 - 언어 차원 수정"""
    
    def __init__(self, processor, vision_dim=1024, language_dim=2048, action_dim=2,
                 hidden_dim=512, dropout=0.2, use_hierarchical_planning=True):
        super().__init__()
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_hierarchical_planning = use_hierarchical_planning
        
        # Kosmos2 모델 로드
        from transformers import AutoModel
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        if torch.cuda.is_available():
            self.kosmos = self.kosmos.cuda()
        
        # 고급 비전 리샘플러
        self.vision_resampler = AdvancedVisionResampler(
            input_dim=vision_dim,
            output_dim=256,
            num_tokens=64,
            num_layers=3
        )
        
        # 향상된 Claw Matrix
        self.claw_matrix = EnhancedClawMatrix(
            vision_dim=256,
            language_dim=256,
            action_dim=2,
            hidden_dim=hidden_dim
        )
        
        # 계층적 계획 모듈
        if use_hierarchical_planning:
            self.hierarchical_planner = HierarchicalPlanner(
                input_dim=hidden_dim,
                hidden_dim=256,
                num_subgoals=3
            )
        
        # 언어 어댑터
        self.language_adapter = nn.Linear(language_dim, 256)  # 2048 → 256
        
        # 액션 임베딩
        self.action_embedding = nn.Linear(action_dim, 256)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def extract_vision_features(self, images):
        """시각 특징 추출"""
        batch_size = len(images)  # PIL 이미지 리스트
        device = next(self.parameters()).device
        
        # Kosmos2로 시각 특징 추출
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            vision_outputs = self.kosmos(**inputs)
            vision_features = vision_outputs.last_hidden_state  # (B, seq_len, vision_dim)
        
        # 비전 리샘플러 적용
        resampled_features = self.vision_resampler(vision_features)  # (B, 64, 256)
        
        return resampled_features
    
    def extract_language_features(self, texts):
        """언어 특징 추출"""
        batch_size = len(texts)
        device = next(self.parameters()).device
        
        # Kosmos2로 언어 특징 추출
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if 'input_ids' in inputs:
                text_outputs = self.kosmos.text_model(inputs['input_ids'])
                language_features = text_outputs.last_hidden_state.mean(dim=1)  # (B, 2048)
            else:
                language_features = torch.zeros(batch_size, 2048).to(device)
        
        # 언어 어댑터 적용
        adapted_language = self.language_adapter(language_features)  # (B, 256)
        adapted_language = adapted_language.unsqueeze(1)  # (B, 1, 256)
        
        return adapted_language
    
    def forward(self, images, texts):
        """
        Args:
            images: PIL 이미지 리스트
            texts: 텍스트 리스트
        Returns:
            actions: (batch_size, 2)
        """
        batch_size = len(images)
        device = next(self.parameters()).device
        
        # 특징 추출
        vision_features = self.extract_vision_features(images)  # (B, 64, 256)
        language_features = self.extract_language_features(texts)  # (B, 1, 256)
        
        # 더미 액션 특징 (훈련 시에는 실제 액션으로 대체)
        dummy_actions = torch.zeros(batch_size, 1, 2).to(device)  # (B, 1, 2)
        action_features = self.action_embedding(dummy_actions)  # (B, 1, 256)
        
        # Claw Matrix로 융합
        fused_features = self.claw_matrix(
            vision_features, language_features, action_features
        )  # (B, hidden_dim)
        
        fused_features = self.dropout(fused_features)
        
        # 계층적 계획 또는 직접 액션 생성
        if self.use_hierarchical_planning:
            actions = self.hierarchical_planner(fused_features)  # (B, 2)
        else:
            # 간단한 MLP
            actions = torch.tanh(fused_features[:, :2])  # (B, 2)
        
        return actions

class AdvancedMultimodalTrainerV2:
    """고급 멀티모달 트레이너 V2"""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_step(self, batch):
        """훈련 스텝"""
        self.model.train()
        
        images = batch['image']  # PIL 이미지 리스트
        actions = batch['action'].to(self.device)
        texts = batch['text']
        
        self.optimizer.zero_grad()
        
        # 순전파
        predicted_actions = self.model(images, texts)
        
        # 손실 계산
        loss = self.criterion(predicted_actions, actions)
        
        # 역전파
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        """검증 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            images = batch['image']  # PIL 이미지 리스트
            actions = batch['action'].to(self.device)
            texts = batch['text']
            
            # 순전파
            predicted_actions = self.model(images, texts)
            
            # 손실 계산
            loss = self.criterion(predicted_actions, actions)
            
            # MAE 계산
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            return loss.item(), mae.item()

def create_advanced_multimodal_model_v2(processor, device, **kwargs):
    """고급 멀티모달 모델 V2 생성"""
    model = AdvancedMultimodalModelV2(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,  # Kosmos2 text 모델 출력 차원
        action_dim=2,
        hidden_dim=512,
        dropout=0.3,
        use_hierarchical_planning=True,
        **kwargs
    ).to(device)
    
    trainer = AdvancedMultimodalTrainerV2(model, device)
    
    return model, trainer
