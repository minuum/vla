#!/usr/bin/env python3
"""
Korean Text Encoder - 한국어 네비게이션 명령어 인코딩
mobile_vla_data_collector.py의 시나리오별 한국어 명령어 처리
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KoreanTextEncoder(nn.Module):
    """
    한국어 텍스트 인코더 (Mobile VLA 시나리오별 명령어 특화)
    - KLUE RoBERTa 기반 한국어 이해
    - 시나리오별 명령어 매핑
    - mobile_vla_data_collector.py 시나리오와 완전 호환
    """
    
    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        hidden_size: int = 768,
        max_length: int = 128,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 한국어 토크나이저 및 모델 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
            logger.info(f"✅ 한국어 모델 로드 완료: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ KLUE 모델 로드 실패, DistilBERT로 대체: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            self.text_encoder = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        
        # 인코더 가중치 고정 옵션
        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("🔒 텍스트 인코더 가중치 고정됨")
        
        # mobile_vla_data_collector.py 시나리오별 한국어 명령어
        self.scenario_instructions = {
            "1box_vert_left": "박스를 왼쪽으로 돌아서 컵까지 가세요",
            "1box_vert_right": "박스를 오른쪽으로 돌아서 컵까지 가세요", 
            "1box_hori_left": "박스를 왼쪽으로 피해서 컵까지 가세요",
            "1box_hori_right": "박스를 오른쪽으로 피해서 컵까지 가세요",
            "2box_vert_left": "두 박스 사이 왼쪽 경로로 컵까지 가세요",
            "2box_vert_right": "두 박스 사이 오른쪽 경로로 컵까지 가세요",
            "2box_hori_left": "두 박스를 왼쪽으로 우회해서 컵까지 가세요", 
            "2box_hori_right": "두 박스를 오른쪽으로 우회해서 컵까지 가세요"
        }
        
        # 시나리오 임베딩 (8가지 시나리오)
        self.scenario_embedding = nn.Embedding(8, hidden_size)
        
        # 시나리오 ID 매핑
        self.scenario_to_id = {
            scenario: idx for idx, scenario in enumerate(self.scenario_instructions.keys())
        }
        
        # 텍스트 특징 프로젝션 (KLUE RoBERTa: 768 → hidden_size)
        encoder_dim = self.text_encoder.config.hidden_size
        if encoder_dim != hidden_size:
            self.text_projection = nn.Linear(encoder_dim, hidden_size)
        else:
            self.text_projection = nn.Identity()
        
        # 시나리오와 텍스트 융합
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"🗣️ Korean Text Encoder 초기화 완료 (Hidden: {hidden_size})")
    
    def forward(
        self, 
        instructions: List[str], 
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            instructions: 한국어 명령어 리스트 ["박스를 왼쪽으로...", ...]
            scenarios: 시나리오 이름 리스트 ["1box_vert_left", ...] (옵션)
            
        Returns:
            dict with:
                - text_features: [B, seq_len, hidden_size] - 텍스트 특징
                - scenario_features: [B, hidden_size] - 시나리오 특징 (scenarios 제공시)
                - fused_features: [B, hidden_size] - 융합된 특징
        """
        batch_size = len(instructions)
        device = next(self.parameters()).device
        
        # 1. 텍스트 토크나이징
        tokenized = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        # 2. 텍스트 인코딩
        with torch.cuda.amp.autocast(enabled=True):
            text_outputs = self.text_encoder(**tokenized)
        
        # 텍스트 특징 추출 및 프로젝션
        text_features = text_outputs.last_hidden_state  # [B, seq_len, encoder_dim]
        text_features = self.text_projection(text_features)  # [B, seq_len, hidden_size]
        
        # 텍스트 풀링 (평균)
        attention_mask = tokenized['attention_mask'].unsqueeze(-1)  # [B, seq_len, 1]
        text_pooled = (text_features * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # [B, hidden_size]
        
        result = {
            "text_features": text_features,
            "text_pooled": text_pooled
        }
        
        # 3. 시나리오 인코딩 (제공된 경우)
        if scenarios is not None:
            scenario_ids = []
            for scenario in scenarios:
                scenario_id = self.scenario_to_id.get(scenario, 0)  # unknown은 0번
                scenario_ids.append(scenario_id)
            
            scenario_ids = torch.tensor(scenario_ids, device=device)  # [B]
            scenario_features = self.scenario_embedding(scenario_ids)  # [B, hidden_size]
            
            # 4. 텍스트-시나리오 융합
            # 시나리오를 쿼리로, 텍스트를 키-밸류로 사용
            scenario_query = scenario_features.unsqueeze(1)  # [B, 1, hidden_size]
            fused_features, attention_weights = self.fusion_layer(
                query=scenario_query,      # [B, 1, hidden_size]
                key=text_features,         # [B, seq_len, hidden_size]
                value=text_features,       # [B, seq_len, hidden_size]
                key_padding_mask=~tokenized['attention_mask'].bool()  # 패딩 마스크
            )
            
            fused_features = fused_features.squeeze(1)  # [B, hidden_size]
            fused_features = self.layer_norm(fused_features)
            
            result.update({
                "scenario_features": scenario_features,
                "fused_features": fused_features,
                "attention_weights": attention_weights
            })
        else:
            # 시나리오 없이는 텍스트 풀링만 사용
            result["fused_features"] = self.layer_norm(text_pooled)
        
        return result
    
    def encode_scenarios_only(self, scenarios: List[str]) -> torch.Tensor:
        """시나리오만으로 임베딩 생성 (빠른 추론용)"""
        device = next(self.parameters()).device
        
        scenario_ids = []
        for scenario in scenarios:
            scenario_id = self.scenario_to_id.get(scenario, 0)
            scenario_ids.append(scenario_id)
        
        scenario_ids = torch.tensor(scenario_ids, device=device)
        scenario_features = self.scenario_embedding(scenario_ids)
        
        return scenario_features
    
    def get_instruction_for_scenario(self, scenario: str) -> str:
        """시나리오에 대응하는 한국어 명령어 반환"""
        return self.scenario_instructions.get(scenario, "컵까지 가세요")
    
    def batch_encode_scenarios(self, scenarios: List[str]) -> Dict[str, torch.Tensor]:
        """시나리오 배치를 한국어 명령어로 변환하여 인코딩"""
        instructions = [self.get_instruction_for_scenario(scenario) for scenario in scenarios]
        return self.forward(instructions, scenarios)


class KoreanTextEncoderLite(nn.Module):
    """
    경량화된 한국어 텍스트 인코더 (Jetson용)
    사전 정의된 시나리오 임베딩만 사용
    """
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 시나리오별 사전 정의된 임베딩 (학습 가능)
        self.scenario_embedding = nn.Embedding(8, hidden_size)
        
        # mobile_vla_data_collector.py 시나리오 매핑
        self.scenario_to_id = {
            "1box_vert_left": 0, "1box_vert_right": 1,
            "1box_hori_left": 2, "1box_hori_right": 3,
            "2box_vert_left": 4, "2box_vert_right": 5,
            "2box_hori_left": 6, "2box_hori_right": 7
        }
        
        logger.info(f"🚀 Korean Text Encoder Lite 초기화 (Hidden: {hidden_size})")
    
    def forward(self, scenarios: List[str]) -> torch.Tensor:
        """시나리오 이름만으로 임베딩 생성"""
        device = next(self.parameters()).device
        
        scenario_ids = []
        for scenario in scenarios:
            scenario_id = self.scenario_to_id.get(scenario, 0)
            scenario_ids.append(scenario_id)
        
        scenario_ids = torch.tensor(scenario_ids, device=device)
        scenario_features = self.scenario_embedding(scenario_ids)
        
        return scenario_features


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Korean Text Encoder 테스트")
    
    # 모델 초기화
    encoder = KoreanTextEncoder(hidden_size=768)
    encoder_lite = KoreanTextEncoderLite(hidden_size=512)
    
    # 테스트 데이터
    test_instructions = [
        "박스를 왼쪽으로 돌아서 컵까지 가세요",
        "두 박스 사이 오른쪽 경로로 컵까지 가세요"
    ]
    test_scenarios = ["1box_vert_left", "2box_vert_right"]
    
    print(f"📊 입력 명령어: {test_instructions}")
    print(f"🎯 입력 시나리오: {test_scenarios}")
    
    # 일반 인코더 테스트
    with torch.no_grad():
        result = encoder(test_instructions, test_scenarios)
        print(f"🗣️ 텍스트 특징: {result['text_features'].shape}")
        print(f"🎯 시나리오 특징: {result['scenario_features'].shape}")
        print(f"🔄 융합 특징: {result['fused_features'].shape}")
        
        # Lite 인코더 테스트
        lite_result = encoder_lite(test_scenarios)
        print(f"🚀 Lite 특징: {lite_result.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in encoder.parameters())
    lite_params = sum(p.numel() for p in encoder_lite.parameters())
    
    print(f"📊 일반 인코더 파라미터: {total_params:,}개 ({total_params/1e6:.1f}M)")
    print(f"🚀 Lite 인코더 파라미터: {lite_params:,}개 ({lite_params/1e6:.1f}M)")
    print(f"💡 파라미터 감소율: {(1 - lite_params/total_params)*100:.1f}%")
