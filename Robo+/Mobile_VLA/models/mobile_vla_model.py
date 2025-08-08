#!/usr/bin/env python3
"""
Mobile VLA Model - Pure Mobile VLM without Calvin dependencies
RoboVLMs의 VLM 기술을 mobile_vla_data_collector.py에 완전 적응
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .encoders.mobile_image_encoder import MobileImageEncoder, MobileImageEncoderLite
    from .encoders.korean_text_encoder import KoreanTextEncoder, KoreanTextEncoderLite
    from .policy_heads.mobile_policy_head import MobilePolicyHead, MobilePolicyHeadLite
except ImportError:
    # 테스트용 절대 임포트
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from encoders.mobile_image_encoder import MobileImageEncoder, MobileImageEncoderLite
    from encoders.korean_text_encoder import KoreanTextEncoder, KoreanTextEncoderLite
    from policy_heads.mobile_policy_head import MobilePolicyHead, MobilePolicyHeadLite

logger = logging.getLogger(__name__)


class MobileVLAModel(nn.Module):
    """
    Pure Mobile VLA Model
    - 입력: mobile_vla_data_collector.py 데이터 형식
    - 출력: 3D 액션 + 이벤트 예측
    - Calvin 의존성 없는 순수 Mobile VLM
    """
    
    def __init__(
        self,
        # 모델 크기 설정
        hidden_size: int = 768,
        
        # 이미지 인코더 설정
        image_backbone: str = "efficientnet_v2_s",
        freeze_image_backbone: bool = False,
        
        # 텍스트 인코더 설정  
        text_model: str = "klue/roberta-base",
        freeze_text_encoder: bool = False,
        
        # 정책 헤드 설정
        use_policy_lstm: bool = True,
        policy_lstm_layers: int = 2,
        
        # 일반 설정
        dropout: float = 0.1,
        use_lite_mode: bool = False  # Jetson용 경량화 모드
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_lite_mode = use_lite_mode
        
        # 경량화 모드에 따른 컴포넌트 선택
        if use_lite_mode:
            # Jetson용 경량화 모델
            self.image_encoder = MobileImageEncoderLite(
                hidden_size=hidden_size // 2,  # 더 작은 hidden_size
                dropout=dropout
            )
            self.text_encoder = KoreanTextEncoderLite(
                hidden_size=hidden_size // 2
            )
            self.policy_head = MobilePolicyHeadLite(
                hidden_size=hidden_size,  # 융합 후에는 원래 크기
                dropout=dropout
            )
            logger.info("🚀 Lite 모드로 초기화됨 (Jetson 최적화)")
        else:
            # 일반 고성능 모델
            self.image_encoder = MobileImageEncoder(
                backbone=image_backbone,
                hidden_size=hidden_size,
                dropout=dropout,
                freeze_backbone=freeze_image_backbone
            )
            self.text_encoder = KoreanTextEncoder(
                model_name=text_model,
                hidden_size=hidden_size,
                freeze_encoder=freeze_text_encoder
            )
            self.policy_head = MobilePolicyHead(
                hidden_size=hidden_size,
                dropout=dropout,
                use_lstm=use_policy_lstm,
                lstm_layers=policy_lstm_layers
            )
            logger.info("💪 Full 모드로 초기화됨 (고성능)")
        
        # 멀티모달 융합 레이어
        if use_lite_mode:
            # 경량화된 융합
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 어텐션 기반 융합
            self.multimodal_fusion = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # 출력 정규화
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # 모델 통계
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🤖 Mobile VLA Model 초기화 완료")
        logger.info(f"   파라미터: {total_params:,}개 ({total_params/1e6:.1f}M)")
        logger.info(f"   Hidden Size: {hidden_size}, Lite Mode: {use_lite_mode}")
    
    def forward(
        self,
        images: torch.Tensor,
        scenarios: List[str],
        instructions: Optional[List[str]] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, T, 3, 224, 224] - 이미지 시퀀스  
            scenarios: List[str] - 시나리오 이름들 ["1box_vert_left", ...]
            instructions: List[str] - 한국어 명령어 (옵션, scenarios에서 자동 생성 가능)
            return_intermediate: 중간 특징들 반환 여부
            
        Returns:
            Dict with:
                - actions: [B, T, 3] - 정규화된 액션
                - actions_denorm: [B, T, 3] - 실제 범위 액션
                - event_logits: [B, T, 3] - 이벤트 분류 로짓
                - predicted_events: [B, T] - 예측된 이벤트
        """
        batch_size = images.shape[0]
        
        # 1. 이미지 인코딩
        image_features = self.image_encoder(images)  # [B, T, hidden_size//2 or hidden_size]
        
        # 2. 텍스트 인코딩
        if self.use_lite_mode:
            # Lite 모드: 시나리오만 사용
            text_features = self.text_encoder(scenarios)  # [B, hidden_size//2]
            # 시간 차원으로 확장
            text_features = text_features.unsqueeze(1).repeat(1, images.shape[1], 1)  # [B, T, hidden_size//2]
        else:
            # Full 모드: 한국어 명령어 사용
            if instructions is None:
                # 시나리오에서 한국어 명령어 자동 생성
                instructions = [
                    self.text_encoder.get_instruction_for_scenario(scenario) 
                    for scenario in scenarios
                ]
            
            text_result = self.text_encoder(instructions, scenarios)
            text_features = text_result["fused_features"]  # [B, hidden_size]
            # 시간 차원으로 확장
            text_features = text_features.unsqueeze(1).repeat(1, images.shape[1], 1)  # [B, T, hidden_size]
        
        # 3. 멀티모달 융합
        if self.use_lite_mode:
            # 경량화된 융합: 단순 concatenation + MLP
            multimodal_features = torch.cat([image_features, text_features], dim=-1)  # [B, T, hidden_size]
            multimodal_features = self.multimodal_fusion(multimodal_features)
        else:
            # 어텐션 기반 융합
            # 이미지를 쿼리로, 텍스트를 키-밸류로 사용
            fused_features, attention_weights = self.multimodal_fusion(
                query=image_features,     # [B, T, hidden_size]
                key=text_features,        # [B, T, hidden_size]  
                value=text_features       # [B, T, hidden_size]
            )
            multimodal_features = fused_features
        
        # 출력 정규화
        multimodal_features = self.output_norm(multimodal_features)  # [B, T, hidden_size]
        
        # 4. 정책 헤드로 액션 예측
        policy_output = self.policy_head(multimodal_features)
        
        # 결과 구성
        result = {
            "actions": policy_output["actions"],                    # [B, T, 3] 정규화된
            "actions_denorm": policy_output["actions_denorm"],      # [B, T, 3] 실제 범위
            "event_logits": policy_output["event_logits"],          # [B, T, 3]
            "event_probs": policy_output.get("event_probs"),        # [B, T, 3]
            "predicted_events": policy_output["predicted_events"]   # [B, T]
        }
        
        # 중간 특징들 (디버깅/분석용)
        if return_intermediate:
            result.update({
                "image_features": image_features,
                "text_features": text_features,
                "multimodal_features": multimodal_features,
                "attention_weights": attention_weights if not self.use_lite_mode else None
            })
        
        return result
    
    def inference_single_step(
        self,
        current_image: torch.Tensor,
        scenario: str,
        hidden_state: Optional[Tuple] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple]]:
        """
        단일 스텝 추론 (실시간 mobile_vla_data_collector 연동용)
        
        Args:
            current_image: [1, 3, 224, 224] - 현재 이미지
            scenario: str - 현재 시나리오
            hidden_state: LSTM hidden state (있다면)
            
        Returns:
            (액션 예측 결과, 새로운 hidden_state)
        """
        with torch.no_grad():
            # 단일 이미지를 시퀀스로 확장
            image_sequence = current_image.unsqueeze(1)  # [1, 1, 3, 224, 224]
            
            # 인코딩
            image_features = self.image_encoder.extract_spatial_features(current_image)  # [1, hidden_size]
            
            if self.use_lite_mode:
                text_features = self.text_encoder([scenario])  # [1, hidden_size//2]
                # 이미지 특징과 텍스트 특징의 차원을 맞춤
                image_features_lite = image_features  # [1, hidden_size//2] (256)
                multimodal_features = torch.cat([image_features_lite, text_features], dim=-1)  # [1, 512]
                multimodal_features = self.multimodal_fusion(multimodal_features)
            else:
                instruction = self.text_encoder.get_instruction_for_scenario(scenario)
                text_result = self.text_encoder([instruction], [scenario])
                text_features = text_result["fused_features"]  # [1, hidden_size]
                
                # 간단한 융합 (어텐션 없이)
                multimodal_features = (image_features + text_features) / 2
            
            multimodal_features = self.output_norm(multimodal_features)
            
            # 정책 헤드로 단일 스텝 예측
            if hasattr(self.policy_head, 'predict_single_step') and not self.use_lite_mode:
                action_result, new_hidden_state = self.policy_head.predict_single_step(
                    multimodal_features, hidden_state
                )
            else:
                action_result = self.policy_head(multimodal_features.unsqueeze(1))
                # 단일 스텝으로 압축
                for key in action_result:
                    if action_result[key].dim() > 1:
                        action_result[key] = action_result[key].squeeze(1)
                new_hidden_state = None
        
        return action_result, new_hidden_state
    
    def get_mobile_vla_action(
        self,
        current_image: torch.Tensor,
        scenario: str
    ) -> Dict[str, float]:
        """
        mobile_vla_data_collector.py 호환 액션 반환
        
        Returns:
            Dict with keys: linear_x, linear_y, angular_z, event_type
        """
        action_result, _ = self.inference_single_step(current_image, scenario)
        
        # 액션 추출 (첫 번째 배치)
        actions = action_result["actions_denorm"][0].cpu().numpy()  # [3]
        predicted_event = action_result["predicted_events"][0].cpu().item()
        
        # 이벤트 타입 매핑
        event_types = ["episode_start", "start_action", "stop_action"]
        event_type = event_types[predicted_event]
        
        return {
            "linear_x": float(actions[0]),
            "linear_y": float(actions[1]),
            "angular_z": float(actions[2]),
            "event_type": event_type
        }


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile VLA Model 테스트")
    
    # 모델 초기화 (Full & Lite)
    model_full = MobileVLAModel(hidden_size=768, use_lite_mode=False)
    model_lite = MobileVLAModel(hidden_size=512, use_lite_mode=True)
    
    # 테스트 데이터 (mobile_vla_data_collector.py 형식)
    batch_size, seq_len = 2, 18
    test_images = torch.randn(batch_size, seq_len, 3, 224, 224)
    test_scenarios = ["1box_vert_left", "2box_hori_right"]
    
    print(f"📊 입력 이미지: {test_images.shape}")
    print(f"🎯 입력 시나리오: {test_scenarios}")
    
    # Full 모델 테스트
    print("\n💪 Full Model 테스트:")
    with torch.no_grad():
        result_full = model_full(test_images, test_scenarios, return_intermediate=True)
        print(f"   액션: {result_full['actions'].shape}")
        print(f"   실제 액션: {result_full['actions_denorm'].shape}")
        print(f"   이벤트: {result_full['predicted_events'].shape}")
    
    # Lite 모델 테스트  
    print("\n🚀 Lite Model 테스트:")
    with torch.no_grad():
        result_lite = model_lite(test_images, test_scenarios)
        print(f"   액션: {result_lite['actions'].shape}")
        print(f"   실제 액션: {result_lite['actions_denorm'].shape}")
        print(f"   이벤트: {result_lite['predicted_events'].shape}")
    
    # 단일 스텝 추론 테스트
    print("\n🔄 단일 스텝 추론 테스트:")
    single_image = torch.randn(1, 3, 224, 224)
    scenario = "1box_vert_left"
    
    mobile_action = model_full.get_mobile_vla_action(single_image, scenario)
    print(f"   Mobile 액션: {mobile_action}")
    
    # 파라미터 수 비교
    full_params = sum(p.numel() for p in model_full.parameters())
    lite_params = sum(p.numel() for p in model_lite.parameters())
    
    print(f"\n📊 모델 크기 비교:")
    print(f"   Full 모델: {full_params:,}개 ({full_params/1e6:.1f}M)")
    print(f"   Lite 모델: {lite_params:,}개 ({lite_params/1e6:.1f}M)")
    print(f"   경량화율: {(1 - lite_params/full_params)*100:.1f}%")
