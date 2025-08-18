#!/usr/bin/env python3
"""
Mobile Policy Head - mobile_vla_data_collector.py의 3D 액션 예측 특화
RoboVLMs의 정책 헤드를 Mobile VLA 액션 공간에 적응
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MobilePolicyHead(nn.Module):
    """
    Mobile VLA 특화 정책 헤드
    - 입력: [B, T, hidden_size] 멀티모달 특징
    - 출력: [B, T, 3] mobile_vla_data_collector.py 호환 액션
    - 액션: [linear_x, linear_y, angular_z] + 이벤트 예측
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        action_dim: int = 3,  # mobile_vla_data_collector.py 액션 차원
        num_event_types: int = 3,  # episode_start, start_action, stop_action
        dropout: float = 0.1,
        use_lstm: bool = True,
        lstm_layers: int = 2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.num_event_types = num_event_types
        
        # mobile_vla_data_collector.py의 액션 범위 (WASD_TO_CONTINUOUS 기준)
        self.action_bounds = {
            "linear_x": 2.0,   # 실제 ±1.15, 여유있게 ±2.0
            "linear_y": 2.0,   # 실제 ±1.15, 여유있게 ±2.0
            "angular_z": 2.0   # 실제 ±1.15, 여유있게 ±2.0
        }
        
        # 시간적 의존성을 위한 LSTM (옵션)
        if use_lstm:
            self.temporal_encoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=False
            )
        else:
            self.temporal_encoder = None
        
        # 연속 액션 예측 헤드 (linear_x, linear_y, angular_z)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh()  # [-1, 1] 범위로 정규화
        )
        
        # 이벤트 타입 예측 헤드 (episode_start, start_action, stop_action)
        self.event_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_event_types)  # 분류 로짓
        )
        
        # 액션 값 분포 예측 (불확실성 추정용)
        self.action_variance_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softplus()  # 양수 분산
        )
        
        # 출력 레이어 정규화
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"🎯 Mobile Policy Head 초기화 완료")
        logger.info(f"   액션 차원: {action_dim}, 이벤트 타입: {num_event_types}")
        logger.info(f"   LSTM 사용: {use_lstm}, Hidden: {hidden_size}")
    
    def forward(
        self, 
        multimodal_features: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            multimodal_features: [B, T, hidden_size] - 이미지+텍스트 융합 특징
            return_uncertainty: 불확실성 추정 포함 여부
            
        Returns:
            dict with:
                - actions: [B, T, 3] - 정규화된 액션 [-1, 1]
                - actions_denorm: [B, T, 3] - 실제 액션 범위
                - event_logits: [B, T, 3] - 이벤트 타입 로짓
                - event_probs: [B, T, 3] - 이벤트 타입 확률
                - action_variance: [B, T, 3] - 액션 불확실성 (옵션)
        """
        # 입력 정규화
        features = self.layer_norm(multimodal_features)  # [B, T, hidden_size]
        
        # 시간적 인코딩 (LSTM 사용시)
        if self.temporal_encoder is not None:
            temporal_features, (hidden, cell) = self.temporal_encoder(features)
        else:
            temporal_features = features
        
        # 연속 액션 예측 (정규화된 [-1, 1] 범위)
        normalized_actions = self.action_head(temporal_features)  # [B, T, 3]
        
        # 실제 액션 범위로 변환
        denormalized_actions = self.denormalize_actions(normalized_actions)
        
        # 이벤트 타입 예측
        event_logits = self.event_head(temporal_features)  # [B, T, 3]
        event_probs = F.softmax(event_logits, dim=-1)
        
        result = {
            "actions": normalized_actions,      # [-1, 1] 정규화된 액션
            "actions_denorm": denormalized_actions,  # 실제 범위 액션
            "event_logits": event_logits,       # 이벤트 분류 로짓
            "event_probs": event_probs,         # 이벤트 확률
            "predicted_events": torch.argmax(event_logits, dim=-1)  # [B, T] 예측된 이벤트
        }
        
        # 불확실성 추정 (옵션)
        if return_uncertainty:
            action_variance = self.action_variance_head(temporal_features)
            result["action_variance"] = action_variance
        
        return result
    
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """정규화된 액션 [-1, 1]을 실제 범위로 변환"""
        # normalized_actions: [B, T, 3] 범위 [-1, 1]
        denormalized = normalized_actions.clone()
        
        # mobile_vla_data_collector.py 범위로 스케일링
        denormalized[..., 0] = normalized_actions[..., 0] * self.action_bounds["linear_x"]    # linear_x
        denormalized[..., 1] = normalized_actions[..., 1] * self.action_bounds["linear_y"]    # linear_y
        denormalized[..., 2] = normalized_actions[..., 2] * self.action_bounds["angular_z"]   # angular_z
        
        return denormalized
    
    def convert_to_robovlms_action(self, mobile_actions: torch.Tensor) -> torch.Tensor:
        """Mobile VLA 3D 액션을 RoboVLMs 7D 액션으로 변환"""
        # mobile_actions: [B, T, 3] - [linear_x, linear_y, angular_z]
        # RoboVLMs format: [B, T, 7] - [x, y, z, rx, ry, rz, gripper]
        B, T, _ = mobile_actions.shape
        device = mobile_actions.device
        
        robovlms_actions = torch.zeros(B, T, 7, device=device, dtype=mobile_actions.dtype)
        
        # Mobile actions을 6DOF pose로 매핑 (gripper 사용 안함)
        robovlms_actions[..., 0] = mobile_actions[..., 0]  # linear_x → x
        robovlms_actions[..., 1] = mobile_actions[..., 1]  # linear_y → y  
        robovlms_actions[..., 2] = 0.0                     # z = 0 (평면 이동)
        robovlms_actions[..., 3] = 0.0                     # rx = 0 (roll)
        robovlms_actions[..., 4] = 0.0                     # ry = 0 (pitch)
        robovlms_actions[..., 5] = mobile_actions[..., 2]  # angular_z → rz (yaw)
        robovlms_actions[..., 6] = 0.0                     # gripper = 0 (사용 안함)
        
        return robovlms_actions
    
    def convert_events_to_action_mask(self, event_indices: torch.Tensor) -> torch.Tensor:
        """Mobile VLA 이벤트를 RoboVLMs action_mask로 변환"""
        # event_indices: [B, T] - [0: episode_start, 1: start_action, 2: stop_action]
        # action_mask: [B, T] - 액션이 유효한지 (1: 유효, 0: 무효)
        
        # start_action(1)만 유효한 액션으로 간주
        action_mask = (event_indices == 1).float()
        
        return action_mask
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """실제 범위 액션을 [-1, 1]로 정규화"""
        normalized = actions.clone()
        
        normalized[..., 0] = actions[..., 0] / self.action_bounds["linear_x"]    # linear_x
        normalized[..., 1] = actions[..., 1] / self.action_bounds["linear_y"]    # linear_y
        normalized[..., 2] = actions[..., 2] / self.action_bounds["angular_z"]   # angular_z
        
        # 클램핑 [-1, 1]
        normalized = torch.clamp(normalized, -1.0, 1.0)
        
        return normalized
    
    def predict_single_step(
        self, 
        single_feature: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """단일 스텝 예측 (실시간 추론용)"""
        # single_feature: [B, hidden_size] 또는 [B, 1, hidden_size]
        if single_feature.dim() == 2:
            single_feature = single_feature.unsqueeze(1)  # [B, 1, hidden_size]
        
        features = self.layer_norm(single_feature)
        
        # LSTM 스텝별 실행 (hidden_state 유지)
        new_hidden_state = None
        if self.temporal_encoder is not None:
            if hidden_state is not None:
                temporal_features, new_hidden_state = self.temporal_encoder(features, hidden_state)
            else:
                temporal_features, new_hidden_state = self.temporal_encoder(features)
        else:
            temporal_features = features
        
        # 액션 및 이벤트 예측
        normalized_actions = self.action_head(temporal_features)  # [B, 1, 3]
        denormalized_actions = self.denormalize_actions(normalized_actions)
        event_logits = self.event_head(temporal_features)  # [B, 1, 3]
        
        result = {
            "actions": normalized_actions.squeeze(1),      # [B, 3]
            "actions_denorm": denormalized_actions.squeeze(1),  # [B, 3]
            "event_logits": event_logits.squeeze(1),       # [B, 3]
            "predicted_events": torch.argmax(event_logits, dim=-1).squeeze(1)  # [B]
        }
        
        return result, new_hidden_state
    
    def compute_action_loss(
        self, 
        predicted_actions: torch.Tensor, 
        target_actions: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """액션 예측 손실 계산"""
        # MSE 손실
        action_loss = F.mse_loss(predicted_actions, target_actions, reduction='none')  # [B, T, 3]
        
        # 시퀀스 마스킹 (패딩된 부분 제외)
        if sequence_mask is not None:
            mask = sequence_mask.unsqueeze(-1).float()  # [B, T, 1]
            action_loss = action_loss * mask
            return action_loss.sum() / mask.sum()
        
        return action_loss.mean()
    
    def compute_event_loss(
        self,
        predicted_event_logits: torch.Tensor,
        target_events: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """이벤트 예측 손실 계산"""
        # Cross-entropy 손실
        B, T, num_classes = predicted_event_logits.shape
        event_loss = F.cross_entropy(
            predicted_event_logits.view(B * T, num_classes),
            target_events.view(B * T),
            reduction='none'
        ).view(B, T)
        
        # 시퀀스 마스킹
        if sequence_mask is not None:
            mask = sequence_mask.float()  # [B, T]
            event_loss = event_loss * mask
            return event_loss.sum() / mask.sum()
        
        return event_loss.mean()


class MobilePolicyHeadLite(nn.Module):
    """
    경량화된 Mobile Policy Head (Jetson용)
    LSTM 없이 더 단순한 MLP만 사용
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # 액션 범위
        self.action_bounds = {
            "linear_x": 2.0, "linear_y": 2.0, "angular_z": 2.0
        }
        
        # 간단한 액션 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        # 간단한 이벤트 헤드
        self.event_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        logger.info(f"🚀 Mobile Policy Head Lite 초기화 (Hidden: {hidden_size})")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_actions = self.action_head(features)
        denormalized_actions = self.denormalize_actions(normalized_actions)
        event_logits = self.event_head(features)
        
        return {
            "actions": normalized_actions,
            "actions_denorm": denormalized_actions,
            "event_logits": event_logits,
            "predicted_events": torch.argmax(event_logits, dim=-1)
        }
    
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        denormalized = normalized_actions.clone()
        denormalized[..., 0] *= self.action_bounds["linear_x"]
        denormalized[..., 1] *= self.action_bounds["linear_y"]
        denormalized[..., 2] *= self.action_bounds["angular_z"]
        return denormalized


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile Policy Head 테스트")
    
    # 모델 초기화
    policy_head = MobilePolicyHead(hidden_size=768)
    policy_head_lite = MobilePolicyHeadLite(hidden_size=512)
    
    # 테스트 데이터
    batch_size, seq_len, hidden_size = 2, 18, 768
    test_features = torch.randn(batch_size, seq_len, hidden_size)
    test_features_lite = torch.randn(batch_size, seq_len, 512)
    
    print(f"📊 입력 특징: {test_features.shape}")
    
    # 일반 정책 헤드 테스트
    with torch.no_grad():
        result = policy_head(test_features, return_uncertainty=True)
        print(f"🎯 액션 출력: {result['actions'].shape}")
        print(f"🎯 실제 액션: {result['actions_denorm'].shape}")
        print(f"⚡ 이벤트 로짓: {result['event_logits'].shape}")
        print(f"📊 액션 분산: {result['action_variance'].shape}")
        
        # Lite 정책 헤드 테스트
        result_lite = policy_head_lite(test_features_lite)
        print(f"🚀 Lite 액션: {result_lite['actions'].shape}")
    
    # 액션 범위 확인
    sample_actions = result['actions_denorm'][0, :3]  # 첫 3프레임
    print(f"📈 샘플 액션 (실제 범위): {sample_actions}")
    print(f"   Linear X: {sample_actions[:, 0].min():.2f} ~ {sample_actions[:, 0].max():.2f}")
    print(f"   Linear Y: {sample_actions[:, 1].min():.2f} ~ {sample_actions[:, 1].max():.2f}")
    print(f"   Angular Z: {sample_actions[:, 2].min():.2f} ~ {sample_actions[:, 2].max():.2f}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in policy_head.parameters())
    lite_params = sum(p.numel() for p in policy_head_lite.parameters())
    
    print(f"📊 일반 정책 헤드 파라미터: {total_params:,}개 ({total_params/1e6:.1f}M)")
    print(f"🚀 Lite 정책 헤드 파라미터: {lite_params:,}개 ({lite_params/1e6:.1f}M)")
    print(f"💡 파라미터 감소율: {(1 - lite_params/total_params)*100:.1f}%")
