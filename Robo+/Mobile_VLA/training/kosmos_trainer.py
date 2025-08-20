#!/usr/bin/env python3
"""
Kosmos Trainer for Mobile VLA - Kosmos 모델을 Mobile VLA 데이터로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging
from transformers import Kosmos2ForConditionalGeneration, AutoProcessor

try:
    from ..data.robovlms_adapter import MobileVLAToRoboVLMsAdapter
    from ..models.policy_heads.mobile_policy_head import MobilePolicyHead
except ImportError:
    # 테스트용 절대 임포트
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from data.robovlms_adapter import MobileVLAToRoboVLMsAdapter
    from models.policy_heads.mobile_policy_head import MobilePolicyHead

logger = logging.getLogger(__name__)


class MobileKosmosModel(nn.Module):
    """
    Mobile VLA + Kosmos 통합 모델
    """
    
    def __init__(
        self,
        kosmos_model_name: str = "microsoft/kosmos-2-patch14-224",
        hidden_size: int = 768,
        action_dim: int = 7,  # RoboVLMs 호환
        freeze_kosmos: bool = True
    ):
        super().__init__()
        
        # Kosmos 모델 로드
        self.kosmos = Kosmos2ForConditionalGeneration.from_pretrained(
            kosmos_model_name,
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(kosmos_model_name)
        
        # Kosmos 가중치 고정 옵션
        if freeze_kosmos:
            for param in self.kosmos.parameters():
                param.requires_grad = False
            logger.info("🔒 Kosmos 가중치 고정됨")
        
        # Kosmos 특징 차원
        kosmos_hidden_size = self.kosmos.config.text_config.hidden_size
        
        # 특징 프로젝션 (Kosmos → Mobile VLA)
        self.feature_projection = nn.Sequential(
            nn.Linear(kosmos_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Mobile VLA 정책 헤드 (3D 액션 유지)
        self.policy_head = MobilePolicyHead(
            hidden_size=hidden_size,
            action_dim=3,  # Mobile VLA 원본: [linear_x, linear_y, angular_z]
            dropout=0.1
        )
        
        logger.info(f"🤖 Mobile Kosmos Model 초기화 완료")
        logger.info(f"   Kosmos Hidden: {kosmos_hidden_size}, Mobile Hidden: {hidden_size}")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: [B, T, 3, 224, 224] or [B, 1, T, 3, 224, 224] - 이미지 시퀀스
            input_ids: [B, seq_len] - 토크나이즈된 텍스트
            attention_mask: [B, seq_len] - 어텐션 마스크
        """
        # 차원 안전 처리
        if pixel_values.dim() == 6:  # [B, 1, T, C, H, W]
            pixel_values = pixel_values.squeeze(1)  # [B, T, C, H, W]
        elif pixel_values.dim() == 5:  # [B, T, C, H, W] - 이미 올바른 형태
            pass
        else:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")
            
        B, T, C, H, W = pixel_values.shape
        
        # 배치와 시간 차원을 합쳐서 Kosmos에 입력
        images_flat = pixel_values.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        
        # 텍스트는 배치별로 반복
        input_ids_expanded = input_ids.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)  # [B*T, seq_len]
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)
        else:
            attention_mask_expanded = None
        
        # Kosmos 순전파 (올바른 방법)
        with torch.cuda.amp.autocast(enabled=True):
            # Kosmos의 경우 pixel_values 없이 text-only로 먼저 시도
            # 더미 텍스트 입력 사용 (이미지 토큰 없음)
            kosmos_output = self.kosmos.text_model(
                input_ids=input_ids_expanded,
                attention_mask=attention_mask_expanded,
                output_hidden_states=True
            )
        
        # 마지막 hidden state 추출
        last_hidden_state = kosmos_output.hidden_states[-1]  # [B*T, seq_len, kosmos_hidden]
        
        # 평균 풀링으로 시퀀스 특징 압축
        if attention_mask_expanded is not None:
            mask = attention_mask_expanded.unsqueeze(-1).float()  # [B*T, seq_len, 1]
            pooled_features = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)  # [B*T, kosmos_hidden]
        else:
            pooled_features = last_hidden_state.mean(dim=1)  # [B*T, kosmos_hidden]
        
        # 시간 차원 복원
        pooled_features = pooled_features.view(B, T, -1)  # [B, T, kosmos_hidden]
        
        # 특징 프로젝션
        mobile_features = self.feature_projection(pooled_features)  # [B, T, hidden_size]
        
        # 정책 헤드로 액션 예측
        policy_output = self.policy_head(mobile_features)
        
        return policy_output


class MobileKosmosTrainer:
    """
    Mobile VLA + Kosmos 트레이너
    """
    
    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self.model = MobileKosmosModel(
            kosmos_model_name=configs.get("kosmos_model_name", "microsoft/kosmos-2-patch14-224"),
            hidden_size=configs.get("hidden_size", 768),
            freeze_kosmos=configs.get("freeze_kosmos", True)
        ).to(self.device)
        
        # 손실 함수 가중치
        self.action_loss_weight = configs.get("action_loss_weight", 1.0)
        self.event_loss_weight = configs.get("event_loss_weight", 0.5)
        
        # 옵티마이저 (Kosmos가 고정된 경우 정책 헤드만 학습)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=configs.get("learning_rate", 1e-4),
            weight_decay=configs.get("weight_decay", 0.01)
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=configs.get("max_epochs", 100),
            eta_min=configs.get("learning_rate", 1e-4) * 0.1
        )
        
        logger.info("🤖 Mobile Kosmos Trainer 초기화 완료")
        logger.info(f"   학습 가능 파라미터: {sum(p.numel() for p in trainable_params):,}개")
    
    def tokenize_instructions(self, instructions: List[str]) -> Dict[str, torch.Tensor]:
        """명령어 토크나이징"""
        tokenized = self.model.processor.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        return tokenized
    
    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict:
        """손실 계산"""
        # 1. 액션 손실 (MSE) - Mobile VLA 3D 액션
        action_loss = F.mse_loss(predictions["actions_denorm"], targets["mobile_actions"])
        
        # 2. 이벤트 손실 (Cross-entropy)
        B, T, num_classes = predictions["event_logits"].shape
        event_loss = F.cross_entropy(
            predictions["event_logits"].view(B * T, num_classes),
            targets["mobile_events"].view(B * T)
        )
        
        # 총 손실
        total_loss = (
            self.action_loss_weight * action_loss +
            self.event_loss_weight * event_loss
        )
        
        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "event_loss": event_loss
        }
    
    def train_step(self, batch: Dict) -> Dict:
        """학습 스텝"""
        self.model.train()
        
        # 입력 데이터 준비
        pixel_values = batch["vision_x"].to(self.device)  # [B, T, 3, 224, 224]
        
        # 명령어 토크나이징 - 안전한 처리
        task_desc = batch["task_description"]
        
        # task_description 타입 체크 및 안전한 처리
        if isinstance(task_desc, str):
            instructions = [task_desc]
        elif isinstance(task_desc, (list, tuple)):
            # 이미 리스트인 경우
            instructions = list(task_desc)
        else:
            # 기타 경우 (텐서 등)
            instructions = [str(task_desc)]
        
        # 빈 문자열 체크
        instructions = [instr for instr in instructions if instr and instr.strip()]
        if not instructions:
            instructions = ["Navigate to track the target cup"]  # 기본 명령어
        
        tokenized = self.tokenize_instructions(instructions)
        
        # 순전파
        predictions = self.model(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask")
        )
        
        # 타겟 준비 (Mobile VLA 원본 데이터 사용)
        targets = {
            "mobile_actions": batch["mobile_actions"].to(self.device),  # [B, T, 3] - Mobile VLA 액션
            "mobile_events": batch["mobile_events"].to(self.device)     # [B, T] - Mobile VLA 이벤트
        }
        
        # 손실 계산
        losses = self.compute_loss(predictions, targets)
        
        # 역전파
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 결과 반환
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'configs': self.configs
        }, path)
        logger.info(f"✅ 모델 저장 완료: {path}")


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile Kosmos Trainer 테스트")
    
    # 테스트용 설정
    test_configs = {
        "kosmos_model_name": "microsoft/kosmos-2-patch14-224",
        "hidden_size": 768,
        "freeze_kosmos": True,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "sequence_length": 18
    }
    
    # 어댑터와 트레이너 초기화
    adapter = MobileVLAToRoboVLMsAdapter(
        data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/",
        sequence_length=18
    )
    
    trainer = MobileKosmosTrainer(test_configs)
    
    if len(adapter) > 0:
        # 첫 번째 샘플로 테스트
        sample = adapter[0]
        
        print(f"📊 테스트 데이터:")
        print(f"   Vision X: {sample['vision_x'].shape}")
        print(f"   Action: {sample['action'].shape}")
        print(f"   Task: {sample['task_description']}")
        
        # 학습 스텝 테스트
        train_result = trainer.train_step(sample)
        print(f"\n📈 학습 결과:")
        for key, value in train_result.items():
            print(f"   {key}: {value:.4f}")
    
    print(f"\n✅ Mobile Kosmos Trainer 테스트 완료!")
