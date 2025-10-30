#!/usr/bin/env python3
"""
Step 2: Mobile VLA 모델 구조 구현
2D 액션 공간, LoRA Fine-tuning, LSTM Policy Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileVLAModel(nn.Module):
    """
    Mobile VLA 모델 구조
    - 2D 액션 공간 (X, Y, Gripper)
    - LoRA Fine-tuning
    - LSTM Policy Head
    """
    
    def __init__(
        self,
        vlm_model_name: str = "microsoft/kosmos-2-patch14-224",
        action_dim: int = 3,  # X, Y, Gripper
        hidden_size: int = 512,
        lstm_layers: int = 2,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # VLM 백본 로드
        self.vlm_model = AutoModel.from_pretrained(vlm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
        
        # LoRA 설정
        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value", "key", "dense"]
        )
        
        # VLM에 LoRA 적용
        self.vlm_model = get_peft_model(self.vlm_model, self.lora_config)
        
        # VLM 출력 차원
        self.vlm_output_dim = self.vlm_model.config.hidden_size
        
        # [LRN] 토큰 (학습 가능한 액션 토큰)
        self.action_token = nn.Parameter(torch.zeros(self.vlm_output_dim))
        
        # LSTM Policy Head
        self.lstm = nn.LSTM(
            input_size=self.vlm_output_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Action Head (2D + Gripper)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, action_dim)
        )
        
        # Gripper Head (Binary Classification)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Mobile VLA 모델 초기화 완료")
        logger.info(f"- VLM: {vlm_model_name}")
        logger.info(f"- 액션 차원: {action_dim}")
        logger.info(f"- LSTM Hidden Size: {hidden_size}")
        logger.info(f"- LoRA r: {lora_r}")
    
    def forward(
        self,
        images: torch.Tensor,
        text: str,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [batch_size, channels, height, width]
            text: 자연어 명령
            return_dict: 결과를 딕셔너리로 반환할지 여부
        
        Returns:
            Dict containing:
                - action_logits: [batch_size, action_dim]
                - gripper_logits: [batch_size, 1]
                - vlm_outputs: VLM 출력
        """
        batch_size = images.shape[0]
        
        # 1. 텍스트 토큰화
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 2. VLM Forward Pass
        vlm_outputs = self.vlm_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=images
        )
        
        # 3. [LRN] 토큰 추가
        # VLM 출력의 마지막 토큰에 [LRN] 토큰 추가
        last_hidden_states = vlm_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # [LRN] 토큰을 배치별로 복제
        action_tokens = self.action_token.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1
        )  # [batch_size, 1, hidden_size]
        
        # [LRN] 토큰을 시퀀스 끝에 추가
        lstm_input = torch.cat([last_hidden_states, action_tokens], dim=1)
        
        # 4. LSTM Forward Pass
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)
        
        # 마지막 LSTM 출력 사용 (시퀀스의 마지막 토큰)
        last_lstm_output = lstm_output[:, -1, :]  # [batch_size, hidden_size]
        
        # 5. Action 예측
        action_logits = self.action_head(last_lstm_output)  # [batch_size, action_dim]
        gripper_logits = self.gripper_head(last_lstm_output)  # [batch_size, 1]
        
        if return_dict:
            return {
                "action_logits": action_logits,
                "gripper_logits": gripper_logits,
                "vlm_outputs": vlm_outputs,
                "lstm_output": lstm_output,
                "hidden_states": (h_n, c_n)
            }
        else:
            return action_logits, gripper_logits
    
    def get_action(self, images: torch.Tensor, text: str) -> torch.Tensor:
        """
        액션 예측 (추론용)
        
        Args:
            images: [batch_size, channels, height, width]
            text: 자연어 명령
        
        Returns:
            actions: [batch_size, action_dim] (X, Y, Gripper)
        """
        with torch.no_grad():
            outputs = self.forward(images, text)
            
            # 2D Movement (X, Y) - Tanh 활성화
            movement = torch.tanh(outputs["action_logits"][:, :2])
            
            # Gripper - Binary (0 or 1)
            gripper = (outputs["gripper_logits"] > 0.5).float()
            
            # 액션 결합
            actions = torch.cat([movement, gripper], dim=1)
            
            return actions
    
    def get_model_size(self) -> Dict[str, int]:
        """모델 크기 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vlm_parameters": sum(p.numel() for p in self.vlm_model.parameters()),
            "lstm_parameters": sum(p.numel() for p in self.lstm.parameters()),
            "action_head_parameters": sum(p.numel() for p in self.action_head.parameters()),
            "gripper_head_parameters": sum(p.numel() for p in self.gripper_head.parameters())
        }

class MobileVLALoss(nn.Module):
    """Mobile VLA Loss 함수"""
    
    def __init__(self, movement_weight: float = 1.0, gripper_weight: float = 0.1):
        super().__init__()
        self.movement_weight = movement_weight
        self.gripper_weight = gripper_weight
        
        # MSE Loss for 2D movement
        self.movement_loss = nn.MSELoss()
        
        # BCE Loss for gripper
        self.gripper_loss = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Loss 계산
        
        Args:
            predictions: 모델 예측값
            targets: 정답값
        
        Returns:
            Dict containing losses
        """
        # 2D Movement Loss (MSE)
        movement_loss = self.movement_loss(
            predictions["action_logits"][:, :2],  # X, Y
            targets["movement_targets"]  # [batch_size, 2]
        )
        
        # Gripper Loss (BCE)
        gripper_loss = self.gripper_loss(
            predictions["gripper_logits"].squeeze(-1),  # [batch_size]
            targets["gripper_targets"]  # [batch_size]
        )
        
        # Total Loss
        total_loss = (
            self.movement_weight * movement_loss +
            self.gripper_weight * gripper_loss
        )
        
        return {
            "total_loss": total_loss,
            "movement_loss": movement_loss,
            "gripper_loss": gripper_loss
        }

def create_mobile_vla_model(
    vlm_model_name: str = "microsoft/kosmos-2-patch14-224",
    action_dim: int = 3,
    hidden_size: int = 512,
    lstm_layers: int = 2,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1
) -> Tuple[MobileVLAModel, MobileVLALoss]:
    """
    Mobile VLA 모델과 Loss 함수 생성
    
    Args:
        vlm_model_name: VLM 모델명
        action_dim: 액션 차원 (기본 3: X, Y, Gripper)
        hidden_size: LSTM hidden size
        lstm_layers: LSTM 레이어 수
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    
    Returns:
        Tuple of (model, loss_function)
    """
    logger.info("🚀 Mobile VLA 모델 생성 중...")
    
    # 모델 생성
    model = MobileVLAModel(
        vlm_model_name=vlm_model_name,
        action_dim=action_dim,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Loss 함수 생성
    loss_fn = MobileVLALoss()
    
    # 모델 크기 정보 출력
    size_info = model.get_model_size()
    logger.info("📊 모델 크기 정보:")
    for key, value in size_info.items():
        logger.info(f"  {key}: {value:,}")
    
    logger.info("✅ Mobile VLA 모델 생성 완료")
    
    return model, loss_fn

def test_mobile_vla_model():
    """Mobile VLA 모델 테스트"""
    logger.info("🧪 Mobile VLA 모델 테스트 시작")
    
    try:
        # 모델 생성
        model, loss_fn = create_mobile_vla_model()
        
        # 테스트 데이터 생성
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text = "go to the red box"
        
        # Forward pass
        logger.info("Forward pass 테스트...")
        outputs = model(images, text)
        
        logger.info(f"✅ Forward pass 성공")
        logger.info(f"  - action_logits shape: {outputs['action_logits'].shape}")
        logger.info(f"  - gripper_logits shape: {outputs['gripper_logits'].shape}")
        
        # 액션 예측 테스트
        logger.info("액션 예측 테스트...")
        actions = model.get_action(images, text)
        logger.info(f"✅ 액션 예측 성공: {actions.shape}")
        
        # Loss 계산 테스트
        logger.info("Loss 계산 테스트...")
        targets = {
            "movement_targets": torch.randn(batch_size, 2),
            "gripper_targets": torch.randint(0, 2, (batch_size,)).float()
        }
        
        losses = loss_fn(outputs, targets)
        logger.info(f"✅ Loss 계산 성공")
        logger.info(f"  - total_loss: {losses['total_loss'].item():.4f}")
        logger.info(f"  - movement_loss: {losses['movement_loss'].item():.4f}")
        logger.info(f"  - gripper_loss: {losses['gripper_loss'].item():.4f}")
        
        logger.info("🎉 Mobile VLA 모델 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🚀 Mobile VLA 모델 구조 구현 시작")
    
    # 모델 테스트 실행
    success = test_mobile_vla_model()
    
    if success:
        logger.info("✅ Mobile VLA 모델 구조 구현 완료")
        logger.info("🎯 다음 단계: 학습 파이프라인 구현")
    else:
        logger.error("❌ Mobile VLA 모델 구조 구현 실패")
        logger.error("🔧 문제를 해결한 후 다시 시도해주세요")

if __name__ == "__main__":
    main()
