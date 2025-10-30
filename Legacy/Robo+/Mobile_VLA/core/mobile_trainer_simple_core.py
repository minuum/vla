#!/usr/bin/env python3
"""
Simple Mobile VLA Trainer - PyTorch Lightning 없이 기본 PyTorch 사용
테스트 및 간단한 학습용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging

# Mobile VLA 모듈들
try:
    from ..models.mobile_vla_model import MobileVLAModel
    from ..data.mobile_dataset import MobileVLADataset
except ImportError:
    # 테스트용 절대 임포트
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from models.mobile_vla_model import MobileVLAModel
    from data.mobile_dataset import MobileVLADataset

logger = logging.getLogger(__name__)


class SimpleMobileVLATrainer:
    """
    간단한 Mobile VLA 트레이너 (PyTorch Lightning 없음)
    테스트 및 프로토타이핑용
    """
    
    def __init__(self, configs: Dict[str, Any]):
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mobile VLA 모델 초기화
        self.model = MobileVLAModel(
            hidden_size=configs.get("hidden_size", 768),
            image_backbone=configs.get("image_backbone", "efficientnet_v2_s"),
            text_model=configs.get("text_model", "klue/roberta-base"),
            use_lite_mode=configs.get("use_lite_mode", False),
            dropout=configs.get("dropout", 0.1)
        ).to(self.device)
        
        # mobile_vla_data_collector.py 시나리오별 가중치
        self.scenario_weights = configs.get("scenario_weights", {
            "1box_vert_left": 1.0,
            "1box_vert_right": 1.0,
            "1box_hori_left": 1.2,
            "1box_hori_right": 1.1,
            "2box_vert_left": 1.5,
            "2box_vert_right": 1.4,
            "2box_hori_left": 1.8,
            "2box_hori_right": 1.6
        })
        
        # 손실 함수 가중치
        self.action_loss_weight = configs.get("action_loss_weight", 1.0)
        self.event_loss_weight = configs.get("event_loss_weight", 0.5)
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=configs.get("learning_rate", 1e-4),
            weight_decay=configs.get("weight_decay", 0.01)
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=configs.get("max_epochs", 100),
            eta_min=configs.get("learning_rate", 1e-4) * 0.1
        )
        
        logger.info("🤖 Simple Mobile VLA Trainer 초기화 완료")
        logger.info(f"   디바이스: {self.device}")
        logger.info(f"   모델 크기: {sum(p.numel() for p in self.model.parameters()):,}개")
    
    def compute_loss(self, predictions: Dict, targets: Dict, scenarios: List[str]) -> Dict:
        """손실 계산"""
        device = predictions["actions"].device
        
        # 1. 액션 손실 (MSE)
        action_loss = F.mse_loss(predictions["actions"], targets["actions"])
        
        # 2. 이벤트 손실 (Cross-entropy)
        B, T, num_classes = predictions["event_logits"].shape
        event_loss = F.cross_entropy(
            predictions["event_logits"].view(B * T, num_classes),
            targets["action_events"].view(B * T)
        )
        
        # 3. 시나리오별 가중치
        scenario_weights = torch.tensor([
            self.scenario_weights.get(scenario, 1.0) for scenario in scenarios
        ], device=device, dtype=torch.float32)
        
        scenario_weight = scenario_weights.mean()
        
        # 총 손실
        total_loss = (
            self.action_loss_weight * action_loss * scenario_weight +
            self.event_loss_weight * event_loss * scenario_weight
        )
        
        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "event_loss": event_loss,
            "scenario_weight": scenario_weight
        }
    
    def compute_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """메트릭 계산"""
        # 액션 정확도 (허용 오차 0.1 이내)
        action_diff = torch.abs(predictions["actions"] - targets["actions"])
        accurate_actions = (action_diff < 0.1).all(dim=-1)
        action_accuracy = accurate_actions.float().mean()
        
        # 이벤트 정확도
        correct_events = (predictions["predicted_events"] == targets["action_events"])
        event_accuracy = correct_events.float().mean()
        
        return {
            "action_accuracy": action_accuracy,
            "event_accuracy": event_accuracy
        }
    
    def train_step(self, batch: Dict) -> Dict:
        """학습 스텝"""
        self.model.train()
        
        # 배치를 디바이스로 이동
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 순전파
        predictions = self.model(
            images=batch["images"],
            scenarios=batch["scenario"],
            instructions=batch.get("instruction")
        )
        
        # 타겟 준비
        targets = {
            "actions": batch["actions"],
            "action_events": batch["action_events"]
        }
        
        # 손실 계산
        losses = self.compute_loss(predictions, targets, batch["scenario"])
        
        # 역전파
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 메트릭 계산
        with torch.no_grad():
            metrics = self.compute_metrics(predictions, targets)
        
        # 결과 반환
        result = {**losses, **metrics}
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    
    def val_step(self, batch: Dict) -> Dict:
        """검증 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            # 배치를 디바이스로 이동
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 순전파
            predictions = self.model(
                images=batch["images"],
                scenarios=batch["scenario"],
                instructions=batch.get("instruction")
            )
            
            # 타겟 준비
            targets = {
                "actions": batch["actions"],
                "action_events": batch["action_events"]
            }
            
            # 손실 및 메트릭 계산
            losses = self.compute_loss(predictions, targets, batch["scenario"])
            metrics = self.compute_metrics(predictions, targets)
            
            result = {**losses, **metrics}
            return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    
    def predict_mobile_action(self, current_image: torch.Tensor, scenario: str) -> Dict[str, float]:
        """mobile_vla_data_collector.py 호환 액션 예측"""
        self.model.eval()
        
        current_image = current_image.to(self.device)
        
        with torch.no_grad():
            mobile_action = self.model.get_mobile_vla_action(current_image, scenario)
        
        return mobile_action
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'configs': self.configs
        }, path)
        logger.info(f"✅ 모델 저장 완료: {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"✅ 모델 로드 완료: {path}")


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Simple Mobile VLA Trainer 테스트")
    
    # 테스트용 설정
    test_configs = {
        "hidden_size": 512,
        "use_lite_mode": True,  # 빠른 테스트를 위해 Lite 모드
        "learning_rate": 1e-4,
        "batch_size": 2,
        "sequence_length": 18,
        "max_epochs": 5
    }
    
    # 트레이너 초기화
    trainer = SimpleMobileVLATrainer(test_configs)
    
    # 테스트 데이터 생성
    batch_size, seq_len = 2, 18
    test_batch = {
        "images": torch.randn(batch_size, seq_len, 3, 224, 224),
        "actions": torch.randn(batch_size, seq_len, 3),  # 정규화된 액션
        "action_events": torch.randint(0, 3, (batch_size, seq_len)),  # 이벤트 인덱스
        "scenario": ["1box_vert_left", "2box_hori_right"],
        "instruction": ["박스를 왼쪽으로 돌아서 컵까지 가세요", "두 박스를 오른쪽으로 우회해서 컵까지 가세요"]
    }
    
    print(f"📊 테스트 배치:")
    print(f"   이미지: {test_batch['images'].shape}")
    print(f"   액션: {test_batch['actions'].shape}")
    print(f"   이벤트: {test_batch['action_events'].shape}")
    print(f"   시나리오: {test_batch['scenario']}")
    
    # 학습 스텝 테스트
    train_result = trainer.train_step(test_batch)
    print(f"\n📈 학습 결과:")
    for key, value in train_result.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 검증 스텝 테스트
    val_result = trainer.val_step(test_batch)
    print(f"\n📊 검증 결과:")
    for key, value in val_result.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 단일 액션 예측 테스트
    single_image = torch.randn(1, 3, 224, 224)
    mobile_action = trainer.predict_mobile_action(single_image, "1box_vert_left")
    print(f"\n🎯 Mobile 액션 예측: {mobile_action}")
    
    print(f"\n✅ Simple Mobile VLA Trainer 테스트 완료!")
