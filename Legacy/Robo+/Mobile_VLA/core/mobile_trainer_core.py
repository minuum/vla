#!/usr/bin/env python3
"""
Mobile VLA Trainer - RoboVLMs의 BaseTrainer를 Mobile VLA에 적응
mobile_vla_data_collector.py 데이터로 직접 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
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


class MobileVLATrainer(pl.LightningModule):
    """
    Mobile VLA 전용 트레이너
    - RoboVLMs BaseTrainer 구조 유지
    - mobile_vla_data_collector.py 데이터 특화
    - 시나리오별 학습 최적화
    """
    
    def __init__(self, configs: Dict[str, Any]):
        super().__init__()
        
        self.configs = configs
        self.save_hyperparameters()
        
        # Mobile VLA 모델 초기화
        self.model = MobileVLAModel(
            hidden_size=configs.get("hidden_size", 768),
            image_backbone=configs.get("image_backbone", "efficientnet_v2_s"),
            text_model=configs.get("text_model", "klue/roberta-base"),
            use_lite_mode=configs.get("use_lite_mode", False),
            dropout=configs.get("dropout", 0.1)
        )
        
        # mobile_vla_data_collector.py 시나리오별 가중치
        self.scenario_weights = configs.get("scenario_weights", {
            "1box_vert_left": 1.0,
            "1box_vert_right": 1.0,
            "1box_hori_left": 1.2,   # 더 어려운 시나리오
            "1box_hori_right": 1.1,
            "2box_vert_left": 1.5,   # 가장 어려운 시나리오
            "2box_vert_right": 1.4,
            "2box_hori_left": 1.8,
            "2box_hori_right": 1.6
        })
        
        # 손실 함수 가중치
        self.action_loss_weight = configs.get("action_loss_weight", 1.0)
        self.event_loss_weight = configs.get("event_loss_weight", 0.5)
        self.scenario_loss_weight = configs.get("scenario_loss_weight", 0.1)
        
        # 학습 메트릭
        self.action_tolerance = configs.get("action_tolerance", 0.1)  # 액션 정확도 허용 오차
        
        logger.info("🤖 Mobile VLA Trainer 초기화 완료")
        logger.info(f"   모델 크기: {sum(p.numel() for p in self.model.parameters()):,}개")
        logger.info(f"   Lite 모드: {configs.get('use_lite_mode', False)}")
        logger.info(f"   시나리오 가중치: {self.scenario_weights}")
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """순전파"""
        return self.model(
            images=batch["images"],
            scenarios=batch["scenario"],
            instructions=batch.get("instruction")
        )
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """학습 스텝"""
        # 모델 예측
        predictions = self.forward(batch)
        
        # 타겟 데이터
        target_actions = batch["actions"]          # [B, T, 3] 정규화된 액션
        target_events = batch["action_events"]     # [B, T] 이벤트 인덱스
        scenarios = batch["scenario"]              # List[str]
        sequence_masks = batch.get("sequence_mask")  # [B, T] (옵션)
        
        # 손실 계산
        losses = self._compute_losses(
            predictions, target_actions, target_events, scenarios, sequence_masks
        )
        
        # 메트릭 계산
        metrics = self._compute_metrics(
            predictions, target_actions, target_events, sequence_masks
        )
        
        # 로깅
        self.log_dict({
            "train_total_loss": losses["total_loss"],
            "train_action_loss": losses["action_loss"],
            "train_event_loss": losses["event_loss"],
            "train_scenario_loss": losses["scenario_loss"],
            "train_action_accuracy": metrics["action_accuracy"],
            "train_event_accuracy": metrics["event_accuracy"],
            "train_scenario_weight_avg": losses["scenario_weight_avg"]
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return losses["total_loss"]
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """검증 스텝"""
        with torch.no_grad():
            predictions = self.forward(batch)
            
            target_actions = batch["actions"]
            target_events = batch["action_events"]
            scenarios = batch["scenario"]
            sequence_masks = batch.get("sequence_mask")
            
            losses = self._compute_losses(
                predictions, target_actions, target_events, scenarios, sequence_masks
            )
            
            metrics = self._compute_metrics(
                predictions, target_actions, target_events, sequence_masks
            )
            
            # 검증 로깅
            self.log_dict({
                "val_total_loss": losses["total_loss"],
                "val_action_loss": losses["action_loss"],
                "val_event_loss": losses["event_loss"],
                "val_action_accuracy": metrics["action_accuracy"],
                "val_event_accuracy": metrics["event_accuracy"]
            }, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return losses["total_loss"]
    
    def _compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        target_events: torch.Tensor,
        scenarios: List[str],
        sequence_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """통합 손실 계산"""
        
        # 1. 액션 손실 (MSE)
        pred_actions = predictions["actions"]  # [B, T, 3] 정규화된 액션
        action_loss = F.mse_loss(pred_actions, target_actions, reduction='none')  # [B, T, 3]
        
        if sequence_masks is not None:
            mask = sequence_masks.unsqueeze(-1).float()  # [B, T, 1]
            action_loss = (action_loss * mask).sum() / mask.sum()
        else:
            action_loss = action_loss.mean()
        
        # 2. 이벤트 손실 (Cross-entropy)
        pred_event_logits = predictions["event_logits"]  # [B, T, 3]
        B, T, num_classes = pred_event_logits.shape
        
        event_loss = F.cross_entropy(
            pred_event_logits.view(B * T, num_classes),
            target_events.view(B * T),
            reduction='none'
        ).view(B, T)
        
        if sequence_masks is not None:
            mask = sequence_masks.float()  # [B, T]
            event_loss = (event_loss * mask).sum() / mask.sum()
        else:
            event_loss = event_loss.mean()
        
        # 3. 시나리오별 가중치 적용
        scenario_weights = torch.tensor([
            self.scenario_weights.get(scenario, 1.0) for scenario in scenarios
        ], device=self.device, dtype=torch.float32)
        
        scenario_weight_avg = scenario_weights.mean()
        
        # 가중치가 적용된 액션 손실
        weighted_action_loss = action_loss * scenario_weight_avg
        weighted_event_loss = event_loss * scenario_weight_avg
        
        # 4. 시나리오 일관성 손실 (같은 시나리오에서 일관된 행동 유도)
        scenario_consistency_loss = self._compute_scenario_consistency_loss(
            predictions, scenarios
        )
        
        # 총 손실
        total_loss = (
            self.action_loss_weight * weighted_action_loss +
            self.event_loss_weight * weighted_event_loss +
            self.scenario_loss_weight * scenario_consistency_loss
        )
        
        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "event_loss": event_loss,
            "scenario_loss": scenario_consistency_loss,
            "scenario_weight_avg": scenario_weight_avg
        }
    
    def _compute_scenario_consistency_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        scenarios: List[str]
    ) -> torch.Tensor:
        """시나리오 일관성 손실"""
        # 같은 시나리오끼리 그룹화
        scenario_groups = {}
        for i, scenario in enumerate(scenarios):
            if scenario not in scenario_groups:
                scenario_groups[scenario] = []
            scenario_groups[scenario].append(i)
        
        if len(scenario_groups) <= 1:
            # 모든 배치가 같은 시나리오면 일관성 손실 0
            return torch.tensor(0.0, device=self.device)
        
        consistency_loss = 0.0
        num_groups = 0
        
        for scenario, indices in scenario_groups.items():
            if len(indices) < 2:
                continue  # 해당 시나리오가 1개뿐이면 스킵
            
            # 같은 시나리오의 액션들 간 분산 최소화
            scenario_actions = predictions["actions"][indices]  # [num_indices, T, 3]
            action_mean = scenario_actions.mean(dim=0, keepdim=True)  # [1, T, 3]
            action_variance = ((scenario_actions - action_mean) ** 2).mean()
            
            consistency_loss += action_variance
            num_groups += 1
        
        if num_groups > 0:
            consistency_loss = consistency_loss / num_groups
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        return consistency_loss
    
    def _compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        target_events: torch.Tensor,
        sequence_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """평가 메트릭 계산"""
        
        # 1. 액션 정확도 (허용 오차 내 예측)
        pred_actions = predictions["actions"]  # [B, T, 3]
        action_diff = torch.abs(pred_actions - target_actions)  # [B, T, 3]
        accurate_actions = (action_diff < self.action_tolerance).all(dim=-1)  # [B, T]
        
        if sequence_masks is not None:
            mask = sequence_masks.bool()
            action_accuracy = (accurate_actions & mask).float().sum() / mask.float().sum()
        else:
            action_accuracy = accurate_actions.float().mean()
        
        # 2. 이벤트 정확도
        pred_events = predictions["predicted_events"]  # [B, T]
        correct_events = (pred_events == target_events)  # [B, T]
        
        if sequence_masks is not None:
            mask = sequence_masks.bool()
            event_accuracy = (correct_events & mask).float().sum() / mask.float().sum()
        else:
            event_accuracy = correct_events.float().mean()
        
        # 3. 액션별 MAE (Mean Absolute Error)
        action_mae = action_diff.mean(dim=(0, 1))  # [3] - 각 액션 축별 MAE
        
        return {
            "action_accuracy": action_accuracy,
            "event_accuracy": event_accuracy,
            "action_mae_x": action_mae[0],
            "action_mae_y": action_mae[1],
            "action_mae_z": action_mae[2]
        }
    
    def configure_optimizers(self):
        """옵티마이저 및 스케줄러 설정"""
        # RoboVLMs BaseTrainer와 유사한 설정
        lr = self.configs.get("learning_rate", 1e-4)
        weight_decay = self.configs.get("weight_decay", 0.01)
        
        # AdamW 옵티마이저
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 코사인 어닐링 스케줄러
        scheduler_type = self.configs.get("scheduler", "cosine")
        
        if scheduler_type == "cosine":
            max_epochs = self.configs.get("max_epochs", 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=lr * 0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif scheduler_type == "step":
            step_size = self.configs.get("step_size", 30)
            gamma = self.configs.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
    
    def train_dataloader(self) -> DataLoader:
        """학습 데이터로더"""
        dataset = MobileVLADataset(
            data_dir=self.configs.get("train_data_dir", "/home/soda/vla/ROS_action/mobile_vla_dataset/"),
            sequence_length=self.configs.get("sequence_length", 18),
            normalize_actions=True,
            scenario_filter=self.configs.get("train_scenarios")
        )
        
        return DataLoader(
            dataset,
            batch_size=self.configs.get("batch_size", 4),
            shuffle=True,
            num_workers=self.configs.get("num_workers", 4),
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """검증 데이터로더"""
        dataset = MobileVLADataset(
            data_dir=self.configs.get("val_data_dir", "/home/soda/vla/ROS_action/mobile_vla_dataset/"),
            sequence_length=self.configs.get("sequence_length", 18),
            normalize_actions=True,
            scenario_filter=self.configs.get("val_scenarios")
        )
        
        return DataLoader(
            dataset,
            batch_size=self.configs.get("val_batch_size", 2),
            shuffle=False,
            num_workers=self.configs.get("num_workers", 2),
            pin_memory=True
        )
    
    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 호출"""
        # 시나리오별 성능 분석 (추후 구현)
        pass
    
    def predict_mobile_action(
        self,
        current_image: torch.Tensor,
        scenario: str
    ) -> Dict[str, float]:
        """
        mobile_vla_data_collector.py 호환 액션 예측
        (실시간 추론용)
        """
        self.eval()
        with torch.no_grad():
            mobile_action = self.model.get_mobile_vla_action(current_image, scenario)
        return mobile_action


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile VLA Trainer 테스트")
    
    # 테스트용 설정
    test_configs = {
        "hidden_size": 512,
        "use_lite_mode": True,  # 빠른 테스트를 위해 Lite 모드
        "learning_rate": 1e-4,
        "batch_size": 2,
        "sequence_length": 18,
        "max_epochs": 5,
        "scheduler": "cosine"
    }
    
    # 트레이너 초기화
    trainer = MobileVLATrainer(test_configs)
    
    # 테스트 데이터 생성
    batch_size, seq_len = 2, 18
    test_batch = {
        "images": torch.randn(batch_size, seq_len, 3, 224, 224),
        "actions": torch.randn(batch_size, seq_len, 3),  # 정규화된 액션
        "action_events": torch.randint(0, 3, (batch_size, seq_len)),  # 이벤트 인덱스
        "scenario": ["1box_vert_left", "2box_hori_right"],
        "instruction": ["박스를 왼쪽으로 돌아서 컵까지 가세요", "두 박스를 오른쪽으로 우회해서 컵까지 가세요"],
        "sequence_mask": torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    print(f"📊 테스트 배치:")
    print(f"   이미지: {test_batch['images'].shape}")
    print(f"   액션: {test_batch['actions'].shape}")
    print(f"   이벤트: {test_batch['action_events'].shape}")
    print(f"   시나리오: {test_batch['scenario']}")
    
    # 순전파 테스트
    predictions = trainer.forward(test_batch)
    print(f"\n🤖 모델 예측:")
    print(f"   액션: {predictions['actions'].shape}")
    print(f"   이벤트: {predictions['predicted_events'].shape}")
    
    # 학습 스텝 테스트
    loss = trainer.training_step(test_batch, 0)
    print(f"\n📈 학습 손실: {loss:.4f}")
    
    # 단일 액션 예측 테스트
    single_image = torch.randn(1, 3, 224, 224)
    mobile_action = trainer.predict_mobile_action(single_image, "1box_vert_left")
    print(f"\n🎯 Mobile 액션 예측: {mobile_action}")
    
    print(f"\n✅ Mobile VLA Trainer 테스트 완료!")
