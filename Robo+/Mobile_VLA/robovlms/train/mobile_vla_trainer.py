# Mobile VLA Trainer based on RoboVLMs BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# RoboVLMs imports (로컬에서 사용 가능한 부분만)
from transformers import AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)


class MobileVLATrainer:
    """
    Mobile VLA 학습을 위한 간소화된 Trainer
    
    RoboVLMs BaseTrainer의 핵심 기능을 Mobile VLA에 맞게 적용
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/kosmos-2-patch14-224",
        action_dim: int = 3,
        window_size: int = 8,
        chunk_size: int = 2,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        precision: str = "fp16",
        **kwargs
    ):
        self.model_name = model_name
        self.action_dim = action_dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.precision = precision
        
        # 모델 초기화
        self._init_model()
        
        # 옵티마이저 초기화
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if precision == "fp16" else None
        
        self.step_count = 0
        
    def _init_model(self):
        """Mobile VLA 모델 초기화"""
        # Kosmos 프로세서
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # 간단한 VLA 모델 (Kosmos + Action Head)
        from transformers import Kosmos2Model
        
        class MobileVLAModel(nn.Module):
            def __init__(self, model_name: str, action_dim: int, window_size: int, chunk_size: int):
                super().__init__()
                self.kosmos = Kosmos2Model.from_pretrained(model_name)
                
                # Hidden size 가져오기
                self.hidden_size = self.kosmos.config.text_config.hidden_size
                
                # Action prediction head
                self.action_head = nn.Sequential(
                    nn.Linear(self.hidden_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, chunk_size * action_dim)  # 2 프레임 * 3 액션
                )
                
                self.window_size = window_size
                self.chunk_size = chunk_size
                self.action_dim = action_dim
            
            def forward(self, pixel_values, input_ids, attention_mask):
                # pixel_values는 [B, T, C, H, W] 형태로 들어옴
                # Kosmos vision model은 [B, C, H, W]를 기대하므로 마지막 프레임만 사용
                if pixel_values.dim() == 5:  # [B, T, C, H, W]
                    # 마지막 프레임 추출
                    last_frame = pixel_values[:, -1, :, :, :]  # [B, C, H, W]
                else:
                    last_frame = pixel_values
                
                # Vision model만 사용해서 이미지 특징 추출
                with torch.no_grad():
                    vision_outputs = self.kosmos.vision_model(pixel_values=last_frame)
                    # vision_outputs에서 pooler_output 또는 last_hidden_state 사용
                    if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                        image_features = vision_outputs.pooler_output  # [B, vision_hidden_size]
                    else:
                        # Global average pooling over patches
                        image_features = vision_outputs.last_hidden_state.mean(dim=1)  # [B, vision_hidden_size]
                
                # 이미지 특징을 action head의 입력 크기에 맞춰 조정
                if image_features.size(-1) != self.hidden_size:
                    # 간단한 linear projection 추가 (필요시)
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = nn.Linear(image_features.size(-1), self.hidden_size)
                        self.image_projection = self.image_projection.to(image_features.device)
                    image_features = self.image_projection(image_features)
                
                # 액션 예측 (이미지 특징만 사용)
                action_logits = self.action_head(image_features)  # [B, chunk_size * action_dim]
                action_preds = action_logits.view(-1, self.chunk_size, self.action_dim)  # [B, chunk_size, action_dim]
                
                # 더미 텍스트 출력 (호환성을 위해)
                dummy_hidden_states = torch.zeros(
                    last_frame.size(0), input_ids.size(1), self.hidden_size,
                    device=last_frame.device
                )
                
                return {
                    'predicted_actions': action_preds,
                    'hidden_states': dummy_hidden_states,
                    'pooled': image_features
                }
        
        self.model = MobileVLAModel(
            model_name=self.model_name,
            action_dim=self.action_dim,
            window_size=self.window_size,
            chunk_size=self.chunk_size
        ).to(self.device)
        
        logger.info(f"Initialized MobileVLAModel with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """액션 예측 손실 계산"""
        predicted_actions = predictions["predicted_actions"]  # [B, chunk_size, action_dim]
        target_actions = targets["action_chunk"]  # [B, chunk_size, action_dim] 또는 [B, window_size, chunk_size, action_dim]
        
        # target_actions shape 조정 (필요한 경우)
        if target_actions.dim() == 4:  # [B, window_size, chunk_size, action_dim]
            # 마지막 window의 chunk만 사용
            target_actions = target_actions[:, -1, :, :]  # [B, chunk_size, action_dim]
        
        # Huber Loss (MSE보다 outlier에 강함)
        action_loss = F.huber_loss(predicted_actions, target_actions)
        
        # 각 차원별 MAE 계산 (로깅용)
        mae_per_dim = torch.abs(predicted_actions - target_actions).mean(dim=(0, 1))
        
        return {
            "total_loss": action_loss,
            "action_loss": action_loss,
            "mae_linear_x": mae_per_dim[0].item() if len(mae_per_dim) > 0 else 0.0,
            "mae_linear_y": mae_per_dim[1].item() if len(mae_per_dim) > 1 else 0.0,
            "mae_angular_z": mae_per_dim[2].item() if len(mae_per_dim) > 2 else 0.0,
            "mae_avg": mae_per_dim.mean().item() if len(mae_per_dim) > 0 else 0.0
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """단일 학습 스텝 - 새로운 collate_fn 호환"""
        self.model.train()
        
        # 데이터 준비 (커스텀 collate_fn에서 이미 텐서로 변환됨)
        images = batch["images"]  # [B, T, C, H, W] - 이미 정규화된 텐서
        actions = batch["actions"]  # [B, T, action_dim]
        
        # Window/Chunk 분할 (RoboVLMs 방식)
        batch_size, sequence_length = images.shape[:2]
        
        if sequence_length >= self.window_size + self.chunk_size:
            # Window: 처음 window_size개 프레임 (관찰용)
            window_images = images[:, :self.window_size]  # [B, window_size, C, H, W]
            # Chunk: 그 다음 chunk_size개 프레임 (예측 타겟)
            chunk_actions = actions[:, self.window_size:self.window_size + self.chunk_size]  # [B, chunk_size, action_dim]
        else:
            # 시퀀스가 짧은 경우 적절히 처리
            window_images = images[:, :min(sequence_length, self.window_size)]
            chunk_actions = actions[:, -self.chunk_size:] if sequence_length >= self.chunk_size else actions
        
        # 텍스트 처리
        task_descriptions = batch.get("task_description", ["Navigate around obstacles to track the target cup"] * batch_size)
        if not isinstance(task_descriptions, list):
            task_descriptions = [task_descriptions] if isinstance(task_descriptions, str) else task_descriptions
        
        text_inputs = self.processor(text=task_descriptions, return_tensors="pt", padding=True, truncation=True)
        
        # 디바이스로 이동
        window_images = window_images.to(self.device)
        chunk_actions = chunk_actions.to(self.device)
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)
        
        # 타겟 준비
        targets = {
            "action_chunk": chunk_actions  # [B, chunk_size, action_dim]
        }
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = self.model(window_images, input_ids, attention_mask)
                loss_dict = self.compute_loss(predictions, targets)
        else:
            predictions = self.model(window_images, input_ids, attention_mask)
            loss_dict = self.compute_loss(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss_dict["total_loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_dict["total_loss"].backward()
            self.optimizer.step()
        
        self.scheduler.step()
        self.step_count += 1
        
        # 현재 학습률 추가
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        
        return loss_dict

    def save_checkpoint(self, save_path: str, epoch: int, metrics: Optional[Dict] = None):
        """체크포인트 저장"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'model_name': self.model_name,
                'action_dim': self.action_dim,
                'window_size': self.window_size,
                'chunk_size': self.chunk_size,
                'learning_rate': self.learning_rate
            }
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step_count = checkpoint.get('step', 0)
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {load_path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

    def evaluate(self, dataloader, num_batches: Optional[int] = None):
        """평가 실행"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches and i >= num_batches:
                    break
                
                try:
                    # Forward pass (train_step과 동일한 로직이지만 gradient 없음)
                    if isinstance(batch["rgb"], list):
                        pixel_values = []
                        for batch_images in batch["rgb"]:
                            if isinstance(batch_images, list):
                                batch_pixel_values = self.processor(images=batch_images, return_tensors="pt")["pixel_values"]
                                pixel_values.append(batch_pixel_values)
                            else:
                                pixel_values.append(batch_images)
                        pixel_values = torch.stack(pixel_values) if len(pixel_values) > 1 else pixel_values[0]
                    else:
                        pixel_values = batch["rgb"]
                    
                    task_descriptions = batch.get("raw_text", ["Navigate around obstacles to track the target cup"] * len(pixel_values))
                    if not isinstance(task_descriptions, list):
                        task_descriptions = [task_descriptions] if isinstance(task_descriptions, str) else task_descriptions.tolist()
                    
                    text_inputs = self.processor(text=task_descriptions, return_tensors="pt", padding=True, truncation=True)
                    
                    pixel_values = pixel_values.to(self.device)
                    input_ids = text_inputs["input_ids"].to(self.device)
                    attention_mask = text_inputs["attention_mask"].to(self.device)
                    
                    targets = {
                        "action_chunk": batch["action_chunck"].to(self.device) if "action_chunck" in batch else batch["action"].to(self.device)
                    }
                    
                    predictions = self.model(pixel_values, input_ids, attention_mask)
                    loss_dict = self.compute_loss(predictions, targets)
                    
                    total_loss += loss_dict["action_loss"].item()
                    total_mae += loss_dict["mae_avg"]
                    count += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation batch {i} failed: {e}")
                    continue
        
        if count > 0:
            return {
                "eval_loss": total_loss / count,
                "eval_mae": total_mae / count,
                "eval_batches": count
            }
        else:
            return {"eval_loss": float('inf'), "eval_mae": float('inf'), "eval_batches": 0}


# Loss tracking utility
class ActionLossTracker:
    """액션 손실 추적"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.total_losses = []
        self.maes = []
        self.lrs = []
        self.step_count = 0
    
    def update(self, loss_dict: Dict[str, float]):
        self.losses.append(loss_dict.get("action_loss", 0.0))
        self.total_losses.append(loss_dict.get("total_loss", 0.0))
        self.maes.append(loss_dict.get("mae_avg", 0.0))
        self.lrs.append(loss_dict.get("lr", 0.0))
        self.step_count += 1
    
    def get_averages(self, last_n: Optional[int] = None):
        if last_n:
            losses = self.losses[-last_n:]
            total_losses = self.total_losses[-last_n:]
            maes = self.maes[-last_n:]
            lrs = self.lrs[-last_n:]
        else:
            losses = self.losses
            total_losses = self.total_losses
            maes = self.maes
            lrs = self.lrs
        
        if not losses:
            return {}
        
        return {
            "avg_action_loss": sum(losses) / len(losses),
            "avg_total_loss": sum(total_losses) / len(total_losses),
            "avg_mae": sum(maes) / len(maes),
            "current_lr": lrs[-1] if lrs else 0.0,
            "steps": len(losses)
        }
