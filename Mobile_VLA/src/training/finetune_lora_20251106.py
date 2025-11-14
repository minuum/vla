#!/usr/bin/env python3
"""
Mobile VLA LoRA Fine-tuning Script for 20251106 Episodes
참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mobile VLA 모델은 기존 구현 사용
from data.mobile_vla_h5_dataset import create_mobile_vla_h5_dataloader
from model.mobile_vla_model import MobileVLAModel, MobileVLALoss

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoRAFineTuner:
    """
    Mobile VLA LoRA Fine-tuning 클래스
    참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
    RoboVLMs의 학습 파이프라인 구조를 참고하여 LoRA Fine-tuning에 맞게 수정
    """
    
    def __init__(
        self,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        LoRA Fine-tuner 초기화
        
        Args:
            config_path: 설정 파일 경로
            device: 디바이스 (cuda/cpu)
        """
        self.device = device
        
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"🚀 LoRA Fine-tuning 초기화 (Device: {self.device})")
        logger.info(f"📄 Config: {config_path}")
        
        # 출력 디렉토리 생성
        self.output_dir = Path(self.config['output_root'])
        self.log_dir = Path(self.config['log_root'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로더 생성
        self._setup_dataloaders()
        
        # 모델 생성
        self._setup_model()
        
        # Optimizer & Scheduler 설정
        self._setup_optimizer()
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epoch_times = []
        
        logger.info("✅ LoRA Fine-tuner 초기화 완료")
    
    def _setup_dataloaders(self):
        """데이터 로더 설정"""
        logger.info("📊 데이터 로더 설정 중...")
        
        train_config = self.config['train_dataset']
        
        self.train_loader, self.val_loader = create_mobile_vla_h5_dataloader(
            data_dir=train_config['data_dir'],
            episode_pattern=train_config['episode_pattern'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            window_size=self.config['window_size'],
            action_chunk_size=self.config['fwd_pred_next_n'],
            train_split=train_config['train_split']
        )
        
        logger.info(f"✅ 데이터 로더 설정 완료")
        logger.info(f"  - Train batches: {len(self.train_loader)}")
        logger.info(f"  - Val batches: {len(self.val_loader)}")
    
    def _setup_model(self):
        """모델 설정"""
        logger.info("🤖 모델 설정 중...")
        
        act_head_config = self.config['act_head']
        train_setup = self.config['train_setup']
        
        # Mobile VLA 모델 생성
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py:34-50
        self.model = MobileVLAModel(
            vlm_model_name=self.config['model_url'],
            action_dim=act_head_config['action_dim'],
            hidden_size=act_head_config['hidden_size'],
            lstm_layers=2,
            lora_r=train_setup['lora_r'],
            lora_alpha=train_setup['lora_alpha'],
            lora_dropout=train_setup['lora_dropout'],
            window_size=self.config['window_size']
        ).to(self.device)
        
        # Loss 함수 생성
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/policy_head/base_policy.py:118-160
        self.loss_fn = MobileVLALoss(
            movement_weight=1.0,
            gripper_weight=0.0  # Mobile VLA는 gripper 없음
        )
        
        # 모델 크기 정보
        size_info = self.model.get_model_size()
        logger.info("✅ 모델 설정 완료")
        logger.info(f"  - Total params: {size_info['total_parameters']:,}")
        logger.info(f"  - Trainable params: {size_info['trainable_parameters']:,}")
        logger.info(f"  - LoRA 비율: {size_info['trainable_parameters'] / size_info['total_parameters'] * 100:.2f}%")
    
    def _setup_optimizer(self):
        """Optimizer & Scheduler 설정"""
        logger.info("⚙️ Optimizer 설정 중...")
        
        # AdamW Optimizer
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Cosine Annealing LR Scheduler
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        total_steps = len(self.train_loader) * self.config['trainer']['max_epochs']
        warmup_steps = int(total_steps * self.config['warmup_epochs'])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config['learning_rate'] * self.config['min_lr_scale']
        )
        
        logger.info("✅ Optimizer 설정 완료")
        logger.info(f"  - Learning rate: {self.config['learning_rate']}")
        logger.info(f"  - Weight decay: {self.config['weight_decay']}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        한 에포크 학습
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        RoboVLMs의 학습 루프 구조 참고
        """
        self.model.train()
        total_loss = 0.0
        movement_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['trainer']['max_epochs']} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            images = batch["images"].to(self.device)  # (B, T, 3, 224, 224)
            actions = batch["actions"].to(self.device)  # (B, T, 2)
            language = batch["language"]
            
            # 타겟 데이터 준비 (2D 액션만)
            # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:884-887
            targets = {
                "movement_targets": actions.mean(dim=1)  # (B, 2) - linear_x, linear_y
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # 배치별로 처리 (텍스트가 다를 수 있음)
            # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py:470-540
            batch_outputs = []
            for i in range(images.shape[0]):
                single_image = images[i:i+1]
                single_text = language[i] if isinstance(language, list) else language
                
                outputs = self.model(single_image, single_text)
                batch_outputs.append(outputs)
            
            # 배치 출력 결합 (2D 액션만)
            combined_outputs = {
                "action_logits": torch.cat([out["action_logits"] for out in batch_outputs])
            }
            
            # Loss 계산
            # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/policy_head/base_policy.py:118-160
            losses = self.loss_fn(combined_outputs, targets)
            
            # Backward pass
            losses["total_loss"].backward()
            
            # Gradient clipping
            # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['trainer']['gradient_clip_val']
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Loss 기록
            total_loss += losses["total_loss"].item()
            movement_loss += losses["movement_loss"].item()
            num_batches += 1
            
            # Progress bar 업데이트
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "Loss": f"{losses['total_loss'].item():.4f}",
                "LR": f"{current_lr:.6f}"
            })
        
        # 평균 Loss 계산
        avg_total_loss = total_loss / num_batches
        avg_movement_loss = movement_loss / num_batches
        
        return {
            "total_loss": avg_total_loss,
            "movement_loss": avg_movement_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        한 에포크 검증
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        RoboVLMs의 검증 루프 구조 참고
        """
        self.model.eval()
        total_loss = 0.0
        movement_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.config['trainer']['max_epochs']} [Val]",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # 데이터를 디바이스로 이동
                images = batch["images"].to(self.device)
                actions = batch["actions"].to(self.device)
                language = batch["language"]
                
                # 타겟 데이터 준비 (2D 액션만)
                targets = {
                    "movement_targets": actions.mean(dim=1)
                }
                
                # Forward pass
                batch_outputs = []
                for i in range(images.shape[0]):
                    single_image = images[i:i+1]
                    single_text = language[i] if isinstance(language, list) else language
                    
                    outputs = self.model(single_image, single_text)
                    batch_outputs.append(outputs)
                
                # 배치 출력 결합 (2D 액션만)
                combined_outputs = {
                    "action_logits": torch.cat([out["action_logits"] for out in batch_outputs])
                }
                
                # Loss 계산
                losses = self.loss_fn(combined_outputs, targets)
                
                # Loss 기록
                total_loss += losses["total_loss"].item()
                movement_loss += losses["movement_loss"].item()
                num_batches += 1
                
                # Progress bar 업데이트
                progress_bar.set_postfix({
                    "Loss": f"{losses['total_loss'].item():.4f}"
                })
        
        # 평균 Loss 계산
        avg_total_loss = total_loss / num_batches
        avg_movement_loss = movement_loss / num_batches
        
        return {
            "total_loss": avg_total_loss,
            "movement_loss": avg_movement_loss
        }
    
    def train(self):
        """
        전체 학습 실행
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        RoboVLMs의 학습 실행 방식 참고
        """
        max_epochs = self.config['trainer']['max_epochs']
        check_val_every_n_epoch = self.config['trainer']['check_val_every_n_epoch']
        
        logger.info(f"🚀 LoRA Fine-tuning 시작 ({max_epochs} 에포크)")
        logger.info(f"📊 에피소드: {self.config['train_dataset']['episode_pattern']}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증 (주기적)
            if (epoch + 1) % check_val_every_n_epoch == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {"total_loss": 0.0, "movement_loss": 0.0}
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 에포크 시간
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # 기록 저장
            self.train_losses.append(train_metrics["total_loss"])
            self.val_losses.append(val_metrics["total_loss"])
            self.learning_rates.append(current_lr)
            
            # 로그 출력
            logger.info(f"Epoch {epoch+1}/{max_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            if (epoch + 1) % check_val_every_n_epoch == 0:
                logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # 최고 모델 저장
            # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
            if (epoch + 1) % check_val_every_n_epoch == 0 and val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                self.save_checkpoint(epoch, val_metrics, "best_model.pth")
                logger.info(f"  ✅ 최고 모델 저장 (Val Loss: {best_val_loss:.4f})")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, f"checkpoint_epoch_{epoch+1}.pth")
        
        total_time = time.time() - start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        logger.info("🎉 LoRA Fine-tuning 완료!")
        logger.info(f"  - 총 시간: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"  - 평균 에포크 시간: {avg_epoch_time:.1f}s")
        logger.info(f"  - 최고 Val Loss: {best_val_loss:.4f}")
        
        # 학습 결과 저장
        self.save_training_results()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, filename: str):
        """
        체크포인트 저장
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/train
        RoboVLMs의 체크포인트 저장 방식 참고
        """
        filepath = self.output_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 체크포인트 저장: {filepath}")
    
    def save_training_results(self):
        """학습 결과 저장"""
        results = {
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "avg_epoch_time": sum(self.epoch_times) / len(self.epoch_times),
            "total_epochs": len(self.train_losses),
            "best_val_loss": min([loss for loss in self.val_losses if loss > 0]),
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = self.log_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 학습 결과 저장: {results_file}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mobile VLA LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/billy/25-1kp/vla/Mobile_VLA/configs/finetune_mobile_vla_lora_20251106.json",
        help="Config file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # LoRA Fine-tuning 실행
    finetuner = LoRAFineTuner(
        config_path=args.config,
        device=args.device
    )
    
    finetuner.train()

if __name__ == "__main__":
    main()

