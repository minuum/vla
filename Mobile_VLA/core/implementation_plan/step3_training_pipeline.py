#!/usr/bin/env python3
"""
Step 3: 학습 파이프라인 구현
데이터 로더, Loss 함수, Optimizer 설정, 학습 루프
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileVLADataset(Dataset):
    """Mobile VLA 데이터셋"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_episodes: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_episodes = max_episodes
        
        # 에피소드 목록 로드
        self.episodes = self._load_episodes()
        
        logger.info(f"📁 {split} 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
    
    def _load_episodes(self) -> List[Dict]:
        """에피소드 목록 로드"""
        episodes = []
        
        # 에피소드 파일들 찾기
        episode_files = list(self.data_dir.glob("episode_*.json"))
        
        if self.max_episodes:
            episode_files = episode_files[:self.max_episodes]
        
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                    episodes.append(episode_data)
            except Exception as e:
                logger.warning(f"에피소드 로드 실패: {episode_file}, {e}")
        
        return episodes
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터 아이템 반환"""
        episode = self.episodes[idx]
        
        # 이미지 로드 (시뮬레이션)
        # 실제로는 HDF5 파일에서 로드
        images = torch.randn(3, 224, 224)  # 시뮬레이션 데이터
        
        # 액션 로드 (시뮬레이션)
        # 실제로는 HDF5 파일에서 로드
        actions = torch.randn(3)  # X, Y, Gripper
        
        # 언어 명령
        language = episode.get("language", "go to the object")
        
        return {
            "images": images,
            "actions": actions,
            "language": language,
            "episode_id": episode.get("episode_id", idx)
        }

class MobileVLATrainer:
    """Mobile VLA 학습기"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"🚀 Mobile VLA 학습기 초기화 완료 (Device: {self.device})")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        movement_loss = 0.0
        gripper_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)
            language = batch["language"]
            
            # 타겟 데이터 준비
            targets = {
                "movement_targets": actions[:, :2],  # X, Y
                "gripper_targets": actions[:, 2]     # Gripper
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # 배치별로 처리 (텍스트가 다를 수 있음)
            batch_outputs = []
            for i in range(images.shape[0]):
                single_image = images[i:i+1]
                single_text = language[i]
                
                outputs = self.model(single_image, single_text)
                batch_outputs.append(outputs)
            
            # 배치 출력 결합
            combined_outputs = {
                "action_logits": torch.cat([out["action_logits"] for out in batch_outputs]),
                "gripper_logits": torch.cat([out["gripper_logits"] for out in batch_outputs])
            }
            
            # Loss 계산
            losses = self.loss_fn(combined_outputs, targets)
            
            # Backward pass
            losses["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Loss 기록
            total_loss += losses["total_loss"].item()
            movement_loss += losses["movement_loss"].item()
            gripper_loss += losses["gripper_loss"].item()
            num_batches += 1
            
            # Progress bar 업데이트
            progress_bar.set_postfix({
                "Loss": f"{losses['total_loss'].item():.4f}",
                "Movement": f"{losses['movement_loss'].item():.4f}",
                "Gripper": f"{losses['gripper_loss'].item():.4f}"
            })
        
        # 평균 Loss 계산
        avg_total_loss = total_loss / num_batches
        avg_movement_loss = movement_loss / num_batches
        avg_gripper_loss = gripper_loss / num_batches
        
        return {
            "total_loss": avg_total_loss,
            "movement_loss": avg_movement_loss,
            "gripper_loss": avg_gripper_loss
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        movement_loss = 0.0
        gripper_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1} [Val]",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # 데이터를 디바이스로 이동
                images = batch["images"].to(self.device)
                actions = batch["actions"].to(self.device)
                language = batch["language"]
                
                # 타겟 데이터 준비
                targets = {
                    "movement_targets": actions[:, :2],  # X, Y
                    "gripper_targets": actions[:, 2]     # Gripper
                }
                
                # Forward pass
                batch_outputs = []
                for i in range(images.shape[0]):
                    single_image = images[i:i+1]
                    single_text = language[i]
                    
                    outputs = self.model(single_image, single_text)
                    batch_outputs.append(outputs)
                
                # 배치 출력 결합
                combined_outputs = {
                    "action_logits": torch.cat([out["action_logits"] for out in batch_outputs]),
                    "gripper_logits": torch.cat([out["gripper_logits"] for out in batch_outputs])
                }
                
                # Loss 계산
                losses = self.loss_fn(combined_outputs, targets)
                
                # Loss 기록
                total_loss += losses["total_loss"].item()
                movement_loss += losses["movement_loss"].item()
                gripper_loss += losses["gripper_loss"].item()
                num_batches += 1
                
                # Progress bar 업데이트
                progress_bar.set_postfix({
                    "Loss": f"{losses['total_loss'].item():.4f}"
                })
        
        # 평균 Loss 계산
        avg_total_loss = total_loss / num_batches
        avg_movement_loss = movement_loss / num_batches
        avg_gripper_loss = gripper_loss / num_batches
        
        return {
            "total_loss": avg_total_loss,
            "movement_loss": avg_movement_loss,
            "gripper_loss": avg_gripper_loss
        }
    
    def train(self, num_epochs: int = 10, save_dir: str = "checkpoints"):
        """전체 학습 실행"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Mobile VLA 학습 시작 ({num_epochs} 에포크)")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate 업데이트
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 기록 저장
            self.train_losses.append(train_metrics["total_loss"])
            self.val_losses.append(val_metrics["total_loss"])
            self.learning_rates.append(current_lr)
            
            # 로그 출력
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # 최고 모델 저장
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                self.save_checkpoint(epoch, val_metrics, save_dir / "best_model.pth")
                logger.info(f"  ✅ 최고 모델 저장 (Val Loss: {best_val_loss:.4f})")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, save_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        logger.info("🎉 학습 완료!")
        self.plot_training_curves(save_dir)
    
    def save_checkpoint(self, epoch: int, metrics: Dict, filepath: Path):
        """체크포인트 저장"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")
    
    def plot_training_curves(self, save_dir: Path):
        """학습 곡선 플롯"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss 곡선
        axes[0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning Rate 곡선
        axes[1].plot(self.learning_rates, color='green')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True)
        
        # Loss 비교 (로그 스케일)
        axes[2].semilogy(self.train_losses, label='Train Loss', color='blue')
        axes[2].semilogy(self.val_losses, label='Val Loss', color='red')
        axes[2].set_title('Training and Validation Loss (Log Scale)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (Log Scale)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"학습 곡선 저장: {save_dir / 'training_curves.png'}")

def create_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = MobileVLADataset(data_dir, split="all")
    
    # Train/Val 분할
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"📊 데이터 로더 생성 완료")
    logger.info(f"  - Train: {len(train_dataset)} 샘플")
    logger.info(f"  - Val: {len(val_dataset)} 샘플")
    logger.info(f"  - Batch Size: {batch_size}")
    
    return train_loader, val_loader

def test_training_pipeline():
    """학습 파이프라인 테스트"""
    logger.info("🧪 Mobile VLA 학습 파이프라인 테스트 시작")
    
    try:
        # 데이터 디렉토리 생성 (테스트용)
        data_dir = Path("test_data")
        data_dir.mkdir(exist_ok=True)
        
        # 테스트 에피소드 생성
        for i in range(10):
            episode_data = {
                "episode_id": i + 1,
                "language": f"test_task_{i+1}",
                "timestamp": i
            }
            
            with open(data_dir / f"episode_{i+1}.json", 'w') as f:
                json.dump(episode_data, f)
        
        # 데이터 로더 생성
        train_loader, val_loader = create_data_loaders(str(data_dir), batch_size=2)
        
        # 모델 생성 (간단한 테스트용)
        from step2_mobile_vla_model import create_mobile_vla_model
        model, loss_fn = create_mobile_vla_model()
        
        # 학습기 생성
        trainer = MobileVLATrainer(model, loss_fn, train_loader, val_loader)
        
        # 짧은 학습 실행
        trainer.train(num_epochs=2, save_dir="test_checkpoints")
        
        logger.info("✅ 학습 파이프라인 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 학습 파이프라인 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🚀 Mobile VLA 학습 파이프라인 구현 시작")
    
    # 학습 파이프라인 테스트 실행
    success = test_training_pipeline()
    
    if success:
        logger.info("✅ Mobile VLA 학습 파이프라인 구현 완료")
        logger.info("🎯 다음 단계: 추론 시스템 구현")
    else:
        logger.error("❌ Mobile VLA 학습 파이프라인 구현 실패")
        logger.error("🔧 문제를 해결한 후 다시 시도해주세요")

if __name__ == "__main__":
    main()
