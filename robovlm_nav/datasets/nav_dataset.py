#!/usr/bin/env python3
"""
RoboVLM-Nav Dataset 진입점

third_party/RoboVLMs에 의존하지 않고,
robovlm_nav/datasets/nav_h5_dataset_impl.py (우리 구현)를 사용합니다.
"""

from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset
from torch.utils.data import DataLoader
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class NavDataset(MobileVLAH5Dataset):
    """
    RoboVLM-Nav 전용 데이터셋 클래스.
    nav_h5_dataset_impl.py의 MobileVLAH5Dataset을 상속하며
    RoboVLM-Nav 프로젝트 전용 커스터마이징을 여기에 추가합니다.
    """
    pass


def create_nav_dataloader(
    data_dir: str,
    episode_pattern: str = "episode_*.h5",
    batch_size: int = 2,
    num_workers: int = 4,
    window_size: int = 8,
    action_chunk_size: int = 10,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """NavDataset 기반 DataLoader 생성"""

    train_dataset = NavDataset(
        data_dir=data_dir,
        episode_pattern=episode_pattern,
        window_size=window_size,
        action_chunk_size=action_chunk_size,
        train_split=train_split,
        is_validation=False,
    )

    val_dataset = NavDataset(
        data_dir=data_dir,
        episode_pattern=episode_pattern,
        window_size=window_size,
        action_chunk_size=action_chunk_size,
        train_split=train_split,
        is_validation=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(f"📊 NavDataset 로더 생성 완료 | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_loader, val_loader


def test_nav_dataset():
    """NavDataset 기본 동작 테스트"""
    logger.info("🧪 NavDataset 테스트 시작")
    pass


if __name__ == "__main__":
    test_nav_dataset()
