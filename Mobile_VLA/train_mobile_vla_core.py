#!/usr/bin/env python3
"""
Mobile VLA 학습 런처 (실제 mobile_vla_data_collector 데이터로 스모크 학습)
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
import importlib.util
from pathlib import Path


def _load_attr_from_file(module_name: str, file_path: str, attr: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return getattr(module, attr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.getenv('MOBILE_VLA_DATA_DIR', '/home/soda/vla/ROS_action/mobile_vla_dataset/'))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sequence_length', type=int, default=18)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--use_lite_mode', action='store_true', default=True)
    parser.add_argument('--max_steps', type=int, default=10, help='스모크: 몇 스텝만 학습')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    configs = {
        'hidden_size': args.hidden_size,
        'use_lite_mode': args.use_lite_mode,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'max_epochs': 1,
    }

    # 데이터셋 (동적 임포트)
    root = Path(__file__).resolve().parents[1]
    dataset_cls = _load_attr_from_file(
        'mobile_dataset',
        str(root / 'data' / 'mobile_dataset.py'),
        'MobileVLADataset'
    )
    dataset = dataset_cls(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        normalize_actions=True
    )

    if len(dataset) == 0:
        print('⚠️ HDF5 데이터가 없습니다. data_dir를 확인하세요:', args.data_dir)
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # 트레이너 (동적 임포트로 패키지 의존성 회피)
    trainer_cls = _load_attr_from_file(
        'mobile_trainer_simple',
        str(root / 'training' / 'mobile_trainer_simple.py'),
        'SimpleMobileVLATrainer'
    )
    trainer = trainer_cls(configs)

    # 스모크 학습 루프
    steps = 0
    for batch in loader:
        # 배치 dict 텐서 외 필드 유지
        result = trainer.train_step(batch)
        print(json.dumps({k: float(v) if isinstance(v, (int, float)) else v for k, v in result.items()}, ensure_ascii=False))
        steps += 1
        if steps >= args.max_steps:
            break

    print('✅ 스모크 학습 완료: steps =', steps)


if __name__ == '__main__':
    main()


