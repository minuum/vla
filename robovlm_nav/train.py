#!/usr/bin/env python3
"""
RoboVLM-Nav Training Entry Point

아키텍처:
- third_party/RoboVLMs : 원본 그대로 유지 (수정 금지)
- robovlm_nav/         : 우리의 모든 커스텀 코드 (datasets, models, trainer 등)

이 스크립트는 robovlm_nav의 커스텀 컴포넌트들을 robovlms 네임스페이스에
동적 주입하여 third_party/RoboVLMs의 main.py가 그대로 작동하게 합니다.
"""

import sys
import os
from pathlib import Path

# ── Path 설정 ─────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "RoboVLMs"))

import robovlms.data
import robovlms.model.policy_head
import robovlms.train

# ── 커스텀 컴포넌트 import ─────────────────────────────────────
# Dataset
from robovlm_nav.datasets.nav_dataset import NavDataset
from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset as NavH5DatasetImpl

# Policy Heads
from robovlm_nav.models.policy_head.nav_policy_impl import (
    MobileVLALSTMDecoder as NavLSTMDecoder,
    MobileVLAClassificationDecoder as NavClassificationDecoder,
)
from robovlm_nav.models.policy_head.hybrid_action_head import HybridActionHead

# Trainer
from robovlm_nav.trainer.nav_trainer import MobileVLATrainer as NavTrainer

# ── robovlms 네임스페이스에 동적 주입 ──────────────────────────
# Datasets (configs에서 type으로 참조하는 이름들)
setattr(robovlms.data, "NavDataset", NavDataset)
setattr(robovlms.data, "MobileVLAH5Dataset", NavH5DatasetImpl)   # upstream과 동일 인터페이스 유지

# Policy Heads
setattr(robovlms.model.policy_head, "NavPolicy", NavClassificationDecoder)
setattr(robovlms.model.policy_head, "NavPolicyRegression", NavLSTMDecoder)
setattr(robovlms.model.policy_head, "MobileVLAClassificationDecoder", NavClassificationDecoder)
setattr(robovlms.model.policy_head, "MobileVLALSTMDecoder", NavLSTMDecoder)
setattr(robovlms.model.policy_head, "HybridActionHead", HybridActionHead)

# Trainer
setattr(robovlms.train, "MobileVLATrainer", NavTrainer)

if __name__ == "__main__":
    # third_party/RoboVLMs/main.py를 그대로 사용
    os.chdir(ROOT_DIR / "third_party" / "RoboVLMs")
    from main import main
    main()
