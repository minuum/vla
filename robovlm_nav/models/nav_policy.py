#!/usr/bin/env python3
"""
RoboVLM-Nav Policy 진입점

third_party/RoboVLMs에 의존하지 않고,
robovlm_nav/models/policy_head/nav_policy_impl.py (우리 구현)를 사용합니다.
"""

from robovlm_nav.models.policy_head.nav_policy_impl import (
    MobileVLAClassificationDecoder,
    MobileVLALSTMDecoder,
)
from robovlm_nav.models.policy_head.hybrid_action_head import HybridActionHead

# 공개 인터페이스 (configs의 type 필드에서 사용)
NavPolicy = MobileVLAClassificationDecoder
NavPolicyRegression = MobileVLALSTMDecoder

__all__ = [
    "NavPolicy",
    "NavPolicyRegression",
    "MobileVLAClassificationDecoder",
    "MobileVLALSTMDecoder",
    "HybridActionHead",
]
