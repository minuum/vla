"""
🏋️ Mobile VLA Training Module

Lightning 및 Simple 트레이너
"""

from .mobile_trainer import MobileVLATrainer
from .mobile_trainer_simple import SimpleMobileVLATrainer

try:
    from .kosmos_trainer import MobileKosmosTrainer
    HAS_KOSMOS_TRAINER = True
except ImportError:
    HAS_KOSMOS_TRAINER = False

__all__ = [
    "MobileVLATrainer",
    "SimpleMobileVLATrainer",
    "MobileKosmosTrainer" if HAS_KOSMOS_TRAINER else None,
]

# Remove None values
__all__ = [item for item in __all__ if item is not None]