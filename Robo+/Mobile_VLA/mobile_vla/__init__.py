"""
📚 Mobile VLA - Pure Mobile Vision-Language-Action System

🎯 Calvin 없는 순수 Mobile VLA 시스템
- mobile_vla_data_collector.py 100% 호환
- RoboVLMs 기술 적용
- 3D Mobile Robot 액션 예측
"""

__version__ = "0.1.0"
__author__ = "Mobile VLA Team"
__email__ = "mobile-vla@example.com"

# 🔥 Core imports
from .models.mobile_vla_model import MobileVLAModel
from .data.mobile_dataset import MobileVLADataset
from .training.mobile_trainer import MobileVLATrainer
from .training.mobile_trainer_simple import SimpleMobileVLATrainer

# 🧪 Kosmos integration
try:
    from .training.kosmos_trainer import MobileKosmosTrainer
    from .data.robovlms_adapter import MobileVLAToRoboVLMsAdapter
    HAS_KOSMOS = True
except ImportError:
    HAS_KOSMOS = False

# 🔧 Utilities
try:
    from .utils.data_utils import claw_matrix, generate_chunk_data, mobile_vla_sequence_chunking
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

__all__ = [
    # Core classes
    "MobileVLAModel",
    "MobileVLADataset", 
    "MobileVLATrainer",
    "SimpleMobileVLATrainer",
    # Kosmos integration (if available)
    "MobileKosmosTrainer" if HAS_KOSMOS else None,
    "MobileVLAToRoboVLMsAdapter" if HAS_KOSMOS else None,
    # Utilities (if available)
    "claw_matrix" if HAS_UTILS else None,
    "generate_chunk_data" if HAS_UTILS else None,
    "mobile_vla_sequence_chunking" if HAS_UTILS else None,
    # Metadata
    "__version__",
]

# Remove None values from __all__
__all__ = [item for item in __all__ if item is not None]

def get_version():
    """Get Mobile VLA version."""
    return __version__

def check_dependencies():
    """Check if all dependencies are available."""
    dependencies = {
        "torch": True,
        "transformers": True,
        "h5py": True,
        "numpy": True,
        "PIL": True,
        "tqdm": True,
        "kosmos": HAS_KOSMOS,
        "utils": HAS_UTILS,
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        dependencies["torch"] = False
        
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        dependencies["transformers"] = False
        
    try:
        import h5py
        dependencies["h5py"] = True
    except ImportError:
        dependencies["h5py"] = False
        
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        dependencies["numpy"] = False
        
    try:
        from PIL import Image
        dependencies["PIL"] = True
    except ImportError:
        dependencies["PIL"] = False
        
    try:
        import tqdm
        dependencies["tqdm"] = True
    except ImportError:
        dependencies["tqdm"] = False
    
    return dependencies
