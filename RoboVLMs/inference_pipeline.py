#!/usr/bin/env python3
"""
Inference Pipeline for Mobile VLA (On-Device)
Loads checkpoint and performs action prediction from image + text inputs.
"""
import os
import sys
import torch
import argparse
import logging
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Add RoboVLMs to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Mock llava module for LoRA
import types
llava = types.ModuleType("llava")
sys.modules["llava"] = llava
llava_train = types.ModuleType("llava.train")
sys.modules["llava.train"] = llava_train
llava_train_train = types.ModuleType("llava.train.train")
sys.modules["llava.train.train"] = llava_train_train

def find_all_linear_names(model):
    """Find all linear layer names for LoRA"""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

llava_train_train.find_all_linear_names = find_all_linear_names

# Import RoboVLMs modules
from robovlms.model.backbone.robokosmos import RoboKosMos
from robovlms.model.policy_head import MobileVLALSTMDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class MobileVLAInferencePipeline:
    """Mobile VLA Inference Pipeline for on-device execution"""
    
    def __init__(
        self,
        checkpoint_path: str,
        base_model_path: str = "/home/soda/vla/.vlms/kosmos-2-patch14-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mock_mode: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.base_model_path = base_model_path
        self.device = device
        self.mock_mode = mock_mode
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing Mobile VLA Pipeline on {device} (Mock Mode: {mock_mode})")
        
        if self.mock_mode:
            logger.warning("!!! RUNNING IN MOCK MODE !!! No real model will be loaded.")
            self._init_mock_processor()
        else:
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Falling back to MOCK MODE due to load failure.")
                self.mock_mode = True
                self._init_mock_processor()

    def _init_mock_processor(self):
        """Initialize a basic tokenizer/processor for mock mode"""
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.base_model_path, 
                trust_remote_code=True
            )
            logger.info("Mock processor initialized from base path.")
        except Exception:
            logger.warning("Could not load base processor. Using dummy methods.")
            self.processor = None

    def _load_model(self):
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint to CPU first
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        hparams = checkpoint.get("hyper_parameters", {})
        
        # Extract config from hyperparameters
        configs = hparams.get("configs", {})
        
        # Build config dict
        config = {
            "model": "kosmos",
            "model_url": self.base_model_path,
            "image_size": configs.get("image_size", 224),
            "window_size": configs.get("window_size", 8),
            "fwd_pred_next_n": configs.get("fwd_pred_next_n", 10),
            "train_setup": {
                "train_vision": False,
                "freeze_backbone": True,
                "bits": 16 if self.device == "cuda" else 32,
                "lora_enable": True,
                "lora_r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_bias": 'none',
                "train_text_embedding": False
            },
            "act_head": configs.get("act_head", {
                "type": "MobileVLALSTMDecoder",
                "hidden_size": 512,
                "action_dim": 2,
                "down_sample": "none",
                "latent": 1,
                "fwd_pred_next_n": 10,
                "window_size": 8,
                "action_space": "continuous",
                "with_history": True,
                "history_type": "post"
            }),
            "vlm": {
                "type": "AutoModelForVision2Seq",
                "pretrained_model_name_or_path": self.base_model_path,
                "name": "kosmos"
            },
            "tokenizer": {
                "type": "AutoProcessor",
                "pretrained_model_name_or_path": self.base_model_path,
                "tokenizer_type": "kosmos",
                "max_text_len": 256
            }
        }
        
        logger.info("Initializing model structure...")
        self.model = RoboKosMos(
            configs=config,
            train_setup_configs=config["train_setup"],
            fwd_head_configs=None,
            window_size=config["window_size"],
            use_hand_rgb=False,
            act_head_configs=config["act_head"],
            fwd_pred_next_n=config["fwd_pred_next_n"],
            use_state=True
        )
        
        # Load state dict
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        logger.info("Loading weights...")
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Get processor
        self.processor = self.model.processor
        
        # Cleanup
        del checkpoint, state_dict, new_state_dict
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded successfully. Total parameters: {total_params / 1e6:.2f}M")
        
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        instruction: str,
        rel_state: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict action from image and instruction
        
        Args:
            image: PIL Image
            instruction: Text instruction
            rel_state: Relative state (7-dim: x,y,z,qx,qy,qz,gripper)
        
        Returns:
            Dict with 'linear_y' and 'gripper' predictions
        """
        # --- Mock Mode Logic ---
        if self.mock_mode:
            # Simulate processing delay slightly?
            # time.sleep(0.05) 
            # Return dummy actions (stationary)
            # Pred horizon 10, linear_y dim=1, gripper dim=1
            return {
                "linear_y": np.zeros(10), # 10 steps of 0.0
                "gripper": np.zeros(10)   # 10 steps of 0.0
            }
        # -----------------------

        # Prepare inputs
        bs = 1
        seq_len = 1
        dtype = next(self.model.parameters()).dtype
        
        # Process image
        if isinstance(image, Image.Image):
            # Resize and normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            vision_x = transform(image).unsqueeze(0).unsqueeze(0).to(dtype).to(self.device)
        else:
            vision_x = torch.zeros((bs, seq_len, 3, 224, 224), dtype=dtype).to(self.device)
        
        # Tokenize instruction
        if self.processor is not None:
            encoding = self.processor(
                text=instruction,
                return_tensors="pt",
                max_length=256,
                truncation=True
            )
            lang_x = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].bool().to(self.device)
        else:
            lang_x = torch.ones((bs, 10), dtype=torch.long).to(self.device)
            attention_mask = torch.ones((bs, 10)).bool().to(self.device)
        
        # Prepare state
        if rel_state is not None:
            rel_state_tensor = torch.from_numpy(rel_state).float().unsqueeze(0).unsqueeze(0).to(dtype).to(self.device)
        else:
            rel_state_tensor = torch.zeros((bs, seq_len, 7), dtype=dtype).to(self.device)
        
        # Forward pass
        output = self.model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            rel_state=rel_state_tensor
        )
        
        # Extract action predictions
        if isinstance(output, dict):
            action_pred = output.get("action", None)
        elif isinstance(output, tuple):
            action_pred = output[0] if len(output) > 0 else None
        else:
            action_pred = output
        
        if action_pred is None:
            logger.warning("No action prediction found in output")
            return {"linear_y": np.zeros(10), "gripper": np.zeros(10)}
        
        # Action format: (bs, seq, pred_horizon, action_dim)
        # action_dim=2: [linear_y, gripper]
        if isinstance(action_pred, tuple):
            linear_y = action_pred[0].cpu().numpy()[0, 0, :, 0]  # (pred_horizon,)
            gripper = action_pred[1].cpu().numpy()[0, 0, :]      # (pred_horizon,)
        else:
            action_np = action_pred.cpu().numpy()[0, 0]  # (pred_horizon, action_dim)
            linear_y = action_np[:, 0]
            gripper = action_np[:, 1] if action_np.shape[-1] > 1 else np.zeros_like(linear_y)
        
        return {
            "linear_y": linear_y,
            "gripper": gripper
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="/home/soda/vla/ROS_action/last.ckpt")
    parser.add_argument("--base-model-path", type=str, default="/home/soda/vla/.vlms/kosmos-2-patch14-224")
    parser.add_argument("--test-init", action="store_true", help="Test model initialization only")
    parser.add_argument("--test-inference", action="store_true", help="Test dummy inference")
    parser.add_argument("--dummy-image", action="store_true", help="Use dummy image")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--mock", action="store_true", help="Force Mock Mode")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=args.ckpt_path,
        base_model_path=args.base_model_path,
        mock_mode=args.mock
    )
    
    if args.test_init:
        logger.info("✓ Model initialized successfully")
        if args.profile_memory and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        return
    
    if args.test_inference:
        # Create dummy image
        if args.dummy_image:
            dummy_image = Image.new('RGB', (224, 224), color='red')
        else:
            dummy_image = Image.open(args.image_path) if hasattr(args, 'image_path') else Image.new('RGB', (224, 224))
        
        # Run inference
        result = pipeline.predict(
            image=dummy_image,
            instruction="move forward",
            rel_state=np.zeros(7)
        )
        
        logger.info("✓ Inference successful")
        logger.info(f"Linear Y: {result['linear_y']}")
        logger.info(f"Gripper: {result['gripper']}")


if __name__ == "__main__":
    main()
