"""
Mobile VLA Inference Pipeline
모델 로딩 및 추론 파이프라인 구현
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys
from robovlms.data.data_utils import unnoramalize_action

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))


class MobileVLAInferencePipeline:
    """
    Mobile VLA 추론 파이프라인
    
    Steps:
        1. Image preprocessing (resize, normalize)
        2. VLM forward pass (frozen or LoRA)
        3. Extract context vector (hidden states)
        4. Action Head prediction
        5. Denormalization
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda"
    ):
        """
        Args:
            checkpoint_path: Lightning checkpoint (.ckpt)
            config_path: Config JSON
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        
        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)
            
        # Load model
        self._load_model()
        
        # Setup transforms
        self._setup_transforms()
        
    def _load_model(self):
        """모델 로딩 from Lightning checkpoint"""
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        print(f"Loading model from {self.checkpoint_path}")
        
        # Load from checkpoint
        self.trainer = MobileVLATrainer.load_from_checkpoint(
            str(self.checkpoint_path),
            config_path=str(self.config_path),
            map_location=self.device
        )
        
        self.trainer.model.to(self.device)
        self.trainer.model.eval()
        
        print("Model loaded successfully")
        
    def _setup_transforms(self):
        """이미지 전처리 설정"""
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['image_mean'],
                std=self.config['image_std']
            )
        ])
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            image_tensor: (1, window_size, 3, 224, 224)
        """
        # Transform single image
        img_tensor = self.image_transform(image)  # (3, 224, 224)
        
        # Window size
        window_size = self.config['window_size']
        
        # Repeat for window (simplified: use same image)
        img_window = img_tensor.unsqueeze(0).repeat(window_size, 1, 1, 1)  # (8, 3, 224, 224)
        
        # Add batch dimension
        img_batch = img_window.unsqueeze(0)  # (1, 8, 3, 224, 224)
        
        return img_batch.to(self.device)
    
    def denormalize_action(self, action_normalized: torch.Tensor) -> np.ndarray:
        """
        Action denormalization
        
        Args:
            action_normalized: Normalized action in [-1, 1]
            
        Returns:
            action_denorm: Denormalized action
        """
        # Data Clipping Compensation (Gain 1.15)
        # 학습 데이터가 [-1, 1]로 잘못 클리핑되었으므로, 추론 시에 [-1.15, 1.15]로 강제 확장하여
        # 원래의 물리적 속도(최대 1.15 m/s)를 복원함.
        target_min = -1.15
        target_max = 1.15
        
        # Denormalize: [-1, 1] -> [-1.15, 1.15]
        action_denorm = unnoramalize_action(
            action_normalized.cpu().numpy(),
            action_min=target_min,
            action_max=target_max
        )
        
        return action_denorm
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        instruction: str
    ) -> dict:
        """
        추론 실행
        
        Args:
            image: PIL Image (RGB)
            instruction: Language instruction
            
        Returns:
            {
                'action': (2,) numpy array [linear_x, linear_y],
                'action_normalized': (2,) normalized action,
                'logits': raw logits (optional)
            }
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Prepare batch
        batch = {
            'images': image_tensor,  # (1, 8, 3, 224, 224)
            'language': [instruction]  # List[str]
        }
        
        # Prepare inputs suitable for model.inference()
        # Tokenize instruction
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            self.config['tokenizer']['pretrained_model_name_or_path'],
            trust_remote_code=True
        )
        
        encoded = processor.tokenizer(
            instruction,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config['tokenizer']['max_text_len'],
            truncation=True
        ).to(self.device)
        
        # Inference
        outputs = self.trainer.model.inference(
            vision_x=image_tensor,
            lang_x=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
        
        # Extract action
        action_pred = outputs['action']  # (1, 1, 2) or (1, 2)
        
        if isinstance(action_pred, tuple):
            action_pred = action_pred[0]
            
        if action_pred.dim() == 3:
            action_pred = action_pred.squeeze(1)  # (1, 2)
            
        action_normalized = action_pred.squeeze(0)  # (2,)
        
        # Denormalize
        action_denorm = self.denormalize_action(action_normalized)
        
        return {
            'action': action_denorm,
            'action_normalized': action_normalized.cpu().numpy(),
            'instruction': instruction
        }
    
    def predict_from_path(self, image_path: str, instruction: str) -> dict:
        """파일 경로에서 이미지 로드 및 추론"""
        image = Image.open(image_path).convert('RGB')
        return self.predict(image, instruction)


def test_pipeline():
    """파이프라인 테스트"""
    # Best LoRA model
    checkpoint_path = "runs/mobile_vla_no_chunk_20251209/checkpoints/epoch=04-val_loss=0.001.ckpt"
    config_path = "configs/mobile_vla_no_chunk_20251209.json"
    
    # Create pipeline
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test with dummy image
    dummy_image = Image.new('RGB', (1280, 720), color='red')
    
    # Test instructions
    instructions = [
        "Navigate around obstacles and reach the front of the beverage bottle on the left",
        "Navigate around obstacles and reach the front of the beverage bottle on the right"
    ]
    
    for instruction in instructions:
        result = pipeline.predict(dummy_image, instruction)
        print(f"\nInstruction: {instruction}")
        print(f"Action: {result['action']}")
        print(f"  linear_x: {result['action'][0]:.3f}")
        print(f"  linear_y: {result['action'][1]:.3f}")


if __name__ == "__main__":
    test_pipeline()
