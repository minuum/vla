"""
FastAPI Inference Server for Mobile VLA
교수님 서버에서 모델 호출을 위한 API 서버

입력: 이미지 + Language instruction
출력: 2DOF actions [linear_x, linear_y]
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import base64
import io
import time
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Inference API", version="1.0.0")

# Global model instance (lazy loading)
model_instance = None


class InferenceRequest(BaseModel):
    """추론 요청 스키마"""
    image: str  # base64 encoded image
    instruction: str  # Language instruction (e.g., "...on the left")
    

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    

class MobileVLAInference:
    """Mobile VLA 추론 파이프라인"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        """
        Args:
            checkpoint_path: LoRA Fine-tuned 모델 checkpoint 경로
            config_path: Config JSON 파일 경로
            device: "cuda" or "cpu"
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        logger.info(f"Loading model from {checkpoint_path}")
        logger.info(f"Using device: {device}")
        
        # Load config
        import json
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self._load_model()
        
        logger.info("Model loaded successfully")
        
    def _load_model(self):
        """모델 로딩"""
        # TODO: RoboVLMs 모델 로딩 로직 구현
        # Lightning checkpoint에서 모델 로드
        import sys
        sys.path.append('RoboVLMs_upstream')
        
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        # Load from checkpoint
        self.model = MobileVLATrainer.load_from_checkpoint(
            self.checkpoint_path,
            config_path=self.config_path
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup image transforms
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['image_mean'],
                std=self.config['image_std']
            )
        ])
        
    def preprocess_image(self, image_base64: str) -> torch.Tensor:
        """Base64 이미지를 모델 입력 형식으로 변환"""
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform
        image_tensor = self.image_transform(image)
        
        # Add batch and window dimensions: (3, 224, 224) -> (1, 1, 3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image_base64: str, instruction: str) -> tuple[np.ndarray, float]:
        """
        추론 실행
        
        Returns:
            (action, latency_ms): 2DOF action과 latency
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Preprocess
            image_tensor = self.preprocess_image(image_base64)
            
            # Forward pass
            # TODO: 실제 모델 forward 로직 구현
            # outputs = self.model(image_tensor, instruction)
            # action = outputs['action']  # (1, 2)
            
            # Placeholder for now
            action = np.array([1.15, 0.5])  # [linear_x, linear_y]
            
        latency_ms = (time.time() - start_time) * 1000
        
        return action, latency_ms


def get_model():
    """모델 인스턴스 가져오기 (lazy loading)"""
    global model_instance
    
    if model_instance is None:
        # Best LoRA model checkpoint
        checkpoint_path = "runs/mobile_vla_no_chunk_20251209/checkpoints/epoch=04-val_loss=0.001.ckpt"
        config_path = "Mobile_VLA/configs/mobile_vla_no_chunk_20251209.json"
        
        model_instance = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    return model_instance


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Mobile VLA Inference API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    추론 엔드포인트
    
    Example:
        {
            "image": "base64_encoded_image_string",
            "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
        }
    """
    try:
        model = get_model()
        
        action, latency_ms = model.predict(
            image_base64=request.image,
            instruction=request.instruction
        )
        
        return InferenceResponse(
            action=action.tolist(),
            latency_ms=latency_ms,
            model_name="mobile_vla_no_chunk_20251209"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint():
    """
    테스트 엔드포인트 - 더미 데이터로 API 테스트
    
    Returns:
        샘플 prediction 결과
    """
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    
    # Create dummy RGB image
    dummy_img = Image.new('RGB', (1280, 720), color=(255, 0, 0))
    
    # Convert to base64
    buffer = BytesIO()
    dummy_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test instruction
    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
    
    # Simulate prediction
    dummy_action = [1.15, 0.319]  # Expected for "left"
    
    return {
        "message": "Test endpoint - using dummy data",
        "instruction": instruction,
        "action": dummy_action,
        "note": "This is a test endpoint. Use POST /predict for real inference."
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
