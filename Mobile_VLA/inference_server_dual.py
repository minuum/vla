"""
Mobile VLA Inference Server with Dual Strategy Support

Supports two inference strategies:
1. Chunk Reuse (default): Fast, 9x speedup, 20 FPS
2. Receding Horizon: Accurate, RoboVLMs-style, 2.2 FPS

Usage:
    export VLA_API_KEY="your-secret-key"
    export VLA_CHECKPOINT_PATH="path/to/checkpoint.ckpt"
    export VLA_CONFIG_PATH="path/to/config.json"
    python inference_server_dual.py
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import base64
import io
import time
from typing import List, Optional, Literal
import logging
import os
import secrets

# Import ActionBuffer
from action_buffer import ActionBuffer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Dual Strategy API", version="2.0.0")

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key():
    """Get API key from environment or generate one"""
    api_key = os.getenv("VLA_API_KEY")
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        logger.warning("="*60)
        logger.warning("⚠️  VLA_API_KEY 환경 변수가 없습니다!")
        logger.warning(f"생성된 API Key: {api_key}")
        logger.warning("다음 명령어로 저장하세요:")
        logger.warning(f'export VLA_API_KEY="{api_key}"')
        logger.warning("="*60)
    return api_key

VALID_API_KEY = get_api_key()

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """API Key 검증"""
    if api_key != VALID_API_KEY:
        logger.warning(f"❌ 인증 실패: {api_key[:10] if api_key else 'None'}...")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Global model instance (lazy loading)
model_instance = None


class InferenceRequest(BaseModel):
    """추론 요청 스키마"""
    image: str  # base64 encoded image
    instruction: str  # Language instruction
    strategy: Literal["chunk_reuse", "receding_horizon"] = "chunk_reuse"


class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]
    latency_ms: float
    model_name: str
    strategy: str
    source: str  # "inferred" or "reused"
    buffer_status: dict


class MobileVLAInference:
    """Mobile VLA 추론 with Dual Strategy"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Initialize action buffer
        self.action_buffer = ActionBuffer(chunk_size=10)
        
        logger.info(f"Loading model from {checkpoint_path}")
        logger.info(f"Using device: {device}")
        
        # Load config
        import json
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self._load_model()
        
        logger.info("✅ Model loaded successfully")
    
    def _load_model(self):
        """모델 로딩"""
        import sys
        sys.path.append('RoboVLMs_upstream')
        
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        self.model = MobileVLATrainer.load_from_checkpoint(
            self.checkpoint_path,
            config_path=self.config_path
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['image_mean'],
                std=self.config['image_std']
            )
        ])
    
    def preprocess_image(self, image_base64: str) -> torch.Tensor:
        """Base64 이미지를 모델 입력으로 변환"""
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.image_transform(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def _tokenize_instruction(self, instruction: str):
        """Instruction을 tokenize"""
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained(
            self.config['tokenizer']['pretrained_model_name_or_path']
        )
        
        encoded = processor.tokenizer(
            instruction,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config['tokenizer']['max_text_len'],
            truncation=True
        )
        
        lang_x = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        return lang_x, attention_mask
    
    def predict(self, image_base64: str, instruction: str, strategy: str = "chunk_reuse"):
        """
        Dual Strategy Prediction
        
        Args:
            image_base64: Base64 encoded image
            instruction: Natural language command
            strategy: "chunk_reuse" or "receding_horizon"
        
        Returns:
            (action, latency_ms, source, buffer_status)
        """
        if strategy == "chunk_reuse":
            return self._predict_chunk_reuse(image_base64, instruction)
        elif strategy == "receding_horizon":
            return self._predict_receding_horizon(image_base64, instruction)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _predict_chunk_reuse(self, image_base64: str, instruction: str):
        """
        Chunk Reuse Strategy (Fast, 9x speedup)
        
        Reuses buffered actions when available.
        Only infers when buffer is empty.
        """
        # Check buffer first
        if not self.action_buffer.is_empty():
            action = self.action_buffer.pop_action()
            buffer_status = self.action_buffer.get_status()
            logger.info(f"📦 Reused action: {action}, buffer: {buffer_status['size']}")
            return action, 0.0, "reused", buffer_status
        
        # Buffer empty - need inference
        start_time = time.time()
        
        try:
            # Get full action chunk
            action_chunk = self._do_inference(image_base64, instruction)
            
            # De-normalize entire chunk
            action_chunk = self._denormalize_actions(action_chunk)
            
            # Use first action
            action = action_chunk[0, :]
            
            # Store rest in buffer
            if len(action_chunk) > 1:
                self.action_buffer.push_chunk(action_chunk[1:, :])
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            action = np.array([0.0, 0.0])
        
        latency_ms = (time.time() - start_time) * 1000
        buffer_status = self.action_buffer.get_status()
        
        logger.info(f"🔄 Inferred chunk: {action}, buffer: {buffer_status['size']}")
        return action, latency_ms, "inferred", buffer_status
    
    def _predict_receding_horizon(self, image_base64: str, instruction: str):
        """
        Receding Horizon Strategy (Accurate, RoboVLMs-style)
        
        Always infers, never uses buffer.
        """
        start_time = time.time()
        
        try:
            # Always infer
            action_chunk = self._do_inference(image_base64, instruction)
            
            # De-normalize
            action_chunk = self._denormalize_actions(action_chunk)
            
            # Use only first action (discard rest)
            action = action_chunk[0, :]
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            action = np.array([0.0, 0.0])
        
        latency_ms = (time.time() - start_time) * 1000
        buffer_status = {"size": 0, "notes": "receding_horizon (no buffer)"}
        
        logger.info(f"🎯 Receding Horizon: {action}")
        return action, latency_ms, "inferred", buffer_status
    
    def _do_inference(self, image_base64: str, instruction: str) -> np.ndarray:
        """
        Perform model inference
        
        Returns:
            action_chunk: (N, 2) numpy array
        """
        with torch.no_grad():
            # Preprocess
            image_tensor = self.preprocess_image(image_base64)
            lang_x, attention_mask = self._tokenize_instruction(instruction)
            
            # Model inference
            outputs = self.model.model.inference(
                vision_x=image_tensor,
                lang_x=lang_x,
                attention_mask=attention_mask
            )
            
            # Extract action chunk
            action_chunk = self._extract_action_chunk(outputs)
            
            return action_chunk
    
    def _extract_action_chunk(self, outputs) -> np.ndarray:
        """
        Extract action chunk from model outputs
        
        Returns:
            action_chunk: (N, 2) numpy array
        """
        if isinstance(outputs, dict) and 'action' in outputs:
            action_out = outputs['action']
            
            if isinstance(action_out, tuple):
                # Tuple: (velocities, gripper)
                velocities = action_out[0]
                
                # Extract chunk based on shape
                if len(velocities.shape) == 4:  # (1, 1, 10, 2)
                    action_chunk = velocities[0, 0, :, :].cpu().numpy()
                elif len(velocities.shape) == 3:  # (1, 10, 2)
                    action_chunk = velocities[0, :, :].cpu().numpy()
                else:
                    action_chunk = velocities.cpu().numpy()
                    if len(action_chunk.shape) == 1:
                        action_chunk = action_chunk.reshape(-1, 2)
            else:
                # Tensor
                if len(action_out.shape) == 4:
                    action_chunk = action_out[0, 0, :, :].cpu().numpy()
                elif len(action_out.shape) == 3:
                    action_chunk = action_out[0, :, :].cpu().numpy()
                else:
                    action_chunk = action_out.cpu().numpy()
                    if len(action_chunk.shape) == 1:
                        action_chunk = action_chunk.reshape(-1, 2)
            
            return action_chunk
        else:
            logger.warning(f"Unexpected outputs type: {type(outputs)}")
            return np.array([[0.0, 0.0]])
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        De-normalize actions from [-1, 1] to robot command space
        
        Args:
            actions: (N, 2) array in [-1, 1]
        
        Returns:
            actions: (N, 2) array in robot space
                - linear_x: [0.0, 2.0] m/s
                - angular_z: [-0.5, 0.5] rad/s
        """
        denorm_actions = actions.copy()
        
        # De-normalize
        denorm_actions[:, 0] = (denorm_actions[:, 0] + 1.0) * 1.0  # [-1,1] -> [0,2]
        denorm_actions[:, 1] = denorm_actions[:, 1] * 0.5          # [-1,1] -> [-0.5,0.5]
        
        # Clip to valid range
        denorm_actions[:, 0] = np.clip(denorm_actions[:, 0], 0.0, 2.0)
        denorm_actions[:, 1] = np.clip(denorm_actions[:, 1], -0.5, 0.5)
        
        return denorm_actions


def get_model():
    """모델 인스턴스 가져오기 (lazy loading)"""
    global model_instance
    
    if model_instance is None:
        checkpoint_path = os.getenv(
            "VLA_CHECKPOINT_PATH",
            "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt"
        )
        config_path = os.getenv(
            "VLA_CONFIG_PATH",
            "Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json"
        )
        
        model_instance = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    return model_instance


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Mobile VLA Dual Strategy API",
        "version": "2.0.0",
        "status": "running",
        "strategies": ["chunk_reuse", "receding_horizon"],
        "auth": "API Key required (X-API-Key header)"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "device_name": torch.cuda.get_device_name(0)
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": gpu_memory,
        "strategies": {
            "chunk_reuse": "Fast (9x), 20 FPS",
            "receding_horizon": "Accurate, 2.2 FPS"
        }
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Dual Strategy Inference Endpoint
    
    Strategies:
        - chunk_reuse (default): Fast, 9x speedup, 20 FPS
        - receding_horizon: Accurate, RoboVLMs-style, 2.2 FPS
    
    Example (Chunk Reuse):
        curl -X POST http://localhost:8000/predict \
          -H "X-API-Key: your-key" \
          -H "Content-Type: application/json" \
          -d '{"image": "...", "instruction": "go left", "strategy": "chunk_reuse"}'
    
    Example (Receding Horizon):
        curl -X POST http://localhost:8000/predict \
          -H "X-API-Key: your-key" \
          -d '{"image": "...", "instruction": "go left", "strategy": "receding_horizon"}'
    """
    try:
        model = get_model()
        
        action, latency_ms, source, buffer_status = model.predict(
            image_base64=request.image,
            instruction=request.instruction,
            strategy=request.strategy
        )
        
        logger.info(
            f"✅ {request.strategy}: {action}, "
            f"source={source}, latency={latency_ms:.1f}ms"
        )
        
        return InferenceResponse(
            action=action.tolist(),
            latency_ms=latency_ms,
            model_name="mobile_vla_left_chunk10",
            strategy=request.strategy,
            source=source,
            buffer_status=buffer_status
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint(api_key: str = Depends(verify_api_key)):
    """테스트 엔드포인트"""
    return {
        "message": "Dual Strategy API Test",
        "strategies": {
            "chunk_reuse": "Default, fast mode",
            "receding_horizon": "Accurate mode"
        },
        "note": "Use POST /predict for real inference"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*60)
    logger.info("🚀 Mobile VLA Dual Strategy Server Starting")
    logger.info("="*60)
    logger.info("Strategies:")
    logger.info("  • chunk_reuse (default): 20 FPS, 9x speedup")
    logger.info("  • receding_horizon: 2.2 FPS, RoboVLMs-style")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
