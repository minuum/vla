"""
FastAPI Inference Server for Mobile VLA
교수님 서버에서 모델 호출을 위한 API 서버

입력: 이미지 + Language instruction
출력: 2DOF actions [linear_x, linear_y]  # 학습 데이터 기준

보안: API Key 인증
교수님 합의: Chunk=1,5,10 모델 지원 (첫 번째 액션만 실행)

Note: 학습 시 linear_x, linear_y로 학습했음 (실제 로봇 제어 시 linear_y를 angular_z로 매핑 필요)
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
from typing import List, Optional
import logging
import os
import secrets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Inference API", version="1.0.0")

# API Key 설정
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 환경 변수에서 API Key 읽기 (없으면 생성)
def get_api_key():
    api_key = os.getenv("VLA_API_KEY")
    if not api_key:
        # API Key 자동 생성 및 출력
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
        logger.warning(f"❌ 인증 실패: {api_key[:10]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

# Global model instance (lazy loading)
model_instance = None


class InferenceRequest(BaseModel):
    """추론 요청 스키마"""
    image: str  # base64 encoded image
    instruction: str  # Language instruction
    

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]  # 학습 데이터 기준
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    chunk_size: int  # 모델의 chunk size (fwd_pred_next_n)
    

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
        import sys
        from pathlib import Path
        
        # Add RoboVLMs to path
        robovlms_path = str(Path(__file__).parent / 'RoboVLMs_upstream')
        if robovlms_path not in sys.path:
            sys.path.insert(0, robovlms_path)
        
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        # Load from Lightning checkpoint
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        logger.info(f"Using config: {self.config_path}")
        
        self.model = MobileVLATrainer.load_from_checkpoint(
            self.checkpoint_path,
            config_path=self.config_path,
            strict=False  # Allow missing keys
        )
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
        
        # Get model info
        self.fwd_pred_next_n = self.config.get('fwd_pred_next_n', 1)
        self.action_dim = self.config.get('action_dim', 2)
        logger.info(f"Model: Chunk={self.fwd_pred_next_n}, Action_dim={self.action_dim}")
        
    def preprocess_image(self, image_base64: str) -> Image.Image:
        """Base64 이미지를 PIL Image로 변환"""
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    
    def predict(self, image_base64: str, instruction: str) -> tuple[np.ndarray, float]:
        """
        추론 실행
        
        Returns:
            (action, latency_ms): 2DOF action [linear_x, linear_y]와 latency (학습 데이터 기준)
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Preprocess image (base64 -> PIL)
            pil_image = self.preprocess_image(image_base64)
            
            # Load Kosmos processor
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                self.config.get('model_path', '.vlms/kosmos-2-patch14-224')
            )
            
            # Prepare inputs (Kosmos format)
            # Image normalization is handled by processor
            inputs = processor(
                images=pil_image,
                text=instruction,
                return_tensors="pt"
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Add batch dimension if needed
            if inputs['pixel_values'].dim() == 3:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)  # (1, C, H, W)
            
            # Add temporal dimension (window_size=1 for single image inference)
            if inputs['pixel_values'].dim() == 4:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(1)  # (1, 1, C, H, W)
            
            # Prepare inputs for model.inference()
            # Based on BaseTrainer.inference_step()
            rgb = inputs['pixel_values']  # (B, T, C, H, W)
            language = inputs['input_ids']  # (B, seq_len)
            text_mask = inputs['attention_mask']  # (B, seq_len)
            
            # Call model.inference() like in BaseTrainer
            # Note: self.model is MobileVLATrainer (Lightning module)
            # The actual model is in trainer_model.model
            # BaseRoboVLM.inference() uses vision_x and lang_x (not rgb and language)
            prediction = self.model.model.inference(
                vision_x=rgb,
                lang_x=language,
                attention_mask=text_mask,
                action_labels=None,
                action_mask=None,
                caption_labels=None,
                caption_mask=None,
                vision_gripper=None
            )
            
            # Extract action from prediction
            logger.info(f"Prediction keys: {prediction.keys()}")
            
            if 'action' in prediction:
                action_output = prediction['action']
                logger.info(f"Action type: {type(action_output)}")
                
                # Handle tuple format - first element is velocity (linear_x, linear_y)
                if isinstance(action_output, tuple):
                    logger.info(f"Action is tuple, extracting velocity")
                    action_output = action_output[0]  # velocity only
                    logger.info(f"Velocity shape: {action_output.shape if hasattr(action_output, 'shape') else type(action_output)}")
                
                # Extract from tensor
                # Expected: (B, seq_len, fwd_pred_next_n, 2) where 2 = [linear_x, linear_y]
                if isinstance(action_output, torch.Tensor):
                    logger.info(f"Action tensor shape: {action_output.shape}")
                    if action_output.dim() == 4:
                        # (B, seq_len, fwd_pred_next_n, 2) -> take first action
                        first_action = action_output[0, 0, 0, :].cpu().numpy()
                    elif action_output.dim() == 3:
                        # (B, fwd_pred_next_n, 2) -> take first action
                        first_action = action_output[0, 0, :].cpu().numpy()
                    elif action_output.dim() == 2:
                        # (B, 2) -> take first batch
                        first_action = action_output[0, :].cpu().numpy()
                    elif action_output.dim() == 1:
                        # (2,) -> use directly
                        first_action = action_output.cpu().numpy()
                    else:
                        logger.error(f"Unexpected action tensor shape: {action_output.shape}")
                        raise ValueError(f"Unexpected action tensor shape: {action_output.shape}")
                else:
                    logger.error(f"Action is not a tensor: {type(action_output)}")
                    raise ValueError(f"Action must be a torch.Tensor, got: {type(action_output)}")
            else:
                logger.error(f"No 'action' key in prediction. Keys: {list(prediction.keys())}")
                raise ValueError("Could not find action in prediction output")
            
            logger.info(f"Extracted action (linear_x, linear_y): {first_action}")
            
        latency_ms = (time.time() - start_time) * 1000
        
        return first_action, latency_ms


def get_model():
    """모델 인스턴스 가져오기 (lazy loading)"""
    global model_instance
    
    if model_instance is None:
        # 환경 변수로 모델 선택 가능
        model_name = os.getenv("VLA_MODEL_NAME", "chunk5_epoch6")
        
        # 모델별 체크포인트 및 config 경로
        model_configs = {
            "chunk5_epoch6": {
                "checkpoint": "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt",
                "config": "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
            },
            "chunk10_epoch8": {
                "checkpoint": "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt",
                "config": "Mobile_VLA/configs/mobile_vla_chunk10_20251217.json"
            }
        }
        
        # Get model config
        if model_name not in model_configs:
            logger.warning(f"Unknown model '{model_name}', defaulting to 'chunk5_epoch6'")
            model_name = "chunk5_epoch6"
        
        config = model_configs[model_name]
        checkpoint_path = os.getenv("VLA_CHECKPOINT_PATH", config["checkpoint"])
        config_path = os.getenv("VLA_CONFIG_PATH", config["config"])
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Config: {config_path}")
        
        model_instance = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    return model_instance


@app.get("/")
async def root():
    """API 정보 (인증 불필요)"""
    return {
        "name": "Mobile VLA Inference API",
        "version": "1.0.0",
        "status": "running",
        "auth": "API Key required (X-API-Key header)"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 (인증 불필요)"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": gpu_memory
    }


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """모델 정보 조회 (API Key 필수)"""
    try:
        model = get_model()
        
        return {
            "model_name": os.getenv("VLA_MODEL_NAME", "chunk5_epoch6"),
            "checkpoint_path": model.checkpoint_path,
            "config_path": model.config_path,
            "fwd_pred_next_n": model.fwd_pred_next_n,
            "action_dim": model.action_dim,
            "freeze_backbone": model.config.get('freeze_backbone', True),
            "lora_enable": model.config.get('lora_enable', False),
            "device": model.device
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/list")
async def list_models(api_key: str = Depends(verify_api_key)):
    """사용 가능한 모델 목록 조회 (API Key 필수)"""
    # 모델별 체크포인트 및 config 경로 (get_model 함수와 동일)
    model_configs = {
        "chunk5_epoch6": {
            "checkpoint": "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt",
            "config": "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json",
            "description": "Chunk5 Epoch 6 - Best model (val_loss=0.067)",
            "fwd_pred_next_n": 5,
            "recommended": True
        },
        "chunk10_epoch8": {
            "checkpoint": "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt",
            "config": "Mobile_VLA/configs/mobile_vla_chunk10_20251217.json",
            "description": "Chunk10 Epoch 8 (val_loss=0.312)",
            "fwd_pred_next_n": 10,
            "recommended": False
        }
    }
    
    current_model = os.getenv("VLA_MODEL_NAME", "chunk5_epoch6")
    
    return {
        "available_models": model_configs,
        "current_model": current_model,
        "model_loaded": model_instance is not None
    }


class ModelSwitchRequest(BaseModel):
    """모델 전환 요청 스키마"""
    model_name: str  # chunk5_epoch6, chunk10_epoch8, no_chunk_epoch4


@app.post("/model/switch")
async def switch_model(request: ModelSwitchRequest, api_key: str = Depends(verify_api_key)):
    """모델 전환 (API Key 필수)
    
    서버 재시작 없이 런타임에 모델 변경 가능
    """
    global model_instance
    
    try:
        available_models = ["chunk5_epoch6", "chunk10_epoch8"]
        
        if request.model_name not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Available: {available_models}"
            )
        
        logger.info(f"🔄 Switching model to: {request.model_name}")
        
        # 기존 모델 메모리 해제
        if model_instance is not None:
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_instance = None
        
        # 환경 변수 업데이트
        os.environ["VLA_MODEL_NAME"] = request.model_name
        
        # 새 모델 로드
        new_model = get_model()
        
        logger.info(f"✅ Model switched successfully to: {request.model_name}")
        
        return {
            "status": "success",
            "previous_model": os.getenv("VLA_MODEL_NAME", "chunk5_epoch6"),
            "current_model": request.model_name,
            "model_info": {
                "checkpoint_path": new_model.checkpoint_path,
                "fwd_pred_next_n": new_model.fwd_pred_next_n,
                "action_dim": new_model.action_dim,
                "device": new_model.device
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    추론 엔드포인트 (API Key 필수)
    
    Example:
        curl -X POST http://localhost:8000/predict \
          -H "X-API-Key: your-api-key" \
          -H "Content-Type: application/json" \
          -d '{"image": "base64_string", "instruction": "..."}'
    """
    try:
        model = get_model()
        
        action, latency_ms = model.predict(
            image_base64=request.image,
            instruction=request.instruction
        )
        
        logger.info(f"✅ Prediction: {action}, Latency: {latency_ms:.1f}ms")
        
        return InferenceResponse(
            action=action.tolist(),
            latency_ms=latency_ms,
            model_name=os.getenv("VLA_MODEL_NAME", "chunk5_epoch6"),
            chunk_size=model.fwd_pred_next_n
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint(api_key: str = Depends(verify_api_key)):
    """
    테스트 엔드포인트 - 더미 데이터로 API 테스트 (API Key 필수)
    
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
    
    # Simulate prediction (2 DOF)
    dummy_action = [1.15, 0.319]  # [linear_x, angular_z]
    
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
