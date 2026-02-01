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
    if api_key is None:
        logger.warning("❌ 인증 실패: API Key가 제공되지 않음")
        raise HTTPException(
            status_code=403,
            detail="API Key required. Please provide X-API-Key header."
        )
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
    snap_to_grid: Optional[bool] = None  # Enable/Disable discrete action filtering
    snap_threshold: Optional[float] = 0.1 # Deadzone threshold (default 0.1)
    

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]  # 학습 데이터 기준
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    chunk_size: int  # 모델의 chunk size (fwd_pred_next_n)
    full_chunk: Optional[List[List[float]]] = None  # Future trajectory
    

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
        
        # Inference settings (from config or defaults)
        self.default_snap_to_grid = self.config.get('snap_to_grid', True)  # Default Enable
        self.default_snap_threshold = self.config.get('snap_threshold', 0.1)
        
        # Load model
        self._load_model()
        
        # History buffer for images
        self.window_size = self.config.get('window_size', 8)
        self.image_history = []
        
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
        
        # Load and cache processor
        from transformers import AutoProcessor
        processor_path = self.config.get('model_path', '.vlms/kosmos-2-patch14-224')
        logger.info(f"Loading processor from {processor_path}")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        
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
    
    def _apply_snap_to_grid(self, action: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        액션을 [-1.15, 0.0, 1.15] 이산 액션셋으로 매핑
        모델의 Tanh() 출력 특성과 데이터 정규화 편차를 고려하여 보정(Gain/Bias) 적용
        """
        targets = np.array([-1.15, 0.0, 1.15])
        snapped = np.zeros_like(action)
        
        # Action Dim: [linear_x, linear_y]
        for i in range(len(action.flat)):
            val = action.flat[i]
            
            # i=0: linear_x (전후), i=1: linear_y (좌우)
            if i % 2 == 0:
                # X축: Tanh() 압축 보상을 위해 1.2배 증폭
                val = val * 1.2
            else:
                # Y축: 모델의 양수 편향(Left 편애) 보정을 위해 시프트 후 증폭
                val = (val - 0.15) * 2.0
            
            # 가장 가까운 타겟값 선택
            closest_idx = np.argmin(np.abs(targets - val))
            snapped.flat[i] = targets[closest_idx]
            
        return snapped

    def predict(self, image_base64: str, instruction: str, 
                snap_to_grid: Optional[bool] = None, 
                snap_threshold: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, float]:
        """
        추론 실행
        
        Args:
            image_base64: Image data
            instruction: Text command
            snap_to_grid: Force enable/disable snap logic (overrides config)
            snap_threshold: Deadzone threshold (overrides config)

        Returns:
            (action, full_chunk, latency_ms): 2DOF action [linear_x, linear_y]와 latency
        """
        start_time = time.time()
        
        # Determine settings
        use_snap = snap_to_grid if snap_to_grid is not None else self.default_snap_to_grid
        threshold = snap_threshold if snap_threshold is not None else self.default_snap_threshold
        
        with torch.no_grad():
            # Preprocess image (base64 -> PIL)
            pil_image = self.preprocess_image(image_base64)
            
            # Prepare inputs (Kosmos format)
            # Use cached processor
            processor = self.processor
            
            # Prepare inputs (Kosmos format)
            # Match RoboVLMs training prompt: "<grounding>An image of a robot {instruction}"
            # Only prepend if not already present to avoid duplication
            clean_instruction = instruction.strip()
            if not clean_instruction.startswith("<grounding>"):
                if not clean_instruction.startswith("An image of a robot"):
                    full_prompt = f"<grounding>An image of a robot {clean_instruction}"
                else:
                    full_prompt = f"<grounding>{clean_instruction}"
            else:
                full_prompt = clean_instruction
                
            # Add to history buffer
            self.image_history.append(pil_image)
            if len(self.image_history) > self.window_size:
                self.image_history.pop(0)
            
            # Pad history if not full
            current_history = self.image_history.copy()
            while len(current_history) < self.window_size:
                current_history.insert(0, current_history[0])
            
            # Process all images in history
            inputs = processor(
                images=current_history,
                text=[full_prompt] * self.window_size,
                return_tensors="pt"
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Add batch dimension (B, T, C, H, W)
            if inputs['pixel_values'].dim() == 4:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)  # (1, T, C, H, W)
            
            # Prepare inputs for model.inference()
            rgb = inputs['pixel_values']  # (B, T, C, H, W)
            language = inputs['input_ids']  # (B, T, seq_len) -> model expects (B, seq_len)?
            # Note: RoboKosMos model typically handles language as (B, seq_len) 
            # and applies it to the visual sequence.
            if language.dim() == 3:
                language = language[:, -1, :] # Take last instruction
            
            # Add hand_rgb dummy (B, T, C, H, W)
            hand_rgb = torch.zeros_like(rgb)
            text_mask = inputs['attention_mask']  # (B, seq_len)
            
            # Call model.inference()
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
            if 'action' in prediction:
                action_output = prediction['action']
                
                # Handle tuple
                if isinstance(action_output, tuple):
                    action_output = action_output[0]
                
                # Extract from tensor
                if isinstance(action_output, torch.Tensor):
                    if action_output.dim() == 4:
                        full_chunk = action_output[0, 0, :, :].cpu().numpy()
                        first_action = full_chunk[0]
                    elif action_output.dim() == 3:
                        full_chunk = action_output[0, :, :].cpu().numpy()
                        first_action = full_chunk[0]
                    elif action_output.dim() == 2:
                        first_action = action_output[0, :].cpu().numpy()
                        full_chunk = np.expand_dims(first_action, axis=0)
                    elif action_output.dim() == 1:
                        first_action = action_output.cpu().numpy()
                        full_chunk = np.expand_dims(first_action, axis=0)
                    else:
                        raise ValueError(f"Unexpected action tensor shape: {action_output.shape}")
                else:
                    raise ValueError(f"Action must be a torch.Tensor, got: {type(action_output)}")
            else:
                raise ValueError("Could not find action in prediction output")
            
            # Apply Snap-to-Grid if enabled
            if use_snap:
                original_action = first_action.copy()
                first_action = self._apply_snap_to_grid(first_action, threshold)
                
                # Also snap the full chunk if needed
                full_chunk = self._apply_snap_to_grid(full_chunk, threshold)
                
                logger.debug(f"Snapped action: {original_action} -> {first_action}")
            
            logger.info(f"Extracted action (linear_x, linear_y): {first_action}")
            logger.info(f"Full chunk shape: {full_chunk.shape}")
            
        latency_ms = (time.time() - start_time) * 1000
        
        return first_action, full_chunk, latency_ms


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
            },
            "basket_chunk5": {
                "checkpoint": "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=04-val_loss=val_loss=0.020.ckpt",
                "config": "Mobile_VLA/configs/mobile_vla_chunk5_basket.json"
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
            "recommended": False
        },
        "chunk10_epoch8": {
            "checkpoint": "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt",
            "config": "Mobile_VLA/configs/mobile_vla_chunk10_20251217.json",
            "description": "Chunk10 Epoch 8 (val_loss=0.312)",
            "fwd_pred_next_n": 10,
            "recommended": False
        },
        "basket_chunk5": {
            "checkpoint": "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=04-val_loss=val_loss=0.020.ckpt",
            "config": "Mobile_VLA/configs/mobile_vla_chunk5_basket.json",
            "description": "Basket Navigation - Chunk5 (val_loss=0.020)",
            "fwd_pred_next_n": 5,
            "recommended": True
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
        available_models = ["chunk5_epoch6", "chunk10_epoch8", "basket_chunk5"]
        
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



@app.post("/reset")
async def reset_episode():
    """에피소드 초기화 (히스토리 버퍼 비우기)"""
    if vla_server:
        vla_server.image_history = []
        logger.info("Episode history buffer reset")
    return {"status": "success", "message": "History buffer cleared"}

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
        
        action, full_chunk, latency_ms = model.predict(
            image_base64=request.image,
            instruction=request.instruction,
            snap_to_grid=request.snap_to_grid,
            snap_threshold=request.snap_threshold
        )
        
        logger.info(f"✅ Prediction: {action}, Chunk: {full_chunk.shape}, Latency: {latency_ms:.1f}ms")
        
        return InferenceResponse(
            action=action.tolist(),
            latency_ms=latency_ms,
            model_name=os.getenv("VLA_MODEL_NAME", "chunk5_epoch6"),
            chunk_size=model.fwd_pred_next_n,
            full_chunk=full_chunk.tolist()
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
    
    # Test instruction (Unified with training data)
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
