"""
FastAPI Inference Server for Mobile VLA
교수님 서버에서 모델 호출을 위한 API 서버

입력: 이미지 + Language instruction
출력: 2DOF actions [linear_x, linear_y]

보안: API Key 인증
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
    strategy: Literal["chunk_reuse", "receding_horizon"] = "chunk_reuse"  # Inference strategy
    

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    strategy: str  # Inference strategy used
    source: str  # "inferred" or "reused"
    buffer_status: dict  # Buffer status for chunk_reuse
    

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
        
        # Initialize action buffer for chunk_reuse strategy
        self.action_buffer = ActionBuffer(chunk_size=10)
        
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
        """모델 로딩 with BitsAndBytes INT8 Quantization (OpenVLA Standard)"""
        import sys
        sys.path.append('RoboVLMs_upstream')
        
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        from transformers import BitsAndBytesConfig
        import torch
        
        logger.info("="*60)
        logger.info("🔧 Loading Model with BitsAndBytes INT8")
        logger.info("   (OpenVLA/BitVLA Standard Method)")
        logger.info("="*60)
        
        # Configure BitsAndBytes INT8 (OpenVLA method)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        logger.info("INT8 Config:")
        logger.info(f"  - load_in_8bit: True")
        logger.info(f"  - threshold: 6.0")
        logger.info(f"  - Method: OpenVLA/BitVLA standard")
        logger.info(f"  - Expected: 73% memory reduction, 27x speedup")
        
        # Load checkpoint first
        logger.info(f"\nLoading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Check if it's quantized checkpoint
        if 'quantization' in checkpoint:
            logger.info("✅ Quantized checkpoint detected")
            logger.info(f"   Quantization info: {checkpoint['quantization']}")
        
        # Load from checkpoint with BitsAndBytes INT8
        logger.info("\nCreating trainer with INT8 quantization...")
        
        # Pass quantization_config directly to trainer
        self.model = MobileVLATrainer(
            self.config,
            quantization_config=bnb_config  # OpenVLA/BitVLA method
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError("No state_dict found in checkpoint")
        
        self.model.load_state_dict(state_dict, strict=False)
        
        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Measure actual GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"\n📊 Actual GPU Memory: {gpu_memory:.3f} GB")
            logger.info(f"   Expected: ~1.7 GB (73% reduction from 6.3GB FP32)")
        
        logger.info("="*60)
        logger.info("✅ Model loaded with BitsAndBytes INT8")
        logger.info("="*60)
        
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
        
    def _tokenize_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Instruction을 tokenize하여 tensor로 변환
        
        Args:
            instruction: Natural language instruction
        
        Returns:
            (lang_x, attention_mask): Tokenized instruction과 attention mask
        """
        from transformers import AutoProcessor
        
        # Kosmos-2 tokenizer 사용
        processor = AutoProcessor.from_pretrained(self.config['tokenizer']['pretrained_model_name_or_path'])
        
        # Tokenize instruction
        # Kosmos-2는 text prompt를 처리할 때 특별한 형식 사용
        encoded = processor.tokenizer(
            instruction,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config['tokenizer']['max_text_len'],
            truncation=True
        )
        
        lang_x = encoded['input_ids'].to(self.device)  # (1, seq_len)
        attention_mask = encoded['attention_mask'].to(self.device)  # (1, seq_len)
        
        return lang_x, attention_mask

    
    def predict(self, image_base64: str, instruction: str) -> tuple[np.ndarray, float]:
        """
        추론 실행
        
        Args:
            image_base64: Base64 encoded RGB image
            instruction: Natural language command
        
        Returns:
            (action, latency_ms): 
                - action: [linear_x, linear_y] numpy array
                  - linear_x (m/s): Forward velocity [0.0, 2.0]
                  - linear_y (rad/s): Angular velocity [-0.5, 0.5]
                - latency_ms: Inference latency in milliseconds
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image_base64)
            
            # Tokenize instruction
            lang_x, attention_mask = self._tokenize_instruction(instruction)
            
            # Model inference
            try:
                # self.model.model은 MobileVLATrainer의 실제 Kosmos-2 모델
                outputs = self.model.model.inference(
                    vision_x=image_tensor,
                    lang_x=lang_x,
                    attention_mask=attention_mask
                )
                
                # Extract action from outputs
                # outputs는 dictionary {'action': tensor or tuple}
                if isinstance(outputs, dict) and 'action' in outputs:
                    action_out = outputs['action']
                    
                    logger.info(f"DEBUG: action_out type: {type(action_out)}")
                    
                    # action_out이 tuple인지 먼저 확인
                    if isinstance(action_out, tuple):
                        # Tuple 형식: (velocities, gripper) 등
                        velocities = action_out[0]
                        logger.info(f"DEBUG: velocities shape: {velocities.shape}")
                        # Flatten to 1D and take first N elements
                        action = velocities.flatten().cpu().numpy()[:2]
                    else:
                        # Tensor 형식
                        logger.info(f"DEBUG: action_out shape: {action_out.shape}")
                        # Flatten and take first 2 elements
                        action = action_out.flatten().cpu().numpy()[:2]
                    
                    logger.info(f"DEBUG: extracted action shape: {action.shape}, values: {action}")
                else:
                    # outputs 자체가 tensor 또는 tuple인 경우 (fallback)
                    logger.warning(f"Unexpected outputs type: {type(outputs)}")
                    action = np.array([0.0, 0.0])
                
                # De-normalize action
                # 모델 출력: [-1, 1] normalized
                # Target range:
                #   - linear_x: [0.0, 2.0] m/s
                #   - linear_y: [-0.5, 0.5] rad/s
                action[0] = (action[0] + 1.0) * 1.0  # [-1,1] -> [0,2]
                action[1] = action[1] * 0.5          # [-1,1] -> [-0.5,0.5]
                
                # Clip to valid range
                action[0] = np.clip(action[0], 0.0, 2.0)
                action[1] = np.clip(action[1], -0.5, 0.5)
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: stop command
                action = np.array([0.0, 0.0])
            
        latency_ms = (time.time() - start_time) * 1000
        
        return action, latency_ms


def get_model():
    """모델 인스턴스 가져오기 (lazy loading)"""
    global model_instance
    
    if model_instance is None:
        # Best LoRA model checkpoint
        checkpoint_path = os.getenv(
            "VLA_CHECKPOINT_PATH",
            "runs/mobile_vla_no_chunk_20251209/checkpoints/epoch=04-val_loss=0.001.ckpt"
        )
        config_path = "Mobile_VLA/configs/mobile_vla_no_chunk_20251209.json"
        
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
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "device_name": torch.cuda.get_device_name(0)
        }
    
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": gpu_memory
    }


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
            model_name="mobile_vla_no_chunk_20251209"
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
