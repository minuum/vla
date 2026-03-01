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
    snap_threshold: Optional[float] = 0.8 # Optimized threshold
    

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]  # 학습 데이터 기준
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    chunk_size: int  # 모델의 chunk size (fwd_pred_next_n)
    full_chunk: Optional[List[List[float]]] = None  # Future trajectory
    raw_action: Optional[List[float]] = None  # Raw action for debugging

# Step 1 Model: Direct VLM Inference
import sys
import os

# Add src to path for imports
sys.path.append(os.path.abspath("src"))

# Import Infrerence Engine from src
try:
    from robovlms_mobile_vla_inference import RoboVLMsInferenceEngine, MobileVLAConfig, ImageBuffer
except ImportError:
    # Try adding current dir if running from src parent
    sys.path.append(os.getcwd())
    from src.robovlms_mobile_vla_inference import RoboVLMsInferenceEngine, MobileVLAConfig, ImageBuffer

# Singleton Instance
vlm_service = None

class VLAInferenceService:
    """Wrapper for Step 1 Model (Engine + History Buffer)"""
    def __init__(self, ckpt_path):
        self.config = MobileVLAConfig(
            model_name="unified_regression_win12",
            checkpoint_path=ckpt_path,
            window_size=8,
            action_dim=2
        )
        self.engine = RoboVLMsInferenceEngine(self.config)
        self.buffer = ImageBuffer(window_size=self.config.window_size)
        
        # Load Model
        logger.info(f"🚀 Loading Step 1 Model from {ckpt_path}...")
        if not self.engine.load_model():
            raise RuntimeError("Failed to load VLM model")
        logger.info("✅ Step 1 Model Loaded!")

    def predict(self, np_image, instruction):
        # Add to buffer
        self.buffer.add_image(np_image)
        images_tensor = self.buffer.get_images()
        
        # Run Inference
        actions, info = self.engine.predict_action(images_tensor, instruction)
        return actions, info

    def reset(self):
        self.buffer.clear()
        
def get_vlm_service():
    global vlm_service
    if vlm_service is None:
        default_ckpt = "runs/unified_regression_win12/kosmos/mobile_vla_unified_finetune/2026-02-05/unified_regression_win12_20260205/epoch=9-step=600.ckpt"
        ckpt_path = os.getenv("VLA_CHECKPOINT_PATH", default_ckpt)
        vlm_service = VLAInferenceService(ckpt_path)
    return vlm_service

class ModelSwitchRequest(BaseModel):
    """모델 전환 요청 스키마"""
    model_name: str  # basket_chunk5, chunk10_epoch8, no_chunk_epoch4


@app.post("/model/switch")
async def switch_model(request: ModelSwitchRequest, api_key: str = Depends(verify_api_key)):
    """Model switching not supported in static Step 1 mode yet"""
    raise HTTPException(status_code=501, detail="Model switching disabled for Step 1 Verification")

@app.post("/reset")
async def reset_episode():
    """Reset History Buffer"""
    try:
        service = get_vlm_service()
        service.reset()
        return {"status": "success", "message": "History Buffer Reset"}
    except Exception as e:
         return {"status": "error", "message": str(e)}

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Step 1 Inference Endpoint (Real VLM)
    """
    try:
        service = get_vlm_service()
        
        # Preprocess Image (Base64 -> Numpy)
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_image = np.array(pil_image)
        
        # Run Inference
        start_time = time.time()
        actions, info = service.predict(np_image, request.instruction)
        latency = (time.time() - start_time) * 1000
        
        # Action Processing
        # actions is (chunk_size, 2)
        # We take the first action for immediate execution, but return full chunk
        first_action = actions[0]
        
        return InferenceResponse(
            action=first_action.tolist(),
            latency_ms=latency,
            model_name="step1_unified_regression",
            chunk_size=len(actions),
            full_chunk=actions.tolist(),
            raw_action=first_action.tolist()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Create dummy BLACK image to trigger "Blindness" check
    dummy_img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    
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
