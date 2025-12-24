"""
Mobile VLA 로컬 추론 서버 (인증 없음)
Jetson에서 로컬로만 사용, API Key 불필요
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
from pathlib import Path
import sys

# RoboVLMs 경로 추가
ROBOVLMS_PATH = Path(__file__).parent / "RoboVLMs"
if str(ROBOVLMS_PATH) not in sys.path:
    sys.path.insert(0, str(ROBOVLMS_PATH))

from src.robovlms_mobile_vla_inference import (
    MobileVLAConfig, 
    RoboVLMsInferenceEngine,
    ImageBuffer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Local Inference", version="1.0.0")

# Global instances
inference_engine = None
image_buffer = None
current_config = None


class InferenceRequest(BaseModel):
    """추론 요청"""
    image: str  # base64 encoded
    instruction: str
    use_abs_action: bool = True


class InferenceResponse(BaseModel):
    """추론 응답"""
    action: List[float]
    latency_ms: float
    chunk_size: int
    full_chunk: List[List[float]]


def get_inference_engine():
    """추론 엔진 가져오기"""
    global inference_engine, image_buffer, current_config
    
    if inference_engine is None:
        import os
        checkpoint_path = os.getenv(
            "VLA_CHECKPOINT_PATH",
            "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
        )
        
        current_config = MobileVLAConfig(
            checkpoint_path=checkpoint_path,
            window_size=2,
            fwd_pred_next_n=10,
            use_abs_action=True
        )
        
        logger.info(f"🚀 추론 엔진 초기화")
        logger.info(f"  체크포인트: {checkpoint_path}")
        
        inference_engine = RoboVLMsInferenceEngine(current_config)
        
        if not inference_engine.load_model():
            raise RuntimeError("모델 로드 실패")
        
        image_buffer = ImageBuffer(
            window_size=current_config.window_size,
            image_size=current_config.image_size
        )
        
        logger.info("✅ 추론 엔진 준비 완료")
    
    return inference_engine, image_buffer, current_config


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Mobile VLA Local Inference",
        "version": "1.0.0",
        "status": "running",
        "note": "로컬 전용 - 인증 불필요"
    }


@app.get("/health")
async def health():
    """헬스 체크"""
    gpu_info = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "allocated_gb": round(allocated, 2)
        }
    
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_info": gpu_info
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """추론 엔드포인트 (인증 불필요)"""
    try:
        engine, buffer, config = get_inference_engine()
        
        # 이미지 디코딩
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(pil_image)
        
        # 버퍼에 추가
        buffer.add_image(image_np)
        images = buffer.get_images()
        
        # 추론
        start_time = time.time()
        actions, info = engine.predict_action(
            images, 
            request.instruction,
            use_abs_action=request.use_abs_action
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # 정규화 해제
        denorm_actions = engine.denormalize_action(actions)
        first_action = denorm_actions[0]
        
        logger.info(f"✅ 추론: {first_action}, 지연: {latency_ms:.1f}ms")
        
        return InferenceResponse(
            action=first_action.tolist(),
            latency_ms=latency_ms,
            chunk_size=config.fwd_pred_next_n,
            full_chunk=denorm_actions.tolist()
        )
        
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_buffer():
    """버퍼 리셋"""
    try:
        engine, buffer, config = get_inference_engine()
        buffer.clear()
        logger.info("🔄 버퍼 리셋")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
