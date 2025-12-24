"""
FastAPI Inference Server for Mobile VLA (RoboVLMs Style)
robovlms_mobile_vla_inference.py 통합 버전

입력: 이미지 + Language instruction
출력: Action chunk [linear_x, linear_y]

보안: API Key 인증
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import base64
import io
import time
from typing import List, Optional, Dict, Any
import logging
import os
import secrets
from pathlib import Path
import sys

# RoboVLMs 경로 추가
ROBOVLMS_PATH = Path(__file__).parent / "RoboVLMs"
if str(ROBOVLMS_PATH) not in sys.path:
    sys.path.insert(0, str(ROBOVLMS_PATH))

# robovlms_mobile_vla_inference import
from src.robovlms_mobile_vla_inference import (
    MobileVLAConfig, 
    RoboVLMsInferenceEngine,
    ImageBuffer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Inference API (RoboVLMs)", version="2.0.0")

# API Key 설정
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key():
    """API Key 가져오기 또는 생성"""
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
inference_engine = None
image_buffer = None
current_config = None


class InferenceRequest(BaseModel):
    """추론 요청 스키마"""
    image: str  # base64 encoded image
    instruction: str  # Language instruction
    use_abs_action: bool = True  # abs_action 전략 사용 여부

class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # First action [linear_x, linear_y]
    latency_ms: float
    model_name: str
    chunk_size: int
    full_chunk: List[List[float]]  # Full action chunk
    direction: Optional[float] = None  # abs_action 방향 
    use_abs_action: bool = False


def get_inference_engine():
    """추론 엔진 가져오기 (lazy loading)"""
    global inference_engine, image_buffer, current_config
    
    if inference_engine is None:
        # 환경변수에서 설정 읽기
        checkpoint_path = os.getenv(
            "VLA_CHECKPOINT_PATH",
            "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
        )
        
        # Config 생성
        current_config = MobileVLAConfig(
            checkpoint_path=checkpoint_path,
            window_size=int(os.getenv("VLA_WINDOW_SIZE", "2")),  # 메모리 절약
            fwd_pred_next_n=int(os.getenv("VLA_CHUNK_SIZE", "10")),
            use_abs_action=os.getenv("VLA_USE_ABS_ACTION", "true").lower() == "true",
            denormalize_strategy=os.getenv("VLA_DENORM_STRATEGY", "safe")
        )
        
        logger.info(f"🚀 추론 엔진 초기화")
        logger.info(f"  체크포인트: {checkpoint_path}")
        logger.info(f"  Window size: {current_config.window_size}")
        logger.info(f"  Chunk size: {current_config.fwd_pred_next_n}")
        logger.info(f"  abs_action: {current_config.use_abs_action}")
        
        # 추론 엔진 생성
        inference_engine = RoboVLMsInferenceEngine(current_config)
        
        # 모델 로드
        if not inference_engine.load_model():
            raise RuntimeError("모델 로드 실패")
        
        # 이미지 버퍼 생성
        image_buffer = ImageBuffer(
            window_size=current_config.window_size,
            image_size=current_config.image_size
        )
        
        logger.info("✅ 추론 엔진 준비 완료")
    
    return inference_engine, image_buffer, current_config


@app.get("/")
async def root():
    """API 정보 (인증 불필요)"""
    return {
        "name": "Mobile VLA Inference API (RoboVLMs Style)",
        "version": "2.0.0",
        "status": "running",
        "auth": "API Key required (X-API-Key header)",
        "inference_engine": "RoboVLMs forward_continuous"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 (인증 불필요)"""
    gpu_info = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2)
        }
    
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_info": gpu_info
    }


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """모델 정보 조회 (API Key 필수)"""
    try:
        engine, buffer, config = get_inference_engine()
        
        return {
            "model_name": "RoboVLMs Mobile VLA",
            "checkpoint_path": config.checkpoint_path,
            "window_size": config.window_size,
            "image_size": config.image_size,
            "action_dim": config.action_dim,
            "fwd_pred_next_n": config.fwd_pred_next_n,
            "use_abs_action": config.use_abs_action,
            "denormalize_strategy": config.denormalize_strategy,
            "max_linear_x": config.max_linear_x,
            "max_linear_y": config.max_linear_y,
            "device": str(engine.device)
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    추론 엔드포인트 (API Key 필수)
    
    RoboVLMs forward_continuous() 사용
    """
    try:
        engine, buffer, config = get_inference_engine()
        
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(request.image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(pil_image)
        
        # 이미지 버퍼에 추가
        buffer.add_image(image_np)
        
        # 버퍼가 준비될 때까지 대기 (첫 몇 프레임)
        if not buffer.is_ready():
            logger.warning(f"⚠️ 버퍼 미준비 ({len(buffer.buffer)}/{config.window_size})")
        
        # 이미지 텐서 가져오기
        images = buffer.get_images()
        
        # 추론 실행
        start_time = time.time()
        actions, info = engine.predict_action(
            images, 
            request.instruction,
            use_abs_action=request.use_abs_action
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # 정규화 해제 (모델 출력 [-1,1] → 실제 속도)
        denorm_actions = engine.denormalize_action(actions)
        
        # 첫 번째 액션
        first_action = denorm_actions[0]
        
        logger.info(f"✅ Prediction: {first_action}, Chunk: {denorm_actions.shape}, Latency: {latency_ms:.1f}ms")
        
        return InferenceResponse(
            action=first_action.tolist(),
            latency_ms=latency_ms,
            model_name="RoboVLMs Mobile VLA",
            chunk_size=config.fwd_pred_next_n,
            full_chunk=denorm_actions.tolist(),
            direction=info.get("direction"),
            use_abs_action=request.use_abs_action
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_buffer(api_key: str = Depends(verify_api_key)):
    """이미지 버퍼 리셋 (새 에피소드 시작 시)"""
    try:
        engine, buffer, config = get_inference_engine()
        buffer.clear()
        logger.info("🔄 이미지 버퍼 리셋 완료")
        return {"status": "success", "message": "Image buffer cleared"}
    except Exception as e:
        logger.error(f"Buffer reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark")
async def benchmark(
    iterations: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """성능 벤치마크"""
    try:
        engine, buffer, config = get_inference_engine()
        
        # 더미 이미지로 버퍼 채우기
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(config.window_size):
            buffer.add_image(dummy_image)
        
        # 벤치마크 실행
        latencies = []
        for _ in range(iterations):
            images = buffer.get_images()
            start = time.time()
            engine.predict_action(images, "move forward")
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies) * 1000
        max_latency = np.max(latencies) * 1000
        min_latency = np.min(latencies) * 1000
        fps = 1000.0 / avg_latency
        
        return {
            "iterations": iterations,
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "fps": round(fps, 2)
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
