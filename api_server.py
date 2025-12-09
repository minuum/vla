#!/usr/bin/env python3
"""
FastAPI 추론 서버 (원격 서버용)
메모리 충분한 환경(32GB+ RAM, 16GB+ VRAM)에서 실행
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import base64
import uvicorn
from typing import List
import sys
from pathlib import Path

# RoboVLMs 경로
ROBOVLMS_PATH = Path(__file__).parent / "RoboVLMs"
sys.path.insert(0, str(ROBOVLMS_PATH))

from src.robovlms_mobile_vla_inference import MobileVLAConfig, MobileVLAInferenceSystem

app = FastAPI(title="Mobile VLA Inference API")

# 전역 추론 시스템
inference_system = None

class InferenceRequest(BaseModel):
    images: List[str]  # Base64 encoded images
    instruction: str = "move forward"

class InferenceResponse(BaseModel):
    actions: List[List[float]]  # (fwd_pred_next_n, action_dim)
    inference_time: float
    fps: float

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global inference_system
    
    print("🚀 모델 로딩 중...")
    
    # 환경 변수에서 체크포인트 경로 가져오기
    import os
    checkpoint_path = os.getenv(
        "VLA_CHECKPOINT_PATH",
        "RoboVLMs_upstream/runs/mobile_vla_kosmos2_aug_abs_20251209/.../last.ckpt"
    )
    
    config = MobileVLAConfig(
        checkpoint_path=checkpoint_path,
        window_size=2,  # 메모리 최적화
        use_abs_action=True  # abs_action 전략 사용
    )
    
    inference_system = MobileVLAInferenceSystem(config)
    
    if not inference_system.inference_engine.load_model():
        raise RuntimeError("모델 로드 실패")
    
    print("✅ 서버 준비 완료!")
    print(f"📌 체크포인트: {checkpoint_path}")
    print(f"🎯 abs_action 전략: 활성화")


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """추론 API 엔드포인트"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # Base64 → numpy 변환
        images_np = []
        for img_b64 in request.images:
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = np.reshape(img_array, (224, 224, 3))
            images_np.append(img)
        
        # 이미지 버퍼 채우기
        inference_system.image_buffer.clear()
        for img in images_np:
            inference_system.image_buffer.add_image(img)
        
        # 추론 실행 (abs_action 전략 사용)
        actions, info = inference_system.inference_engine.predict_action(
            inference_system.image_buffer.get_images(),
            request.instruction,
            use_abs_action=inference_system.config.use_abs_action
        )
        
        return InferenceResponse(
            actions=actions.tolist(),
            inference_time=info["inference_time"],
            fps=info["fps"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": inference_system is not None
    }

if __name__ == "__main__":
    # 사용법:
    # 1. 체크포인트 경로 수정 (line 30)
    # 2. 실행: python3 api_server.py
    # 3. 접속: http://0.0.0.0:8000/docs
    uvicorn.run(app, host="0.0.0.0", port=8000)
