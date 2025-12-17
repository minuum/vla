#!/usr/bin/env python3
"""
Mobile VLA Inference API Server
Chunk5 Epoch 6 Best Model (Val Loss 0.067)
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import json
import sys
import uvicorn
from typing import Optional

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer

app = FastAPI(
    title="Mobile VLA Inference API",
    description="Action prediction API for Mobile VLA model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance = None
config_instance = None

# API Key for authentication
API_KEY = "vla_mobile_robot_2025"  # Change this in production


class PredictionRequest(BaseModel):
    """Prediction request schema"""
    image: str  # Base64 encoded image
    instruction: str
    

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    linear_x: float
    linear_y: float
    instruction: str
    model_name: str


def verify_api_key(x_api_key: str = Header(...)):
    """API Key verification"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


def load_model():
    """Load the trained model"""
    global model_instance, config_instance
    
    if model_instance is not None:
        return model_instance
    
    # Model paths - Chunk5 Epoch 6 Best Model
    checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load config
    with open(config_path) as f:
        config_instance = json.load(f)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileVLATrainer.load_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        map_location=device
    )
    
    model.model.to(device)
    model.model.eval()
    
    model_instance = {
        'trainer': model,
        'device': device,
        'config': config_instance
    }
    
    print(f"✅ Model loaded successfully on {device}")
    return model_instance


def preprocess_image(image_b64: str) -> torch.Tensor:
    """Preprocess base64 image to tensor"""
    # Decode base64
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # To tensor and normalize
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    img_tensor = transform(image)  # (3, 224, 224)
    
    # Create window (repeat for window_size=8)
    img_window = img_tensor.unsqueeze(0).repeat(8, 1, 1, 1)  # (8, 3, 224, 224)
    img_batch = img_window.unsqueeze(0)  # (1, 8, 3, 224, 224)
    
    return img_batch


@torch.no_grad()
def predict_action(image_tensor: torch.Tensor, instruction: str, model_info: dict) -> dict:
    """Predict action from image and instruction using model.predict_step()"""
    device = model_info['device']
    trainer = model_info['trainer']
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Prepare batch in the format expected by predict_step
    batch = {
        'images': image_tensor,  # (1, 8, 3, 224, 224)
        'language': [instruction]  # List[str]
    }
    
    # Forward pass using predict_step
    outputs = trainer.model.predict_step(batch, batch_idx=0)
    
    # Extract action
    action_pred = outputs['action']  # (1, 1, 2) or (1, 2)
    
    if action_pred.dim() == 3:
        action_pred = action_pred.squeeze(1)  # (1, 2)
    
    action_normalized = action_pred.squeeze(0)  # (2,)
    action = action_normalized.cpu().numpy()  # (2,)
    
    return {
        'linear_x': float(action[0]),
        'linear_y': float(action[1])
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Mobile VLA Inference API",
        "version": "1.0.0",
        "model": "Chunk5 Epoch 6 (Best)",
        "status": "ready"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model_instance is not None
    return {
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "device": model_instance['device'] if model_loaded else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict action from image and instruction
    
    Args:
        request: PredictionRequest with base64 image and instruction
        api_key: API key for authentication
        
    Returns:
        PredictionResponse with predicted action
    """
    try:
        # Load model if not loaded
        model_info = load_model()
        
        # Preprocess image
        image_tensor = preprocess_image(request.image)
        
        # Predict
        result = predict_action(image_tensor, request.instruction, model_info)
        
        return PredictionResponse(
            linear_x=result['linear_x'],
            linear_y=result['linear_y'],
            instruction=request.instruction,
            model_name="mobile_vla_chunk5_epoch06_best"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """Get model information"""
    model_info = load_model()
    config = model_info['config']
    
    return {
        "model_name": config.get('exp_name', 'unknown'),
        "fwd_pred_next_n": config.get('fwd_pred_next_n', 0),
        "window_size": config.get('window_size', 0),
        "freeze_backbone": config['train_setup'].get('freeze_backbone', False),
        "lora_enable": config['train_setup'].get('lora_enable', False),
        "device": model_info['device']
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mobile VLA Inference API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting Mobile VLA Inference API Server on {args.host}:{args.port}")
    print(f"API Key: {API_KEY}")
    print(f"Model: Chunk5 Epoch 6 (Best - Val Loss 0.067)")
    print("")
    print("Endpoints:")
    print(f"  - Docs: http://{args.host}:{args.port}/docs")
    print(f"  - Health: http://{args.host}:{args.port}/health")
    print(f"  - Predict: POST http://{args.host}:{args.port}/predict")
    print("")
    
    uvicorn.run(
        "inference_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
