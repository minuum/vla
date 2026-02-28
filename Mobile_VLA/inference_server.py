"""
FastAPI Inference Server for Mobile VLA
교수님 서버에서 모델 호출을 위한 API 서버

입력: 이미지 + Language instruction
출력: 2DOF actions [linear_x, linear_y]

보안: API Key 인증
"""

import sys
import os
import time
import io
import base64
import logging
import secrets
import json
from typing import List, Optional, Literal
from pathlib import Path
import gc
from datetime import datetime

# 1. Add project root and RoboVLMs to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Handle RoboVLMs path (try both RoboVLMs and RoboVLMs_upstream)
robovlms_path = os.path.join(project_root, 'RoboVLMs')
if os.path.exists(robovlms_path) and robovlms_path not in sys.path:
    sys.path.insert(0, robovlms_path)

robovlms_upstream_path = os.path.join(project_root, 'RoboVLMs_upstream')
if os.path.exists(robovlms_upstream_path) and robovlms_upstream_path not in sys.path:
    sys.path.insert(0, robovlms_upstream_path)

# Third-party imports
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Project imports
try:
    from Mobile_VLA.action_buffer import ActionBuffer
    from robovlms.data.data_utils import unnoramalize_action
except ImportError as e:
    print(f"FATAL ERROR: Failed to import project modules. sys.path: {sys.path}")
    raise e

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
        self.model_name = Path(config_path).stem
        
        # Initialize action buffer for chunk_reuse strategy
        self.action_buffer = ActionBuffer(chunk_size=10)
        self.inference_count = 0
        
        # Driving log history
        self.episode_log = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(project_root, "docs", "inference_reports")
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Loading model from {checkpoint_path}")
        logger.info(f"Using device: {device}")
        
        # [INTEGRATION] Load config and set inference mode
        try:
            from robovlms.utils.config_utils import load_config
            self.config = load_config(config_path)
        except ImportError:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        self.mode = self.config.get("inference_mode", "regression")
        self.scale_factor = self.config.get("scale_factor", 1.15)
        self.class_map = self.config.get("class_map", {
            0: [0.0, 0.0],    # Stop
            1: [1.15, 0.0],   # Forward
            2: [-1.15, 0.0],  # Backward
            3: [0.0, 1.15],   # Left
            4: [0.0, -1.15],  # Right
            5: [1.15, 1.15],  # Forward-Left
            6: [1.15, -1.15], # Forward-Right
            7: [-1.15, 1.15], # Backward-Left
            8: [-1.15, -1.15] # Backward-Right
        })
        
        logger.info(f"🔄 [INIT] Inference Mode: {self.mode.upper()}")
        
        # Load model
        self._load_model()
        
        # [OPTIMIZATION] Initialize processor once
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.config['tokenizer']['pretrained_model_name_or_path']
        )
        logger.info(f"Processor initialized from {self.config['tokenizer']['pretrained_model_name_or_path']}")
        
        logger.info("Model loaded successfully")
        
    def _load_model(self):
        """모델 로딩 (VLA_QUANTIZE=false면 FP16, true면 INT8로 로드)"""
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        self.use_quant = os.getenv("VLA_QUANTIZE", "false").lower() == "true"
        
        if self.use_quant:
            from transformers import BitsAndBytesConfig
            logger.info("🔧 [MODE] BitsAndBytes INT8 Quantization (Memory Efficient)")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            self.model = MobileVLATrainer(self.config, quantization_config=bnb_config)
        else:
            logger.info("🚀 [MODE] High-Precision BF16/FP16 (Action Diversity Prioritized)")
            self.model = MobileVLATrainer(self.config)
            
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        if state_dict is None:
            raise ValueError("No state_dict found in checkpoint")
        
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('model.'):
            if not hasattr(self.model, 'model'):
                logger.info("✂️ Stripping 'model.' prefix from state_dict keys")
                state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            logger.warning(f"⚠️ Missing keys during load: {len(missing_keys)}")
            act_head_missing = [k for k in missing_keys if 'act_head' in k or 'policy_head' in k]
            if act_head_missing:
                logger.error(f"❌ CRITICAL: Action head weights NOT loaded! Example: {act_head_missing[:3]}")
        
        if not self.use_quant:
            # Force Float16 for inference to avoid LSTM BFloat16 implementation gaps
            self.model.model.half()
            logger.info("✅ Model converted to Float16 (Optimized for LSTM consistency)")
            
        self.model.to(self.device).eval()
        
        self.policy_head = None
        potential_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        for name in ['act_head', 'policy_head']:
            if hasattr(potential_model, name):
                self.policy_head = getattr(potential_model, name)
                logger.info(f"✅ Verified: policy_head found at model.{name}")
                break
        
        if self.policy_head is None:
            logger.warning("⚠️ Warning: policy_head (act_head) not found in model structure!")
            
        del state_dict
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"📊 Actual GPU Memory: {gpu_memory:.3f} GB")
        
        from torchvision import transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.get('image_mean', [0.481, 0.457, 0.408]),
                std=self.config.get('image_std', [0.268, 0.261, 0.275])
            )
        ])
        
    def preprocess_image(self, image_input: str | np.ndarray) -> torch.Tensor:
        """이미지를 모델 입력 형식으로 변환"""
        if isinstance(image_input, str):
            image_bytes = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
             raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        image_tensor = self.image_transform(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor.to(self.device)
        
    def _tokenize_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Instruction을 tokenize하여 tensor로 변환 (학습 데이터와 동일한 Grounding Prefix 포함)
        학습 데이터(V2) 형식: "<grounding>An image of a robot {instruction}"
        """
        clean_instr = instruction.strip()
        if not clean_instr.startswith("<grounding>"):
            if not clean_instr.startswith("An image of a robot"):
                full_prompt = f"<grounding>An image of a robot {clean_instr}"
            else:
                full_prompt = f"<grounding>{clean_instr}"
        else:
            full_prompt = clean_instr
            
        logger.debug(f"📝 [INPUT] Prompt: {full_prompt}")
            
        encoded = self.processor.tokenizer(
            full_prompt,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config['tokenizer']['max_text_len'],
            truncation=True
        )
        
        lang_x = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        return lang_x, attention_mask

    def reset(self):
        """추론 히스토리 초기화 및 로그 저장"""
        try:
            # Save episode log if exists
            if self.episode_log:
                log_path = os.path.join(self.log_dir, f"inference_{self.session_id}_{self.inference_count}.json")
                with open(log_path, 'w') as f:
                    json.dump(self.episode_log, f, indent=2)
                logger.info(f"✅ Driving log saved to {log_path}")
                self.episode_log = []
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.action_buffer.clear()
            # LSTM hidden state 및 history 초기화 (policy_head가 있을 경우)
            potential_model = self.model.model if hasattr(self.model, 'model') else self.model
            for head_name in ['act_head', 'policy_head']:
                if hasattr(potential_model, head_name):
                    head = getattr(potential_model, head_name)
                    if hasattr(head, 'reset'):
                        head.reset()
                        logger.info(f"✅ {head_name} history reset")
            
            self.inference_count = 0
            logger.info("✅ Model history and logging reset (Internal)")
        except Exception as e:
            logger.error(f"Reset failed: {e}")

    def decode_action(self, logits):
        """
        통합 액션 디코더: LSTM/Chunk 구조 대응
        - logits: (B, T, K, D) 또는 (B, L, D) 형태의 텐서
        """
        # 1. Temporal/Chunk 처리 (Receding Horizon: 첫 번째 액션만 선택)
        # (B, T, K, D) -> (B, K, D) -> (K, D) -> (D,)
        if logits.dim() >= 3:
            # 마지막 타임스텝의 예측값 선택 (LSTM 사용 시)
            current_logits = logits[0, -1] 
            # 청크가 있을 경우 첫 번째 액션 선택
            if current_logits.dim() >= 2:
                current_logits = current_logits[0]
        else:
            current_logits = logits.flatten()

        # 2. 모드에 따른 디코딩
        if self.mode == "classification":
            class_idx = torch.argmax(current_logits, dim=-1).item()
            # str key handling (json load makes keys strings)
            action = np.array(self.class_map.get(class_idx, self.class_map.get(str(class_idx), [0.0, 0.0])))
            logger.debug(f"🎯 [CLASS] Index {class_idx} -> Velocity: {action}")
        else:
            # Regression 모드
            raw_val = current_logits.cpu().numpy()[:2]
            
            norm_action = self.config.get('norm_action', False)
            if not norm_action:
                action = raw_val * self.scale_factor
            else:
                target_min = -1.15
                target_max = 1.15
                action = unnoramalize_action(raw_val, action_min=target_min, action_max=target_max)
            logger.debug(f"📈 [REG] Raw {raw_val} -> Final {action}")
            
        return action

    def predict(self, image_base64: str, instruction: str) -> tuple[np.ndarray, float]:
        """추론 실행 (3-Omniwheel 스케일 및 방향 보정 포함)"""
        start_time = time.time()
        
        # 0. First-Frame Zero Enforcement (Hidden State Priming)
        is_first_frame = (self.inference_count == 0)
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_base64)
            lang_x, attention_mask = self._tokenize_instruction(instruction)
            
            try:
                # 1. Model Inference
                outputs = self.model.model.inference(
                    vision_x=image_tensor,
                    lang_x=lang_x,
                    attention_mask=attention_mask
                )
                
                if isinstance(outputs, dict) and 'action' in outputs:
                    action_out = outputs['action']
                    
                    # Logits 추출 (tuple 형태일 경우 첫 번째 요소가 logits)
                    logits = action_out[0] if isinstance(action_out, tuple) else action_out
                    
                    # [INTEGRATED] 통합 디코더 호출
                    action = self.decode_action(logits)
                else:
                    action = np.array([0.0, 0.0])
                
                # 2. First-frame Safety
                if is_first_frame:
                    logger.info("🛡️ First-Frame Zero Enforcement 적용")
                    action = np.array([0.0, 0.0])
                
                raw_action = action.copy()
                
                # 4. Clipping
                action[0] = np.clip(action[0], -1.5, 1.5)
                action[1] = np.clip(action[1], -1.5, 1.5)
                
                logger.info(f"📤 Action: {'[CLASS]' if self.mode == 'classification' else '[REG]'} Raw[{raw_action[0]:.3f}, {raw_action[1]:.3f}] -> Final[{action[0]:.3f}, {action[1]:.3f}]")
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                action = np.array([0.0, 0.0])
            
        latency_ms = (time.time() - start_time) * 1000
        
        # Log step
        self.episode_log.append({
            "step": self.inference_count,
            "timestamp": time.time(),
            "instruction": instruction,
            "action": action.tolist(),
            "latency_ms": latency_ms
        })
        
        self.inference_count += 1
        return action, latency_ms


def get_model():
    """모델 인스턴스 가져오기 (lazy loading)"""
    global model_instance
    if model_instance is None:
        checkpoint_path = os.getenv(
            "VLA_CHECKPOINT_PATH",
            "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
        )
        config_path = os.getenv(
            "VLA_CONFIG_PATH", 
            "Mobile_VLA/configs/mobile_vla_exp17_win8_k1.json"
        )
        model_instance = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return model_instance


@app.get("/")
async def root():
    return {"name": "Mobile VLA Inference API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    gpu_memory = None
    model_name = "Unknown"
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "device_name": torch.cuda.get_device_name(0)
        }
    if model_instance:
        model_name = model_instance.model_name
    
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "model_name": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": gpu_memory
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    try:
        model = get_model()
        action, latency_ms = model.predict(image_base64=request.image, instruction=request.instruction)
        return InferenceResponse(
            action=action.tolist(),
            latency_ms=latency_ms,
            model_name=model.model_name,
            strategy="receding_horizon",
            source="inferred",
            buffer_status={}
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_history(api_key: str = Depends(verify_api_key)):
    model = get_model()
    try:
        model.reset()
        return {"status": "success", "message": "History and buffer cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint(api_key: str = Depends(verify_api_key)):
    instruction = "Navigate to the brown pot"
    dummy_action = [0.0, 0.0]
    return {"message": "Test endpoint", "instruction": instruction, "action": dummy_action}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
