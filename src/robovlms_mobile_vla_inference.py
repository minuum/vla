#!/usr/bin/env python3
"""
RoboVLMs 기반 Mobile VLA 추론 시스템
학습 코드 (MobileVLATrainer, forward_continuous)와 일치하는 추론 파이프라인

핵심 인사이트:
- Frozen VLM + Action Head가 LoRA보다 성능 우수 (92.5% vs 50%)
- forward_continuous() 사용하여 언어 토큰 포함 필수
- window_size=8, fwd_pred_next_n=10, action_dim=2
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import sys
import time
import os
from dataclasses import dataclass
from collections import deque

# RoboVLMs 경로 추가
ROBOVLMS_PATH = Path(__file__).parent.parent / "RoboVLMs"
if str(ROBOVLMS_PATH) not in sys.path:
    sys.path.insert(0, str(ROBOVLMS_PATH))



import transformers

print(f"📦 Transformers version: {transformers.__version__}")
if transformers.__version__ not in ["4.35.0", "4.41.2"]:
    print(f"⚠️ Warning: transformers version mismatch. Found {transformers.__version__}")
    print(f"   Expected 4.35.0 (INT8 support) or 4.41.2 (RoboVLMs default)")


def extract_direction_from_instruction(instruction: str) -> float:
    """
    abs_action 전략: 언어 명령에서 방향 추출
    
    좌표계:
    - Left: Positive linear_y (+1.0)
    - Right: Negative linear_y (-1.0)
    - Straight/Unclear: 0.0
    
    Args:
        instruction: 언어 명령 (예: "Navigate to the left bottle")
        
    Returns:
        direction: 방향 부호 (-1.0, 0.0, +1.0)
    """
    instr_lower = instruction.lower()
    
    if 'left' in instr_lower:
        return 1.0
    elif 'right' in instr_lower:
        return -1.0
    else:
        # 방향이 명확하지 않으면 직진
        return 0.0


@dataclass
class MobileVLAConfig:
    """Mobile VLA 추론 설정 (학습 config와 일치)"""
    # 모델 설정
    model_name: str = "kosmos"
    checkpoint_path: str = ""
    
    # 입력 설정 (학습과 일치)
    window_size: int = 2  # 메모리 절약: 8→2 (75% 감소)
    image_size: int = 224
    
    # 출력 설정
    action_dim: int = 2  # linear_x, linear_y
    fwd_pred_next_n: int = 10  # 10 action chunk 예측
    
    # 정규화 설정 (모델 출력: [-1, 1])
    norm_min: float = -1.0
    norm_max: float = 1.0
    
    # 실제 로봇 속도 범위 (정규화 해제 시 사용)
    robot_max_linear_x: float = 1.15  # 실제 로봇 최대 속도 (m/s)
    robot_max_linear_y: float = 1.15  # 실제 로봇 최대 속도 (m/s)
    
    # 정규화 해제 전략 ("scale", "minmax", "safe")
    denormalize_strategy: str = "safe"  # 기본: 안전 모드 (권장)
    
    # abs_action 전략 (Case 4, 5에서 사용)
    use_abs_action: bool = True  # 언어에서 방향 추출, 모델은 크기만 예측
    
    # INT8 Quantization
    use_int8: bool = False
    
    # 추론 설정
    inference_interval: float = 0.3  # 300ms (학습 데이터 수집 주기와 일치)
    action_chunk_execution: int = 1  # 한 번에 실행할 action chunk 수
    
    # 속도 제한 (안전 모드에서 사용되는 최대 속도)
    max_linear_x: float = 0.5  # m/s - 안전 최대 속도
    max_linear_y: float = 0.5  # m/s - 안전 최대 속도



class ImageBuffer:
    """이미지 버퍼 (window_size 만큼 유지)"""
    
    def __init__(self, window_size: int = 8, image_size: int = 224):
        self.window_size = window_size
        self.image_size = image_size
        self.buffer = deque(maxlen=window_size)
        
        # 이미지 전처리
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
    
    def add_image(self, image: np.ndarray) -> None:
        """이미지 추가 (BGR -> RGB 변환 포함)"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            image = image[:, :, ::-1].copy()
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        tensor = self.transform(pil_image)
        self.buffer.append(tensor)
    
    def get_images(self) -> torch.Tensor:
        """버퍼의 이미지들을 텐서로 반환 (패딩 포함)"""
        images = list(self.buffer)
        
        # 패딩 (window_size 미만일 경우)
        while len(images) < self.window_size:
            if len(images) > 0:
                images.insert(0, images[0].clone())  # 첫 이미지 복제
            else:
                images.append(torch.zeros(3, self.image_size, self.image_size))
        
        # (window_size, 3, H, W) -> (1, window_size, 3, H, W)
        return torch.stack(images).unsqueeze(0)
    
    def is_ready(self) -> bool:
        """버퍼가 충분히 채워졌는지 확인"""
        return len(self.buffer) >= self.window_size
    
    def clear(self) -> None:
        """버퍼 초기화"""
        self.buffer.clear()


class RoboVLMsInferenceEngine:
    """RoboVLMs 기반 추론 엔진"""
    
    def __init__(self, config: MobileVLAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.text_fn = None
        
        print(f"🔧 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    def load_model(self, checkpoint_path: str = None) -> bool:
        """모델 로드 (메모리 최적화)"""
        if checkpoint_path:
            self.config.checkpoint_path = checkpoint_path
        
        if not self.config.checkpoint_path:
            print("❌ 체크포인트 경로가 지정되지 않았습니다.")
            return False
        
        print(f"🚀 모델 로딩: {self.config.checkpoint_path}")
        print("⚙️  메모리 최적화 모드: CPU 로드 → FP16 변환 → GPU 전송")
        
        try:
            import gc
            # Robo+ 디렉토리 구조에 맞게 수정
            import sys
            from pathlib import Path
            
            # Robo+ 경로 추가
            robo_plus_path = Path(__file__).parent.parent / "Robo+"
            if str(robo_plus_path) not in sys.path:
                sys.path.insert(0, str(robo_plus_path))
            
            from Mobile_VLA.core.train_core.mobile_vla_trainer import MobileVLATrainer
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 체크포인트를 CPU로 먼저 로드 (메모리 피크 감소)
            print("📥 체크포인트 로딩 (CPU)...")
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
            
            # config에서 필요한 정보 추출
            ckpt_config = checkpoint.get('config', {})
            model_name = ckpt_config.get('model_name', 'microsoft/kosmos-2-patch14-224')
            action_dim = ckpt_config.get('action_dim', self.config.action_dim)
            window_size = ckpt_config.get('window_size', self.config.window_size)
            chunk_size = ckpt_config.get('chunk_size', self.config.fwd_pred_next_n)
            
            # window_size보다 chunk_size가 크면 IndexError 발생 (MobileVLATrainer 구조상)
            # window_size보다 chunk_size가 크면 IndexError 발생 (MobileVLATrainer 구조상)
            if chunk_size > window_size:
                print(f"⚠️ [Constraint] MobileVLATrainer 구조상 chunk_size({chunk_size})는 window_size({window_size})를 초과할 수 없습니다.")
                print(f"🔄 chunk_size를 {window_size}로 자동 조정합니다.")
                chunk_size = window_size
            
            print(f"📝 [Model Config] Action Dim: {action_dim} | Window Size: {window_size} | Chunk Size: {chunk_size}")
            
            # MobileVLATrainer 생성 (모델만 필요)
            print("🔧 MobileVLATrainer 초기화...")
            trainer = MobileVLATrainer(
                model_name=model_name,
                action_dim=action_dim,
                window_size=window_size,
                chunk_size=chunk_size,
                device='cpu'  # 먼저 CPU에 로드
            )
            
            # 체크포인트의 state_dict 로드
            print("📦 State dict 로딩...")
            # 'model_state_dict' 또는 'state_dict' 키 지원
            if 'model_state_dict' in checkpoint:
                state_dict_key = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            else:
                raise KeyError("체크포인트에 'model_state_dict' 또는 'state_dict' 키가 없습니다")
            
            trainer.model.load_state_dict(checkpoint[state_dict_key], strict=False)
            
            # 체크포인트 메모리 해제 (중요: OOM 방지)
            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # FP16 변환 (CPU에서)
            print("🔄 FP16 변환 중... (잠시 대기)")
            trainer.model = trainer.model.half()
            
            # 변환 후 다시 GC
            gc.collect()
            
            trainer.model.eval()
            
            # GPU로 전송
            print(f"🎮 GPU로 전송 중... (디바이스: {self.device})")
            trainer.model = trainer.model.to(self.device)
            
            # 모델과 토크나이저 저장
            self.model = trainer.model
            self.tokenizer = trainer.processor.tokenizer  # AutoProcessor에서 tokenizer 추출
            
            # text_fn 정의 (간단한 토크나이징)
            def simple_text_fn(texts):
                """간단한 텍스트 토크나이징"""
                inputs = trainer.processor(
                    text=texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                return inputs['input_ids'], inputs['attention_mask']
            
            self.text_fn = simple_text_fn
            self.processor = trainer.processor
            
            # 다시 한번 캐시 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            print("✅ 모델 로드 완료!")
            
            # 메모리 사용량 출력
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"💾 GPU 메모리: {allocated:.2f}GB 할당됨, {reserved:.2f}GB 예약됨")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def tokenize_instruction(self, instruction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """언어 지시문 토크나이징"""
        text_tokens, text_mask = self.text_fn([instruction])
        return text_tokens.to(self.device), text_mask.to(self.device)
    
    @torch.no_grad()
    def predict_action(
        self, 
        images: torch.Tensor, 
        instruction: str,
        use_abs_action: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        액션 예측 (학습 코드와 일치하는 방식)
        
        Args:
            images: (1, window_size, 3, H, W) 이미지 텐서
            instruction: 언어 지시문 (예: "Move to the target on the right")
            use_abs_action: abs_action 전략 사용 여부 (기본 True)
            
        Returns:
            actions: (fwd_pred_next_n, action_dim) 예측된 액션 시퀀스
            info: 추가 정보 (추론 시간 등)
        """
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        start_time = time.time()
        
        # abs_action 전략: 방향 추출
        direction = 0.0
        if use_abs_action:
            direction = extract_direction_from_instruction(instruction)
        
        # 이미지를 GPU로 이동 (FP16)
        images = images.to(self.device, dtype=torch.float16)
        
        # 언어 토크나이징
        text_tokens, text_mask = self.tokenize_instruction(instruction)
        
        # MobileVLATrainer의 모델 forward 호출
        # 입력: pixel_values (B, T, C, H, W), input_ids, attention_mask
        # 출력: {'predicted_actions': (B, chunk_size, action_dim), ...}
        with torch.cuda.amp.autocast():
            result = self.model(
                pixel_values=images,  # (1, window_size, 3, H, W)
                input_ids=text_tokens,
                attention_mask=text_mask
            )
        
        # 결과 처리 - MobileVLATrainer는 dict 반환
        if isinstance(result, dict):
            actions = result['predicted_actions']  # (B, chunk_size, action_dim)
        elif isinstance(result, tuple):
            actions = result[0]
        else:
            actions = result
        
        # 배치 차원 제거: (1, chunk_size, action_dim) -> (chunk_size, action_dim)
        if actions.dim() == 3:
            actions = actions[0]  # (chunk_size, action_dim)
        
        # numpy로 변환
        actions = actions.cpu().numpy()  # (chunk_size, action_dim)
        
        # abs_action 적용: [linear_x, abs_linear_y] -> [linear_x, linear_y * direction]
        if use_abs_action and direction != 0.0:
            # linear_y 값에 절댓값 취한 후 방향 부호 곱하기
            actions[:, 1] = np.abs(actions[:, 1]) * direction
        
        inference_time = time.time() - start_time
        
        info = {
            "inference_time": inference_time,
            "fps": 1.0 / inference_time if inference_time > 0 else 0,
            "instruction": instruction,
            "device": str(self.device),
            "direction": direction if use_abs_action else None,
            "use_abs_action": use_abs_action
        }
        
        return actions, info

    
    def denormalize_action(
        self, 
        action: np.ndarray,
        strategy: str = None
    ) -> np.ndarray:
        """
        액션 정규화 해제: 모델 출력 [-1, 1] → 실제 로봇 속도
        
        Args:
            action: (N, 2) 모델 출력 액션 [linear_x, linear_y]
            strategy: 정규화 해제 전략 (config 값 사용 시 None)
                - "scale": 단순 스케일링 (robot_max * action)
                - "minmax": Min-Max 역변환 (RoboVLMs 방식)
                - "safe": 안전 제한 + 스케일링 (기본값, 권장)
        
        Returns:
            denorm_action: 실제 로봇 속도 [m/s]
        
        Examples:
            # scale 방식: action=1.0 → 1.15 m/s
            # minmax 방식: action=1.0 → 1.15 m/s (동일)
            # safe 방식: action=1.0 → max_linear (0.5 m/s, 안전 제한)
        """
        if strategy is None:
            strategy = self.config.denormalize_strategy
        
        action = action.copy()
        
        if strategy == "scale":
            # Option A: 단순 스케일링
            # 모델 [-1, 1] → 로봇 [-robot_max, robot_max]
            action[:, 0] = action[:, 0] * self.config.robot_max_linear_x
            action[:, 1] = action[:, 1] * self.config.robot_max_linear_y
            
        elif strategy == "minmax":
            # Option B: Min-Max 역변환 (RoboVLMs 방식)
            # 공식: 0.5 * (action + 1) * (max - min) + min
            robot_min_x = -self.config.robot_max_linear_x
            robot_max_x = self.config.robot_max_linear_x
            robot_min_y = -self.config.robot_max_linear_y
            robot_max_y = self.config.robot_max_linear_y
            
            action[:, 0] = 0.5 * (action[:, 0] + 1) * (robot_max_x - robot_min_x) + robot_min_x
            action[:, 1] = 0.5 * (action[:, 1] + 1) * (robot_max_y - robot_min_y) + robot_min_y
            
        elif strategy == "safe":
            # Option C: 안전 제한 + 스케일링 (기본값, 권장)
            # 1. 클리핑으로 안전 보장
            # 2. 조절 가능한 max_speed 파라미터 사용
            action = np.clip(action, -1.0, 1.0)
            action[:, 0] = action[:, 0] * self.config.max_linear_x
            action[:, 1] = action[:, 1] * self.config.max_linear_y
            
        else:
            raise ValueError(f"알 수 없는 denormalize 전략: {strategy}. "
                           f"'scale', 'minmax', 'safe' 중 선택")
        
        return action


class MobileVLAInferenceSystem:
    def __init__(self, config: MobileVLAConfig):
        self.config = config
        self.image_buffer = ImageBuffer(
            window_size=config.window_size,
            image_size=config.image_size
        )
        self.inference_engine = RoboVLMsInferenceEngine(config)
        self.action_chunk_buffer = None
        self.action_chunk_index = 0
        
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """성능 벤치마크"""
        print(f"⏱️ 벤치마크 시작 ({num_iterations}회)...")
        
        if self.inference_engine.model is None:
             if not self.inference_engine.load_model():
                 return {}

        # 웜업
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        # 버퍼 채우기
        for _ in range(8):
            self.image_buffer.add_image(dummy_image)
            
        # 웜업 실행
        try:
            self.inference_engine.predict_action(self.image_buffer.get_images(), "move forward")
        except Exception as e:
            print(f"❌ 웜업 실패: {e}")
            return {}
        
        latencies = []
        for _ in range(num_iterations):
            start = time.time()
            # 벤치마크용 예측 (이미지 추가 과정 생략)
            self.inference_engine.predict_action(self.image_buffer.get_images(), "move forward")
            latencies.append(time.time() - start)
            
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        fps = 1.0 / avg_latency
        
        print(f"✅ 벤치마크 완료:")
        print(f"  - 평균 지연 시간: {avg_latency*1000:.2f} ms")
        print(f"  - 최대 지연 시간: {max_latency*1000:.2f} ms")
        print(f"  - FPS: {fps:.2f}")
        
        return {
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "fps": fps
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mobile VLA Inference System")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        config = MobileVLAConfig(checkpoint_path=args.checkpoint)
        system = MobileVLAInferenceSystem(config)
        
        if args.benchmark:
            system.benchmark()
        else:
            print("ℹ️ 일반 모드. 모델 로드 테스트:")
            if system.inference_engine.load_model():
                print("✅ 모델 로드 성공. 실제 실행은 ROS2 노드를 이용하세요.")
    else:
        parser.print_help() 