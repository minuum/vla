#!/usr/bin/env python3
"""
Action Chunk 기반 Mobile VLA 추론 시스템

주요 기능:
- 200ms마다 10개의 액션 청크 예측
- 20ms마다 개별 액션 실행
- 0.4s마다 2DOF 입력 수집
- 파인튜닝된 모델 로드 및 추론
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys
from typing import Optional, Tuple, Dict, Any, List
import json
from pathlib import Path
from collections import deque
import threading
import queue


class ActionScheduler:
    """액션 청크 관리 및 스케줄링"""
    
    def __init__(self, chunk_size: int = 10, inference_interval: float = 0.2):
        """
        Args:
            chunk_size: 한 번에 예측할 액션 개수 (기본 10개)
            inference_interval: 추론 주기 (초, 기본 0.2초 = 200ms)
        """
        self.chunk_size = chunk_size
        self.inference_interval = inference_interval
        self.action_interval = 0.02  # 20ms
        
        self.action_chunk = None
        self.chunk_index = 0
        self.last_inference_time = 0
        
        self.is_initialized = False
        
    def should_infer(self, current_time: float) -> bool:
        """추론이 필요한지 확인"""
        if not self.is_initialized:
            return True
        
        return (current_time - self.last_inference_time) >= self.inference_interval
    
    def get_current_action(self) -> Optional[np.ndarray]:
        """현재 실행할 액션 반환"""
        if self.action_chunk is None:
            return None
        
        if self.chunk_index >= len(self.action_chunk):
            return None
        
        action = self.action_chunk[self.chunk_index]
        self.chunk_index += 1
        return action
    
    def update_chunk(self, new_chunk: np.ndarray):
        """새로운 액션 청크로 업데이트"""
        assert new_chunk.shape == (self.chunk_size, 2), \
            f"Expected shape ({self.chunk_size}, 2), got {new_chunk.shape}"
        
        self.action_chunk = new_chunk
        self.chunk_index = 0
        self.last_inference_time = time.time()
        
        if not self.is_initialized:
            self.is_initialized = True
    
    def reset(self):
        """스케줄러 리셋"""
        self.action_chunk = None
        self.chunk_index = 0
        self.last_inference_time = 0
        self.is_initialized = False


class InputManager:
    """센서 입력 수집 및 관리"""
    
    def __init__(self, velocity_interval: float = 0.4):
        """
        Args:
            velocity_interval: 속도 입력 수집 주기 (초, 기본 0.4초)
        """
        self.velocity_interval = velocity_interval
        self.last_velocity_time = 0
        
        self.velocity_buffer = deque(maxlen=10)  # 최근 10개 유지
        self.current_image = None
        self.text_command = ""
        
        self.initial_distance = None
        
    def should_collect_velocity(self, current_time: float) -> bool:
        """속도 입력 수집이 필요한지 확인"""
        return (current_time - self.last_velocity_time) >= self.velocity_interval
    
    def add_velocity(self, velocity: Tuple[float, float]):
        """속도 입력 추가"""
        self.velocity_buffer.append({
            'timestamp': time.time(),
            'velocity': velocity
        })
        self.last_velocity_time = time.time()
    
    def get_recent_velocities(self, n: int = 5) -> List[Tuple[float, float]]:
        """최근 n개의 속도 반환"""
        velocities = [item['velocity'] for item in list(self.velocity_buffer)[-n:]]
        return velocities
    
    def set_initial_input(self, image: np.ndarray, text: str, distance: float):
        """초기 입력 설정"""
        self.current_image = image
        self.text_command = text
        self.initial_distance = distance
    
    def update_image(self, image: np.ndarray):
        """이미지 업데이트"""
        self.current_image = image


class ModelInference:
    """VLA 모델 추론 엔진"""
    
    def __init__(self, checkpoint_path: str, chunk_size: int = 10):
        """
        Args:
            checkpoint_path: 파인튜닝된 모델 체크포인트 경로
            chunk_size: 액션 청크 크기
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        
        print(f"Using device: {self.device}")
        
        # 모델 로드
        self.model = self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path: str):
        """파인튜닝된 모델 로드"""
        print(f"Loading model from {checkpoint_path}...")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 아키텍처 생성
        model = self._create_model_architecture()
        
        # 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def _create_model_architecture(self):
        """모델 아키텍처 생성"""
        # 실제 모델 아키텍처는 학습 시 사용한 것과 동일해야 함
        # 여기서는 간단한 예시
        
        class ActionChunkPredictor(nn.Module):
            def __init__(self, chunk_size=10):
                super().__init__()
                self.chunk_size = chunk_size
                
                # 특징 추출 (간단한 CNN)
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                # 액션 예측 헤드
                self.action_head = nn.Sequential(
                    nn.Linear(64 * 7 * 7, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, chunk_size * 2)  # chunk_size개의 2DOF 액션
                )
            
            def forward(self, image):
                # 특징 추출
                features = self.feature_extractor(image)
                features = features.view(features.size(0), -1)
                
                # 액션 예측
                actions = self.action_head(features)
                actions = actions.view(-1, self.chunk_size, 2)
                
                return actions
        
        return ActionChunkPredictor(chunk_size=self.chunk_size)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 리사이즈
        image = cv2.resize(image, (224, 224))
        
        # 정규화
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Numpy to Tensor
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict_action_chunk(self, image: np.ndarray, text: str = None, 
                            state: Dict = None) -> np.ndarray:
        """
        액션 청크 예측
        
        Args:
            image: 입력 이미지 (H, W, C)
            text: 텍스트 명령 (선택)
            state: 로봇 상태 (선택)
        
        Returns:
            action_chunk: shape (chunk_size, 2) - chunk_size개의 2DOF 액션
        """
        # 이미지 전처리
        image_tensor = self.preprocess_image(image)
        
        # 추론
        with torch.no_grad():
            action_chunk = self.model(image_tensor)
        
        # Tensor to Numpy
        action_chunk = action_chunk.squeeze(0).cpu().numpy()
        
        return action_chunk


class InferenceValidator:
    """추론 결과 검증"""
    
    def __init__(self, max_linear_vel: float = 1.0, max_angular_vel: float = 1.0):
        """
        Args:
            max_linear_vel: 최대 선속도
            max_angular_vel: 최대 각속도
        """
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        
        self.action_history = []
        
    def validate_action_chunk(self, action_chunk: np.ndarray) -> bool:
        """액션 청크 유효성 검사"""
        # 형태 확인
        if len(action_chunk.shape) != 2 or action_chunk.shape[1] != 2:
            print(f"Invalid shape: {action_chunk.shape}")
            return False
        
        # 값 범위 확인
        for i, action in enumerate(action_chunk):
            x, y = action
            
            # NaN/Inf 확인
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                print(f"Action {i}: Invalid value (NaN or Inf)")
                return False
            
            # 속도 제한 확인
            if abs(x) > self.max_linear_vel or abs(y) > self.max_angular_vel:
                print(f"Action {i}: Exceeds velocity limits (x={x}, y={y})")
                return False
        
        return True
    
    def log_action(self, action: np.ndarray, timestamp: float):
        """액션 로깅"""
        self.action_history.append({
            'timestamp': timestamp,
            'action': action.tolist()
        })
    
    def save_action_log(self, filepath: str):
        """액션 로그 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.action_history, f, indent=2)
        print(f"Action log saved to {filepath}")


class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self.inference_times = []
        self.action_execution_times = []
        self.total_actions = 0
        self.failed_actions = 0
        
    def record_inference_time(self, time_ms: float):
        """추론 시간 기록"""
        self.inference_times.append(time_ms)
    
    def record_action_execution(self, success: bool):
        """액션 실행 기록"""
        self.total_actions += 1
        if not success:
            self.failed_actions += 1
    
    def get_statistics(self) -> Dict:
        """통계 계산"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'std_inference_time_ms': np.std(self.inference_times),
            'total_actions': self.total_actions,
            'failed_actions': self.failed_actions,
            'success_rate': 1 - (self.failed_actions / max(self.total_actions, 1))
        }
    
    def print_report(self):
        """성능 보고서 출력"""
        stats = self.get_statistics()
        
        if not stats:
            print("No statistics available")
            return
        
        print("\n" + "="*50)
        print("Performance Report")
        print("="*50)
        print(f"Average inference time: {stats['avg_inference_time_ms']:.2f}ms")
        print(f"Max inference time: {stats['max_inference_time_ms']:.2f}ms")
        print(f"Min inference time: {stats['min_inference_time_ms']:.2f}ms")
        print(f"Std inference time: {stats['std_inference_time_ms']:.2f}ms")
        print(f"Total actions: {stats['total_actions']}")
        print(f"Failed actions: {stats['failed_actions']}")
        print(f"Success rate: {stats['success_rate']*100:.2f}%")
        print("="*50 + "\n")


class ActionChunkInferenceSystem:
    """액션 청크 기반 VLA 추론 시스템"""
    
    def __init__(self, checkpoint_path: str, config: Dict = None):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            config: 설정 딕셔너리
        """
        # 기본 설정
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 10)
        self.inference_interval = self.config.get('inference_interval', 0.2)
        self.velocity_interval = self.config.get('velocity_interval', 0.4)
        
        # 컴포넌트 초기화
        self.model = ModelInference(checkpoint_path, chunk_size=self.chunk_size)
        self.scheduler = ActionScheduler(
            chunk_size=self.chunk_size,
            inference_interval=self.inference_interval
        )
        self.input_manager = InputManager(velocity_interval=self.velocity_interval)
        self.validator = InferenceValidator()
        self.monitor = PerformanceMonitor()
        
        self.is_running = False
        
    def initialize(self, image: np.ndarray, text: str, distance: float):
        """시스템 초기화"""
        print("Initializing inference system...")
        
        # 초기 입력 설정
        self.input_manager.set_initial_input(image, text, distance)
        
        # 첫 추론 수행
        print("Performing initial inference...")
        start_time = time.time()
        action_chunk = self.model.predict_action_chunk(image, text)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"Initial inference time: {inference_time:.2f}ms")
        print(f"Action chunk shape: {action_chunk.shape}")
        print(f"First action: {action_chunk[0]}")
        
        # 검증
        if not self.validator.validate_action_chunk(action_chunk):
            raise ValueError("Initial action chunk validation failed")
        
        # 스케줄러 업데이트
        self.scheduler.update_chunk(action_chunk)
        
        # 성능 기록
        self.monitor.record_inference_time(inference_time)
        
        print("Initialization complete!")
    
    def step(self) -> Optional[np.ndarray]:
        """
        한 스텝 실행
        
        Returns:
            현재 실행할 액션 또는 None
        """
        current_time = time.time()
        
        # 추론이 필요한 경우
        if self.scheduler.should_infer(current_time):
            self._perform_inference()
        
        # 현재 액션 가져오기
        action = self.scheduler.get_current_action()
        
        if action is not None:
            # 액션 로깅
            self.validator.log_action(action, current_time)
            self.monitor.record_action_execution(success=True)
        
        return action
    
    def _perform_inference(self):
        """모델 추론 수행"""
        # 현재 이미지 가져오기
        image = self.input_manager.current_image
        text = self.input_manager.text_command
        
        if image is None:
            print("Warning: No image available for inference")
            return
        
        # 추론
        start_time = time.time()
        action_chunk = self.model.predict_action_chunk(image, text)
        inference_time = (time.time() - start_time) * 1000
        
        # 검증
        if not self.validator.validate_action_chunk(action_chunk):
            print("Warning: Action chunk validation failed")
            self.monitor.record_action_execution(success=False)
            return
        
        # 스케줄러 업데이트
        self.scheduler.update_chunk(action_chunk)
        
        # 성능 기록
        self.monitor.record_inference_time(inference_time)
        
        print(f"Inference completed in {inference_time:.2f}ms")
    
    def run_demo(self, duration: float = 10.0):
        """
        데모 실행
        
        Args:
            duration: 실행 시간 (초)
        """
        print(f"\nRunning demo for {duration} seconds...")
        
        start_time = time.time()
        action_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                # 스텝 실행
                action = self.step()
                
                if action is not None:
                    action_count += 1
                    print(f"Action {action_count}: [{action[0]:.4f}, {action[1]:.4f}]")
                
                # 20ms 대기 (실제 액션 실행 주기)
                time.sleep(0.02)
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        # 성능 보고서
        self.monitor.print_report()
        
        # 액션 로그 저장
        log_path = "logs/demo_actions.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.validator.save_action_log(log_path)


def main():
    """메인 함수"""
    print("="*60)
    print("Action Chunk Based VLA Inference System")
    print("="*60)
    
    # 체크포인트 경로 (실제 경로로 수정 필요)
    checkpoint_path = "checkpoints/mobile_vla_finetuned.pth"
    
    # 체크포인트 찾기
    if not os.path.exists(checkpoint_path):
        # 자동 탐색
        possible_paths = [
            "checkpoints/mobile_vla_finetuned.pth",
            "checkpoints/best_model.pth",
            "checkpoints/latest.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        else:
            print("Error: No checkpoint found")
            print("Please specify the correct checkpoint path")
            return False
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # 설정
    config = {
        'chunk_size': 10,
        'inference_interval': 0.2,  # 200ms
        'velocity_interval': 0.4,   # 400ms
    }
    
    # 시스템 초기화
    try:
        system = ActionChunkInferenceSystem(checkpoint_path, config)
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False
    
    # 더미 입력으로 초기화
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_text = "1box_hori_right"
    dummy_distance = 1.5
    
    try:
        system.initialize(dummy_image, dummy_text, dummy_distance)
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False
    
    # 데모 실행
    system.run_demo(duration=5.0)
    
    print("\nDemo completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
