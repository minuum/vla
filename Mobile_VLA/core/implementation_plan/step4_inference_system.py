#!/usr/bin/env python3
"""
Step 4: 추론 시스템 구현
실시간 추론, Jetson 최적화, Docker 컨테이너
"""

import torch
import torch.jit
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import threading
import queue
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """추론 설정"""
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_fps: int = 10
    image_size: Tuple[int, int] = (224, 224)
    memory_limit_gb: float = 14.0
    use_torchscript: bool = True
    use_fp16: bool = True

class MobileVLAInference:
    """Mobile VLA 실시간 추론 시스템"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.history_memory = []
        self.history_size = 8
        
        # 성능 모니터링
        self.inference_times = []
        self.memory_usage = []
        
        # 모델 로드
        self._load_model()
        
        logger.info(f"🚀 Mobile VLA 추론 시스템 초기화 완료")
        logger.info(f"  - Device: {self.config.device}")
        logger.info(f"  - Batch Size: {self.config.batch_size}")
        logger.info(f"  - Max FPS: {self.config.max_fps}")
        logger.info(f"  - TorchScript: {self.config.use_torchscript}")
        logger.info(f"  - FP16: {self.config.use_fp16}")
    
    def _load_model(self):
        """모델 로드 및 최적화"""
        try:
            # 모델 로드
            if self.config.use_torchscript:
                self.model = torch.jit.load(self.config.model_path)
                logger.info("✅ TorchScript 모델 로드 완료")
            else:
                # 일반 PyTorch 모델 로드
                checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
                self.model = checkpoint['model']
                self.model.eval()
                logger.info("✅ PyTorch 모델 로드 완료")
            
            # 디바이스로 이동
            self.model = self.model.to(self.config.device)
            
            # FP16 최적화
            if self.config.use_fp16 and self.config.device == "cuda":
                self.model = self.model.half()
                logger.info("✅ FP16 최적화 적용")
            
            # TorchScript 최적화 (추가)
            if self.config.use_torchscript and not isinstance(self.model, torch.jit.ScriptModule):
                self.model = torch.jit.optimize_for_inference(self.model)
                logger.info("✅ TorchScript 추론 최적화 적용")
            
            # 메모리 사용량 확인
            self._check_memory_usage()
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def _check_memory_usage(self):
        """메모리 사용량 확인"""
        if self.config.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            logger.info(f"📊 GPU 메모리 사용량:")
            logger.info(f"  - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  - Reserved: {memory_reserved:.2f} GB")
            
            if memory_allocated > self.config.memory_limit_gb:
                logger.warning(f"⚠️  메모리 사용량이 제한을 초과했습니다: {memory_allocated:.2f} GB > {self.config.memory_limit_gb} GB")
        else:
            logger.info("📊 CPU 모드에서 실행 중")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # OpenCV 이미지를 RGB로 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 리사이징
        image = cv2.resize(image, self.config.image_size)
        
        # 정규화 (ImageNet 표준)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # 텐서로 변환
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # 디바이스로 이동
        image_tensor = image_tensor.to(self.config.device)
        
        # FP16 변환
        if self.config.use_fp16 and self.config.device == "cuda":
            image_tensor = image_tensor.half()
        
        return image_tensor
    
    def update_history(self, action: torch.Tensor):
        """히스토리 업데이트"""
        self.history_memory.append(action.cpu().numpy())
        
        # 히스토리 크기 제한
        if len(self.history_memory) > self.history_size:
            self.history_memory.pop(0)
    
    def predict_action(self, image: np.ndarray, text: str) -> Dict[str, np.ndarray]:
        """액션 예측"""
        start_time = time.time()
        
        try:
            # 이미지 전처리
            image_tensor = self.preprocess_image(image)
            
            # 모델 추론
            with torch.no_grad():
                if self.config.use_torchscript:
                    # TorchScript 모델 추론
                    action_tensor = self.model(image_tensor, text)
                else:
                    # 일반 모델 추론
                    action_tensor = self.model.get_action(image_tensor, text)
            
            # 액션 후처리
            action = action_tensor.cpu().numpy().squeeze()
            
            # 히스토리 업데이트
            self.update_history(action_tensor)
            
            # 추론 시간 기록
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # FPS 제한
            target_time = 1.0 / self.config.max_fps
            if inference_time < target_time:
                time.sleep(target_time - inference_time)
            
            return {
                "action": action,
                "inference_time": inference_time,
                "fps": 1.0 / inference_time,
                "history_length": len(self.history_memory)
            }
            
        except Exception as e:
            logger.error(f"❌ 액션 예측 실패: {e}")
            return {
                "action": np.zeros(3),
                "inference_time": 0.0,
                "fps": 0.0,
                "history_length": len(self.history_memory),
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        if not self.inference_times:
            return {"avg_fps": 0.0, "avg_inference_time": 0.0}
        
        avg_inference_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return {
            "avg_fps": avg_fps,
            "avg_inference_time": avg_inference_time,
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "total_inferences": len(self.inference_times)
        }

class RealTimeInferenceServer:
    """실시간 추론 서버"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.inference_engine = MobileVLAInference(config)
        self.is_running = False
        self.image_queue = queue.Queue(maxsize=10)
        self.action_queue = queue.Queue(maxsize=10)
        
        # 스레드
        self.inference_thread = None
        self.camera_thread = None
        
        logger.info("🚀 실시간 추론 서버 초기화 완료")
    
    def start_camera(self, camera_id: int = 0):
        """카메라 시작"""
        def camera_worker():
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            while self.is_running:
                ret, frame = cap.read()
                if ret:
                    try:
                        self.image_queue.put_nowait(frame)
                    except queue.Full:
                        # 큐가 가득 찬 경우 오래된 프레임 제거
                        try:
                            self.image_queue.get_nowait()
                            self.image_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                else:
                    logger.warning("카메라 프레임 읽기 실패")
                    time.sleep(0.1)
            
            cap.release()
            logger.info("카메라 종료")
        
        self.camera_thread = threading.Thread(target=camera_worker)
        self.camera_thread.start()
        logger.info(f"카메라 시작 (ID: {camera_id})")
    
    def start_inference(self, text_command: str = "go to the object"):
        """추론 시작"""
        def inference_worker():
            while self.is_running:
                try:
                    # 이미지 가져오기
                    image = self.image_queue.get(timeout=1.0)
                    
                    # 액션 예측
                    result = self.inference_engine.predict_action(image, text_command)
                    
                    # 결과 저장
                    self.action_queue.put_nowait(result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"추론 오류: {e}")
                    time.sleep(0.1)
        
        self.inference_thread = threading.Thread(target=inference_worker)
        self.inference_thread.start()
        logger.info(f"추론 시작 (명령: {text_command})")
    
    def start(self, camera_id: int = 0, text_command: str = "go to the object"):
        """서버 시작"""
        self.is_running = True
        
        # 카메라 시작
        self.start_camera(camera_id)
        
        # 추론 시작
        self.start_inference(text_command)
        
        logger.info("🚀 실시간 추론 서버 시작")
    
    def stop(self):
        """서버 중지"""
        self.is_running = False
        
        # 스레드 종료 대기
        if self.camera_thread:
            self.camera_thread.join()
        if self.inference_thread:
            self.inference_thread.join()
        
        logger.info("🛑 실시간 추론 서버 중지")
    
    def get_latest_action(self) -> Optional[Dict]:
        """최신 액션 가져오기"""
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 가져오기"""
        return self.inference_engine.get_performance_stats()

class JetsonOptimizer:
    """Jetson 최적화 도구"""
    
    @staticmethod
    def optimize_for_jetson(model: torch.nn.Module) -> torch.nn.Module:
        """Jetson용 모델 최적화"""
        logger.info("🔧 Jetson 최적화 적용 중...")
        
        # 1. FP16 변환
        if torch.cuda.is_available():
            model = model.half()
            logger.info("✅ FP16 변환 완료")
        
        # 2. TorchScript 최적화
        model = torch.jit.script(model)
        model = torch.jit.optimize_for_inference(model)
        logger.info("✅ TorchScript 최적화 완료")
        
        # 3. 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info("✅ 메모리 최적화 완료")
        
        return model
    
    @staticmethod
    def benchmark_model(model: torch.nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """모델 벤치마크"""
        logger.info(f"📊 모델 벤치마크 시작 ({num_runs}회 실행)")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, *input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            if hasattr(model, 'half'):
                dummy_input = dummy_input.half()
        
        # 워밍업
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # 벤치마크 실행
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            times.append(time.time() - start_time)
        
        # 통계 계산
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        results = {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "min_inference_time": min_time,
            "max_inference_time": max_time,
            "fps": fps
        }
        
        logger.info(f"📊 벤치마크 결과:")
        logger.info(f"  - 평균 추론 시간: {avg_time*1000:.2f}ms")
        logger.info(f"  - FPS: {fps:.2f}")
        logger.info(f"  - 최소 시간: {min_time*1000:.2f}ms")
        logger.info(f"  - 최대 시간: {max_time*1000:.2f}ms")
        
        return results

def test_inference_system():
    """추론 시스템 테스트"""
    logger.info("🧪 Mobile VLA 추론 시스템 테스트 시작")
    
    try:
        # 설정
        config = InferenceConfig(
            model_path="test_model.pth",
            device="cpu",  # 테스트용 CPU
            batch_size=1,
            max_fps=10,
            use_torchscript=False,
            use_fp16=False
        )
        
        # 더미 모델 생성 (테스트용)
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3*224*224, 3)
            
            def forward(self, x, text):
                x = x.view(x.size(0), -1)
                return self.linear(x)
        
        dummy_model = DummyModel()
        torch.save(dummy_model.state_dict(), "test_model.pth")
        
        # 추론 엔진 생성
        inference_engine = MobileVLAInference(config)
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_text = "go to the red box"
        
        # 액션 예측 테스트
        logger.info("액션 예측 테스트...")
        result = inference_engine.predict_action(test_image, test_text)
        
        logger.info(f"✅ 액션 예측 성공:")
        logger.info(f"  - Action: {result['action']}")
        logger.info(f"  - Inference Time: {result['inference_time']:.4f}s")
        logger.info(f"  - FPS: {result['fps']:.2f}")
        
        # 성능 통계
        stats = inference_engine.get_performance_stats()
        logger.info(f"📊 성능 통계: {stats}")
        
        # 정리
        Path("test_model.pth").unlink()
        
        logger.info("✅ 추론 시스템 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 추론 시스템 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🚀 Mobile VLA 추론 시스템 구현 시작")
    
    # 추론 시스템 테스트 실행
    success = test_inference_system()
    
    if success:
        logger.info("✅ Mobile VLA 추론 시스템 구현 완료")
        logger.info("🎯 다음 단계: 성능 평가 시스템 구현")
    else:
        logger.error("❌ Mobile VLA 추론 시스템 구현 실패")
        logger.error("🔧 문제를 해결한 후 다시 시도해주세요")

if __name__ == "__main__":
    main()
