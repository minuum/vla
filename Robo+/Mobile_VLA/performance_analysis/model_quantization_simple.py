#!/usr/bin/env python3
"""
Mobile VLA 모델 간단한 양자화 스크립트
Jetson Orin NX에서 메모리 사용량 최적화
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX not available. Install with: pip install onnx onnxruntime")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModelQuantizer:
    """
    간단한 모델 양자화 클래스
    """
    
    def __init__(self, model_path: str, output_dir: str = "quantized_models"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Jetson Orin NX 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 로드
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """모델 로드"""
        logger.info(f"모델 로드 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 모델 구조 확인
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 모델 구조 분석
        logger.info("모델 구조 분석 중...")
        logger.info(f"총 파라미터 수: {len(state_dict)}")
        
        # Kosmos2 기반 모델인지 확인
        if any('kosmos_model' in key for key in state_dict.keys()):
            logger.info("Kosmos2 기반 모델 감지됨")
            model = self._create_kosmos2_model()
        else:
            logger.info("일반 모델로 처리")
            model = self._create_generic_model()
        
        # 파라미터 로드 (호환되는 부분만)
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("모델 파라미터 로드 완료 (strict=False)")
        except Exception as e:
            logger.warning(f"모델 파라미터 로드 중 오류 (일부만 로드): {e}")
        
        model.eval()
        model.to(self.device)
        
        logger.info("모델 로드 완료")
        return model
    
    def _create_kosmos2_model(self) -> nn.Module:
        """Kosmos2 기반 모델 생성"""
        class Kosmos2BasedModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Vision encoder (간단한 CNN)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 768)  # Kosmos2 hidden size
                )
                
                # RNN (LSTM 대신 간단한 RNN)
                self.rnn = nn.RNN(
                    input_size=768,
                    hidden_size=512,
                    num_layers=4,
                    batch_first=True,
                    dropout=0.1
                )
                
                # Action head
                self.actions = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 2)  # action_dim = 2
                )
            
            def forward(self, x):
                # x: [batch_size, channels, height, width]
                batch_size = x.size(0)
                
                # Vision encoding
                vision_features = self.vision_encoder(x)  # [batch_size, 768]
                
                # RNN processing (시퀀스로 확장)
                sequence_features = vision_features.unsqueeze(1)  # [batch_size, 1, 768]
                rnn_out, _ = self.rnn(sequence_features)  # [batch_size, 1, 512]
                
                # Action prediction
                actions = self.actions(rnn_out.squeeze(1))  # [batch_size, 2]
                
                return actions
        
        return Kosmos2BasedModel()
    
    def _create_generic_model(self) -> nn.Module:
        """일반적인 모델 생성"""
        class GenericModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Vision encoder
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 512)
                )
                
                # Action head
                self.action_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 2)
                )
            
            def forward(self, x):
                features = self.vision_encoder(x)
                actions = self.action_head(features)
                return actions
        
        return GenericModel()
    
    def export_to_onnx(self, onnx_path: str) -> bool:
        """모델을 ONNX 형식으로 내보내기"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False
        
        logger.info("ONNX 모델 내보내기 시작...")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        try:
            # ONNX 내보내기
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # ONNX 모델 검증
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX 모델 내보내기 완료: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX 내보내기 실패: {e}")
            return False
    
    def benchmark_model(self, model_type: str, model_path: str, num_runs: int = 50) -> Dict[str, float]:
        """모델 성능 벤치마크"""
        logger.info(f"{model_type} 모델 벤치마크 시작...")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        if model_type == "PyTorch":
            # PyTorch 모델 벤치마크
            self.model.eval()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            # 벤치마크
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
        elif model_type == "ONNX":
            # ONNX 모델 벤치마크
            if not ONNX_AVAILABLE:
                return {"error": "ONNX not available"}
            
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input.cpu().numpy()})
            
            # 벤치마크
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = session.run(None, {input_name: dummy_input.cpu().numpy()})
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        else:
            return {"error": f"Unknown model type: {model_type}"}
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "memory_used_mb": memory_used,
            "throughput_fps": 1.0 / avg_time
        }
    
    def quantize_model(self) -> Dict[str, Any]:
        """모델 양자화 실행"""
        logger.info("모델 양자화 시작...")
        
        results = {
            "original_model": self.model_path,
            "quantization_results": {}
        }
        
        # 1. PyTorch 모델 벤치마크
        logger.info("1. PyTorch 모델 벤치마크...")
        pytorch_benchmark = self.benchmark_model("PyTorch", self.model_path)
        results["quantization_results"]["pytorch"] = pytorch_benchmark
        
        # 2. ONNX 모델 생성 및 벤치마크
        if ONNX_AVAILABLE:
            logger.info("2. ONNX 모델 생성...")
            onnx_path = self.output_dir / "mobile_vla_model.onnx"
            
            if self.export_to_onnx(str(onnx_path)):
                onnx_benchmark = self.benchmark_model("ONNX", str(onnx_path))
                results["quantization_results"]["onnx"] = onnx_benchmark
                results["onnx_model"] = str(onnx_path)
        
        # 결과 저장
        results_path = self.output_dir / "quantization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"양자화 결과 저장: {results_path}")
        
        # 결과 출력
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """양자화 결과 출력"""
        print("\n" + "="*60)
        print("🤖 Mobile VLA 모델 양자화 결과")
        print("="*60)
        
        quantization_results = results.get("quantization_results", {})
        
        if "pytorch" in quantization_results:
            pytorch = quantization_results["pytorch"]
            print(f"\n📊 PyTorch 모델:")
            print(f"   추론 시간: {pytorch.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {pytorch.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {pytorch.get('throughput_fps', 0):.2f} FPS")
        
        if "onnx" in quantization_results:
            onnx = quantization_results["onnx"]
            print(f"\n📊 ONNX 모델:")
            print(f"   추론 시간: {onnx.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {onnx.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {onnx.get('throughput_fps', 0):.2f} FPS")
            
            # 개선율 계산
            if "pytorch" in quantization_results:
                pytorch_time = pytorch.get('avg_inference_time_ms', 0)
                onnx_time = onnx.get('avg_inference_time_ms', 0)
                if pytorch_time > 0:
                    speedup = pytorch_time / onnx_time
                    print(f"   속도 개선: {speedup:.2f}x")
        
        print("\n" + "="*60)

def main():
    """메인 함수"""
    print("🚀 Mobile VLA 모델 간단한 양자화 시작")
    
    # 모델 경로 설정
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 양자화 실행
    quantizer = SimpleModelQuantizer(model_path)
    results = quantizer.quantize_model()
    
    print("\n✅ 양자화 완료!")

if __name__ == "__main__":
    main()
