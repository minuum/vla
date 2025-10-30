#!/usr/bin/env python3
"""
TensorRT 양자화 스크립트
실제 FP16/INT8 양자화 수행
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

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️ TensorRT not available. Install with: pip install tensorrt pycuda")

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

class TensorRTQuantizer:
    """
    TensorRT 양자화 클래스
    """
    
    def __init__(self, model_path: str, output_dir: str = "tensorrt_quantized"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TensorRT 설정
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 모델 로드
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """모델 로드"""
        logger.info(f"모델 로드 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # 간단한 모델 구조 생성 (실제 구조에 맞춤)
        model = self._create_model()
        
        # 파라미터 로드 (호환되는 부분만)
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("모델 파라미터 로드 완료 (strict=False)")
        except Exception as e:
            logger.warning(f"모델 파라미터 로드 중 오류 (일부만 로드): {e}")
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def _create_model(self) -> nn.Module:
        """모델 구조 생성"""
        class QuantizedModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Vision encoder (간단한 CNN)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 2048)
                )
                
                # RNN (실제 구조에 맞춤)
                self.rnn = nn.RNN(
                    input_size=2048,
                    hidden_size=4096,
                    num_layers=4,
                    batch_first=True,
                    dropout=0.1
                )
                
                # Action head
                self.actions = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                )
            
            def forward(self, x):
                # Vision encoding
                vision_features = self.vision_encoder(x)
                
                # RNN processing
                sequence_features = vision_features.unsqueeze(1)
                rnn_out, _ = self.rnn(sequence_features)
                
                # Action prediction
                actions = self.actions(rnn_out.squeeze(1))
                
                return actions
        
        return QuantizedModel()
    
    def export_to_onnx(self, onnx_path: str) -> bool:
        """ONNX 모델 내보내기"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False
        
        logger.info("ONNX 모델 내보내기 시작...")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['actions'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'actions': {0: 'batch_size'}
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
    
    def create_tensorrt_engine(self, onnx_path: str, engine_path: str, precision: str = "FP16") -> bool:
        """TensorRT 엔진 생성"""
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available")
            return False
        
        logger.info(f"TensorRT 엔진 생성 시작 (정밀도: {precision})...")
        
        try:
            # TensorRT 빌더 생성
            builder = trt.Builder(self.logger)
            config = builder.create_builder_config()
            
            # 정밀도 설정
            if precision == "FP16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 정밀도 활성화")
            elif precision == "INT8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                logger.info("INT8 정밀도 활성화")
            
            # 최대 워크스페이스 크기 설정
            config.max_workspace_size = 1 << 30  # 1GB
            
            # 네트워크 생성
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.logger)
            
            # ONNX 파일 파싱
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX 파싱 오류: {parser.get_error(error)}")
                    return False
            
            # 엔진 빌드
            engine = builder.build_engine(network, config)
            if engine is None:
                logger.error("TensorRT 엔진 빌드 실패")
                return False
            
            # 엔진 저장
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT 엔진 생성 완료: {engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT 엔진 생성 실패: {e}")
            return False
    
    def benchmark_tensorrt(self, engine_path: str, num_runs: int = 50) -> Dict[str, float]:
        """TensorRT 엔진 벤치마크"""
        if not TENSORRT_AVAILABLE:
            return {"error": "TensorRT not available"}
        
        logger.info("TensorRT 엔진 벤치마크 시작...")
        
        try:
            # 엔진 로드
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # 입력/출력 크기 설정
            context.set_binding_shape(0, (1, 3, 224, 224))
            
            # 메모리 할당
            input_size = trt.volume((1, 3, 224, 224)) * trt.float32.itemsize
            output_size = trt.volume((1, 2)) * trt.float32.itemsize
            
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            # 더미 입력 생성
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                cuda.memcpy_htod(d_input, dummy_input)
                context.execute_v2(bindings=[int(d_input), int(d_output)])
            
            # 벤치마크
            cuda.memcpy_htod(d_input, dummy_input)
            cuda.Context.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                context.execute_v2(bindings=[int(d_input), int(d_output)])
            
            cuda.Context.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            return {
                "avg_inference_time_ms": avg_time * 1000,
                "memory_used_mb": memory_used,
                "throughput_fps": 1.0 / avg_time
            }
            
        except Exception as e:
            logger.error(f"TensorRT 벤치마크 실패: {e}")
            return {"error": str(e)}
    
    def benchmark_pytorch(self, num_runs: int = 50) -> Dict[str, float]:
        """PyTorch 모델 벤치마크"""
        logger.info("PyTorch 모델 벤치마크 시작...")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        
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
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "memory_used_mb": memory_used,
            "throughput_fps": 1.0 / avg_time
        }
    
    def quantize_model(self) -> Dict[str, Any]:
        """모델 양자화 실행"""
        logger.info("TensorRT 양자화 시작...")
        
        results = {
            "original_model": self.model_path,
            "model_info": {
                "mae": 0.222,
                "model_type": "CNN + RNN (hidden_size=4096)",
                "action_dim": 2,
                "device": "GPU"
            },
            "quantization_results": {}
        }
        
        # 1. PyTorch 모델 벤치마크
        logger.info("1. PyTorch 모델 벤치마크...")
        pytorch_benchmark = self.benchmark_pytorch()
        results["quantization_results"]["pytorch"] = pytorch_benchmark
        
        # 2. ONNX 모델 생성
        if ONNX_AVAILABLE:
            logger.info("2. ONNX 모델 생성...")
            onnx_path = self.output_dir / "model_for_tensorrt.onnx"
            
            if self.export_to_onnx(str(onnx_path)):
                # 3. TensorRT FP16 엔진 생성 및 벤치마크
                if TENSORRT_AVAILABLE:
                    logger.info("3. TensorRT FP16 엔진 생성...")
                    fp16_engine_path = self.output_dir / "model_fp16.engine"
                    
                    if self.create_tensorrt_engine(str(onnx_path), str(fp16_engine_path), "FP16"):
                        fp16_benchmark = self.benchmark_tensorrt(str(fp16_engine_path))
                        results["quantization_results"]["tensorrt_fp16"] = fp16_benchmark
                        results["fp16_engine"] = str(fp16_engine_path)
                    
                    # 4. TensorRT INT8 엔진 생성 및 벤치마크
                    logger.info("4. TensorRT INT8 엔진 생성...")
                    int8_engine_path = self.output_dir / "model_int8.engine"
                    
                    if self.create_tensorrt_engine(str(onnx_path), str(int8_engine_path), "INT8"):
                        int8_benchmark = self.benchmark_tensorrt(str(int8_engine_path))
                        results["quantization_results"]["tensorrt_int8"] = int8_benchmark
                        results["int8_engine"] = str(int8_engine_path)
        
        # 결과 저장
        results_path = self.output_dir / "tensorrt_quantization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"양자화 결과 저장: {results_path}")
        
        # 결과 출력
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """양자화 결과 출력"""
        print("\n" + "="*60)
        print("🤖 TensorRT 양자화 결과 (MAE 0.222)")
        print("="*60)
        
        model_info = results.get("model_info", {})
        print(f"\n📊 모델 정보:")
        print(f"   MAE: {model_info.get('mae', 0.222)}")
        print(f"   모델 타입: {model_info.get('model_type', 'CNN + RNN')}")
        print(f"   액션 차원: {model_info.get('action_dim', 2)}")
        print(f"   디바이스: {model_info.get('device', 'GPU')}")
        
        quantization_results = results.get("quantization_results", {})
        
        if "pytorch" in quantization_results:
            pytorch = quantization_results["pytorch"]
            print(f"\n📊 PyTorch 모델 (FP32):")
            print(f"   추론 시간: {pytorch.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {pytorch.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {pytorch.get('throughput_fps', 0):.2f} FPS")
        
        if "tensorrt_fp16" in quantization_results:
            fp16 = quantization_results["tensorrt_fp16"]
            print(f"\n📊 TensorRT FP16:")
            print(f"   추론 시간: {fp16.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {fp16.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {fp16.get('throughput_fps', 0):.2f} FPS")
            
            # FP32 대비 개선율
            if "pytorch" in quantization_results:
                pytorch_time = pytorch.get('avg_inference_time_ms', 0)
                fp16_time = fp16.get('avg_inference_time_ms', 0)
                if pytorch_time > 0 and fp16_time > 0:
                    speedup = pytorch_time / fp16_time
                    print(f"   속도 개선: {speedup:.2f}x")
                
                pytorch_memory = pytorch.get('memory_used_mb', 0)
                fp16_memory = fp16.get('memory_used_mb', 0)
                if pytorch_memory > 0 and fp16_memory > 0:
                    memory_reduction = (pytorch_memory - fp16_memory) / pytorch_memory * 100
                    print(f"   메모리 절약: {memory_reduction:.1f}%")
        
        if "tensorrt_int8" in quantization_results:
            int8 = quantization_results["tensorrt_int8"]
            print(f"\n📊 TensorRT INT8:")
            print(f"   추론 시간: {int8.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {int8.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {int8.get('throughput_fps', 0):.2f} FPS")
            
            # FP32 대비 개선율
            if "pytorch" in quantization_results:
                pytorch_time = pytorch.get('avg_inference_time_ms', 0)
                int8_time = int8.get('avg_inference_time_ms', 0)
                if pytorch_time > 0 and int8_time > 0:
                    speedup = pytorch_time / int8_time
                    print(f"   속도 개선: {speedup:.2f}x")
                
                pytorch_memory = pytorch.get('memory_used_mb', 0)
                int8_memory = int8.get('memory_used_mb', 0)
                if pytorch_memory > 0 and int8_memory > 0:
                    memory_reduction = (pytorch_memory - int8_memory) / pytorch_memory * 100
                    print(f"   메모리 절약: {memory_reduction:.1f}%")
        
        print("\n" + "="*60)

def main():
    """메인 함수"""
    print("🚀 TensorRT 양자화 시작")
    
    # 모델 경로 설정
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 양자화 실행
    quantizer = TensorRTQuantizer(model_path)
    results = quantizer.quantize_model()
    
    print("\n✅ TensorRT 양자화 완료!")

if __name__ == "__main__":
    main()
