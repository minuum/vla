#!/usr/bin/env python3
"""
Mobile VLA 모델 TensorRT 양자화 스크립트
Jetson Orin NX에서 TensorRT 8.6.2.3을 활용한 모델 최적화
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple

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

class MobileVLAQuantizer:
    """
    Mobile VLA 모델 양자화 클래스
    TensorRT와 ONNX를 사용한 모델 최적화
    """
    
    def __init__(self, model_path: str, output_dir: str = "quantized_models"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Jetson Orin NX 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = "fp16"  # Jetson Orin NX에서 권장
        
        # 모델 로드
        self.model = self._load_model()
        
        # TensorRT 설정
        self.trt_logger = trt.Logger(trt.Logger.WARNING) if TENSORRT_AVAILABLE else None
        self.max_batch_size = 1
        self.max_workspace_size = 1 << 30  # 1GB
        
    def _load_model(self) -> nn.Module:
        """모델 로드"""
        logger.info(f"모델 로드 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        # 모델 로드 (간단한 LSTM 모델 구조)
        model = self._create_simple_lstm_model()
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        logger.info("모델 로드 완료")
        return model
    
    def _create_simple_lstm_model(self) -> nn.Module:
        """간단한 LSTM 모델 구조 생성 (실제 모델과 유사하게)"""
        class SimpleLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 768
                self.lstm_hidden_size = 512
                self.action_dim = 2
                
                # Vision encoder (Kosmos2 대신 간단한 CNN)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, self.hidden_size)
                )
                
                # LSTM
                self.lstm = nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1
                )
                
                # Action head
                self.action_head = nn.Sequential(
                    nn.Linear(self.lstm_hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, self.action_dim)
                )
            
            def forward(self, x):
                # x: [batch_size, channels, height, width]
                batch_size = x.size(0)
                
                # Vision encoding
                vision_features = self.vision_encoder(x)  # [batch_size, hidden_size]
                
                # LSTM processing (시퀀스로 확장)
                sequence_features = vision_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
                lstm_out, _ = self.lstm(sequence_features)  # [batch_size, 1, lstm_hidden_size]
                
                # Action prediction
                actions = self.action_head(lstm_out.squeeze(1))  # [batch_size, action_dim]
                
                return actions
        
        return SimpleLSTMModel()
    
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
    
    def create_tensorrt_engine(self, onnx_path: str, engine_path: str, precision: str = "fp16") -> bool:
        """TensorRT 엔진 생성"""
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available")
            return False
        
        logger.info(f"TensorRT 엔진 생성 시작 (precision: {precision})...")
        
        try:
            # TensorRT 빌더 생성
            builder = trt.Builder(self.trt_logger)
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            # Precision 설정
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                logger.info("INT8 precision enabled")
            
            # 네트워크 생성
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.trt_logger)
            
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
    
    def benchmark_model(self, model_type: str, model_path: str, num_runs: int = 100) -> Dict[str, float]:
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
            
        elif model_type == "TensorRT":
            # TensorRT 모델 벤치마크
            if not TENSORRT_AVAILABLE:
                return {"error": "TensorRT not available"}
            
            # TensorRT 엔진 로드
            with open(model_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # 메모리 할당
            input_shape = (1, 3, 224, 224)
            output_shape = (1, 2)  # action_dim = 2
            
            d_input = cuda.mem_alloc(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4)
            d_output = cuda.mem_alloc(output_shape[0] * output_shape[1] * 4)
            
            bindings = [int(d_input), int(d_output)]
            
            # Warmup
            for _ in range(10):
                cuda.memcpy_htod(d_input, dummy_input.cpu().numpy())
                context.execute_v2(bindings)
                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(output, d_output)
            
            # 벤치마크
            start_time = time.time()
            
            for _ in range(num_runs):
                cuda.memcpy_htod(d_input, dummy_input.cpu().numpy())
                context.execute_v2(bindings)
                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(output, d_output)
            
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
        
        # 3. TensorRT FP16 엔진 생성 및 벤치마크
        if TENSORRT_AVAILABLE and ONNX_AVAILABLE:
            logger.info("3. TensorRT FP16 엔진 생성...")
            fp16_engine_path = self.output_dir / "mobile_vla_model_fp16.trt"
            
            if self.create_tensorrt_engine(str(onnx_path), str(fp16_engine_path), "fp16"):
                fp16_benchmark = self.benchmark_model("TensorRT", str(fp16_engine_path))
                results["quantization_results"]["tensorrt_fp16"] = fp16_benchmark
                results["tensorrt_fp16_engine"] = str(fp16_engine_path)
        
        # 4. TensorRT INT8 엔진 생성 및 벤치마크 (선택적)
        if TENSORRT_AVAILABLE and ONNX_AVAILABLE:
            logger.info("4. TensorRT INT8 엔진 생성...")
            int8_engine_path = self.output_dir / "mobile_vla_model_int8.trt"
            
            if self.create_tensorrt_engine(str(onnx_path), str(int8_engine_path), "int8"):
                int8_benchmark = self.benchmark_model("TensorRT", str(int8_engine_path))
                results["quantization_results"]["tensorrt_int8"] = int8_benchmark
                results["tensorrt_int8_engine"] = str(int8_engine_path)
        
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
        
        if "tensorrt_fp16" in quantization_results:
            fp16 = quantization_results["tensorrt_fp16"]
            print(f"\n📊 TensorRT FP16:")
            print(f"   추론 시간: {fp16.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {fp16.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {fp16.get('throughput_fps', 0):.2f} FPS")
        
        if "tensorrt_int8" in quantization_results:
            int8 = quantization_results["tensorrt_int8"]
            print(f"\n📊 TensorRT INT8:")
            print(f"   추론 시간: {int8.get('avg_inference_time_ms', 0):.2f} ms")
            print(f"   메모리 사용량: {int8.get('memory_used_mb', 0):.2f} MB")
            print(f"   처리량: {int8.get('throughput_fps', 0):.2f} FPS")
        
        print("\n" + "="*60)

def main():
    """메인 함수"""
    print("🚀 Mobile VLA 모델 TensorRT 양자화 시작")
    
    # 모델 경로 설정
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 양자화 실행
    quantizer = MobileVLAQuantizer(model_path)
    results = quantizer.quantize_model()
    
    print("\n✅ 양자화 완료!")

if __name__ == "__main__":
    main()
