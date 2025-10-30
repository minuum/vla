#!/usr/bin/env python3
"""
Jetson Orin NX용 TensorRT 양자화 스크립트
MAE 0.222 모델을 TensorRT로 최적화
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

# TensorRT imports (Jetson에서만 사용 가능)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("⚠️ TensorRT not available. This script is for Jetson Orin NX")

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

class JetsonTensorRTQuantizer:
    """
    Jetson Orin NX용 TensorRT 양자화 클래스
    """
    
    def __init__(self, onnx_path: str, output_dir: str = "jetson_tensorrt_models"):
        self.onnx_path = onnx_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Jetson Orin NX 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TensorRT 설정
        if TENSORRT_AVAILABLE:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            self.max_batch_size = 1
            self.max_workspace_size = 1 << 30  # 1GB
        
    def create_tensorrt_engine(self, engine_path: str, precision: str = "fp16") -> bool:
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
            with open(self.onnx_path, 'rb') as model:
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
        """TensorRT 모델 벤치마크"""
        if not TENSORRT_AVAILABLE:
            return {"error": "TensorRT not available"}
        
        logger.info("TensorRT 모델 벤치마크 시작...")
        
        try:
            # TensorRT 엔진 로드
            with open(engine_path, 'rb') as f:
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
            
            # 더미 입력 생성
            dummy_input = torch.randn(input_shape, device=self.device)
            
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
            
            return {
                "avg_inference_time_ms": avg_time * 1000,
                "memory_used_mb": memory_used,
                "throughput_fps": 1.0 / avg_time
            }
            
        except Exception as e:
            logger.error(f"TensorRT 벤치마크 실패: {e}")
            return {"error": str(e)}
    
    def quantize_for_jetson(self) -> Dict[str, Any]:
        """Jetson용 양자화 실행"""
        logger.info("Jetson Orin NX용 TensorRT 양자화 시작...")
        
        results = {
            "onnx_model": self.onnx_path,
            "model_info": {
                "mae": 0.222,
                "model_type": "Simple CNN + RNN",
                "action_dim": 2,
                "target_device": "Jetson Orin NX"
            },
            "tensorrt_results": {}
        }
        
        # 1. TensorRT FP16 엔진 생성 및 벤치마크
        if TENSORRT_AVAILABLE:
            logger.info("1. TensorRT FP16 엔진 생성...")
            fp16_engine_path = self.output_dir / "mae0222_model_fp16.trt"
            
            if self.create_tensorrt_engine(str(fp16_engine_path), "fp16"):
                fp16_benchmark = self.benchmark_tensorrt(str(fp16_engine_path))
                results["tensorrt_results"]["fp16"] = fp16_benchmark
                results["fp16_engine"] = str(fp16_engine_path)
        
        # 2. TensorRT INT8 엔진 생성 및 벤치마크
        if TENSORRT_AVAILABLE:
            logger.info("2. TensorRT INT8 엔진 생성...")
            int8_engine_path = self.output_dir / "mae0222_model_int8.trt"
            
            if self.create_tensorrt_engine(str(int8_engine_path), "int8"):
                int8_benchmark = self.benchmark_tensorrt(str(int8_engine_path))
                results["tensorrt_results"]["int8"] = int8_benchmark
                results["int8_engine"] = str(int8_engine_path)
        
        # 결과 저장
        results_path = self.output_dir / "jetson_tensorrt_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Jetson 양자화 결과 저장: {results_path}")
        
        # 결과 출력
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """양자화 결과 출력"""
        print("\n" + "="*60)
        print("🤖 Jetson Orin NX TensorRT 양자화 결과 (MAE 0.222)")
        print("="*60)
        
        model_info = results.get("model_info", {})
        print(f"\n📊 모델 정보:")
        print(f"   MAE: {model_info.get('mae', 0.222)}")
        print(f"   모델 타입: {model_info.get('model_type', 'Simple CNN + RNN')}")
        print(f"   액션 차원: {model_info.get('action_dim', 2)}")
        print(f"   타겟 디바이스: {model_info.get('target_device', 'Jetson Orin NX')}")
        
        tensorrt_results = results.get("tensorrt_results", {})
        
        if "fp16" in tensorrt_results:
            fp16 = tensorrt_results["fp16"]
            if "error" not in fp16:
                print(f"\n📊 TensorRT FP16:")
                print(f"   추론 시간: {fp16.get('avg_inference_time_ms', 0):.2f} ms")
                print(f"   메모리 사용량: {fp16.get('memory_used_mb', 0):.2f} MB")
                print(f"   처리량: {fp16.get('throughput_fps', 0):.2f} FPS")
            else:
                print(f"\n❌ TensorRT FP16 오류: {fp16['error']}")
        
        if "int8" in tensorrt_results:
            int8 = tensorrt_results["int8"]
            if "error" not in int8:
                print(f"\n📊 TensorRT INT8:")
                print(f"   추론 시간: {int8.get('avg_inference_time_ms', 0):.2f} ms")
                print(f"   메모리 사용량: {int8.get('memory_used_mb', 0):.2f} MB")
                print(f"   처리량: {int8.get('throughput_fps', 0):.2f} FPS")
            else:
                print(f"\n❌ TensorRT INT8 오류: {int8['error']}")
        
        print("\n" + "="*60)

def main():
    """메인 함수"""
    print("🚀 Jetson Orin NX TensorRT 양자화 시작")
    
    # ONNX 모델 경로 설정
    onnx_path = "quantized_models_cpu/mae0222_model_cpu.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 모델 파일을 찾을 수 없습니다: {onnx_path}")
        print("먼저 CPU 양자화를 실행하여 ONNX 모델을 생성하세요.")
        return
    
    # Jetson 양자화 실행
    quantizer = JetsonTensorRTQuantizer(onnx_path)
    results = quantizer.quantize_for_jetson()
    
    print("\n✅ Jetson 양자화 완료!")

if __name__ == "__main__":
    main()
