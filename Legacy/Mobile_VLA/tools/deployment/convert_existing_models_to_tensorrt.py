#!/usr/bin/env python3
"""
기존 양자화된 모델들을 TensorRT로 변환
- 이미 있는 ONNX 모델들을 TensorRT 엔진으로 변환
- FP16/INT8 양자화 지원
"""

import os
import subprocess
import json
import time
from typing import Dict, Any

class ExistingModelsTensorRTConverter:
    """기존 모델들을 TensorRT로 변환하는 클래스"""
    
    def __init__(self):
        self.base_dir = "Robo+/Mobile_VLA"
        self.output_dir = "Robo+/Mobile_VLA/tensorrt_engines"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 기존 ONNX 모델들
        self.existing_models = {
            'accurate_gpu': 'accurate_gpu_quantized/accurate_gpu_model.onnx',
            'simple_gpu': 'simple_gpu_quantized/simple_gpu_model.onnx',
            'cpu_mae0222': 'quantized_models_cpu/mae0222_model_cpu.onnx'
        }
        
    def check_trtexec_availability(self):
        """trtexec 사용 가능 여부 확인"""
        try:
            result = subprocess.run(['trtexec', '--help'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ trtexec is available")
                return True
            else:
                print("❌ trtexec is not working properly")
                return False
        except FileNotFoundError:
            print("❌ trtexec not found. Please install TensorRT.")
            return False
    
    def convert_onnx_to_tensorrt(self, model_name: str, onnx_path: str, precision: str = "fp16"):
        """ONNX 모델을 TensorRT 엔진으로 변환"""
        print(f"🔨 Converting {model_name} to TensorRT {precision.upper()}")
        
        # 출력 경로
        engine_path = os.path.join(self.output_dir, f"{model_name}_{precision}.engine")
        
        # trtexec 명령어 구성
        cmd = [
            'trtexec',
            '--onnx=' + onnx_path,
            '--saveEngine=' + engine_path,
            '--workspace=1024',
            '--verbose'
        ]
        
        # 정밀도 설정
        if precision == "fp16":
            cmd.append('--fp16')
        elif precision == "int8":
            cmd.append('--int8')
        
        # 동적 배치 크기 설정 (필요시)
        cmd.extend([
            '--minShapes=pixel_values:1x3x224x224',
            '--optShapes=pixel_values:1x3x224x224',
            '--maxShapes=pixel_values:4x3x224x224'
        ])
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ TensorRT engine created: {engine_path}")
                
                # 엔진 파일 크기 확인
                if os.path.exists(engine_path):
                    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                    print(f"📊 Engine size: {size_mb:.1f} MB")
                
                return engine_path
            else:
                print(f"❌ TensorRT conversion failed for {model_name}")
                print(f"Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error converting {model_name}: {e}")
            return None
    
    def convert_all_models(self):
        """모든 기존 모델을 TensorRT로 변환"""
        print("🚀 Starting conversion of existing models to TensorRT")
        
        # trtexec 확인
        if not self.check_trtexec_availability():
            print("⚠️ Skipping TensorRT conversion due to missing trtexec")
            return
        
        results = {}
        
        for model_name, onnx_relative_path in self.existing_models.items():
            onnx_path = os.path.join(self.base_dir, onnx_relative_path)
            
            if not os.path.exists(onnx_path):
                print(f"⚠️ ONNX model not found: {onnx_path}")
                continue
            
            print(f"\n📁 Processing {model_name}: {onnx_path}")
            
            # ONNX 파일 크기 확인
            onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"📊 ONNX size: {onnx_size_mb:.1f} MB")
            
            model_results = {}
            
            # FP16 변환
            fp16_engine = self.convert_onnx_to_tensorrt(model_name, onnx_path, "fp16")
            if fp16_engine:
                model_results['fp16'] = fp16_engine
            
            # INT8 변환 (선택적)
            try:
                int8_engine = self.convert_onnx_to_tensorrt(model_name, onnx_path, "int8")
                if int8_engine:
                    model_results['int8'] = int8_engine
            except Exception as e:
                print(f"⚠️ INT8 conversion failed for {model_name}: {e}")
            
            results[model_name] = model_results
        
        return results
    
    def create_benchmark_script(self, results: Dict[str, Any]):
        """벤치마크 스크립트 생성"""
        print("🔧 Creating benchmark script")
        
        benchmark_script = '''#!/usr/bin/env python3
"""
TensorRT 엔진 벤치마크 스크립트
- 생성된 TensorRT 엔진들의 성능 측정
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import json
import os

def benchmark_engine(engine_path: str, num_runs: int = 100):
    """TensorRT 엔진 벤치마크"""
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return None
    
    try:
        # 엔진 로드
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # 메모리 할당
        input_size = 1 * 3 * 224 * 224 * 4  # float32
        output_size = 1 * 3 * 4  # float32
        
        input_mem = cuda.mem_alloc(input_size)
        output_mem = cuda.mem_alloc(output_size)
        
        # 테스트 데이터
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # 워밍업
        for _ in range(10):
            cuda.memcpy_htod(input_mem, test_input)
            context.execute_v2(bindings=[int(input_mem), int(output_mem)])
        
        # 벤치마크
        times = []
        for i in range(num_runs):
            start_time = time.time()
            
            cuda.memcpy_htod(input_mem, test_input)
            context.execute_v2(bindings=[int(input_mem), int(output_mem)])
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_runs}")
        
        # 결과 분석
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        return {
            "average_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "fps": fps,
            "num_runs": num_runs
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed for {engine_path}: {e}")
        return None

def main():
    """메인 함수"""
    print("🚀 Starting TensorRT Engine Benchmark")
    
    # 엔진 디렉토리
    engine_dir = "Robo+/Mobile_VLA/tensorrt_engines"
    
    if not os.path.exists(engine_dir):
        print(f"❌ Engine directory not found: {engine_dir}")
        return
    
    # 모든 엔진 파일 찾기
    engine_files = []
    for file in os.listdir(engine_dir):
        if file.endswith('.engine'):
            engine_files.append(os.path.join(engine_dir, file))
    
    if not engine_files:
        print("❌ No TensorRT engines found")
        return
    
    print(f"📁 Found {len(engine_files)} TensorRT engines")
    
    # 벤치마크 실행
    results = {}
    for engine_path in engine_files:
        engine_name = os.path.basename(engine_path)
        print(f"\n🧪 Benchmarking {engine_name}")
        
        benchmark_result = benchmark_engine(engine_path, num_runs=50)
        if benchmark_result:
            results[engine_name] = benchmark_result
            
            print(f"📊 Results for {engine_name}:")
            print(f"  Average: {benchmark_result['average_time_ms']:.2f} ms")
            print(f"  FPS: {benchmark_result['fps']:.1f}")
    
    # 결과 저장
    if results:
        results_path = os.path.join(engine_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Benchmark results saved: {results_path}")
        
        # 성능 비교
        print("\n📊 Performance Comparison:")
        for engine_name, result in results.items():
            print(f"  {engine_name}: {result['average_time_ms']:.2f} ms ({result['fps']:.1f} FPS)")

if __name__ == "__main__":
    main()
'''
        
        benchmark_path = os.path.join(self.output_dir, "benchmark_engines.py")
        with open(benchmark_path, "w") as f:
            f.write(benchmark_script)
        
        os.chmod(benchmark_path, 0o755)
        print(f"✅ Benchmark script created: {benchmark_path}")
        
        return benchmark_path
    
    def create_usage_guide(self, results: Dict[str, Any]):
        """사용 가이드 생성"""
        print("📖 Creating usage guide")
        
        guide = f"""# 기존 모델 TensorRT 변환 가이드

## 개요
기존에 양자화된 ONNX 모델들을 TensorRT 엔진으로 변환하여 고성능 추론을 수행합니다.

## 변환된 엔진들
"""
        
        for model_name, model_results in results.items():
            guide += f"\n### {model_name}\n"
            for precision, engine_path in model_results.items():
                if os.path.exists(engine_path):
                    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                    guide += f"- **{precision.upper()}**: `{os.path.basename(engine_path)}` ({size_mb:.1f} MB)\n"
        
        guide += f"""
## 사용 방법

### 1. 벤치마크 실행
```bash
cd {self.output_dir}
python benchmark_engines.py
```

### 2. ROS 노드에서 사용
```bash
# FP16 엔진 사용
ros2 run mobile_vla_package tensorrt_inference_node --ros-args \\
    -p engine_path:={self.output_dir}/accurate_gpu_fp16.engine

# INT8 엔진 사용
ros2 run mobile_vla_package tensorrt_inference_node --ros-args \\
    -p engine_path:={self.output_dir}/simple_gpu_int8.engine
```

### 3. 성능 예상
- **FP16**: 2-5x 속도 향상
- **INT8**: 5-10x 속도 향상
- **메모리 사용량**: 50-80% 감소

## 요구사항
- NVIDIA GPU
- TensorRT 8.x
- CUDA 11.x 이상

## 문제 해결
1. TensorRT 설치: `pip install tensorrt`
2. 권한 문제: `chmod +x benchmark_engines.py`
3. 메모리 부족: 워크스페이스 크기 조정
"""
        
        guide_path = os.path.join(self.output_dir, "README.md")
        with open(guide_path, "w") as f:
            f.write(guide)
        
        print(f"✅ Usage guide created: {guide_path}")

def main():
    """메인 함수"""
    print("🚀 Starting Existing Models TensorRT Conversion")
    
    # 변환기 초기화
    converter = ExistingModelsTensorRTConverter()
    
    try:
        # 모든 모델 변환
        results = converter.convert_all_models()
        
        if results:
            # 벤치마크 스크립트 생성
            print("\n🔧 Creating benchmark script...")
            converter.create_benchmark_script(results)
            
            # 사용 가이드 생성
            print("\n📖 Creating usage guide...")
            converter.create_usage_guide(results)
            
            print("\n✅ TensorRT conversion completed!")
            print(f"\n📁 Output directory: {converter.output_dir}")
            print("🔧 Next steps:")
            print("  1. cd Robo+/Mobile_VLA/tensorrt_engines")
            print("  2. python benchmark_engines.py")
            print("  3. Use the generated TensorRT engines in ROS nodes")
        else:
            print("\n⚠️ No models were successfully converted")
        
    except Exception as e:
        print(f"❌ TensorRT conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()
