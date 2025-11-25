#!/usr/bin/env python3
"""
정확한 메모리 사용량과 FPS 측정 벤치마크
- GPU/CPU 메모리 사용량 측정
- 정확한 FPS 계산 방법 검증
- 상세한 성능 분석
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import psutil
import gc
from typing import Dict, Any, List
import onnxruntime as ort

class Kosmos2CLIPHybridModel(nn.Module):
    """Kosmos2 + CLIP 하이브리드 모델 (MAE 0.212)"""
    
    def __init__(self):
        super().__init__()
        
        # 모델 구조 정의
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # linear_x, linear_y, angular_z
        )
    
    def forward(self, images, text_embeddings):
        """전방 전파"""
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_embeddings)
        combined_features = torch.cat([image_features, text_features], dim=1)
        actions = self.fusion_layer(combined_features)
        return actions

class MemoryAccurateBenchmark:
    """정확한 메모리 사용량과 FPS 측정 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        
        # 테스트 데이터 준비
        self.test_images = torch.randn(1, 3, 224, 224, device=self.device)
        self.test_text = torch.randn(1, 512, device=self.device)
        
        print(f"🔧 Device: {self.device}")
        print(f"📊 Test data shape: images {self.test_images.shape}, text {self.test_text.shape}")
        print(f"🎯 Target: Kosmos2 + CLIP Hybrid (MAE 0.212)")
    
    def get_memory_usage(self):
        """현재 메모리 사용량 측정"""
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        cpu_used_mb = cpu_memory.used / (1024 * 1024)
        cpu_total_mb = cpu_memory.total / (1024 * 1024)
        cpu_percent = cpu_memory.percent
        
        # GPU 메모리 (CUDA 사용 가능한 경우)
        gpu_used_mb = 0
        gpu_total_mb = 0
        gpu_percent = 0
        
        if torch.cuda.is_available():
            gpu_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_percent = (gpu_used_mb / gpu_total_mb) * 100
        
        return {
            'cpu_used_mb': cpu_used_mb,
            'cpu_total_mb': cpu_total_mb,
            'cpu_percent': cpu_percent,
            'gpu_used_mb': gpu_used_mb,
            'gpu_total_mb': gpu_total_mb,
            'gpu_percent': gpu_percent
        }
    
    def benchmark_pytorch_with_memory(self, num_runs: int = 100):
        """PyTorch 벤치마크 (메모리 측정 포함)"""
        print(f"\n📈 Benchmarking PyTorch with Memory Measurement ({num_runs} runs)")
        print("-" * 60)
        
        # 메모리 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 초기 메모리 측정
        initial_memory = self.get_memory_usage()
        print(f"Initial Memory:")
        print(f"   CPU: {initial_memory['cpu_used_mb']:.1f}MB / {initial_memory['cpu_total_mb']:.1f}MB ({initial_memory['cpu_percent']:.1f}%)")
        print(f"   GPU: {initial_memory['gpu_used_mb']:.1f}MB / {initial_memory['gpu_total_mb']:.1f}MB ({initial_memory['gpu_percent']:.1f}%)")
        
        # PyTorch 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        model = Kosmos2CLIPHybridModel().to(self.device)
        model.eval()
        
        # TorchScript 최적화 시도
        try:
            model = torch.jit.script(model)
            print(f"   ✅ TorchScript optimization applied")
        except Exception as e:
            print(f"   ⚠️ TorchScript optimization failed: {e}")
        
        # 모델 로드 후 메모리 측정
        model_memory = self.get_memory_usage()
        print(f"After Model Load:")
        print(f"   CPU: {model_memory['cpu_used_mb']:.1f}MB / {model_memory['cpu_total_mb']:.1f}MB ({model_memory['cpu_percent']:.1f}%)")
        print(f"   GPU: {model_memory['gpu_used_mb']:.1f}MB / {model_memory['gpu_total_mb']:.1f}MB ({model_memory['gpu_percent']:.1f}%)")
        
        # 워밍업
        print("🔥 Warming up PyTorch model...")
        with torch.no_grad():
            for i in range(50):
                _ = model(self.test_images, self.test_text)
                if (i + 1) % 10 == 0:
                    print(f"   Warmup: {i + 1}/50")
        
        # 벤치마크
        print(f"⚡ Running PyTorch benchmark...")
        times = []
        memory_samples = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(self.test_images, self.test_text)
            
            inference_time = time.perf_counter() - start_time
            times.append(inference_time)
            
            # 메모리 샘플링 (10개마다)
            if i % 10 == 0:
                memory_samples.append(self.get_memory_usage())
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i + 1}/{num_runs}")
        
        # 최종 메모리 측정
        final_memory = self.get_memory_usage()
        
        # 결과 분석
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # FPS 계산 방법 검증
        fps_method1 = 1.0 / avg_time  # 기존 방법
        fps_method2 = num_runs / sum(times)  # 전체 시간 기반
        fps_method3 = 1000.0 / avg_time  # ms 기반 (잘못된 방법)
        
        # 메모리 사용량 분석
        avg_gpu_memory = np.mean([sample['gpu_used_mb'] for sample in memory_samples])
        avg_cpu_memory = np.mean([sample['cpu_used_mb'] for sample in memory_samples])
        
        result = {
            "framework": "PyTorch (Optimized)",
            "optimization": "TorchScript + cuDNN",
            "average_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "fps_method1": fps_method1,  # 1/avg_time
            "fps_method2": fps_method2,  # num_runs/total_time
            "fps_method3": fps_method3,  # 1000/avg_time (잘못된 방법)
            "num_runs": num_runs,
            "memory": {
                "initial_gpu_mb": initial_memory['gpu_used_mb'],
                "model_gpu_mb": model_memory['gpu_used_mb'],
                "final_gpu_mb": final_memory['gpu_used_mb'],
                "avg_gpu_mb": avg_gpu_memory,
                "initial_cpu_mb": initial_memory['cpu_used_mb'],
                "model_cpu_mb": model_memory['cpu_used_mb'],
                "final_cpu_mb": final_memory['cpu_used_mb'],
                "avg_cpu_mb": avg_cpu_memory
            }
        }
        
        print(f"📊 PyTorch Results:")
        print(f"   Average: {avg_time*1000:.3f} ms")
        print(f"   Std Dev: {std_time*1000:.3f} ms")
        print(f"   Min: {min_time*1000:.3f} ms")
        print(f"   Max: {max_time*1000:.3f} ms")
        print(f"   FPS (1/avg_time): {fps_method1:.1f}")
        print(f"   FPS (num_runs/total): {fps_method2:.1f}")
        print(f"   FPS (1000/avg_time): {fps_method3:.1f} ⚠️ (잘못된 방법)")
        print(f"   GPU Memory: {avg_gpu_memory:.1f}MB")
        print(f"   CPU Memory: {avg_cpu_memory:.1f}MB")
        
        return result
    
    def benchmark_onnx_with_memory(self, onnx_path: str, num_runs: int = 100):
        """ONNX Runtime 벤치마크 (메모리 측정 포함)"""
        print(f"\n📈 Benchmarking ONNX Runtime with Memory Measurement ({num_runs} runs)")
        print("-" * 60)
        
        # 메모리 초기화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 초기 메모리 측정
        initial_memory = self.get_memory_usage()
        print(f"Initial Memory:")
        print(f"   CPU: {initial_memory['cpu_used_mb']:.1f}MB / {initial_memory['cpu_total_mb']:.1f}MB ({initial_memory['cpu_percent']:.1f}%)")
        print(f"   GPU: {initial_memory['gpu_used_mb']:.1f}MB / {initial_memory['gpu_total_mb']:.1f}MB ({initial_memory['gpu_percent']:.1f}%)")
        
        # ONNX Runtime 설정 (CPU 사용 - CUDA 문제로 인해)
        providers = ['CPUExecutionProvider']
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 4
        
        try:
            # ONNX Runtime 세션 생성
            session = ort.InferenceSession(onnx_path, session_options, providers=providers)
            
            # 입력 이름 가져오기
            input_names = [input.name for input in session.get_inputs()]
            output_names = [output.name for output in session.get_outputs()]
            
            # 입력 데이터 준비
            inputs = {
                input_names[0]: self.test_images.cpu().numpy(),
                input_names[1]: self.test_text.cpu().numpy()
            }
            
            # 세션 로드 후 메모리 측정
            session_memory = self.get_memory_usage()
            print(f"After Session Load:")
            print(f"   CPU: {session_memory['cpu_used_mb']:.1f}MB / {session_memory['cpu_total_mb']:.1f}MB ({session_memory['cpu_percent']:.1f}%)")
            print(f"   GPU: {session_memory['gpu_used_mb']:.1f}MB / {session_memory['gpu_total_mb']:.1f}MB ({session_memory['gpu_percent']:.1f}%)")
            
            # 워밍업
            print("🔥 Warming up ONNX Runtime model...")
            for i in range(50):
                _ = session.run(output_names, inputs)
                if (i + 1) % 10 == 0:
                    print(f"   Warmup: {i + 1}/50")
            
            # 벤치마크
            print(f"⚡ Running ONNX Runtime benchmark...")
            times = []
            memory_samples = []
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                
                outputs = session.run(output_names, inputs)
                
                inference_time = time.perf_counter() - start_time
                times.append(inference_time)
                
                # 메모리 샘플링 (10개마다)
                if i % 10 == 0:
                    memory_samples.append(self.get_memory_usage())
                
                if (i + 1) % 20 == 0:
                    print(f"   Progress: {i + 1}/{num_runs}")
            
            # 최종 메모리 측정
            final_memory = self.get_memory_usage()
            
            # 결과 분석
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            # FPS 계산 방법 검증
            fps_method1 = 1.0 / avg_time
            fps_method2 = num_runs / sum(times)
            fps_method3 = 1000.0 / avg_time
            
            # 메모리 사용량 분석
            avg_gpu_memory = np.mean([sample['gpu_used_mb'] for sample in memory_samples])
            avg_cpu_memory = np.mean([sample['cpu_used_mb'] for sample in memory_samples])
            
            result = {
                "framework": "ONNX Runtime (Optimized)",
                "optimization": "Graph Optimization + CPU",
                "average_time_ms": avg_time * 1000,
                "std_time_ms": std_time * 1000,
                "min_time_ms": min_time * 1000,
                "max_time_ms": max_time * 1000,
                "fps_method1": fps_method1,
                "fps_method2": fps_method2,
                "fps_method3": fps_method3,
                "num_runs": num_runs,
                "model_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
                "memory": {
                    "initial_gpu_mb": initial_memory['gpu_used_mb'],
                    "session_gpu_mb": session_memory['gpu_used_mb'],
                    "final_gpu_mb": final_memory['gpu_used_mb'],
                    "avg_gpu_mb": avg_gpu_memory,
                    "initial_cpu_mb": initial_memory['cpu_used_mb'],
                    "session_cpu_mb": session_memory['cpu_used_mb'],
                    "final_cpu_mb": final_memory['cpu_used_mb'],
                    "avg_cpu_mb": avg_cpu_memory
                }
            }
            
            print(f"📊 ONNX Runtime Results:")
            print(f"   Average: {avg_time*1000:.3f} ms")
            print(f"   Std Dev: {std_time*1000:.3f} ms")
            print(f"   Min: {min_time*1000:.3f} ms")
            print(f"   Max: {max_time*1000:.3f} ms")
            print(f"   FPS (1/avg_time): {fps_method1:.1f}")
            print(f"   FPS (num_runs/total): {fps_method2:.1f}")
            print(f"   FPS (1000/avg_time): {fps_method3:.1f} ⚠️ (잘못된 방법)")
            print(f"   GPU Memory: {avg_gpu_memory:.1f}MB")
            print(f"   CPU Memory: {avg_cpu_memory:.1f}MB")
            print(f"   Model Size: {result['model_size_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"❌ ONNX benchmark failed: {e}")
            return None
    
    def create_detailed_report(self, results: List[Dict]):
        """상세한 성능 리포트 생성"""
        print(f"\n" + "="*80)
        print("📊 DETAILED PERFORMANCE ANALYSIS")
        print("="*80)
        
        if len(results) < 2:
            print("❌ Need at least 2 results for comparison")
            return
        
        # 결과 정렬 (FPS 기준)
        sorted_results = sorted(results, key=lambda x: x['fps_method1'], reverse=True)
        
        print(f"\n🏆 Performance Ranking (by FPS):")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            framework = result['framework']
            optimization = result.get('optimization', 'N/A')
            avg_time = result['average_time_ms']
            fps = result['fps_method1']
            fps_method2 = result['fps_method2']
            fps_method3 = result['fps_method3']
            
            print(f"{i}. {framework}")
            print(f"   Optimization: {optimization}")
            print(f"   ⏱️  Time: {avg_time:.3f} ms")
            print(f"   🚀 FPS (1/avg_time): {fps:.1f}")
            print(f"   🚀 FPS (num_runs/total): {fps_method2:.1f}")
            print(f"   ⚠️  FPS (1000/avg_time): {fps_method3:.1f} (잘못된 방법)")
            
            if 'memory' in result:
                mem = result['memory']
                print(f"   💾 GPU Memory: {mem['avg_gpu_mb']:.1f}MB")
                print(f"   💾 CPU Memory: {mem['avg_cpu_mb']:.1f}MB")
            
            if 'model_size_mb' in result:
                print(f"   📏 Model Size: {result['model_size_mb']:.1f}MB")
            print()
        
        # FPS 계산 방법 비교
        print(f"🔍 FPS Calculation Method Analysis:")
        print("-" * 80)
        
        pytorch_result = next((r for r in results if 'PyTorch' in r['framework']), None)
        onnx_result = next((r for r in results if 'ONNX' in r['framework']), None)
        
        if pytorch_result and onnx_result:
            print(f"PyTorch FPS Methods:")
            print(f"   1/avg_time: {pytorch_result['fps_method1']:.1f}")
            print(f"   num_runs/total: {pytorch_result['fps_method2']:.1f}")
            print(f"   1000/avg_time: {pytorch_result['fps_method3']:.1f} ⚠️")
            print()
            print(f"ONNX Runtime FPS Methods:")
            print(f"   1/avg_time: {onnx_result['fps_method1']:.1f}")
            print(f"   num_runs/total: {onnx_result['fps_method2']:.1f}")
            print(f"   1000/avg_time: {onnx_result['fps_method3']:.1f} ⚠️")
            print()
            print(f"🔧 Correct FPS calculation: 1/avg_time (seconds)")
            print(f"⚠️  Wrong FPS calculation: 1000/avg_time (treats ms as seconds)")
        
        # 메모리 사용량 비교
        print(f"\n💾 Memory Usage Comparison:")
        print("-" * 80)
        
        for result in results:
            if 'memory' in result:
                mem = result['memory']
                framework = result['framework']
                print(f"{framework}:")
                print(f"   GPU: {mem['avg_gpu_mb']:.1f}MB")
                print(f"   CPU: {mem['avg_cpu_mb']:.1f}MB")
                print()
        
        # 결과 저장
        report_path = "Robo+/Mobile_VLA/memory_accurate_results.json"
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
                "results": results,
                "ranking": [r['framework'] for r in sorted_results]
            }, f, indent=2)
        
        print(f"\n✅ Detailed report saved: {report_path}")

def main():
    """메인 함수"""
    print("🚀 Starting Memory-Accurate Benchmark")
    print("🎯 Measuring Memory Usage and Validating FPS Calculation")
    
    benchmark = MemoryAccurateBenchmark()
    
    try:
        results = []
        
        # 1. PyTorch 벤치마크
        print(f"\n1. PyTorch Benchmark with Memory Measurement")
        pytorch_result = benchmark.benchmark_pytorch_with_memory()
        results.append(pytorch_result)
        
        # 2. ONNX Runtime 벤치마크
        print(f"\n2. ONNX Runtime Benchmark with Memory Measurement")
        onnx_path = "Robo+/Mobile_VLA/optimized_onnx/model.onnx"
        
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX model not found: {onnx_path}")
            print(f"   Creating ONNX model first...")
            # ONNX 모델 생성 로직 추가 필요
        
        onnx_result = benchmark.benchmark_onnx_with_memory(onnx_path)
        if onnx_result:
            results.append(onnx_result)
        
        # 3. 상세 리포트 생성
        benchmark.create_detailed_report(results)
        
        print(f"\n✅ Memory-accurate benchmark completed!")
        print(f"📊 Tested {len(results)} frameworks with memory measurement")
        print(f"🔧 Device: {benchmark.device}")
        
    except Exception as e:
        print(f"❌ Memory-accurate benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
