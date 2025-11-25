#!/usr/bin/env python3
"""
Mobile VLA 모델 TensorRT 변환기 (고급)
- 실제 모델 구조 분석 및 TensorRT 변환
- Torch2TRT 사용으로 간편한 변환
- 다양한 양자화 옵션 지원
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
import numpy as np
import os
import json
import time
from typing import Dict, Any, Optional
from PIL import Image
import cv2

# Torch2TRT import (설치 필요: pip install torch2trt)
try:
    from torch2trt import torch2trt, TRTModule
    TORCH2TRT_AVAILABLE = True
except ImportError:
    print("Warning: torch2trt not available. Install with: pip install torch2trt")
    TORCH2TRT_AVAILABLE = False

class MobileVLAModelWrapper(nn.Module):
    """Mobile VLA 모델을 TensorRT 변환용으로 래핑"""
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        
    def forward(self, images, text_embeddings):
        """전방 전파 (TensorRT 변환용)"""
        # 모델의 forward 함수를 직접 호출
        outputs = self.model(
            pixel_values=images,
            input_ids=text_embeddings
        )
        
        # 액션 로짓 반환
        return outputs.action_logits

class MobileVLATensorRTConverterAdvanced:
    """고급 TensorRT 변환기"""
    
    def __init__(self, model_name: str = "minium/mobile-vla-omniwheel"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 출력 디렉토리
        self.output_dir = "Robo+/Mobile_VLA/tensorrt_quantized"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 모델 로드
        self.load_original_model()
        
    def load_original_model(self):
        """원본 모델 로드"""
        print(f"🔄 Loading original model: {self.model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if self.device.type == 'cuda':
                self.model = self.model.cuda()
            
            self.model.eval()
            print("✅ Original model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            raise
    
    def create_model_wrapper(self):
        """모델 래퍼 생성"""
        return MobileVLAModelWrapper(self.model, self.processor)
    
    def prepare_sample_inputs(self, batch_size: int = 1):
        """샘플 입력 데이터 준비"""
        # 테스트 이미지 생성
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image)
        
        # 프로세서로 전처리
        inputs = self.processor(
            images=pil_image,
            text="Navigate around obstacles to track the target cup",
            return_tensors="pt"
        )
        
        # GPU로 이동
        if self.device.type == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        return inputs['pixel_values'], inputs['input_ids']
    
    def convert_to_tensorrt_torch2trt(self, precision: str = "fp16"):
        """Torch2TRT를 사용한 TensorRT 변환"""
        if not TORCH2TRT_AVAILABLE:
            print("❌ Torch2TRT not available")
            return None
        
        print(f"🔨 Converting to TensorRT using Torch2TRT (precision: {precision})")
        
        # 모델 래퍼 생성
        model_wrapper = self.create_model_wrapper()
        model_wrapper.eval()
        
        # 샘플 입력 준비
        sample_images, sample_text = self.prepare_sample_inputs()
        
        # TensorRT 변환 설정
        fp16_mode = precision == "fp16"
        max_workspace_size = 1 << 30  # 1GB
        
        # 변환 실행
        model_trt = torch2trt(
            model_wrapper,
            [sample_images, sample_text],
            fp16_mode=fp16_mode,
            max_workspace_size=max_workspace_size,
            use_onnx=True  # ONNX를 통한 변환
        )
        
        # 변환된 모델 저장
        model_path = os.path.join(self.output_dir, f"mobile_vla_torch2trt_{precision}.pth")
        torch.save(model_trt.state_dict(), model_path)
        
        print(f"✅ Torch2TRT model saved: {model_path}")
        return model_trt, model_path
    
    def test_torch2trt_inference(self, model_trt, num_runs: int = 10):
        """Torch2TRT 추론 테스트"""
        print("🧪 Testing Torch2TRT inference")
        
        model_trt.eval()
        
        # 테스트 데이터 준비
        test_images, test_text = self.prepare_sample_inputs()
        
        # 워밍업
        with torch.no_grad():
            for _ in range(5):
                _ = model_trt(test_images, test_text)
        
        # 벤치마크
        times = []
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model_trt(test_images, test_text)
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{num_runs}")
        
        # 결과 분석
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"🎯 Torch2TRT inference: {avg_time*1000:.2f} ms ({fps:.1f} FPS)")
        print(f"📊 Action output shape: {outputs.shape}")
        
        return avg_time, fps, outputs
    
    def convert_to_onnx_then_tensorrt(self, precision: str = "fp16"):
        """ONNX를 통한 TensorRT 변환"""
        print(f"🔨 Converting to ONNX then TensorRT (precision: {precision})")
        
        # 모델 래퍼 생성
        model_wrapper = self.create_model_wrapper()
        model_wrapper.eval()
        
        # 샘플 입력 준비
        sample_images, sample_text = self.prepare_sample_inputs()
        
        # ONNX 모델 저장
        onnx_path = os.path.join(self.output_dir, "mobile_vla_model.onnx")
        
        torch.onnx.export(
            model_wrapper,
            (sample_images, sample_text),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images', 'text_embeddings'],
            output_names=['action_logits'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'text_embeddings': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved: {onnx_path}")
        
        # ONNX 모델을 TensorRT로 변환 (trtexec 사용)
        engine_path = self.convert_onnx_to_tensorrt(onnx_path, precision)
        
        return engine_path
    
    def convert_onnx_to_tensorrt(self, onnx_path: str, precision: str = "fp16"):
        """ONNX 모델을 TensorRT 엔진으로 변환"""
        print(f"🔨 Converting ONNX to TensorRT engine (precision: {precision})")
        
        # trtexec 명령어 구성
        engine_path = os.path.join(self.output_dir, f"mobile_vla_{precision}.engine")
        
        # 기본 trtexec 명령어
        cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
        
        # 정밀도 설정
        if precision == "fp16":
            cmd += " --fp16"
        elif precision == "int8":
            cmd += " --int8"
        
        # 추가 옵션
        cmd += " --workspace=1024 --verbose"
        
        print(f"Running command: {cmd}")
        
        # 명령어 실행
        import subprocess
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ TensorRT engine created: {engine_path}")
                return engine_path
            else:
                print(f"❌ TensorRT conversion failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Error running trtexec: {e}")
            return None
    
    def benchmark_original_vs_tensorrt(self, model_trt, num_runs: int = 50):
        """원본 모델 vs TensorRT 성능 비교"""
        print("📈 Benchmarking Original vs TensorRT")
        
        # 원본 모델 준비
        model_wrapper = self.create_model_wrapper()
        model_wrapper.eval()
        
        # 테스트 데이터 준비
        test_images, test_text = self.prepare_sample_inputs()
        
        # 원본 모델 벤치마크
        print("Testing original model...")
        original_times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                outputs_orig = model_wrapper(test_images, test_text)
                inference_time = time.time() - start_time
                original_times.append(inference_time)
        
        # TensorRT 모델 벤치마크
        print("Testing TensorRT model...")
        trt_times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                outputs_trt = model_trt(test_images, test_text)
                inference_time = time.time() - start_time
                trt_times.append(inference_time)
        
        # 결과 분석
        orig_avg = np.mean(original_times)
        trt_avg = np.mean(trt_times)
        speedup = orig_avg / trt_avg
        
        results = {
            "original": {
                "average_time_ms": orig_avg * 1000,
                "fps": 1.0 / orig_avg
            },
            "tensorrt": {
                "average_time_ms": trt_avg * 1000,
                "fps": 1.0 / trt_avg
            },
            "speedup": speedup
        }
        
        print(f"📊 Benchmark Results:")
        print(f"  Original: {orig_avg*1000:.2f} ms ({1.0/orig_avg:.1f} FPS)")
        print(f"  TensorRT: {trt_avg*1000:.2f} ms ({1.0/trt_avg:.1f} FPS)")
        print(f"  Speedup: {speedup:.2f}x")
        
        # 결과 저장
        results_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Benchmark results saved: {results_path}")
        return results
    
    def create_tensorrt_inference_node(self, engine_path: str):
        """TensorRT 추론 노드 생성"""
        print("🔧 Creating TensorRT inference node")
        
        node_code = f'''#!/usr/bin/env python3
"""
TensorRT 추론 노드 (자동 생성)
- {engine_path} 엔진 사용
- 고성능 실시간 추론
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
import numpy as np
from PIL import Image as PILImage
import json
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInferenceNode(Node):
    def __init__(self):
        super().__init__('tensorrt_inference_node')
        
        # TensorRT 엔진 로드
        self.engine_path = "{engine_path}"
        self.load_tensorrt_engine()
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.inference_count = 0
        self.last_inference_time = 0.0
        
    def load_tensorrt_engine(self):
        """TensorRT 엔진 로드"""
        try:
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            
            # 메모리 할당
            self.input_images = cuda.mem_alloc(1 * 3 * 224 * 224 * 4)  # float32
            self.input_text = cuda.mem_alloc(1 * 512 * 4)  # float32
            self.output = cuda.mem_alloc(1 * 3 * 4)  # float32
            
            self.get_logger().info("✅ TensorRT engine loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load TensorRT engine: {{e}}")
            raise
    
    def setup_ros(self):
        """ROS 설정"""
        # 이미지 서브스크라이버
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )
        
        # 액션 퍼블리셔
        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 결과 퍼블리셔
        self.result_pub = self.create_publisher(
            String,
            '/tensorrt/inference_result',
            10
        )
        
    def image_callback(self, msg):
        """이미지 콜백"""
        try:
            # 이미지 처리
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환 및 리사이즈
            pil_image = PILImage.fromarray(image_rgb)
            pil_image = pil_image.resize((224, 224))
            
            # numpy 배열로 변환
            image_array = np.array(pil_image, dtype=np.float32) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
            image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가
            
            # 텍스트 임베딩 (간단한 더미 데이터)
            text_embedding = np.random.randn(1, 512).astype(np.float32)
            
            # TensorRT 추론
            action = self.run_tensorrt_inference(image_array, text_embedding)
            
            # 액션 실행
            self.execute_action(action)
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in image callback: {{e}}")
    
    def run_tensorrt_inference(self, image_array, text_embedding):
        """TensorRT 추론 실행"""
        try:
            # GPU 메모리로 데이터 복사
            cuda.memcpy_htod(self.input_images, image_array)
            cuda.memcpy_htod(self.input_text, text_embedding)
            
            # 추론 실행
            start_time = time.time()
            self.context.execute_v2(bindings=[
                int(self.input_images),
                int(self.input_text),
                int(self.output)
            ])
            inference_time = time.time() - start_time
            
            # 결과 가져오기
            result = np.empty((1, 3), dtype=np.float32)
            cuda.memcpy_dtoh(result, self.output)
            
            self.inference_count += 1
            self.last_inference_time = inference_time
            
            # 결과 발행
            self.publish_result(result[0], inference_time)
            
            return result[0]
            
        except Exception as e:
            self.get_logger().error(f"❌ TensorRT inference error: {{e}}")
            return np.array([0.0, 0.0, 0.0])
    
    def execute_action(self, action):
        """액션 실행"""
        try:
            twist = Twist()
            twist.linear.x = float(action[0])
            twist.linear.y = float(action[1])
            twist.angular.z = float(action[2])
            
            self.action_pub.publish(twist)
            
        except Exception as e:
            self.get_logger().error(f"❌ Error executing action: {{e}}")
    
    def publish_result(self, action, inference_time):
        """결과 발행"""
        try:
            result = {{
                "timestamp": time.time(),
                "inference_time": inference_time,
                "action": action.tolist(),
                "inference_count": self.inference_count,
                "engine": "{engine_path}"
            }}
            
            msg = String()
            msg.data = json.dumps(result)
            self.result_pub.publish(msg)
            
            self.get_logger().info(f"🎯 TensorRT Inference #{{self.inference_count}}: {{inference_time*1000:.2f}}ms")
            
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing result: {{e}}")

def main(args=None):
    rclpy.init(args=args)
    node = TensorRTInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
'''
        
        # 노드 파일 저장
        node_path = os.path.join(self.output_dir, "tensorrt_inference_node.py")
        with open(node_path, "w") as f:
            f.write(node_code)
        
        print(f"✅ TensorRT inference node created: {node_path}")
        return node_path

def main():
    """메인 함수"""
    print("🚀 Starting Advanced Mobile VLA TensorRT Conversion")
    
    # 변환기 초기화
    converter = MobileVLATensorRTConverterAdvanced()
    
    try:
        # Torch2TRT 변환
        if TORCH2TRT_AVAILABLE:
            print("\n🔨 Converting with Torch2TRT...")
            model_trt, model_path = converter.convert_to_tensorrt_torch2trt(precision="fp16")
            
            if model_trt is not None:
                # 성능 테스트
                print("\n🧪 Testing Torch2TRT performance...")
                converter.test_torch2trt_inference(model_trt)
                
                # 벤치마크
                print("\n📈 Running benchmark...")
                converter.benchmark_original_vs_tensorrt(model_trt)
        
        # ONNX -> TensorRT 변환
        print("\n🔨 Converting with ONNX -> TensorRT...")
        engine_path = converter.convert_to_onnx_then_tensorrt(precision="fp16")
        
        if engine_path:
            # TensorRT 추론 노드 생성
            print("\n🔧 Creating TensorRT inference node...")
            converter.create_tensorrt_inference_node(engine_path)
        
        print("\n✅ Advanced TensorRT conversion completed!")
        
    except Exception as e:
        print(f"❌ Advanced TensorRT conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()
