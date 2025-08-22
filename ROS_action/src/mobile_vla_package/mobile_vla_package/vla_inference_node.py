#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import numpy as np
import cv2
from PIL import Image
import time
from typing import Optional, Dict, Any

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from camera_interfaces.srv import GetImage

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Using mock inference.")
    TRANSFORMERS_AVAILABLE = False

class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_inference_node')
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # 추론 설정
        self.inference_interval = 0.5  # 0.5초마다 추론
        self.last_inference_time = 0.0
        self.confidence_threshold = 0.7
        
        # ROS 설정
        self.setup_ros_components()
        
        # 모델 로드
        self.load_vla_model()
        
        self.get_logger().info(f"🚀 VLA 추론 노드 시작 - Device: {self.device}")
        self.get_logger().info(f"📊 추론 간격: {self.inference_interval}초")
        
    def setup_ros_components(self):
        """ROS 컴포넌트 설정"""
        # 서비스 클라이언트 (카메라에서 이미지 가져오기)
        self.get_image_client = self.create_client(GetImage, 'get_image_service')
        
        # 발행자들
        self.inference_result_pub = self.create_publisher(
            String, 'vla_inference_result', 10
        )
        self.action_command_pub = self.create_publisher(
            Twist, 'vla_action_command', 10
        )
        self.confidence_pub = self.create_publisher(
            Float32MultiArray, 'vla_confidence', 10
        )
        
        # 타이머 (주기적 추론)
        self.inference_timer = self.create_timer(
            self.inference_interval, self.inference_callback
        )
        
        self.get_logger().info("✅ ROS 컴포넌트 설정 완료")
        
    def load_vla_model(self):
        """VLA 모델 로드"""
        if not TRANSFORMERS_AVAILABLE:
            self.get_logger().warn("⚠️ Transformers 없음 - Mock 모드로 실행")
            self.model_loaded = True
            return
            
        try:
            self.get_logger().info("🔄 VLA 모델 로딩 중...")
            
            # 모델 경로 설정 (로컬 또는 HuggingFace)
            model_name = "microsoft/kosmos-2-patch14-224"
            
            # 프로세서와 모델 로드
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            if self.device.type == 'cuda':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            
            self.get_logger().info("✅ VLA 모델 로드 완료!")
            
        except Exception as e:
            self.get_logger().error(f"❌ 모델 로드 실패: {e}")
            self.model_loaded = False
            
    def get_latest_image(self) -> Optional[np.ndarray]:
        """카메라 서비스에서 최신 이미지 가져오기"""
        try:
            if not self.get_image_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("⚠️ 카메라 서비스 연결 실패")
                return None
                
            request = GetImage.Request()
            future = self.get_image_client.call_async(request)
            
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.done():
                response = future.result()
                if response.image.data:
                    cv_image = self.bridge.imgmsg_to_cv2(response.image, "bgr8")
                    return cv_image
                    
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 가져오기 실패: {e}")
            
        return None
        
    def preprocess_image(self, cv_image: np.ndarray) -> Optional[torch.Tensor]:
        """이미지 전처리"""
        try:
            # BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = Image.fromarray(rgb_image)
            
            # 프로세서로 전처리
            if self.processor:
                inputs = self.processor(
                    images=pil_image,
                    return_tensors="pt"
                )
                
                # GPU로 이동
                if self.device.type == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                return inputs
                
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 전처리 실패: {e}")
            
        return None
        
    def run_inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """VLA 모델 추론 실행"""
        if not self.model_loaded or self.model is None:
            return self.mock_inference()
            
        try:
            with torch.no_grad():
                # 추론 실행
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
                # 결과 디코딩
                generated_text = self.processor.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # 신뢰도 계산 (간단한 휴리스틱)
                confidence = self.calculate_confidence(outputs[0])
                
                return {
                    "text": generated_text,
                    "confidence": confidence,
                    "raw_output": outputs[0].cpu().numpy()
                }
                
        except Exception as e:
            self.get_logger().error(f"❌ 추론 실행 실패: {e}")
            return self.mock_inference()
            
    def mock_inference(self) -> Dict[str, Any]:
        """Mock 추론 (모델 없을 때)"""
        return {
            "text": "Mock VLA inference result",
            "confidence": 0.8,
            "raw_output": np.array([1, 2, 3, 4, 5])
        }
        
    def calculate_confidence(self, output_tokens: torch.Tensor) -> float:
        """추론 결과 신뢰도 계산"""
        # 간단한 휴리스틱: 토큰 길이와 특수 토큰 비율
        token_length = len(output_tokens)
        special_tokens = torch.sum(output_tokens == self.processor.tokenizer.eos_token_id)
        
        # 길이가 적당하고 특수 토큰이 적으면 높은 신뢰도
        if 10 <= token_length <= 30 and special_tokens <= 2:
            return 0.9
        elif 5 <= token_length <= 50:
            return 0.7
        else:
            return 0.5
            
    def parse_action_from_text(self, text: str) -> Optional[Dict[str, float]]:
        """텍스트에서 액션 파싱"""
        text_lower = text.lower()
        
        # 간단한 키워드 기반 파싱
        action_mapping = {
            "forward": {"linear_x": 1.0, "linear_y": 0.0, "angular_z": 0.0},
            "backward": {"linear_x": -1.0, "linear_y": 0.0, "angular_z": 0.0},
            "left": {"linear_x": 0.0, "linear_y": 1.0, "angular_z": 0.0},
            "right": {"linear_x": 0.0, "linear_y": -1.0, "angular_z": 0.0},
            "turn left": {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 1.0},
            "turn right": {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -1.0},
            "stop": {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        
        for keyword, action in action_mapping.items():
            if keyword in text_lower:
                return action
                
        return None
        
    def publish_inference_result(self, result: Dict[str, Any]):
        """추론 결과 발행"""
        # 텍스트 결과 발행
        text_msg = String()
        text_msg.data = result["text"]
        self.inference_result_pub.publish(text_msg)
        
        # 신뢰도 발행
        confidence_msg = Float32MultiArray()
        confidence_msg.data = [result["confidence"]]
        self.confidence_pub.publish(confidence_msg)
        
        # 액션 명령 파싱 및 발행
        action = self.parse_action_from_text(result["text"])
        if action and result["confidence"] > self.confidence_threshold:
            twist_msg = Twist()
            twist_msg.linear.x = float(action["linear_x"])
            twist_msg.linear.y = float(action["linear_y"])
            twist_msg.angular.z = float(action["angular_z"])
            self.action_command_pub.publish(twist_msg)
            
            self.get_logger().info(f"🎯 추론: {result['text']} → 액션: {action}")
        else:
            self.get_logger().info(f"📝 추론: {result['text']} (신뢰도: {result['confidence']:.2f})")
            
    def inference_callback(self):
        """주기적 추론 콜백"""
        current_time = time.time()
        
        # 추론 간격 체크
        if current_time - self.last_inference_time < self.inference_interval:
            return
            
        # 이미지 가져오기
        image = self.get_latest_image()
        if image is None:
            return
            
        # 전처리
        inputs = self.preprocess_image(image)
        if inputs is None:
            return
            
        # 추론 실행
        result = self.run_inference(inputs)
        
        # 결과 발행
        self.publish_inference_result(result)
        
        self.last_inference_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = VLAInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
