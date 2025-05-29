#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import torch
import cv2
import numpy as np
from PIL import Image as PilImage
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from cv_bridge import CvBridge
import os
from pathlib import Path

class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_node')
        self.get_logger().info("🤖 VLA 추론 노드 초기화 중...")

        # 모델 설정
        self.declare_parameter('model_id', "google/paligemma-3b-mix-224")
        self.declare_parameter('model_cache_dir', ".vla_models_cache")
        self.declare_parameter('max_new_tokens', 128)
        self.declare_parameter('device_preference', "cuda")

        self.model_id = self.get_parameter('model_id').get_parameter_value().string_value
        self.model_cache_dir = self.get_parameter('model_cache_dir').get_parameter_value().string_value
        self.max_new_tokens = self.get_parameter('max_new_tokens').get_parameter_value().integer_value
        self.device_preference = self.get_parameter('device_preference').get_parameter_value().string_value

        # 디바이스 설정
        if self.device_preference == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.get_logger().info(f"🎯 사용 디바이스: {self.device}")

        # 모델 로드
        self.model = None
        self.processor = None
        self.load_model()

        # 상태 변수
        self.bridge = CvBridge()
        self.current_image = None
        self.current_text = None
        self.image_width = 640
        self.image_height = 480

        # 구독자 & 발행자 (기존 카메라 노드와 호환)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(String, '/stt/text', self.text_callback, 10)  # STT 결과 토픽
        
        # 출력 토픽
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla/status', 10)

        self.get_logger().info("✅ VLA 추론 노드 초기화 완료")

    def load_model(self):
        """VLA 모델 로드"""
        try:
            self.get_logger().info(f"📥 모델 로딩 중: {self.model_id}")
            
            model_save_path = Path(self.model_cache_dir) / self.model_id.split('/')[-1]
            model_save_path.mkdir(parents=True, exist_ok=True)

            # 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                cache_dir=model_save_path
            )

            # 모델 로드
            model_kwargs = {
                "cache_dir": model_save_path,
                "low_cpu_mem_usage": True
            }
            
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id, 
                **model_kwargs
            )
            
            if self.device.type != "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            self.get_logger().info("✅ VLA 모델 로딩 완료")
            
        except Exception as e:
            self.get_logger().error(f"❌ 모델 로딩 실패: {e}")
            raise

    def image_callback(self, msg):
        """카메라 이미지 수신 (기존 카메라 노드 호환)"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.image_width = msg.width
            self.image_height = msg.height
            
            self.get_logger().debug(f"📷 이미지 수신: {self.image_width}x{self.image_height}")
            
            # 이미지와 텍스트가 모두 있으면 추론 실행
            self.try_inference()
                
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 처리 오류: {e}")

    def text_callback(self, msg):
        """STT 텍스트 수신"""
        self.current_text = msg.data
        self.get_logger().info(f"🎯 STT 텍스트 수신: '{self.current_text}'")
        
        # 이미지와 텍스트가 모두 있으면 추론 실행
        self.try_inference()

    def try_inference(self):
        """이미지와 텍스트가 모두 있으면 VLA 추론 실행"""
        if self.current_image is None or self.current_text is None:
            return

        try:
            self.get_logger().info(f"🧠 VLA 추론 실행: '{self.current_text}'")
            
            # 이미지 전처리
            pil_image = PilImage.fromarray(self.current_image)
            
            # 텍스트 명령에 따른 추론
            linear_x, linear_y, angular_z = self.perform_vla_inference(pil_image, self.current_text)

            # Twist 메시지 발행
            self.publish_cmd_vel(linear_x, linear_y, angular_z)
            
            # 상태 리셋 (한 번 추론 후 초기화)
            self.current_image = None
            self.current_text = None
            
        except Exception as e:
            self.get_logger().error(f"❌ VLA 추론 오류: {e}")

    def perform_vla_inference(self, image, text_command):
        """VLA 모델 추론"""
        try:
            # 간단한 명령어 매핑 먼저 확인
            simple_commands = self.check_simple_commands(text_command)
            if simple_commands is not None:
                return simple_commands

            # VLA 모델을 통한 복잡한 추론
            if "navigate to" in text_command.lower() or "go to" in text_command.lower():
                return self.perform_navigation_inference(image, text_command)
            elif "avoid" in text_command.lower() or "obstacle" in text_command.lower():
                return self.perform_obstacle_avoidance_inference(image, text_command)
            else:
                return self.perform_general_inference(image, text_command)
            
        except Exception as e:
            self.get_logger().error(f"❌ VLA 추론 오류: {e}")
            return 0.0, 0.0, 0.0

    def check_simple_commands(self, text_command):
        """간단한 명령어 직접 처리"""
        command_lower = text_command.lower()
        
        if "stop" in command_lower or "halt" in command_lower:
            self.get_logger().info("🛑 정지 명령")
            return 0.0, 0.0, 0.0
        elif "move forward" in command_lower or "go forward" in command_lower:
            self.get_logger().info("➡️ 전진 명령")
            return 0.3, 0.0, 0.0
        elif "move backward" in command_lower or "go backward" in command_lower:
            self.get_logger().info("⬅️ 후진 명령")
            return -0.3, 0.0, 0.0
        elif "turn left" in command_lower:
            self.get_logger().info("↪️ 좌회전 명령")
            return 0.0, 0.0, 0.5
        elif "turn right" in command_lower:
            self.get_logger().info("↩️ 우회전 명령")
            return 0.0, 0.0, -0.5
        
        return None  # 복잡한 명령은 VLA 모델 사용

    def perform_navigation_inference(self, image, text_command):
        """내비게이션 추론"""
        try:
            # 목표물 찾기
            target = text_command.lower().replace("navigate to", "").replace("go to", "").strip()
            prompt = f"find {target} in the image and determine robot movement direction"
            
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            self.get_logger().info(f"🎯 내비게이션 결과: {result}")
            return self.parse_action_to_twist(result)
            
        except Exception as e:
            self.get_logger().error(f"❌ 내비게이션 추론 오류: {e}")
            return 0.0, 0.0, 0.0

    def perform_obstacle_avoidance_inference(self, image, text_command):
        """장애물 회피 추론"""
        try:
            prompt = "detect obstacles and suggest safe movement direction"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # 장애물이 감지되면 안전한 속도로 이동
            if "obstacle" in result.lower() or "blocked" in result.lower():
                self.get_logger().info("🛑 장애물 감지 - 정지")
                return 0.0, 0.0, 0.0
            else:
                self.get_logger().info("✅ 경로 안전 - 천천히 전진")
                return 0.1, 0.0, 0.0
                
        except Exception as e:
            self.get_logger().error(f"❌ 장애물 회피 추론 오류: {e}")
            return 0.0, 0.0, 0.0

    def perform_general_inference(self, image, text_command):
        """일반적인 VLA 추론"""
        try:
            prompt = f"Robot action for command: {text_command}"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            self.get_logger().info(f"🤖 일반 VLA 결과: {result}")
            return self.parse_action_to_twist(result)
            
        except Exception as e:
            self.get_logger().error(f"❌ 일반 VLA 추론 오류: {e}")
            return 0.0, 0.0, 0.0

    def parse_action_to_twist(self, action_text):
        """VLA 결과를 Twist 명령으로 변환"""
        linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
        
        action_lower = action_text.lower()
        
        if "forward" in action_lower or "ahead" in action_lower:
            linear_x = 0.2
        elif "backward" in action_lower or "back" in action_lower:
            linear_x = -0.2
        elif "left" in action_lower:
            angular_z = 0.5
        elif "right" in action_lower:
            angular_z = -0.5
        elif "stop" in action_lower or "halt" in action_lower:
            linear_x, linear_y, angular_z = 0.0, 0.0, 0.0
            
        return linear_x, linear_y, angular_z

    def publish_cmd_vel(self, linear_x, linear_y, angular_z):
        """cmd_vel 메시지 발행"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"🚀 cmd_vel 발행: x={linear_x:.2f}, y={linear_y:.2f}, z={angular_z:.2f}")
        
        # 상태 발행
        status_msg = String()
        status_msg.data = f"VLA inference completed: linear_x={linear_x:.2f}, linear_y={linear_y:.2f}, angular_z={angular_z:.2f}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VLAInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ VLA 노드 오류: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
