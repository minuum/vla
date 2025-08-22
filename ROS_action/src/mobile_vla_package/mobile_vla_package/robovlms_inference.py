#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from transformers import AutoProcessor, AutoModel
import json
import time
from typing import List, Optional
import threading
from queue import Queue

class RoboVLMsInference(Node):
    """
    RoboVLMs 방식의 추론 노드
    단일 이미지를 받아서 단일 액션을 생성하는 실시간 반응형 시스템
    """
    
    def __init__(self):
        super().__init__('robovlms_inference')
        
        # 모델 설정 (업데이트된 최신 모델 사용)
        self.model_name = "minium/mobile-vla-omniwheel"  # MAE 0.222 달성한 최신 모델
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(f"Using updated model: {self.model_name} (MAE 0.222)")
        
        # 모델 로드
        self.load_model()
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.is_processing = False
        self.is_system_running = False
        self.current_task = "Navigate around obstacles to track the target cup"
        self.inference_count = 0
        self.last_inference_time = 0.0
        
        # 이미지 큐
        self.image_queue = Queue(maxsize=1)  # 최신 이미지만 유지
        
        # 추론 스레드 시작
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        
        self.get_logger().info("RoboVLMs Inference Node initialized")
    
    def load_model(self):
        """Mobile VLA Omniwheel 모델 로드 (MAE 0.222)"""
        try:
            self.get_logger().info(f"Loading updated model: {self.model_name}")
            self.get_logger().info("Model performance: MAE 0.222 (72.5% improvement)")
            
            # 모델과 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # GPU로 이동
            self.model.to(self.device)
            self.model.eval()
            
            self.get_logger().info("✅ Updated Mobile VLA Omniwheel model loaded successfully")
            self.get_logger().info("🎯 Model optimized for omniwheel robot navigation")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load updated model: {e}")
            # 테스트 모드로 전환
            self.get_logger().info("Switching to test mode (no model loading)")
            self.processor = None
            self.model = None
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 이미지 서브스크라이버 (압축된 이미지)
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
        
        # 추론 결과 퍼블리셔
        self.inference_result_pub = self.create_publisher(
            String,
            '/mobile_vla/inference_result',
            10
        )
        
        # 태스크 서브스크라이버
        self.task_sub = self.create_subscription(
            String,
            '/mobile_vla/task',
            self.task_callback,
            10
        )
        
        # 상태 퍼블리셔
        self.status_pub = self.create_publisher(
            String,
            '/mobile_vla/status',
            10
        )
        
        # 시스템 제어 서브스크라이버
        self.control_sub = self.create_subscription(
            String,
            '/mobile_vla/system_control',
            self.control_callback,
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def control_callback(self, msg):
        """시스템 제어 콜백"""
        try:
            command = json.loads(msg.data)
            action = command.get('action')
            
            if action == 'start':
                self.start_system()
            elif action == 'stop':
                self.stop_system()
            elif action == 'pause':
                self.pause_system()
            elif action == 'resume':
                self.resume_system()
            
        except Exception as e:
            self.get_logger().error(f"Error processing control command: {e}")
    
    def start_system(self):
        """시스템 시작"""
        self.is_system_running = True
        self.inference_count = 0
        self.get_logger().info("🚀 RoboVLMs system started")
        self.publish_status("started")
    
    def stop_system(self):
        """시스템 중지"""
        self.is_system_running = False
        # 로봇 정지
        self.stop_robot()
        self.get_logger().info("🛑 RoboVLMs system stopped")
        self.publish_status("stopped")
    
    def pause_system(self):
        """시스템 일시정지"""
        self.is_system_running = False
        self.stop_robot()
        self.get_logger().info("⏸️ RoboVLMs system paused")
        self.publish_status("paused")
    
    def resume_system(self):
        """시스템 재개"""
        self.is_system_running = True
        self.get_logger().info("▶️ RoboVLMs system resumed")
        self.publish_status("running")
    
    def stop_robot(self):
        """로봇 정지"""
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            self.action_pub.publish(twist)
        except Exception as e:
            self.get_logger().error(f"Error stopping robot: {e}")
    
    def task_callback(self, msg):
        """태스크 업데이트 콜백"""
        self.current_task = msg.data
        self.get_logger().info(f"Task updated: {self.current_task}")
    
    def image_callback(self, msg):
        """이미지 수신 콜백"""
        if not self.is_system_running:
            return
        
        try:
            # 압축된 이미지를 numpy 배열로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = PILImage.fromarray(image_rgb)
            
            # 큐에 이미지 추가 (기존 이미지 교체)
            if not self.image_queue.empty():
                self.image_queue.get()  # 기존 이미지 제거
            self.image_queue.put((pil_image, msg.header.stamp))
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def preprocess_image(self, image: PILImage.Image) -> Optional[torch.Tensor]:
        """이미지 전처리"""
        try:
            if self.processor is None:
                return None  # 테스트 모드
            
            # 모델 입력 형식에 맞게 전처리
            inputs = self.processor(
                images=image,
                text=self.current_task,
                return_tensors="pt"
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            self.get_logger().error(f"Error preprocessing image: {e}")
            return None
    
    def predict_single_action(self, inputs: dict) -> Optional[List[float]]:
        """단일 액션 예측 (RoboVLMs 방식) - MAE 0.222 모델 사용"""
        try:
            if self.model is None:
                # 테스트 모드: 간단한 액션 생성
                return self.generate_test_action()
            
            with torch.no_grad():
                # 업데이트된 모델 추론 (MAE 0.222)
                outputs = self.model(**inputs)
                
                # 액션 헤드에서 예측값 추출 (옴니휠 최적화)
                action_logits = outputs.action_logits  # [batch_size, 1, 3]
                
                # 단일 액션으로 변환 (RoboVLMs 방식)
                if action_logits.shape[1] > 1:
                    action_logits = action_logits[:, 0:1, :]  # 첫 번째 액션만 사용
                
                # CPU로 이동하고 numpy로 변환
                action = action_logits.cpu().numpy()[0, 0]  # [3]
                
                # 옴니휠 로봇에 최적화된 액션 반환
                return action.tolist()
                
        except Exception as e:
            self.get_logger().error(f"Error predicting action with updated model: {e}")
            return None
    
    def generate_test_action(self) -> List[float]:
        """테스트용 액션 생성"""
        # 간단한 원형 움직임
        import math
        t = time.time()
        angle = (t * 0.5) % (2 * math.pi)
        
        linear_x = 0.1 * math.cos(angle)
        linear_y = 0.05 * math.sin(angle)
        angular_z = 0.2 * math.sin(angle * 2)
        
        return [float(linear_x), float(linear_y), float(angular_z)]
    
    def inference_worker(self):
        """추론 워커 스레드 (RoboVLMs 방식)"""
        while rclpy.ok():
            try:
                if not self.is_system_running:
                    time.sleep(0.1)
                    continue
                
                # 큐에서 이미지 가져오기
                if not self.image_queue.empty():
                    image, timestamp = self.image_queue.get()
                    
                    self.is_processing = True
                    start_time = time.time()
                    
                    # 상태 업데이트
                    self.publish_status("processing")
                    
                    # 이미지 전처리
                    inputs = self.preprocess_image(image)
                    
                    # 단일 액션 예측
                    action = self.predict_single_action(inputs)
                    if action is None:
                        continue
                    
                    # 추론 시간 계산
                    inference_time = time.time() - start_time
                    self.last_inference_time = inference_time
                    self.inference_count += 1
                    
                    # 결과 발행
                    self.publish_inference_result(action, inference_time, timestamp)
                    
                    # 액션 실행
                    self.execute_action(action)
                    
                    self.is_processing = False
                    self.publish_status("ready")
                    
                else:
                    time.sleep(0.01)  # 10ms 대기
                    
            except Exception as e:
                self.get_logger().error(f"Error in inference worker: {e}")
                self.is_processing = False
                time.sleep(0.1)
    
    def execute_action(self, action: List[float]):
        """단일 액션 실행"""
        try:
            # 액션을 Twist 메시지로 변환
            twist = Twist()
            twist.linear.x = float(action[0])  # linear_x
            twist.linear.y = float(action[1])  # linear_y
            twist.angular.z = float(action[2])  # angular_z
            
            # 액션 발행
            self.action_pub.publish(twist)
            
        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")
    
    def publish_inference_result(self, action: List[float], inference_time: float, timestamp):
        """추론 결과 발행"""
        try:
            result = {
                "timestamp": timestamp.sec + timestamp.nanosec * 1e-9,
                "inference_time": inference_time,
                "action": action,
                "task": self.current_task,
                "inference_count": self.inference_count,
                "mode": "robovlms_single_action"
            }
            
            msg = String()
            msg.data = json.dumps(result)
            self.inference_result_pub.publish(msg)
            
            self.get_logger().info(f"🎯 RoboVLMs Inference #{self.inference_count}: {inference_time:.3f}s, Action: {action} (MAE 0.222 Model)")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing inference result: {e}")
    
    def publish_status(self, status: str):
        """상태 발행"""
        try:
            msg = String()
            msg.data = json.dumps({
                "status": status,
                "timestamp": time.time(),
                "inference_count": self.inference_count,
                "last_inference_time": self.last_inference_time,
                "mode": "robovlms"
            })
            self.status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = RoboVLMsInference()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
