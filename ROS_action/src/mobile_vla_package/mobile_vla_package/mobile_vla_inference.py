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
from typing import List, Tuple
import threading
from queue import Queue

class MobileVLAInference(Node):
    """
    Mobile VLA 모델을 사용한 추론 노드
    단일 이미지를 받아서 18프레임의 액션 시퀀스를 예측
    """
    
    def __init__(self):
        super().__init__('mobile_vla_inference')
        
        # 모델 설정
        self.model_name = "minium/mobile-vla"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # 모델 로드
        self.load_model()
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.is_processing = False
        self.action_queue = Queue()
        self.current_task = "Navigate around obstacles to track the target cup"
        
        # 추론 스레드 시작
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        
        self.get_logger().info("Mobile VLA Inference Node initialized")
    
    def load_model(self):
        """Mobile VLA 모델 로드"""
        try:
            self.get_logger().info(f"Loading model: {self.model_name}")
            
            # 모델과 프로세서 로드
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # GPU로 이동
            self.model.to(self.device)
            self.model.eval()
            
            self.get_logger().info("Model loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
    
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
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def task_callback(self, msg):
        """태스크 업데이트 콜백"""
        self.current_task = msg.data
        self.get_logger().info(f"Task updated: {self.current_task}")
    
    def image_callback(self, msg):
        """이미지 수신 콜백"""
        if self.is_processing:
            return
        
        try:
            # 압축된 이미지를 numpy 배열로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            pil_image = PILImage.fromarray(image_rgb)
            
            # 추론 큐에 추가
            self.action_queue.put((pil_image, msg.header.stamp))
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def preprocess_image(self, image: PILImage.Image) -> torch.Tensor:
        """이미지 전처리"""
        try:
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
    
    def predict_actions(self, inputs: dict) -> List[List[float]]:
        """18프레임 액션 시퀀스 예측"""
        try:
            with torch.no_grad():
                # 모델 추론
                outputs = self.model(**inputs)
                
                # 액션 헤드에서 예측값 추출
                action_logits = outputs.action_logits  # [batch_size, 18, 3]
                
                # 시퀀스 길이 확인 및 조정
                if action_logits.shape[1] < 18:
                    # 패딩으로 18프레임 맞추기
                    padding = torch.zeros(action_logits.shape[0], 18 - action_logits.shape[1], 3, device=self.device)
                    action_logits = torch.cat([action_logits, padding], dim=1)
                elif action_logits.shape[1] > 18:
                    # 18프레임으로 자르기
                    action_logits = action_logits[:, :18, :]
                
                # CPU로 이동하고 numpy로 변환
                actions = action_logits.cpu().numpy()[0]  # [18, 3]
                
                return actions.tolist()
                
        except Exception as e:
            self.get_logger().error(f"Error predicting actions: {e}")
            return None
    
    def inference_worker(self):
        """추론 워커 스레드"""
        while rclpy.ok():
            try:
                # 큐에서 이미지 가져오기
                if not self.action_queue.empty():
                    image, timestamp = self.action_queue.get()
                    
                    self.is_processing = True
                    start_time = time.time()
                    
                    # 상태 업데이트
                    self.publish_status("processing")
                    
                    # 이미지 전처리
                    inputs = self.preprocess_image(image)
                    if inputs is None:
                        continue
                    
                    # 액션 예측
                    actions = self.predict_actions(inputs)
                    if actions is None:
                        continue
                    
                    # 추론 시간 계산
                    inference_time = time.time() - start_time
                    
                    # 결과 발행
                    self.publish_inference_result(actions, inference_time, timestamp)
                    
                    # 첫 번째 액션 실행
                    if actions:
                        self.execute_action(actions[0])
                    
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
    
    def publish_inference_result(self, actions: List[List[float]], inference_time: float, timestamp):
        """추론 결과 발행"""
        try:
            result = {
                "timestamp": timestamp.sec + timestamp.nanosec * 1e-9,
                "inference_time": inference_time,
                "actions": actions,
                "task": self.current_task,
                "frame_count": len(actions)
            }
            
            msg = String()
            msg.data = json.dumps(result)
            self.inference_result_pub.publish(msg)
            
            self.get_logger().info(f"Inference completed: {inference_time:.3f}s, {len(actions)} frames")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing inference result: {e}")
    
    def publish_status(self, status: str):
        """상태 발행"""
        try:
            msg = String()
            msg.data = json.dumps({
                "status": status,
                "timestamp": time.time()
            })
            self.status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = MobileVLAInference()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
