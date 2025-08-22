#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
import numpy as np
import json
import time
import threading
from queue import Queue

class SimpleInferenceTest(Node):
    """
    간단한 테스트 추론 노드 (모델 로딩 없이)
    """
    
    def __init__(self):
        super().__init__('simple_inference_test')
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.is_processing = False
        self.action_queue = Queue()
        self.current_task = "Navigate around obstacles to track the target cup"
        
        # 추론 스레드 시작
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        
        self.get_logger().info("Simple Inference Test Node initialized")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
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
        
        # 추론 결과 퍼블리셔
        self.inference_result_pub = self.create_publisher(
            String,
            '/mobile_vla/inference_result',
            10
        )
        
        # 상태 퍼블리셔
        self.status_pub = self.create_publisher(
            String,
            '/mobile_vla/status',
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def image_callback(self, msg):
        """이미지 수신 콜백"""
        if self.is_processing:
            return
        
        try:
            # 압축된 이미지를 numpy 배열로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 이미지 크기 확인
            height, width = image.shape[:2]
            
            # 추론 큐에 추가
            self.action_queue.put((image, msg.header.stamp))
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def generate_test_actions(self, image):
        """테스트용 액션 생성 (18프레임)"""
        try:
            # 이미지 중앙 좌표 계산
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # 간단한 액션 생성 (원형 움직임)
            actions = []
            for i in range(18):
                angle = (i / 18) * 2 * np.pi
                linear_x = 0.1 * np.cos(angle)  # 전진/후진
                linear_y = 0.1 * np.sin(angle)  # 좌우 이동
                angular_z = 0.2 * np.sin(angle)  # 회전
                
                actions.append([float(linear_x), float(linear_y), float(angular_z)])
            
            return actions
            
        except Exception as e:
            self.get_logger().error(f"Error generating test actions: {e}")
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
                    
                    # 테스트 액션 생성
                    actions = self.generate_test_actions(image)
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
    
    def execute_action(self, action):
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
    
    def publish_inference_result(self, actions, inference_time, timestamp):
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
            
            self.get_logger().info(f"🎯 Test Inference: {inference_time:.3f}s, {len(actions)} frames")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing inference result: {e}")
    
    def publish_status(self, status):
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
    
    node = SimpleInferenceTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
