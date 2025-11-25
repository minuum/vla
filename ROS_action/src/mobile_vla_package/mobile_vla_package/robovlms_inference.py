#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
import time
import threading
from queue import Queue

class MinimalInferenceNode(Node):
    """
    최소 추론 노드 - 실제 이미지를 받아서 액션 생성
    """
    
    def __init__(self):
        super().__init__('minimal_inference_node')
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.is_processing = False
        self.inference_count = 0
        self.current_task = "Navigate around obstacles to track the target cup"
        
        # 이미지 큐
        self.image_queue = Queue(maxsize=1)
        
        # 추론 스레드 시작
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        
        self.get_logger().info("🧠 최소 추론 노드 시작")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 이미지 서브스크라이버
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
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
            '/inference/result',
            10
        )
        
        # 상태 퍼블리셔
        self.status_pub = self.create_publisher(
            String,
            '/inference/status',
            10
        )
        
        self.get_logger().info("✅ ROS 인터페이스 설정 완료")
    
    def image_callback(self, msg):
        """이미지 수신 콜백"""
        if self.is_processing:
            return
        
        try:
            # ROS 이미지를 OpenCV로 변환
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 큐에 이미지 추가 (기존 이미지 교체)
            if not self.image_queue.empty():
                self.image_queue.get()  # 기존 이미지 제거
            self.image_queue.put((cv_image, msg.header.stamp))
            
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 처리 오류: {e}")
    
    def analyze_image(self, image):
        """이미지 분석 및 액션 생성"""
        try:
            # 이미지 크기 확인
            height, width = image.shape[:2]
            
            # 간단한 이미지 분석 (중앙 영역 확인)
            center_x, center_y = width // 2, height // 2
            
            # 중앙 영역의 색상 분석
            center_region = image[center_y-50:center_y+50, center_x-50:center_x+50]
            avg_color = np.mean(center_region, axis=(0, 1))
            
            # 색상에 따른 액션 결정
            if avg_color[2] > 100:  # 빨간색이 많으면 (컵이 보이면)
                # 전진
                action = [0.3, 0.0, 0.0]
                action_type = "전진 (컵 발견)"
            elif avg_color[1] > 100:  # 녹색이 많으면
                # 좌회전
                action = [0.1, 0.0, 0.5]
                action_type = "좌회전"
            else:
                # 우회전
                action = [0.1, 0.0, -0.5]
                action_type = "우회전"
            
            return action, action_type
            
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 분석 오류: {e}")
            return [0.0, 0.0, 0.0], "정지"
    
    def inference_worker(self):
        """추론 워커 스레드"""
        while rclpy.ok():
            try:
                # 큐에서 이미지 가져오기
                if not self.image_queue.empty():
                    image, timestamp = self.image_queue.get()
                    
                    self.is_processing = True
                    start_time = time.time()
                    
                    # 상태 업데이트
                    self.publish_status("processing")
                    
                    # 이미지 분석 및 액션 생성
                    action, action_type = self.analyze_image(image)
                    
                    # 추론 시간 계산
                    inference_time = time.time() - start_time
                    self.inference_count += 1
                    
                    # 결과 발행
                    self.publish_inference_result(action, inference_time, timestamp, action_type)
                    
                    # 액션 실행
                    self.execute_action(action)
                    
                    self.is_processing = False
                    self.publish_status("ready")
                    
                else:
                    time.sleep(0.01)  # 10ms 대기
                    
            except Exception as e:
                self.get_logger().error(f"❌ 추론 워커 오류: {e}")
                self.is_processing = False
                time.sleep(0.1)
    
    def execute_action(self, action):
        """액션 실행"""
        try:
            # 액션을 Twist 메시지로 변환
            twist = Twist()
            twist.linear.x = float(action[0])  # linear_x
            twist.linear.y = float(action[1])  # linear_y
            twist.angular.z = float(action[2])  # angular_z
            
            # 액션 발행
            self.action_pub.publish(twist)
            
        except Exception as e:
            self.get_logger().error(f"❌ 액션 실행 오류: {e}")
    
    def publish_inference_result(self, action, inference_time, timestamp, action_type):
        """추론 결과 발행"""
        try:
            result = {
                "timestamp": timestamp.sec + timestamp.nanosec * 1e-9,
                "inference_time": inference_time,
                "action": action,
                "action_type": action_type,
                "task": self.current_task,
                "inference_count": self.inference_count
            }
            
            msg = String()
            msg.data = json.dumps(result)
            self.inference_result_pub.publish(msg)
            
            self.get_logger().info(f"🧠 추론 #{self.inference_count}: {inference_time*1000:.1f}ms, {action_type}")
            
        except Exception as e:
            self.get_logger().error(f"❌ 추론 결과 발행 오류: {e}")
    
    def publish_status(self, status):
        """상태 발행"""
        try:
            msg = String()
            msg.data = json.dumps({
                "status": status,
                "timestamp": time.time(),
                "inference_count": self.inference_count
            })
            self.status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ 상태 발행 오류: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = MinimalInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
