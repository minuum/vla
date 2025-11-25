#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json
import time
from typing import List, Optional
import threading
from queue import Queue

class ActionSequenceExecutor(Node):
    """
    18프레임 액션 시퀀스를 순차적으로 실행하는 노드
    """
    
    def __init__(self):
        super().__init__('action_sequence_executor')
        
        # ROS 설정
        self.setup_ros()
        
        # 상태 변수
        self.is_executing = False
        self.current_sequence = []
        self.current_frame = 0
        self.frame_rate = 10.0  # 10Hz (100ms per frame)
        self.execution_thread = None
        
        # 액션 큐
        self.action_queue = Queue()
        
        # 실행 스레드 시작
        self.execution_thread = threading.Thread(target=self.execution_worker, daemon=True)
        self.execution_thread.start()
        
        self.get_logger().info("Action Sequence Executor initialized")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 추론 결과 서브스크라이버
        self.inference_sub = self.create_subscription(
            String,
            '/mobile_vla/inference_result',
            self.inference_callback,
            10
        )
        
        # 액션 퍼블리셔
        self.action_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 실행 상태 퍼블리셔
        self.execution_status_pub = self.create_publisher(
            String,
            '/mobile_vla/execution_status',
            10
        )
        
        # 제어 명령 서브스크라이버
        self.control_sub = self.create_subscription(
            String,
            '/mobile_vla/control',
            self.control_callback,
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def inference_callback(self, msg):
        """추론 결과 수신 콜백"""
        try:
            result = json.loads(msg.data)
            actions = result.get('actions', [])
            
            if actions:
                # 새로운 액션 시퀀스를 큐에 추가
                self.action_queue.put(actions)
                self.get_logger().info(f"Received new action sequence: {len(actions)} frames")
            
        except Exception as e:
            self.get_logger().error(f"Error processing inference result: {e}")
    
    def control_callback(self, msg):
        """제어 명령 수신 콜백"""
        try:
            command = json.loads(msg.data)
            action = command.get('action')
            
            if action == 'stop':
                self.stop_execution()
            elif action == 'pause':
                self.pause_execution()
            elif action == 'resume':
                self.resume_execution()
            elif action == 'set_frame_rate':
                self.frame_rate = command.get('frame_rate', 10.0)
                self.get_logger().info(f"Frame rate updated: {self.frame_rate} Hz")
            
        except Exception as e:
            self.get_logger().error(f"Error processing control command: {e}")
    
    def execution_worker(self):
        """액션 실행 워커 스레드"""
        while rclpy.ok():
            try:
                # 큐에서 액션 시퀀스 가져오기
                if not self.action_queue.empty() and not self.is_executing:
                    actions = self.action_queue.get()
                    self.execute_sequence(actions)
                
                time.sleep(0.01)  # 10ms 대기
                
            except Exception as e:
                self.get_logger().error(f"Error in execution worker: {e}")
                time.sleep(0.1)
    
    def execute_sequence(self, actions: List[List[float]]):
        """액션 시퀀스 실행"""
        try:
            self.is_executing = True
            self.current_sequence = actions
            self.current_frame = 0
            
            self.get_logger().info(f"Starting sequence execution: {len(actions)} frames")
            self.publish_execution_status("started", len(actions))
            
            frame_interval = 1.0 / self.frame_rate
            
            for i, action in enumerate(actions):
                if not self.is_executing:
                    break
                
                # 액션 실행
                self.execute_action(action)
                self.current_frame = i + 1
                
                # 진행 상태 발행
                progress = (i + 1) / len(actions) * 100
                self.publish_execution_status("executing", len(actions), i + 1, progress)
                
                # 프레임 간격 대기
                time.sleep(frame_interval)
            
            if self.is_executing:
                self.get_logger().info("Sequence execution completed")
                self.publish_execution_status("completed", len(actions))
            
            self.is_executing = False
            self.current_sequence = []
            self.current_frame = 0
            
        except Exception as e:
            self.get_logger().error(f"Error executing sequence: {e}")
            self.is_executing = False
            self.publish_execution_status("error", 0)
    
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
    
    def stop_execution(self):
        """실행 중지"""
        self.is_executing = False
        self.get_logger().info("Execution stopped")
        self.publish_execution_status("stopped", 0)
    
    def pause_execution(self):
        """실행 일시정지"""
        # 현재는 stop과 동일하게 처리
        self.stop_execution()
        self.get_logger().info("Execution paused")
    
    def resume_execution(self):
        """실행 재개"""
        # 현재는 새로운 시퀀스가 들어올 때까지 대기
        self.get_logger().info("Execution resumed")
    
    def publish_execution_status(self, status: str, total_frames: int, current_frame: int = 0, progress: float = 0.0):
        """실행 상태 발행"""
        try:
            result = {
                "status": status,
                "total_frames": total_frames,
                "current_frame": current_frame,
                "progress": progress,
                "timestamp": time.time()
            }
            
            msg = String()
            msg.data = json.dumps(result)
            self.execution_status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing execution status: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = ActionSequenceExecutor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
