#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from typing import Dict, Any
import threading

class RoboVLMsController(Node):
    """
    RoboVLMs 시스템 컨트롤러
    시스템 시작, 중지, 태스크 관리 등을 담당
    """
    
    def __init__(self):
        super().__init__('robovlms_controller')
        
        # ROS 설정
        self.setup_ros()
        
        # 시스템 상태
        self.system_state = {
            'status': 'stopped',
            'task': 'Navigate around obstacles to track the target cup',
            'inference_count': 0,
            'start_time': None,
            'last_action_time': None
        }
        
        # 제어 명령 처리
        self.control_thread = threading.Thread(target=self.control_worker, daemon=True)
        self.control_thread.start()
        
        self.get_logger().info("RoboVLMs Controller initialized")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 시스템 제어 퍼블리셔
        self.system_control_pub = self.create_publisher(
            String,
            '/mobile_vla/system_control',
            10
        )
        
        # 태스크 퍼블리셔
        self.task_pub = self.create_publisher(
            String,
            '/mobile_vla/task',
            10
        )
        
        # 상태 서브스크라이버
        self.status_sub = self.create_subscription(
            String,
            '/mobile_vla/status',
            self.status_callback,
            10
        )
        
        # 추론 결과 서브스크라이버
        self.inference_result_sub = self.create_subscription(
            String,
            '/mobile_vla/inference_result',
            self.inference_result_callback,
            10
        )
        
        # 사용자 명령 서브스크라이버
        self.user_command_sub = self.create_subscription(
            String,
            '/robovlms/user_command',
            self.user_command_callback,
            10
        )
        
        # 시스템 상태 퍼블리셔
        self.system_status_pub = self.create_publisher(
            String,
            '/robovlms/system_status',
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def status_callback(self, msg):
        """상태 업데이트 콜백"""
        try:
            data = json.loads(msg.data)
            status = data.get('status', 'unknown')
            
            if status == 'started':
                self.system_state['status'] = 'running'
                self.system_state['start_time'] = time.time()
                self.get_logger().info("🟢 System started successfully")
            elif status == 'stopped':
                self.system_state['status'] = 'stopped'
                self.get_logger().info("🔴 System stopped")
            elif status == 'paused':
                self.system_state['status'] = 'paused'
                self.get_logger().info("🟡 System paused")
            elif status == 'running':
                self.system_state['status'] = 'running'
            
            # 추론 카운트 업데이트
            if 'inference_count' in data:
                self.system_state['inference_count'] = data['inference_count']
            
            # 시스템 상태 발행
            self.publish_system_status()
            
        except Exception as e:
            self.get_logger().error(f"Error processing status: {e}")
    
    def inference_result_callback(self, msg):
        """추론 결과 콜백"""
        try:
            data = json.loads(msg.data)
            self.system_state['last_action_time'] = time.time()
            
            # 추론 카운트 업데이트
            if 'inference_count' in data:
                self.system_state['inference_count'] = data['inference_count']
            
        except Exception as e:
            self.get_logger().error(f"Error processing inference result: {e}")
    
    def user_command_callback(self, msg):
        """사용자 명령 콜백"""
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
            elif action == 'set_task':
                task = command.get('task', self.system_state['task'])
                self.set_task(task)
            elif action == 'get_status':
                self.publish_system_status()
            
        except Exception as e:
            self.get_logger().error(f"Error processing user command: {e}")
    
    def start_system(self):
        """시스템 시작"""
        try:
            command = {
                'action': 'start',
                'timestamp': time.time()
            }
            
            msg = String()
            msg.data = json.dumps(command)
            self.system_control_pub.publish(msg)
            
            self.get_logger().info("🚀 Starting RoboVLMs system...")
            
        except Exception as e:
            self.get_logger().error(f"Error starting system: {e}")
    
    def stop_system(self):
        """시스템 중지"""
        try:
            command = {
                'action': 'stop',
                'timestamp': time.time()
            }
            
            msg = String()
            msg.data = json.dumps(command)
            self.system_control_pub.publish(msg)
            
            self.get_logger().info("🛑 Stopping RoboVLMs system...")
            
        except Exception as e:
            self.get_logger().error(f"Error stopping system: {e}")
    
    def pause_system(self):
        """시스템 일시정지"""
        try:
            command = {
                'action': 'pause',
                'timestamp': time.time()
            }
            
            msg = String()
            msg.data = json.dumps(command)
            self.system_control_pub.publish(msg)
            
            self.get_logger().info("⏸️ Pausing RoboVLMs system...")
            
        except Exception as e:
            self.get_logger().error(f"Error pausing system: {e}")
    
    def resume_system(self):
        """시스템 재개"""
        try:
            command = {
                'action': 'resume',
                'timestamp': time.time()
            }
            
            msg = String()
            msg.data = json.dumps(command)
            self.system_control_pub.publish(msg)
            
            self.get_logger().info("▶️ Resuming RoboVLMs system...")
            
        except Exception as e:
            self.get_logger().error(f"Error resuming system: {e}")
    
    def set_task(self, task: str):
        """태스크 설정"""
        try:
            self.system_state['task'] = task
            
            msg = String()
            msg.data = task
            self.task_pub.publish(msg)
            
            self.get_logger().info(f"📝 Task updated: {task}")
            
        except Exception as e:
            self.get_logger().error(f"Error setting task: {e}")
    
    def publish_system_status(self):
        """시스템 상태 발행"""
        try:
            # 실행 시간 계산
            runtime = 0.0
            if self.system_state['start_time']:
                runtime = time.time() - self.system_state['start_time']
            
            # 마지막 액션 시간 계산
            time_since_last_action = 0.0
            if self.system_state['last_action_time']:
                time_since_last_action = time.time() - self.system_state['last_action_time']
            
            status_data = {
                'status': self.system_state['status'],
                'task': self.system_state['task'],
                'inference_count': self.system_state['inference_count'],
                'runtime': runtime,
                'time_since_last_action': time_since_last_action,
                'timestamp': time.time(),
                'mode': 'robovlms'
            }
            
            msg = String()
            msg.data = json.dumps(status_data)
            self.system_status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing system status: {e}")
    
    def control_worker(self):
        """제어 워커 스레드"""
        while rclpy.ok():
            try:
                # 주기적으로 시스템 상태 발행
                self.publish_system_status()
                
                # 1초마다 업데이트
                time.sleep(1.0)
                
            except Exception as e:
                self.get_logger().error(f"Error in control worker: {e}")
                time.sleep(1.0)
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        try:
            runtime = 0.0
            if self.system_state['start_time']:
                runtime = time.time() - self.system_state['start_time']
            
            return {
                'status': self.system_state['status'],
                'task': self.system_state['task'],
                'inference_count': self.system_state['inference_count'],
                'runtime': runtime,
                'mode': 'robovlms'
            }
            
        except Exception as e:
            self.get_logger().error(f"Error getting system info: {e}")
            return {'status': 'error', 'error': str(e)}

def main(args=None):
    rclpy.init(args=args)
    
    node = RoboVLMsController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
