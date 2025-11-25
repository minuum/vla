#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from typing import Dict, Any
import threading

class SystemMonitor(Node):
    """
    Mobile VLA 시스템 모니터링 노드
    전체 시스템의 상태를 모니터링하고 관리
    """
    
    def __init__(self):
        super().__init__('system_monitor')
        
        # 시스템 상태
        self.system_status = {
            'inference_node': 'unknown',
            'execution_node': 'unknown',
            'camera_node': 'unknown',
            'overall_status': 'unknown',
            'last_update': time.time()
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            'inference_time': 0.0,
            'execution_progress': 0.0,
            'frame_count': 0,
            'total_frames': 0
        }
        
        # ROS 설정
        self.setup_ros()
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self.monitoring_worker, daemon=True)
        self.monitor_thread.start()
        
        self.get_logger().info("System Monitor initialized")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 상태 서브스크라이버들
        self.inference_status_sub = self.create_subscription(
            String,
            '/mobile_vla/status',
            self.inference_status_callback,
            10
        )
        
        self.execution_status_sub = self.create_subscription(
            String,
            '/mobile_vla/execution_status',
            self.execution_status_callback,
            10
        )
        
        # 시스템 상태 퍼블리셔
        self.system_status_pub = self.create_publisher(
            String,
            '/mobile_vla/system_status',
            10
        )
        
        # 성능 메트릭 퍼블리셔
        self.performance_pub = self.create_publisher(
            String,
            '/mobile_vla/performance_metrics',
            10
        )
        
        # 추론 결과 서브스크라이버
        self.inference_result_sub = self.create_subscription(
            String,
            '/mobile_vla/inference_result',
            self.inference_result_callback,
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def inference_status_callback(self, msg):
        """추론 노드 상태 콜백"""
        try:
            data = json.loads(msg.data)
            status = data.get('status', 'unknown')
            self.system_status['inference_node'] = status
            self.update_overall_status()
            
        except Exception as e:
            self.get_logger().error(f"Error processing inference status: {e}")
    
    def execution_status_callback(self, msg):
        """실행 노드 상태 콜백"""
        try:
            data = json.loads(msg.data)
            status = data.get('status', 'unknown')
            self.system_status['execution_node'] = status
            
            # 성능 메트릭 업데이트
            self.performance_metrics['execution_progress'] = data.get('progress', 0.0)
            self.performance_metrics['current_frame'] = data.get('current_frame', 0)
            self.performance_metrics['total_frames'] = data.get('total_frames', 0)
            
            self.update_overall_status()
            
        except Exception as e:
            self.get_logger().error(f"Error processing execution status: {e}")
    
    def inference_result_callback(self, msg):
        """추론 결과 콜백"""
        try:
            data = json.loads(msg.data)
            
            # 성능 메트릭 업데이트
            self.performance_metrics['inference_time'] = data.get('inference_time', 0.0)
            self.performance_metrics['frame_count'] = data.get('frame_count', 0)
            
        except Exception as e:
            self.get_logger().error(f"Error processing inference result: {e}")
    
    def update_overall_status(self):
        """전체 시스템 상태 업데이트"""
        try:
            # 각 노드 상태 확인
            inference_ok = self.system_status['inference_node'] in ['ready', 'processing']
            execution_ok = self.system_status['execution_node'] in ['ready', 'executing', 'completed']
            
            # 전체 상태 결정
            if inference_ok and execution_ok:
                if self.system_status['inference_node'] == 'processing' or self.system_status['execution_node'] == 'executing':
                    overall_status = 'active'
                else:
                    overall_status = 'ready'
            elif self.system_status['inference_node'] == 'error' or self.system_status['execution_node'] == 'error':
                overall_status = 'error'
            else:
                overall_status = 'unknown'
            
            self.system_status['overall_status'] = overall_status
            self.system_status['last_update'] = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error updating overall status: {e}")
    
    def monitoring_worker(self):
        """모니터링 워커 스레드"""
        while rclpy.ok():
            try:
                # 시스템 상태 발행
                self.publish_system_status()
                
                # 성능 메트릭 발행
                self.publish_performance_metrics()
                
                # 1초마다 업데이트
                time.sleep(1.0)
                
            except Exception as e:
                self.get_logger().error(f"Error in monitoring worker: {e}")
                time.sleep(1.0)
    
    def publish_system_status(self):
        """시스템 상태 발행"""
        try:
            msg = String()
            msg.data = json.dumps(self.system_status)
            self.system_status_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing system status: {e}")
    
    def publish_performance_metrics(self):
        """성능 메트릭 발행"""
        try:
            msg = String()
            msg.data = json.dumps(self.performance_metrics)
            self.performance_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing performance metrics: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 정보 반환"""
        try:
            current_time = time.time()
            last_update = self.system_status.get('last_update', 0)
            
            # 노드 응답성 확인
            node_responsive = (current_time - last_update) < 5.0  # 5초 이내 업데이트
            
            health_info = {
                'overall_status': self.system_status['overall_status'],
                'node_responsive': node_responsive,
                'inference_time': self.performance_metrics['inference_time'],
                'execution_progress': self.performance_metrics['execution_progress'],
                'last_update': last_update
            }
            
            return health_info
            
        except Exception as e:
            self.get_logger().error(f"Error getting system health: {e}")
            return {'overall_status': 'error', 'error': str(e)}

def main(args=None):
    rclpy.init(args=args)
    
    node = SystemMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
