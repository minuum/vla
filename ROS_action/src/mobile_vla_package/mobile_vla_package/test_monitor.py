#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from typing import Dict, Any

class TestMonitor(Node):
    """
    테스트용 모니터링 도구
    시스템 상태와 성능을 실시간으로 모니터링
    """
    
    def __init__(self):
        super().__init__('test_monitor')
        
        # 서브스크라이버들
        self.system_status_sub = self.create_subscription(
            String,
            '/mobile_vla/system_status',
            self.system_status_callback,
            10
        )
        
        self.performance_sub = self.create_subscription(
            String,
            '/mobile_vla/performance_metrics',
            self.performance_callback,
            10
        )
        
        self.inference_result_sub = self.create_subscription(
            String,
            '/mobile_vla/inference_result',
            self.inference_result_callback,
            10
        )
        
        self.execution_status_sub = self.create_subscription(
            String,
            '/mobile_vla/execution_status',
            self.execution_status_callback,
            10
        )
        
        # 상태 변수
        self.system_status = {}
        self.performance_metrics = {}
        self.last_inference_time = 0
        self.inference_count = 0
        self.execution_count = 0
        
        # 타이머 설정 (1초마다 상태 출력)
        self.timer = self.create_timer(1.0, self.print_status)
        
        self.get_logger().info("Test Monitor initialized")
    
    def system_status_callback(self, msg):
        """시스템 상태 콜백"""
        try:
            self.system_status = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error parsing system status: {e}")
    
    def performance_callback(self, msg):
        """성능 메트릭 콜백"""
        try:
            self.performance_metrics = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Error parsing performance metrics: {e}")
    
    def inference_result_callback(self, msg):
        """추론 결과 콜백"""
        try:
            result = json.loads(msg.data)
            self.last_inference_time = time.time()
            self.inference_count += 1
            
            # 추론 결과 요약 출력
            inference_time = result.get('inference_time', 0)
            frame_count = result.get('frame_count', 0)
            task = result.get('task', 'Unknown')
            
            self.get_logger().info(f"🎯 Inference #{self.inference_count}: {inference_time:.3f}s, {frame_count} frames, Task: {task[:50]}...")
            
        except Exception as e:
            self.get_logger().error(f"Error parsing inference result: {e}")
    
    def execution_status_callback(self, msg):
        """실행 상태 콜백"""
        try:
            status = json.loads(msg.data)
            status_type = status.get('status', 'unknown')
            
            if status_type == 'started':
                self.execution_count += 1
                total_frames = status.get('total_frames', 0)
                self.get_logger().info(f"🚀 Execution #{self.execution_count} started: {total_frames} frames")
            elif status_type == 'executing':
                progress = status.get('progress', 0)
                current_frame = status.get('current_frame', 0)
                total_frames = status.get('total_frames', 0)
                self.get_logger().info(f"⏳ Execution progress: {progress:.1f}% ({current_frame}/{total_frames})")
            elif status_type == 'completed':
                self.get_logger().info(f"✅ Execution completed successfully")
            elif status_type == 'error':
                self.get_logger().error(f"❌ Execution failed")
            
        except Exception as e:
            self.get_logger().error(f"Error parsing execution status: {e}")
    
    def print_status(self):
        """상태 출력"""
        try:
            # 시스템 상태 출력
            overall_status = self.system_status.get('overall_status', 'unknown')
            inference_status = self.system_status.get('inference_node', 'unknown')
            execution_status = self.system_status.get('execution_node', 'unknown')
            
            # 성능 메트릭 출력
            inference_time = self.performance_metrics.get('inference_time', 0)
            execution_progress = self.performance_metrics.get('execution_progress', 0)
            
            # 상태 표시
            status_symbols = {
                'ready': '🟢',
                'processing': '🟡',
                'executing': '🟡',
                'active': '🟡',
                'completed': '🟢',
                'error': '🔴',
                'unknown': '⚪'
            }
            
            print(f"\n{'='*60}")
            print(f"📊 Mobile VLA System Status")
            print(f"{'='*60}")
            print(f"Overall Status: {status_symbols.get(overall_status, '⚪')} {overall_status}")
            print(f"Inference Node: {status_symbols.get(inference_status, '⚪')} {inference_status}")
            print(f"Execution Node: {status_symbols.get(execution_status, '⚪')} {execution_status}")
            print(f"{'='*60}")
            print(f"Performance Metrics:")
            print(f"  • Last Inference Time: {inference_time:.3f}s")
            print(f"  • Execution Progress: {execution_progress:.1f}%")
            print(f"  • Total Inferences: {self.inference_count}")
            print(f"  • Total Executions: {self.execution_count}")
            print(f"{'='*60}")
            
        except Exception as e:
            self.get_logger().error(f"Error printing status: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TestMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
