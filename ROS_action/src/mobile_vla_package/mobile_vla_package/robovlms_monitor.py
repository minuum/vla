#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from typing import Dict, Any
import threading

class RoboVLMsMonitor(Node):
    """
    RoboVLMs 시스템 모니터
    실시간으로 시스템 상태, 추론 결과, 성능을 모니터링
    """
    
    def __init__(self):
        super().__init__('robovlms_monitor')
        
        # ROS 설정
        self.setup_ros()
        
        # 모니터링 데이터
        self.monitoring_data = {
            'system_status': {},
            'inference_results': [],
            'performance_metrics': {
                'total_inferences': 0,
                'avg_inference_time': 0.0,
                'max_inference_time': 0.0,
                'min_inference_time': float('inf'),
                'start_time': time.time(),
                'last_inference_time': 0.0
            },
            'last_update': time.time()
        }
        
        # 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(target=self.monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.get_logger().info("RoboVLMs Monitor initialized")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 시스템 상태 서브스크라이버
        self.system_status_sub = self.create_subscription(
            String,
            '/robovlms/system_status',
            self.system_status_callback,
            10
        )
        
        # 추론 결과 서브스크라이버
        self.inference_result_sub = self.create_subscription(
            String,
            '/mobile_vla/inference_result',
            self.inference_result_callback,
            10
        )
        
        # 상태 서브스크라이버
        self.status_sub = self.create_subscription(
            String,
            '/mobile_vla/status',
            self.status_callback,
            10
        )
        
        # 모니터링 결과 퍼블리셔
        self.monitoring_result_pub = self.create_publisher(
            String,
            '/robovlms/monitoring_result',
            10
        )
        
        self.get_logger().info("ROS interfaces setup completed")
    
    def system_status_callback(self, msg):
        """시스템 상태 콜백"""
        try:
            data = json.loads(msg.data)
            self.monitoring_data['system_status'] = data
            self.monitoring_data['last_update'] = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing system status: {e}")
    
    def inference_result_callback(self, msg):
        """추론 결과 콜백"""
        try:
            data = json.loads(msg.data)
            
            # 추론 결과 저장 (최근 10개만 유지)
            self.monitoring_data['inference_results'].append(data)
            if len(self.monitoring_data['inference_results']) > 10:
                self.monitoring_data['inference_results'].pop(0)
            
            # 성능 메트릭 업데이트
            self.update_performance_metrics(data)
            
            self.monitoring_data['last_update'] = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing inference result: {e}")
    
    def status_callback(self, msg):
        """상태 콜백"""
        try:
            data = json.loads(msg.data)
            
            # 시스템 상태 업데이트
            if 'status' in data:
                self.monitoring_data['system_status']['status'] = data['status']
            
            self.monitoring_data['last_update'] = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing status: {e}")
    
    def update_performance_metrics(self, inference_data: Dict[str, Any]):
        """성능 메트릭 업데이트"""
        try:
            metrics = self.monitoring_data['performance_metrics']
            
            # 추론 시간 업데이트
            inference_time = inference_data.get('inference_time', 0.0)
            
            metrics['total_inferences'] += 1
            metrics['last_inference_time'] = time.time()
            
            # 평균 추론 시간 계산
            if metrics['total_inferences'] == 1:
                metrics['avg_inference_time'] = inference_time
            else:
                metrics['avg_inference_time'] = (
                    (metrics['avg_inference_time'] * (metrics['total_inferences'] - 1) + inference_time) 
                    / metrics['total_inferences']
                )
            
            # 최대/최소 추론 시간 업데이트
            metrics['max_inference_time'] = max(metrics['max_inference_time'], inference_time)
            metrics['min_inference_time'] = min(metrics['min_inference_time'], inference_time)
            
        except Exception as e:
            self.get_logger().error(f"Error updating performance metrics: {e}")
    
    def print_monitoring_summary(self):
        """모니터링 요약 출력"""
        try:
            system_status = self.monitoring_data['system_status']
            metrics = self.monitoring_data['performance_metrics']
            
            # 실행 시간 계산
            runtime = time.time() - metrics['start_time']
            
            # 상태 이모지
            status_emoji = {
                'running': '🟢',
                'stopped': '🔴',
                'paused': '🟡',
                'processing': '🔄',
                'ready': '✅'
            }.get(system_status.get('status', 'unknown'), '❓')
            
            print("\n" + "="*80)
            print("🤖 RoboVLMs System Monitoring Dashboard")
            print("="*80)
            
            # 시스템 상태
            print(f"📊 System Status: {status_emoji} {system_status.get('status', 'unknown').upper()}")
            print(f"⏱️  Runtime: {runtime:.1f}s")
            print(f"📝 Task: {system_status.get('task', 'N/A')}")
            
            # 추론 통계
            print(f"\n🎯 Inference Statistics:")
            print(f"   Total Inferences: {metrics['total_inferences']}")
            print(f"   Avg Time: {metrics['avg_inference_time']:.3f}s")
            print(f"   Max Time: {metrics['max_inference_time']:.3f}s")
            print(f"   Min Time: {metrics['min_inference_time']:.3f}s")
            
            # 최근 추론 결과
            if self.monitoring_data['inference_results']:
                latest = self.monitoring_data['inference_results'][-1]
                print(f"\n🔄 Latest Inference:")
                print(f"   Count: #{latest.get('inference_count', 0)}")
                print(f"   Time: {latest.get('inference_time', 0):.3f}s")
                print(f"   Action: {latest.get('action', [0, 0, 0])}")
                print(f"   Mode: {latest.get('mode', 'unknown')}")
            
            # 성능 지표
            if metrics['total_inferences'] > 0:
                fps = metrics['total_inferences'] / runtime if runtime > 0 else 0
                print(f"\n⚡ Performance Metrics:")
                print(f"   FPS: {fps:.1f}")
                print(f"   Efficiency: {'🟢 Good' if metrics['avg_inference_time'] < 0.1 else '🟡 Normal' if metrics['avg_inference_time'] < 0.5 else '🔴 Slow'}")
            
            print("="*80)
            
        except Exception as e:
            self.get_logger().error(f"Error printing monitoring summary: {e}")
    
    def monitoring_worker(self):
        """모니터링 워커 스레드"""
        while rclpy.ok():
            try:
                # 모니터링 요약 출력
                self.print_monitoring_summary()
                
                # 모니터링 결과 발행
                self.publish_monitoring_result()
                
                # 2초마다 업데이트
                time.sleep(2.0)
                
            except Exception as e:
                self.get_logger().error(f"Error in monitoring worker: {e}")
                time.sleep(2.0)
    
    def publish_monitoring_result(self):
        """모니터링 결과 발행"""
        try:
            monitoring_result = {
                'timestamp': time.time(),
                'system_status': self.monitoring_data['system_status'],
                'performance_metrics': self.monitoring_data['performance_metrics'],
                'recent_inferences': len(self.monitoring_data['inference_results']),
                'last_update': self.monitoring_data['last_update']
            }
            
            msg = String()
            msg.data = json.dumps(monitoring_result)
            self.monitoring_result_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing monitoring result: {e}")
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """모니터링 데이터 반환"""
        return self.monitoring_data.copy()

def main(args=None):
    rclpy.init(args=args)
    
    node = RoboVLMsMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
