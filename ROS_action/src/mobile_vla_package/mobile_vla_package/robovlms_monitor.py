#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading

class MinimalMonitorNode(Node):
    """
    최소 모니터링 노드
    시스템 상태와 추론 결과를 모니터링
    """
    
    def __init__(self):
        super().__init__('minimal_monitor_node')
        
        # ROS 설정
        self.setup_ros()
        
        # 모니터링 데이터
        self.monitoring_data = {
            'camera_status': 'unknown',
            'inference_status': 'unknown',
            'control_status': 'unknown',
            'last_camera_time': 0.0,
            'last_inference_time': 0.0,
            'last_control_time': 0.0,
            'inference_count': 0,
            'start_time': time.time()
        }
        
        # 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(target=self.monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.get_logger().info("📊 최소 모니터링 노드 시작")
    
    def setup_ros(self):
        """ROS 퍼블리셔/서브스크라이버 설정"""
        
        # 카메라 상태 서브스크라이버
        self.camera_status_sub = self.create_subscription(
            String,
            '/camera/status',
            self.camera_status_callback,
            10
        )
        
        # 추론 결과 서브스크라이버
        self.inference_result_sub = self.create_subscription(
            String,
            '/inference/result',
            self.inference_result_callback,
            10
        )
        
        # 추론 상태 서브스크라이버
        self.inference_status_sub = self.create_subscription(
            String,
            '/inference/status',
            self.inference_status_callback,
            10
        )
        
        # 제어 상태 서브스크라이버
        self.control_status_sub = self.create_subscription(
            String,
            '/control/status',
            self.control_status_callback,
            10
        )
        
        # 모니터링 결과 퍼블리셔
        self.monitoring_result_pub = self.create_publisher(
            String,
            '/monitor/result',
            10
        )
        
        self.get_logger().info("✅ ROS 인터페이스 설정 완료")
    
    def camera_status_callback(self, msg):
        """카메라 상태 콜백"""
        try:
            data = json.loads(msg.data)
            self.monitoring_data['camera_status'] = data.get('status', 'unknown')
            self.monitoring_data['last_camera_time'] = time.time()
        except Exception as e:
            self.get_logger().error(f"❌ 카메라 상태 처리 오류: {e}")
    
    def inference_result_callback(self, msg):
        """추론 결과 콜백"""
        try:
            data = json.loads(msg.data)
            self.monitoring_data['inference_count'] = data.get('inference_count', 0)
            self.monitoring_data['last_inference_time'] = time.time()
        except Exception as e:
            self.get_logger().error(f"❌ 추론 결과 처리 오류: {e}")
    
    def inference_status_callback(self, msg):
        """추론 상태 콜백"""
        try:
            data = json.loads(msg.data)
            self.monitoring_data['inference_status'] = data.get('status', 'unknown')
        except Exception as e:
            self.get_logger().error(f"❌ 추론 상태 처리 오류: {e}")
    
    def control_status_callback(self, msg):
        """제어 상태 콜백"""
        try:
            data = json.loads(msg.data)
            self.monitoring_data['control_status'] = data.get('status', 'unknown')
            self.monitoring_data['last_control_time'] = time.time()
        except Exception as e:
            self.get_logger().error(f"❌ 제어 상태 처리 오류: {e}")
    
    def print_monitoring_summary(self):
        """모니터링 요약 출력"""
        try:
            runtime = time.time() - self.monitoring_data['start_time']
            
            # 상태 이모지
            status_emoji = {
                'ready': '🟢',
                'processing': '🔄',
                'error': '🔴',
                'unknown': '❓'
            }
            
            print("\n" + "="*60)
            print("📊 최소 모니터링 대시보드")
            print("="*60)
            
            # 노드별 상태
            camera_emoji = status_emoji.get(self.monitoring_data['camera_status'], '❓')
            inference_emoji = status_emoji.get(self.monitoring_data['inference_status'], '❓')
            control_emoji = status_emoji.get(self.monitoring_data['control_status'], '❓')
            
            print(f"📷 카메라 노드: {camera_emoji} {self.monitoring_data['camera_status'].upper()}")
            print(f"🧠 추론 노드: {inference_emoji} {self.monitoring_data['inference_status'].upper()}")
            print(f"🤖 제어 노드: {control_emoji} {self.monitoring_data['control_status'].upper()}")
            
            # 통계
            print(f"\n📈 통계:")
            print(f"   실행 시간: {runtime:.1f}초")
            print(f"   추론 횟수: {self.monitoring_data['inference_count']}회")
            
            # 마지막 업데이트 시간
            time_since_camera = time.time() - self.monitoring_data['last_camera_time']
            time_since_inference = time.time() - self.monitoring_data['last_inference_time']
            time_since_control = time.time() - self.monitoring_data['last_control_time']
            
            print(f"\n⏰ 마지막 업데이트:")
            print(f"   카메라: {time_since_camera:.1f}초 전")
            print(f"   추론: {time_since_inference:.1f}초 전")
            print(f"   제어: {time_since_control:.1f}초 전")
            
            # 시스템 상태 판단
            all_ready = all([
                self.monitoring_data['camera_status'] == 'ready',
                self.monitoring_data['inference_status'] == 'ready',
                self.monitoring_data['control_status'] == 'ready'
            ])
            
            system_status = "🟢 정상" if all_ready else "🟡 부분 정상" if any(s == 'ready' for s in [self.monitoring_data['camera_status'], self.monitoring_data['inference_status'], self.monitoring_data['control_status']]) else "🔴 오류"
            print(f"\n🏁 시스템 상태: {system_status}")
            
            print("="*60)
            
        except Exception as e:
            self.get_logger().error(f"❌ 모니터링 요약 출력 오류: {e}")
    
    def monitoring_worker(self):
        """모니터링 워커 스레드"""
        while rclpy.ok():
            try:
                # 모니터링 요약 출력
                self.print_monitoring_summary()
                
                # 모니터링 결과 발행
                self.publish_monitoring_result()
                
                # 3초마다 업데이트
                time.sleep(3.0)
                
            except Exception as e:
                self.get_logger().error(f"❌ 모니터링 워커 오류: {e}")
                time.sleep(3.0)
    
    def publish_monitoring_result(self):
        """모니터링 결과 발행"""
        try:
            monitoring_result = {
                'timestamp': time.time(),
                'runtime': time.time() - self.monitoring_data['start_time'],
                'camera_status': self.monitoring_data['camera_status'],
                'inference_status': self.monitoring_data['inference_status'],
                'control_status': self.monitoring_data['control_status'],
                'inference_count': self.monitoring_data['inference_count'],
                'last_camera_time': self.monitoring_data['last_camera_time'],
                'last_inference_time': self.monitoring_data['last_inference_time'],
                'last_control_time': self.monitoring_data['last_control_time']
            }
            
            msg = String()
            msg.data = json.dumps(monitoring_result)
            self.monitoring_result_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ 모니터링 결과 발행 오류: {e}")
    
    def get_monitoring_data(self):
        """모니터링 데이터 반환"""
        return self.monitoring_data.copy()

def main(args=None):
    rclpy.init(args=args)
    
    node = MinimalMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
