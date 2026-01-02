#!/usr/bin/env python3
"""
Mobile VLA 추론 노드 (ROS2)
mobile_vla_data_collector.py 구조 기반

추론 흐름:
1. 카메라 이미지 취득 (get_image_service)
2. Mobile VLA 모델 추론 (로컬)
3. 로봇 제어 명령 발행 (/cmd_vel)
"""

import rclpy
from rclpy.node import Node
import sys, tty, termios
import time
import numpy as np
import cv2
import threading
from pathlib import Path
from typing import Optional

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera_interfaces.srv import GetImage

# 로컬 추론 엔진 import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.robovlms_mobile_vla_inference import (
    MobileVLAConfig,
    RoboVLMsInferenceEngine,
    ImageBuffer
)

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available")
    ROBOT_AVAILABLE = False


class MobileVLAInferenceNode(Node):
    """Mobile VLA 추론 노드"""
    
    def __init__(self):
        super().__init__('mobile_vla_inference_node')
        
        # 추론 설정
        self.inference_active = False
        self.current_instruction = "Navigate to the target"
       
        # 로봇 제어
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50
        else:
            self.driver = None
        
        # ROS2 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cv_bridge = CvBridge()
        
        # Camera service
        self.get_image_client = self.create_client(GetImage, 'get_image_service')
        while not self.get_image_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('get_image_service 대기 중...')
            if not rclpy.ok():
                sys.exit()
        
        self.get_logger().info('✅ Camera service 연결 완료!')
        
        # 추론 엔진 초기화 (lazy loading)
        self.inference_engine = None
        self.image_buffer = None
        self.config = None
        
        # 추론 주기 (300ms)
        self.inference_interval = 0.3
        self.last_inference_time = 0
        
        # 통계
        self.total_inferences = 0
        self.inference_times = []
        
        self.get_logger().info("🤖 Mobile VLA 추론 노드 준비 완료!")
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("   S: 추론 시작/중지")
        self.get_logger().info("   1-4: 시나리오 선택 (언어 지시문 변경)")
        self.get_logger().info("   P: 통계 표시")
        self.get_logger().info("   Ctrl+C: 종료")
        
        # 키보드 스레드
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # 추론 스레드
        self.inference_thread = None
        
    def init_inference_engine(self):
        """추론 엔진 초기화 (첫 추론 시)"""
        if self.inference_engine is not None:
            return True
        
        self.get_logger().info("🚀 추론 엔진 초기화 중...")
        
        try:
        # Config 설정
            self.config = MobileVLAConfig(
                # Fine-tuned 모델 체크포인트
                checkpoint_path="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt",
                window_size=2,
                fwd_pred_next_n=10,
                use_abs_action=True,
                denormalize_strategy="safe",
                max_linear_x=1.15,  # data_collector와 일치 (실제 로봇 속도 범위)
                max_linear_y=1.15
            )
            
            # 추론 엔진 생성
            self.inference_engine = RoboVLMsInferenceEngine(self.config)
            
            # 모델 로드
            if not self.inference_engine.load_model():
                self.get_logger().error("❌ 모델 로드 실패")
                return False
            
            # 이미지 버퍼 생성
            self.image_buffer = ImageBuffer(
                window_size=self.config.window_size,
                image_size=self.config.image_size
            )
            
            self.get_logger().info("✅ 추론 엔진 준비 완료!")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 추론 엔진 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_camera_image(self) -> Optional[np.ndarray]:
        """카메라 이미지 취득 (data_collector와 동일한 방식)"""
        try:
            request = GetImage.Request()
            future = self.get_image_client.call_async(request)
            
            # 동기 대기 (최대 1초)
            start_time = time.time()
            while not future.done():
                rclpy.spin_once(self, timeout_sec=0.01)
                if time.time() - start_time > 1.0:
                    self.get_logger().warn("⚠️ 이미지 취득 타임아웃")
                    return None
            
            response = future.result()
            if response and response.success:
                # ROS Image -> OpenCV
                cv_image = self.cv_bridge.imgmsg_to_cv2(
                    response.image, 
                    desired_encoding='bgr8'
                )
                # BGR -> RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                return rgb_image
            else:
                self.get_logger().warn("⚠️ 이미지 취득 실패")
                return None
                
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 취득 에러: {e}")
            return None
    
    def publish_cmd_vel(self, action: np.ndarray):
        """로봇 제어 명령 발행 (data_collector와 동일한 방식)"""
        linear_x = float(action[0])
        linear_y = float(action[1])
        
        # 1. ROS2 Twist 메시지 발행
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        self.cmd_pub.publish(msg)
        
        # 2. 하드웨어 드라이버 제어 (data_collector 방식)
        # 속도 크기가 아닌 방향(angle)을 계산하여 고정 속도(throttle)로 이동
        if ROBOT_AVAILABLE and self.driver is not None:
            try:
                # 움직임이 미미하면 정지
                if abs(linear_x) < 0.1 and abs(linear_y) < 0.1:
                    self.driver.stop()
                    return

                # 각도 계산 (atan2(y, x))
                angle = np.degrees(np.arctan2(linear_y, linear_x))
                if angle < 0:
                    angle += 360
                
                # data_collector는 항상 고정된 throttle 사용
                # self.throttle은 __init__에서 50으로 설정됨
                self.driver.move(int(angle), self.throttle)
                
            except Exception as e:
                self.get_logger().error(f"❌ 하드웨어 제어 에러: {e}")
    
    def stop_robot(self):
        """로봇 정지"""
        msg = Twist()
        self.cmd_pub.publish(msg)
        
        if ROBOT_AVAILABLE and self.driver is not None:
            try:
                self.driver.stop()
            except:
                pass
    
    def inference_loop(self):
        """추론 루프 (별도 스레드)"""
        self.get_logger().info("🎯 추론 루프 시작!")
        
        # 추론 엔진 초기화
        if not self.init_inference_engine():
            self.get_logger().error("❌ 추론 엔진 초기화 실패")
            self.inference_active = False
            return
        
        while self.inference_active and rclpy.ok():
            try:
                current_time = time.time()
                
                # 추론 주기 체크
                if current_time - self.last_inference_time < self.inference_interval:
                    time.sleep(0.01)
                    continue
                
                # 1. 이미지 취득
                image = self.get_camera_image()
                if image is None:
                    self.get_logger().warn("⚠️ 이미지 없음, 건너뜀")
                    time.sleep(0.1)
                    continue
                
                # 2. 이미지 버퍼에 추가
                self.image_buffer.add_image(image)
                
                if not self.image_buffer.is_ready():
                    self.get_logger().info(f"⏳ 버퍼 채우는 중... ({len(self.image_buffer.buffer)}/{self.config.window_size})")
                    self.last_inference_time = current_time
                    continue
                
                # 3. 추론 실행
                start_time = time.time()
                images = self.image_buffer.get_images()
                
                actions, info = self.inference_engine.predict_action(
                    images,
                    self.current_instruction,
                    use_abs_action=True
                )
                
                # 정규화 해제
                denorm_actions = self.inference_engine.denormalize_action(actions)
                first_action = denorm_actions[0]
                
                inference_time = (time.time() - start_time) * 1000
                
                # 4. 로봇 제어
                self.publish_cmd_vel(first_action)
                
                # 5. 통계
                self.total_inferences += 1
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
                
                # 6. 로그
                self.get_logger().info(
                    f"✅ [{self.total_inferences}] "
                    f"액션: [{first_action[0]:.3f}, {first_action[1]:.3f}] | "
                    f"지연: {inference_time:.1f}ms | "
                    f"방향: {info.get('direction', 'N/A')}"
                )
                
                self.last_inference_time = current_time
                
            except Exception as e:
                self.get_logger().error(f"❌ 추론 루프 에러: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        # 종료 시 정지
        self.stop_robot()
        self.get_logger().info("🛑 추론 루프 종료")
    
    def start_inference(self):
        """추론 시작"""
        if self.inference_active:
            self.get_logger().warn("⚠️ 이미 추론 중입니다")
            return
        
        self.inference_active = True
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.get_logger().info("🚀 추론 시작!")
        self.get_logger().info(f"📝 지시문: {self.current_instruction}")
    
    def stop_inference(self):
        """추론 중지"""
        if not self.inference_active:
            self.get_logger().warn("⚠️ 추론이 실행되고 있지 않습니다")
            return
        
        self.inference_active = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2.0)
        
        self.stop_robot()
        self.get_logger().info("🛑 추론 중지!")
    
    def show_stats(self):
        """통계 표시"""
        if len(self.inference_times) == 0:
            self.get_logger().info("📊 통계: 아직 추론 없음")
            return
        
        avg_time = np.mean(self.inference_times)
        max_time = np.max(self.inference_times)
        min_time = np.min(self.inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("📊 추론 통계")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"총 추론 횟수: {self.total_inferences}")
        self.get_logger().info(f"평균 지연: {avg_time:.1f} ms")
        self.get_logger().info(f"최대 지연: {max_time:.1f} ms")
        self.get_logger().info(f"최소 지연: {min_time:.1f} ms")
        self.get_logger().info(f"평균 FPS: {fps:.2f}")
        self.get_logger().info(f"현재 지시문: {self.current_instruction}")
        self.get_logger().info("=" * 60)
    
    def set_instruction(self, scenario_num: str):
        """시나리오별 언어 지시문 설정"""
        scenarios = {
            '1': "Navigate around obstacles and reach the left bottle",
            '2': "Navigate around obstacles and reach the right bottle",
            '3': "Navigate around two boxes and reach the left bottle",
            '4': "Navigate around two boxes and reach the right bottle"
        }
        
        if scenario_num in scenarios:
            self.current_instruction = scenarios[scenario_num]
            self.get_logger().info(f"📝 지시문 변경: {self.current_instruction}")
    
    def keyboard_loop(self):
        """키보드 입력 루프"""
        while rclpy.ok():
            key = self.get_key()
            self.handle_key_input(key)
            time.sleep(0.01)
    
    def handle_key_input(self, key: str):
        """키 입력 처리"""
        if key == '\x03':  # Ctrl+C
            self.stop_inference()
            sys.exit()
        elif key == 's':
            if self.inference_active:
                self.stop_inference()
            else:
                self.start_inference()
        elif key in ['1', '2', '3', '4']:
            self.set_instruction(key)
        elif key == 'p':
            self.show_stats()
    
    def get_key(self):
        """키보드 입력 읽기 (data_collector와 동일)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MobileVLAInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
