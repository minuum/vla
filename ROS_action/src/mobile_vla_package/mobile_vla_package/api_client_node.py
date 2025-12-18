#!/usr/bin/env python3
"""
Mobile VLA API Client - RoboVLMs 학습 로직 기반 추론 클라이언트

추론 프로세스:
1. 시작 시 (0,0) 정지 상태
2. 카메라에서 이미지 획득
3. API 서버로 이미지 + instruction 전송
4. 응답으로 받은 2DOF action [linear_x, linear_y] 실행
5. 반복 (10Hz)

키보드 제어:
- W/A/S/D/Q/E/Z/C: 수동 제어 (추론 OFF)
- I: 추론 모드 ON/OFF
- T: 시스템 테스트
- R: 추론 테스트 (단일)
- Space: 정지
- Ctrl+C: 종료
"""
import rclpy
from rclpy.node import Node
import sys, tty, termios
import os
import time
import threading
import requests
import base64
import cv2
import numpy as np
from collections import deque

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from camera_interfaces.srv import GetImage

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available. Using ROS only.")
    ROBOT_AVAILABLE = False


class MobileVLAAPIClient(Node):
    def __init__(self):
        super().__init__('mobile_vla_api_client')
        
        # ==================== RoboVLMs 학습 설정 기준 ====================
        # 학습 시 사용된 액션: [linear_x, linear_y] (2DOF)
        # angular_z는 항상 0 (non-holonomic 가정)
        # 액션 범위: 약 -1.15 ~ 1.15 (학습 데이터 기준)
        # =================================================================
        
        # data_collector 스타일 액션 매핑 (학습 데이터와 동일)
        self.WASD_TO_CONTINUOUS = {
            'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},   # 전진
            'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},  # 좌측
            's': {"linear_x": -1.15, "linear_y": 0.0, "angular_z": 0.0}, # 후진
            'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0}, # 우측
            'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0}, # 전진-좌측
            'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},# 전진-우측
            'z': {"linear_x": -1.15, "linear_y": 1.15, "angular_z": 0.0},# 후진-좌측
            'c': {"linear_x": -1.15, "linear_y": -1.15, "angular_z": 0.0},# 후진-우측
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}    # 정지
        }
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        
        # 기본 instruction (학습 시 사용된 것과 동일)
        self.default_instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
        
        # 추론 상태
        self.inference_mode = False
        self.inference_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.latency_history = deque(maxlen=100)
        self.last_action = [0.0, 0.0]  # 마지막 액션 저장
        
        # API 설정
        self.api_server_url = os.getenv("VLA_API_SERVER", "http://localhost:8000")
        self.api_key = os.getenv("VLA_API_KEY", "")
        
        # 로봇 드라이버 (하드웨어)
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50
        else:
            self.driver = None
        
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ROS Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 서비스 클라이언트 (카메라)
        try:
            self.get_image_client = self.create_client(GetImage, 'get_image_service')
            
            while not self.get_image_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info('get_image_service 서비스 대기 중...')
                if not rclpy.ok():
                    self.get_logger().error("ROS2 컨텍스트가 종료됨.")
                    sys.exit()
            
            self.get_logger().info('✅ get_image_service 서비스 연결 완료!')
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 클라이언트 시작 실패: {e}")
            rclpy.shutdown()
        
        self.cv_bridge = CvBridge()
        
        # 추론 타이머 (100ms = 10Hz, RoboVLMs 기준)
        self.inference_timer = self.create_timer(0.1, self.inference_timer_callback)
        
        # ==================== 시작 시 정지 상태 ====================
        self.publish_cmd_vel(self.STOP_ACTION, "init")
        self.get_logger().info("🛑 시작 시 정지 상태 (0, 0)")
        
        # 환영 메시지
        self.print_welcome()
        
        # 키보드 스레드
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        self.get_logger().info(f"✅ 시스템 준비 완료! API: {self.api_server_url}")
    
    def print_welcome(self):
        """환영 메시지"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 Mobile VLA API Client - RoboVLMs Style")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("  W/A/S/D: 전후좌우 이동 (수동)")
        self.get_logger().info("  Q/E/Z/C: 대각선 이동 (수동)")
        self.get_logger().info("  Space: 정지")
        self.get_logger().info("  I: 추론 모드 ON/OFF (자동 주행)")
        self.get_logger().info("  R: 추론 테스트 (단일 실행)")
        self.get_logger().info("  T: 시스템 테스트")
        self.get_logger().info("  Ctrl+C: 종료")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"🎮 추론 모드: {'🟢 ON' if self.inference_mode else '🔴 OFF'}")
    
    def keyboard_loop(self):
        """키보드 입력 루프"""
        while rclpy.ok():
            key = self.get_key()
            self.handle_key_input(key)
            time.sleep(0.01)
    
    def get_key(self):
        """키 입력"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def handle_key_input(self, key: str):
        """키 입력 처리"""
        if key == '\x03':  # Ctrl+C
            sys.exit()
        
        elif key == 'i':
            # 추론 모드 토글
            self.inference_mode = not self.inference_mode
            status = "🟢 ON" if self.inference_mode else "🔴 OFF"
            self.get_logger().info(f"🎮 추론 모드: {status}")
            if not self.inference_mode:
                # 추론 OFF 시 정지
                self.publish_cmd_vel(self.STOP_ACTION, "inference_off")
                self.get_logger().info("🛑 추론 OFF - 정지")
        
        elif key == 'r':
            # 추론 테스트 (단일 실행)
            self.run_single_inference_test()
        
        elif key == 't':
            # 시스템 테스트
            self.run_system_test()
        
        elif key in self.WASD_TO_CONTINUOUS:
            # 수동 제어 (추론 OFF 시에만)
            if not self.inference_mode:
                action = self.WASD_TO_CONTINUOUS[key]
                self.publish_cmd_vel(action, f"manual_{key}")
                if key != ' ':
                    self.get_logger().info(
                        f"🎮 수동: [{key.upper()}] x={action['linear_x']:.2f}, y={action['linear_y']:.2f}"
                    )
            else:
                self.get_logger().info("⚠️  추론 모드 ON - 수동 제어 불가")
    
    def publish_cmd_vel(self, action: dict, source: str):
        """
        cmd_vel 발행 (RoboVLMs 학습 데이터와 동일한 형식)
        
        action: {"linear_x": float, "linear_y": float, "angular_z": float}
        - linear_x: 전/후 이동 (-1.15 ~ 1.15)
        - linear_y: 좌/우 이동 (-1.15 ~ 1.15)
        - angular_z: 회전 (항상 0, 학습 데이터 기준)
        """
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action.get("angular_z", 0.0))
        
        # ROS2 publish
        self.cmd_pub.publish(twist)
        
        # 마지막 액션 저장
        self.last_action = [action["linear_x"], action["linear_y"]]
        
        # 하드웨어 제어 (data_collector와 동일)
        if ROBOT_AVAILABLE and self.driver:
            try:
                if any(abs(v) > 0.1 for v in action.values()):
                    if abs(action.get("angular_z", 0.0)) > 0.1:
                        spin_speed = int(action["angular_z"] * self.throttle)
                        self.driver.spin(spin_speed)
                    elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                        angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                        self.driver.move(int(angle), self.throttle)
                else:
                    self.driver.stop()
            except Exception as e:
                if not hasattr(self, '_hw_error_logged'):
                    self._hw_error_logged = True
                    self.get_logger().warn(f"하드웨어 제어 실패: {e}")
    
    def get_latest_image_via_service(self, max_retries: int = 3):
        """서비스로 카메라 이미지 가져오기"""
        for attempt in range(max_retries):
            try:
                self.get_logger().info(f"📸 [시도 {attempt+1}/{max_retries}] 이미지 서비스 요청 중...")
                request = GetImage.Request()
                future = self.get_image_client.call_async(request)
                
                # 타임아웃 1.0초로 단축 (너무 오래 대기 방지)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                
                if future.done():
                    response = future.result()
                    
                    if response is None:
                        self.get_logger().warn(f"📸 [시도 {attempt+1}] 응답이 None입니다.")
                        continue
                        
                    data_len = len(response.image.data)
                    height = response.image.height
                    width = response.image.width
                    encoding = response.image.encoding
                    
                    self.get_logger().info(f"📸 [시도 {attempt+1}] 서비스 응답 수신: {width}x{height}, {encoding}, {data_len} bytes")
                    
                    if data_len > 0:
                        cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                        self.get_logger().info(f"✅ 이미지 디코딩 성공: {cv_image.shape}")
                        return cv_image
                    else:
                        self.get_logger().warn(f"⚠️ [시도 {attempt+1}] 이미지 데이터 크기가 0입니다.")
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                else:
                    self.get_logger().warn(f"⚠️ [시도 {attempt+1}] 서비스 호출 타임아웃 (1.0s)")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
            except Exception as e:
                self.get_logger().error(f"❌ [시도 {attempt+1}] 예외 발생: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        self.get_logger().error(f"❌ {max_retries}번 시도 모두 실패")
        return None
    
    def inference_timer_callback(self):
        """
        추론 타이머 (10Hz)
        
        RoboVLMs 추론 프로세스:
        1. 카메라에서 이미지 획득
        2. 224x224로 리사이즈 (모델 입력 크기)
        3. Base64 인코딩
        4. API 서버로 전송 (이미지 + instruction)
        5. 응답: action [linear_x, linear_y]
        6. 로봇에 action 적용
        """
        if not self.inference_mode:
            return
        
        # 1. 카메라 이미지 획득
        image = self.get_latest_image_via_service()
        
        if image is None:
            return
        
        # 첫 이미지 수신 로그
        if self.inference_count == 0 and self.success_count == 0:
            self.get_logger().info(f"📸 첫 이미지 수신! Shape: {image.shape}")
        
        # 2. 224x224로 리사이즈 (RoboVLMs 모델 입력 크기)
        resized = cv2.resize(image, (224, 224))
        
        # 3-6. API 추론 호출
        self.run_inference(resized)
    
    def run_inference(self, image: np.ndarray):
        """
        API 서버에 추론 요청
        
        Args:
            image: 224x224 BGR 이미지
        
        API 서버 응답:
            - action: [linear_x, linear_y] (2DOF)
            - latency_ms: 추론 시간
            - model_name: 모델 이름
            - chunk_size: fwd_pred_next_n 값
        """
        try:
            # Base64 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # API 요청 헤더
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            # API 요청
            response = requests.post(
                f"{self.api_server_url}/predict",
                json={
                    "image": img_b64,
                    "instruction": self.default_instruction
                },
                headers=headers,
                timeout=1.0
            )
            
            if response.status_code == 200:
                data = response.json()
                action_list = data["action"]  # [linear_x, linear_y]
                latency = data["latency_ms"]
                
                # 통계 업데이트
                self.inference_count += 1
                self.success_count += 1
                self.latency_history.append(latency)
                
                # 액션 변환 (API 응답 → cmd_vel 형식)
                action = {
                    "linear_x": float(action_list[0]),
                    "linear_y": float(action_list[1]),
                    "angular_z": 0.0  # 학습 데이터 기준: angular_z 항상 0
                }
                
                # 로봇에 액션 적용
                self.publish_cmd_vel(action, "inference")
                
                # 10회마다 로그
                if self.inference_count % 10 == 0:
                    avg = sum(self.latency_history) / len(self.latency_history)
                    self.get_logger().info(
                        f"✅ #{self.inference_count} | Latency: {latency:.0f}ms (avg: {avg:.0f}ms) | "
                        f"Action: [{action_list[0]:.3f}, {action_list[1]:.3f}]"
                    )
            else:
                self.fail_count += 1
                if self.fail_count % 10 == 0:
                    self.get_logger().error(f"API 에러: {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.fail_count += 1
            if self.fail_count % 10 == 0:
                self.get_logger().warn(f"⚠️ API 타임아웃 (실패: {self.fail_count}회)")
        except Exception as e:
            self.fail_count += 1
            if self.fail_count % 10 == 0:
                self.get_logger().error(f"추론 실패: {e}")
    
    def run_single_inference_test(self):
        """
        추론 테스트 (R 키)
        
        단일 추론을 실행하고 결과를 상세히 출력
        """
        self.get_logger().info("=" * 60)
        self.get_logger().info("🧪 추론 테스트 (단일)")
        self.get_logger().info("=" * 60)
        
        # 1. 카메라 이미지 획득
        self.get_logger().info("📸 카메라 이미지 획득 중...")
        image = self.get_latest_image_via_service()
        
        if image is None:
            self.get_logger().error("❌ 이미지 획득 실패")
            return
        
        self.get_logger().info(f"✅ 이미지: {image.shape}")
        
        # 2. 리사이즈
        resized = cv2.resize(image, (224, 224))
        self.get_logger().info(f"✅ 리사이즈: {resized.shape}")
        
        # 3. API 요청
        self.get_logger().info(f"📡 API 서버: {self.api_server_url}")
        self.get_logger().info(f"📝 Instruction: {self.default_instruction[:50]}...")
        
        try:
            _, buffer = cv2.imencode('.jpg', resized)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_server_url}/predict",
                json={
                    "image": img_b64,
                    "instruction": self.default_instruction
                },
                headers=headers,
                timeout=5.0
            )
            total_time = (time.time() - start_time) * 1000
            
            self.get_logger().info(f"📊 HTTP 상태: {response.status_code}")
            self.get_logger().info(f"⏱️ 총 응답 시간: {total_time:.0f}ms")
            
            if response.status_code == 200:
                data = response.json()
                self.get_logger().info(f"✅ 추론 성공!")
                self.get_logger().info(f"   Action: {data['action']}")
                self.get_logger().info(f"   Model Latency: {data['latency_ms']:.0f}ms")
                self.get_logger().info(f"   Model: {data.get('model_name', 'unknown')}")
                self.get_logger().info(f"   Chunk Size: {data.get('chunk_size', 'unknown')}")
                
                # 액션 적용 (선택)
                self.get_logger().info(f"🎮 액션 적용: linear_x={data['action'][0]:.3f}, linear_y={data['action'][1]:.3f}")
                action = {
                    "linear_x": float(data['action'][0]),
                    "linear_y": float(data['action'][1]),
                    "angular_z": 0.0
                }
                self.publish_cmd_vel(action, "inference_test")
            else:
                self.get_logger().error(f"❌ API 에러: {response.status_code}")
                self.get_logger().error(f"   {response.text}")
                
        except Exception as e:
            self.get_logger().error(f"❌ 추론 실패: {e}")
        
        self.get_logger().info("=" * 60)
    
    def run_system_test(self):
        """시스템 테스트 (T 키)"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("🧪 시스템 테스트")
        self.get_logger().info("=" * 60)
        
        # 1. 카메라
        image = self.get_latest_image_via_service()
        if image is not None:
            self.get_logger().info(f"✅ 카메라: 정상 (Shape: {image.shape})")
        else:
            self.get_logger().info("❌ 카메라: 이미지 없음")
        
        # 2. API 서버
        try:
            resp = requests.get(f"{self.api_server_url}/health", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                model_loaded = data.get("model_loaded", False)
                device = data.get("device", "unknown")
                
                if model_loaded:
                    self.get_logger().info(f"✅ API 서버: 연결됨, 모델 로드됨 ({device})")
                else:
                    self.get_logger().info(f"⚠️ API 서버: 연결됨, 모델 미로드 ({device})")
                    self.get_logger().info("   → Billy 서버에서 모델 시작 필요!")
            else:
                self.get_logger().info(f"⚠️ API 서버: 응답 이상 ({resp.status_code})")
        except:
            self.get_logger().info("❌ API 서버: 연결 실패")
            self.get_logger().info("   → SSH 터널 확인: ssh -N -f -L 8000:localhost:8000 billy@100.86.152.29 -p 10022")
        
        # 3. 통계
        self.get_logger().info(f"📊 추론: {self.inference_count}회 (성공: {self.success_count}, 실패: {self.fail_count})")
        if self.latency_history:
            avg = sum(self.latency_history) / len(self.latency_history)
            self.get_logger().info(f"📈 평균 Latency: {avg:.0f}ms")
        self.get_logger().info(f"🎯 마지막 액션: [{self.last_action[0]:.3f}, {self.last_action[1]:.3f}]")
        
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = MobileVLAAPIClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
