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
import os  # 메모리 측정용
import json # 데이터 저장용
import numpy as np
import cv2
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

# 메모리 프로파일링 자료구조용
try:
    import psutil
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("Warning: psutil not found. Memory profiling disabled.")

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera_interfaces.srv import GetImage

# 로컬 추론 엔진 import
# ROS2 실행 환경 대응을 위해 절대 경로 추가
project_root = "/home/soda/vla"
if project_root not in sys.path:
    sys.path.append(project_root)

# 이제 import 가능
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
        
        # =================================================================
        # 📊 [설정] 메모리 프로파일링 모드 (사용자 요청 기능)
        # =================================================================
        self.ENABLE_MEMORY_PROFILING = True  # True로 설정 시 그래프 생성
        self.memory_stats = {'step': [], 'cpu': [], 'gpu': []}
        
        # 🏎️ [설정] 이동 궤적 시각화
        self.trajectory_stats = {'step': [], 'x': [0.0], 'y': [0.0], 'action_x': [], 'action_y': []}
        self.profiling_dir = Path("/home/soda/vla/docs/memory_analysis")
        
        if self.ENABLE_MEMORY_PROFILING and PROFILING_AVAILABLE:
            try:
                self.profiling_dir.mkdir(parents=True, exist_ok=True)
                self.get_logger().info(f"📊 메모리 분석 활성화됨: {self.profiling_dir}")
            except Exception as e:
                self.get_logger().warn(f"⚠️ 프로파일링 폴더 생성 실패: {e}")
                self.ENABLE_MEMORY_PROFILING = False
        else:
            self.ENABLE_MEMORY_PROFILING = False
        
        # 🔧 [설정] 좌표계 보정 (데이터셋 분석 결과 기반)
        # 분석 결과: 데이터는 X>0=FORWARD, 하지만 모델은 X<0 출력
        self.INVERT_X_AXIS = True  # X축 부호 반전 (후진→전진 수정)
        self.INVERT_Y_AXIS = False # Y축은 유지 (현재 정상)
        
        # 추론 설정
        self.declare_parameter('auto_start', False)
        self.inference_active = self.get_parameter('auto_start').get_parameter_value().bool_value
        self.current_instruction = "가장 왼쪽 외곽으로 돌아 컵까지 가세요"  # 학습 데이터와 동일 (Korean)
       
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
        # 서비스 대기 (Blocking 제거, Warning만 출력) - Dry Run 테스트용
        if not self.get_image_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('⚠️ get_image_service를 찾을 수 없습니다. 나중에 다시 시도합니다.')
        else:
            self.get_logger().info('✅ Camera service 연결 완료!')
        
        # 추론 엔진 초기화 (lazy loading -> init loading)
        self.inference_engine = None
        self.image_buffer = None
        self.config = None
        
        # 추론 주기 (300ms)
        self.inference_interval = 0.3
        self.last_inference_time = 0
        
        # 액션 증폭 계수 (모델 출력이 너무 작을 경우 증폭)
        # 액션 증폭 계수 (모델 출력이 너무 작을 경우 증폭)
        self.declare_parameter('action_gain', 60.0)
        self.action_gain = self.get_parameter('action_gain').get_parameter_value().double_value
        self.get_logger().info(f"⚡ Action Gain: {self.action_gain}")
        
        # 통계
        self.total_inferences = 0
        self.inference_times = []
        
        self.get_logger().info("🤖 Mobile VLA 추론 노드 준비 완료!")
        
        # 모델 로딩 시도
        self.init_inference_engine()
        
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
                # Fine-tuned 모델 체크포인트 (절대 경로 필수)
                checkpoint_path="/home/soda/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt",
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
            # GetImage.srv에는 success 필드가 없음, image 필드만 존재
            if response and response.image.header.frame_id:  # 간단한 유효성 검사
                # ROS Image -> OpenCV
                cv_image = self.cv_bridge.imgmsg_to_cv2(
                    response.image, 
                    desired_encoding='bgr8'
                )
                # BGR -> RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                return rgb_image
            else:
                self.get_logger().warn("⚠️ 이미지 취득 실패 (빈 응답)")
                return None
                
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 취득 에러: {e}")
            return None
    
    def execute_step_action(self, action: np.ndarray):
        """로봇 제어 명령 발행 (data_collector 스타일: move -> sleep -> stop)"""
        linear_x = float(action[0])
        linear_y = float(action[1])
        
        # 🔧 좌표계 보정 (2026-01-02 분석 결과 반영)
        if self.INVERT_X_AXIS:
            linear_x = -linear_x
        if self.INVERT_Y_AXIS:
            linear_y = -linear_y
        
        # 1. ROS2 Twist 메시지 발행 (시각화용)
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        self.cmd_pub.publish(msg)
        
        # 2. 하드웨어 드라이버 제어
        # Deadzone: 0.15 (data_collector는 0.1 사용)
        if abs(linear_x) < 0.15 and abs(linear_y) < 0.15:
            self.get_logger().info(f"🚫 [Deadzone] 입력 미달: x={linear_x:.4f}, y={linear_y:.4f} (기준 0.15)")
            if ROBOT_AVAILABLE and self.driver:
                self.driver.stop()
            return

        if ROBOT_AVAILABLE and self.driver is not None:
            try:
                # 각도 계산 (atan2(y, x))
                angle = np.degrees(np.arctan2(linear_y, linear_x))
                if angle < 0:
                    angle += 360
                
                int_angle = int(angle)
                int_angle = int(angle)
                move_duration = 0.4
                
                self.get_logger().info(f"🏎️ 이동 실행: 각도={int_angle}°, 입력(x={linear_x:.2f}, y={linear_y:.2f})")
                
                # 끊어서 이동 (Move -> Sleep -> Stop)
                self.driver.move(int_angle, self.throttle)
                time.sleep(move_duration)
                self.driver.stop()
                
            except Exception as e:
                self.get_logger().error(f"❌ 하드웨어 제어 에러: {e}")
        else:
            # 시뮬레이션 모드에서도 타이밍 맞춤
            time.sleep(0.4)
    
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
        
        self.get_logger().info("🎯 추론 루프 시작!")
        
        executed_frames_total = 0
        target_frames = 18
        
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
                
                # 액션 증폭 (Gain 적용)
                denorm_actions = denorm_actions * self.action_gain
                
                inference_time = (time.time() - start_time) * 1000
                
                # 로깅: 모델 스펙 및 진행 상황
                chunk_len = len(denorm_actions)
                self.get_logger().info(
                    f"🔮 추론 수행 (Window: {self.config.window_size}, Chunk: {chunk_len}) | "
                    f"누적 진행: {executed_frames_total}/{target_frames}"
                )

                # 4. 로봇 제어 (청크 실행 및 누적 카운트)
                for i, action in enumerate(denorm_actions):
                    if not self.inference_active:
                        break
                        
                    if executed_frames_total >= target_frames:
                        self.get_logger().info(f"🛑 목표 프레임({target_frames}) 도달 완료. 종료합니다.")
                        self.inference_active = False
                        break

                    self.execute_step_action(action)
                    executed_frames_total += 1
                    
                    # 📊 메모리 기록 (사용자 요청)
                    self.record_memory_usage(executed_frames_total)
                    
                    # 🏎️ 궤적 기록
                    self.record_trajectory(executed_frames_total, float(action[0]), float(action[1]))

                    self.get_logger().info(
                        f"🏃 [{executed_frames_total}/{target_frames}] 완료"
                    )
                    # move 내부에 sleep(0.3) 포함됨
                
                # 목표 달성 시 루프 탈출
                if not self.inference_active or executed_frames_total >= target_frames:
                     self.stop_robot()
                     self.inference_active = False
                     break

                # 5. 통계
                self.total_inferences += 1
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)

                self.last_inference_time = current_time
                
            except Exception as e:
                self.get_logger().error(f"❌ 추론 루프 에러: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
        
        # 종료 시 정지
        # 종료 시 정지
        self.stop_robot()
        
        # 📊 [데이터 저장] 메모리 및 궤적 데이터 저장 (18스텝 완료 후)
        if self.ENABLE_MEMORY_PROFILING:
             self.save_inference_data_json()

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
        """시나리오별 언어 지시문 설정 (학습 데이터와 동일: 한국어)"""
        # RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:L151-160과 일치
        scenarios = {
            '1': "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
            '2': "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
            '3': "가장 왼쪽 외곽으로 돌아 컵까지 가세요",  # 2box는 동일 instruction 사용
            '4': "가장 오른쪽 외곽으로 돌아 컵까지 가세요"
        }
        
        if scenario_num in scenarios:
            self.current_instruction = scenarios[scenario_num]
            self.get_logger().info(f"📝 지시문 변경: {self.current_instruction}")
    
    def record_memory_usage(self, step: int):
        """현재 시스템/GPU 메모리 사용량을 기록"""
        if not self.ENABLE_MEMORY_PROFILING:
            return

        try:
            # CPU RAM
            process = psutil.Process(os.getpid())
            cpu_mem_gb = process.memory_info().rss / (1024 ** 3)  # GB 단위
            
            # GPU VRAM (Torch 기준)
            gpu_mem_gb = 0.0
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            
            self.memory_stats['step'].append(step)
            self.memory_stats['cpu'].append(cpu_mem_gb)
            self.memory_stats['gpu'].append(gpu_mem_gb)
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ 메모리 측정 실패: {e}")

    def save_inference_data_json(self):
        """추론 데이터를 JSON으로 저장 (그래프는 별도 스크립트로 생성)"""
        if not self.ENABLE_MEMORY_PROFILING:
            return

        try:
            # 데이터 병합
            data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "memory_stats": self.memory_stats,
                "trajectory_stats": {
                    "step": self.trajectory_stats['step'],
                    "x": self.trajectory_stats['x'],
                    "y": self.trajectory_stats['y'],
                    "action_x": self.trajectory_stats['action_x'],
                    "action_y": self.trajectory_stats['action_y']
                }
            }
            
            # JSON 파일 저장
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_data_{timestamp_str}.json"
            save_path = self.profiling_dir / filename
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.get_logger().info(f"💾 [데이터 저장] JSON 저장 완료: {save_path}")
            
            # 🖼️ 그래프 생성 스크립트 자동 실행 (subprocess)
            try:
                import subprocess
                vis_script = Path("/home/soda/vla/scripts/visualize_inference_log.py")
                if vis_script.exists():
                    # 백그라운드 실행을 위해 Popen 사용. 노드 종료 후에도 실행되도록.
                    subprocess.Popen(["python3", str(vis_script), str(save_path)], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.get_logger().info(f"🚀 [자동 실행] 그래프 생성 프로세스 시작됨")
                else:
                    self.get_logger().warn(f"⚠️ 시각화 스크립트 없음: {vis_script}")
            except Exception as e:
                self.get_logger().error(f"❌ 시각화 스크립트 실행 실패: {e}")

            # 초기화
            self.memory_stats = {'step': [], 'cpu': [], 'gpu': []}
            self.trajectory_stats = {'step': [], 'x': [0.0], 'y': [0.0], 'action_x': [], 'action_y': []}
            
        except Exception as e:
            self.get_logger().error(f"❌ 데이터 저장 실패: {e}")

    def record_trajectory(self, step: int, action_x: float, action_y: float):
        """이동 궤적 데이터 기록"""
        # 현재 위치 (이전 위치 + 이동량)
        # 로봇 좌표계: X(전진), Y(좌측)
        # 0.4초 이동 가정 (속도 * 시간 = 거리)
        # 실제 거리는 속도에 비례하지만, 여기서는 상대적인 궤적 형태만 확인
        # 모델 출력값 자체가 속도(m/s) 혹은 정규화된 값이므로 이를 변위로 간주
        
        last_x = self.trajectory_stats['x'][-1]
        last_y = self.trajectory_stats['y'][-1]
        
        # 누적 이동량 계산
        new_x = last_x + action_x
        new_y = last_y + action_y
        
        self.trajectory_stats['step'].append(step)
        self.trajectory_stats['x'].append(new_x)
        self.trajectory_stats['y'].append(new_y)
        self.trajectory_stats['action_x'].append(action_x)
        self.trajectory_stats['action_y'].append(action_y)

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
        elif key == 'm':
            self.test_simple_move()

    def test_simple_move(self):
        """간단한 로봇 이동 테스트 (0.4초 전진)"""
        if self.inference_active:
            self.get_logger().warn("⚠️ 추론 중에는 테스트할 수 없습니다. S키로 멈추고 시도하세요.")
            return

        if not self.driver:
            self.get_logger().warn("⚠️ 로봇 드라이버가 없습니다.")
            return

        self.get_logger().info("🏎️ 로봇 테스트: 0.4초 전진...")
        try:
            # simple_move_robot.py 로직
            angle = 90  # 전진
            throttle = 50
            
            # 이동
            self.driver.move(angle, throttle)
            time.sleep(0.4)
            # 정지
            self.driver.stop()
            self.get_logger().info("🛑 로봇 테스트 완료")
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 테스트 실패: {e}")
            self.driver.stop()
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
