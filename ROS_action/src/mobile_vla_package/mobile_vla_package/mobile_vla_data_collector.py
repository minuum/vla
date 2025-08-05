#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys, tty, termios
import time
import numpy as np
import cv2
import h5py
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import threading

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from camera_interfaces.srv import GetImage

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available. Using simulation mode.")
    ROBOT_AVAILABLE = False

class MobileVLADataCollector(Node):
    def __init__(self):
        super().__init__('mobile_vla_data_collector')
        self.WASD_TO_CONTINUOUS = {
            'w': {"linear_x": 0.5, "linear_y": 0.0, "angular_z": 0.0},
            'a': {"linear_x": 0.0, "linear_y": 0.5, "angular_z": 0.0},
            's': {"linear_x": -0.5, "linear_y": 0.0, "angular_z": 0.0},
            'd': {"linear_x": 0.0, "linear_y": -0.5, "angular_z": 0.0},
            'q': {"linear_x": 0.5, "linear_y": 0.5, "angular_z": 0.0},
            'e': {"linear_x": 0.5, "linear_y": -0.5, "angular_z": 0.0},
            'z': {"linear_x": -0.5, "linear_y": 0.5, "angular_z": 0.0},
            'c': {"linear_x": -0.5, "linear_y": -0.5, "angular_z": 0.0},
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.5},
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -0.5},
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        
        self.episode_data = []
        self.collecting = False
        self.episode_name = ""
        self.episode_start_time = None
        self.action_chunk_size = 8

        self.current_action = self.STOP_ACTION.copy()
        self.movement_timer = None

        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50
        else:
            self.driver = None
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        try:
            self.get_image_client = self.create_client(GetImage, 'get_image_service')
            while not self.get_image_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info('get_image_service 서비스 대기 중...')
                if not rclpy.ok():
                    self.get_logger().error("ROS2 컨텍스트가 종료되어 서비스 대기를 중단합니다.")
                    sys.exit()
            self.get_logger().info('✅ get_image_service 서비스 연결 완료!')
        except Exception as e:
            self.get_logger().error(f"❌ GetImage 서비스 클라이언트 시작 실패: {e}. 'colcon build' 후 'source install/setup.bash'를 다시 실행했는지, 그리고 패키지 구조가 올바른지 확인하세요.")
            rclpy.shutdown()


        self.cv_bridge = CvBridge()
        self.data_dir = Path("mobile_vla_dataset")
        self.data_dir.mkdir(exist_ok=True)
        self.get_logger().info("🤖 Mobile VLA Data Collector 준비 완료!")
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("   W/A/S/D: 이동, Q/E/Z/C: 대각선")
        self.get_logger().info("   R/T: 회전, 스페이스바: 정지")
        self.get_logger().info("   F/G: 속도 조절, N: 새 에피소드 시작")
        self.get_logger().info("   M: 에피소드 종료, Ctrl+C: 프로그램 종료")
        
        self.get_logger().info("⏳ 키보드 입력 대기 중...")
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        self.get_logger().info("✅ 시스템 준비 완료! 'n' 키를 눌러 새 에피소드를 시작하세요.")

    def keyboard_loop(self):
        """Separate thread loop for handling keyboard input"""
        while rclpy.ok():
            key = self.get_key()
            self.handle_key_input(key)
            time.sleep(0.01)

    def handle_key_input(self, key: str):
        """Execute logic based on keyboard input"""
        if key == '\x03':
            if self.collecting:
                self.stop_episode()
            sys.exit()
        elif key == 'n':
            if self.collecting:
                self.stop_episode()
            self.start_episode()
        elif key == 'm':
            if self.collecting:
                self.stop_episode()
        elif key == 'f':
            if ROBOT_AVAILABLE:
                self.throttle = max(10, self.throttle - 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key == 'g':
            if ROBOT_AVAILABLE:
                self.throttle = min(100, self.throttle + 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key in self.WASD_TO_CONTINUOUS:
            action = self.WASD_TO_CONTINUOUS[key]
            
            if self.movement_timer and self.movement_timer.is_alive():
                self.movement_timer.cancel()
                if self.current_action != self.STOP_ACTION: 
                    self.stop_movement_internal(collect_data=True) 

            self.current_action = action.copy()
            self.publish_cmd_vel(action)
            self.get_logger().info(f"🔴 {'수집중' if self.collecting else '대기중'} | Key: {key.upper()} → Action: ({action['linear_x']:+.1f}, {action['linear_y']:+.1f}, {action['angular_z']:+.1f})")

            # if self.collecting:
            #     self.collect_data_point("start_action")

            self.movement_timer = threading.Timer(0.3, self.stop_movement_timed)
            self.movement_timer.start()
            self.get_logger().info(f"🚀 움직임 시작: 0.3초")
            
        elif key == ' ':
            self.stop_movement_internal(collect_data=True) 
            self.get_logger().info("🛑 정지")

    def stop_movement_timed(self):
        """Stop function called by the timer"""
        self.stop_movement_internal(collect_data=True)

    def stop_movement_internal(self, collect_data: bool):
        """
        Internal function to stop robot movement and collect data if needed.
        collect_data: If True, collects data at the time of stopping.
        """
        if not collect_data and self.current_action == self.STOP_ACTION:
            return

        self.current_action = self.STOP_ACTION.copy()
        self.publish_cmd_vel(self.STOP_ACTION)
        self.get_logger().info("🛑 움직임 완료")

        if self.collecting and collect_data:
            self.collect_data_point("stop_action")

    def publish_cmd_vel(self, action: Dict[str, float]):
        """Publishes Twist message and controls the actual robot"""
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action["angular_z"])
        self.cmd_pub.publish(twist)

        if ROBOT_AVAILABLE and self.driver:
            if any(abs(v) > 0.1 for v in action.values()):
                if abs(action["angular_z"]) > 0.1:
                    spin_speed = int(action["angular_z"] * self.throttle)
                    self.driver.spin(spin_speed)
                elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                    angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                    if angle < 0:
                        angle += 360
                    self.driver.move(int(angle), self.throttle)
            else:
                self.driver.stop()

    def get_latest_image_via_service(self) -> np.ndarray | None:
        """
        GetImage 서비스를 호출하여 최신 이미지를 가져옵니다.
        서비스 호출에 실패하거나 타임아웃되면 None을 반환합니다.
        """
        request = GetImage.Request()
        future = self.get_image_client.call_async(request)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.done():
            try:
                response = future.result()
                if response.image.data:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                    self.get_logger().info("✅ 서비스로부터 이미지 수신 완료!")
                    return cv_image
                else:
                    self.get_logger().warn("⚠️ 서비스로부터 빈 이미지 데이터를 수신했습니다.")
                    return None
            except Exception as e:
                self.get_logger().error(f"서비스 응답 처리 중 에러 발생: {e}")
                return None
        else:
            self.get_logger().warn("⚠️ GetImage 서비스 호출 타임아웃 또는 서비스 응답 실패.")
            return None

    def collect_data_point(self, action_event_type: str):
        """
        Collects data at the time of the event.
        Now fetches image synchronously via service call.
        """
        current_image = self.get_latest_image_via_service()

        if current_image is None:
            self.get_logger().warn(f"⚠️ {action_event_type} - 서비스로부터 이미지를 가져오지 못해 데이터 포인트를 건너뜁니다.")
            return
            
        frame_data = {
            "image": current_image.copy(),
            "action": self.current_action.copy(),
            "action_event_type": action_event_type
        }
        self.episode_data.append(frame_data)
        self.get_logger().info(f"💾 {action_event_type} 이벤트 기반 데이터 프레임 수집: {len(self.episode_data)}개")

    def get_key(self) -> str:
        """Reads key input from the terminal"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()

    def start_episode(self, episode_name: str = None):
        """Starts a new episode collection"""
        if episode_name is None:
            self.episode_name = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.episode_name = episode_name

        self.episode_data = []
        
        self.get_logger().info("⏳ 에피소드 시작 전 첫 이미지 서비스 요청 중...")
        initial_image = self.get_latest_image_via_service()
        
        if initial_image is None:
            self.get_logger().error("❌ 에피소드 시작을 위한 첫 이미지를 가져오지 못했습니다. 서비스 서버(카메라 노드)를 확인하세요.")
            return

        self.collecting = True
        self.episode_start_time = time.time()
        
        self.get_logger().info(f"🎬 에피소드 시작: {self.episode_name}")
        self.get_logger().info(f"🔍 수집 상태: collecting={self.collecting}, 최신이미지={initial_image is not None}")

    def stop_episode(self):
        """Ends episode collection and saves data"""
        if not self.collecting:
            self.get_logger().warn("수집 중이 아닙니다.")
            return

        if self.movement_timer and self.movement_timer.is_alive():
            self.movement_timer.cancel()
            self.stop_movement_internal(collect_data=True)
        else:
            self.stop_movement_internal(collect_data=True)

        self.collecting = False
        
        end_time = time.time()
        total_duration = end_time - self.episode_start_time
        
        save_path = self.save_episode_data(self.episode_data, self.episode_name, total_duration)
        self.get_logger().info(f"✅ 에피소드 완료: {total_duration:.1f}초, 총 프레임 수: {len(self.episode_data)}")
        self.get_logger().info(f"💾 저장됨: {save_path}")

        self.publish_cmd_vel(self.STOP_ACTION)

    def save_episode_data(self, episode_data: List[Dict], episode_name: str, total_duration: float) -> Path:
        """Saves collected episode data to an HDF5 file"""
        save_path = self.data_dir / f"{episode_name}.h5"
        
        if not episode_data:
            self.get_logger().warn("⚠️ 저장할 프레임이 없습니다. 파일을 생성하지 않습니다.")
            return save_path

        images = []
        actions = []
        action_event_types = []

        for d in episode_data:
            images.append(d['image'])
            actions.append([d['action']['linear_x'], d['action']['linear_y'], d['action']['angular_z']])
            action_event_types.append(d['action_event_type'])
        
        images = np.stack(images)
        actions = np.array(actions, dtype=np.float32)
        action_event_types = np.array(action_event_types, dtype=h5py.string_dtype(encoding='utf-8'))

        self.get_logger().info(f"📊 생성된 데이터: 이미지 {images.shape}, 액션 {actions.shape}, 이벤트 타입 {action_event_types.shape}")

        with h5py.File(save_path, 'w') as f:
            f.attrs['episode_name'] = episode_name
            f.attrs['total_duration'] = total_duration
            f.attrs['num_frames'] = images.shape[0]
            f.attrs['action_chunk_size'] = self.action_chunk_size

            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('actions', data=actions, compression='gzip')
            f.create_dataset('action_event_types', data=action_event_types, compression='gzip')

        return save_path

def main(args=None):
    rclpy.init(args=args)
    collector = MobileVLADataCollector()
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop_episode() 
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
