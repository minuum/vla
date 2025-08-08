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
import json
from collections import defaultdict

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from camera_interfaces.srv import GetImage
from std_srvs.srv import Empty

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
            'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},
            'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},
            's': {"linear_x": -1.15, "linear_y": 0.0, "angular_z": 0.0},
            'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},
            'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},
            'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},
            'z': {"linear_x": -1.15, "linear_y": 1.15, "angular_z": 0.0},
            'c': {"linear_x": -1.15, "linear_y": -1.15, "angular_z": 0.0},
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 1.15},
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -1.15},
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        
        self.episode_data = []
        self.collecting = False
        self.episode_name = ""
        self.episode_start_time = None
        self.action_chunk_size = 8
        
        # 분류 및 모니터링 설정
        self.categories = {
            "short": {"min": 1, "max": 10, "target": 50, "description": "짧은 에피소드"},
            "medium": {"min": 11, "max": 25, "target": 100, "description": "중간 에피소드"},  
            "long": {"min": 26, "max": 50, "target": 30, "description": "긴 에피소드"},
            "extra_long": {"min": 51, "max": float('inf'), "target": 10, "description": "매우 긴 에피소드"}
        }
        
        # 8가지 컵 도달 시나리오 목표 설정
        self.cup_scenarios = {
            "1box_vert_left": {"target": 15, "description": "1박스-세로-왼쪽경로", "key": "1"},
            "1box_vert_right": {"target": 15, "description": "1박스-세로-오른쪽경로", "key": "2"},
            "1box_hori_left": {"target": 15, "description": "1박스-가로-왼쪽경로", "key": "3"},
            "1box_hori_right": {"target": 15, "description": "1박스-가로-오른쪽경로", "key": "4"},
            "2box_vert_left": {"target": 15, "description": "2박스-세로-왼쪽경로", "key": "5"},
            "2box_vert_right": {"target": 15, "description": "2박스-세로-오른쪽경로", "key": "6"},
            "2box_hori_left": {"target": 15, "description": "2박스-가로-왼쪽경로", "key": "7"},
            "2box_hori_right": {"target": 15, "description": "2박스-가로-오른쪽경로", "key": "8"}
        }
        
        self.dataset_stats = defaultdict(int)
        self.scenario_stats = defaultdict(int)
        
        # 시나리오 선택 모드
        self.scenario_selection_mode = False

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
            self.reset_camera_client = self.create_client(Empty, 'reset_camera_service')
            
            while not self.get_image_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info('get_image_service 서비스 대기 중...')
                if not rclpy.ok():
                    self.get_logger().error("ROS2 컨텍스트가 종료되어 서비스 대기를 중단합니다.")
                    sys.exit()
            
            while not self.reset_camera_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().info('reset_camera_service 서비스 대기 중...')
                if not rclpy.ok():
                    self.get_logger().error("ROS2 컨텍스트가 종료되어 서비스 대기를 중단합니다.")
                    sys.exit()
                    
            self.get_logger().info('✅ get_image_service 서비스 연결 완료!')
            self.get_logger().info('✅ reset_camera_service 서비스 연결 완료!')
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 클라이언트 시작 실패: {e}. 'colcon build' 후 'source install/setup.bash'를 다시 실행했는지, 그리고 패키지 구조가 올바른지 확인하세요.")
            rclpy.shutdown()


        self.cv_bridge = CvBridge()
        self.data_dir = Path("mobile_vla_dataset")
        self.data_dir.mkdir(exist_ok=True)
        
        # 진행상황 저장 파일 (data_dir 정의 후)
        self.progress_file = self.data_dir / "scenario_progress.json"
        
        # 데이터셋 통계 로드
        self.load_dataset_stats()
        self.load_scenario_progress()
        
        self.get_logger().info("🤖 Mobile VLA Data Collector 준비 완료!")
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("   W/A/S/D: 이동, Q/E/Z/C: 대각선")
        self.get_logger().info("   R/T: 회전, 스페이스바: 정지")
        self.get_logger().info("   F/G: 속도 조절, N: 새 에피소드 시작")
        self.get_logger().info("   M: 에피소드 종료, P: 현재 진행 상황 확인")
        self.get_logger().info("🎯 컵 도달 시나리오 선택:")
        self.get_logger().info("   1: 1박스-세로-왼쪽, 2: 1박스-세로-오른쪽")
        self.get_logger().info("   3: 1박스-가로-왼쪽, 4: 1박스-가로-오른쪽")
        self.get_logger().info("   5: 2박스-세로-왼쪽, 6: 2박스-세로-오른쪽")
        self.get_logger().info("   7: 2박스-가로-왼쪽, 8: 2박스-가로-오른쪽")
        self.get_logger().info("   Ctrl+C: 프로그램 종료")
        
        self.get_logger().info("⏳ 키보드 입력 대기 중...")
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        self.get_logger().info("✅ 시스템 준비 완료!")
        self.get_logger().info("🎯 'N' 키를 눌러 시나리오 선택 메뉴를 확인하세요!")

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
            self.show_scenario_selection()
        elif key == 'm':
            if self.collecting:
                self.stop_episode()
        elif key == 'p':
            self.resync_and_show_progress()
        elif key in ['1', '2', '3', '4', '5', '6', '7', '8']:
            if self.scenario_selection_mode:
                # 시나리오 선택 모드에서 숫자키 입력
                scenario_map = {
                    '1': "1box_vert_left", '2': "1box_vert_right",
                    '3': "1box_hori_left", '4': "1box_hori_right", 
                    '5': "2box_vert_left", '6': "2box_vert_right",
                    '7': "2box_hori_left", '8': "2box_hori_right"
                }
                scenario_id = scenario_map[key]
                self.scenario_selection_mode = False  # 선택 모드 해제
                self.start_episode_with_scenario(scenario_id)
            else:
                self.get_logger().info("⚠️ 먼저 'N' 키를 눌러 에피소드 시작을 해주세요.")
        elif key == 'f':
            if ROBOT_AVAILABLE:
                self.throttle = max(10, self.throttle - 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key == 'g':
            if ROBOT_AVAILABLE:
                self.throttle = min(100, self.throttle + 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key in self.WASD_TO_CONTINUOUS:
            if self.scenario_selection_mode:
                self.scenario_selection_mode = False
                self.get_logger().info("🚫 시나리오 선택이 취소되었습니다.")
                return
                
            action = self.WASD_TO_CONTINUOUS[key]
            
            if self.movement_timer and self.movement_timer.is_alive():
                self.movement_timer.cancel()
                if self.current_action != self.STOP_ACTION: 
                    self.stop_movement_internal(collect_data=False)  # 이전 액션 중단시 데이터 수집 안함 

            self.current_action = action.copy()
            self.publish_cmd_vel(action)
            self.get_logger().info(f"🔴 {'수집중' if self.collecting else '대기중'} | Key: {key.upper()} → Action: ({action['linear_x']:+.1f}, {action['linear_y']:+.1f}, {action['angular_z']:+.1f})")

            if self.collecting:
                self.collect_data_point_with_action("start_action", action)

            self.movement_timer = threading.Timer(0.3, self.stop_movement_timed)
            self.movement_timer.start()
            self.get_logger().info(f"🚀 움직임 시작: 0.3초 타이머 설정됨 (타이머 ID: {id(self.movement_timer)})")
            
        elif key == ' ':
            self.stop_movement_internal(collect_data=True) 
            self.get_logger().info("🛑 정지")

    def stop_movement_timed(self):
        """Stop function called by the timer - NO data collection for auto-stop"""
        self.get_logger().info(f"⏰ 타이머 호출: 0.3초 후 자동 정지 (타이머 ID: {id(threading.current_thread())})")
        self.stop_movement_internal(collect_data=False)

    def stop_movement_internal(self, collect_data: bool):
        """
        Internal function to stop robot movement and collect data if needed.
        collect_data: If True, collects data at the time of stopping.
        """
        self.get_logger().info(f"🔧 stop_movement_internal 호출: collect_data={collect_data}, current_action={self.current_action}")
        
        if not collect_data and self.current_action == self.STOP_ACTION:
            self.get_logger().info("🔧 이미 정지 상태이므로 리턴")
            return

        self.current_action = self.STOP_ACTION.copy()
        self.publish_cmd_vel(self.STOP_ACTION)
        self.get_logger().info("🛑 움직임 완료")

        if self.collecting and collect_data:
            self.collect_data_point_with_action("stop_action", self.STOP_ACTION)

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

    def get_latest_image_via_service(self, max_retries: int = 3) -> np.ndarray | None:
        """
        GetImage 서비스를 호출하여 최신 이미지를 가져옵니다.
        서비스 호출에 실패하거나 타임아웃되면 재시도합니다.
        """
        for attempt in range(max_retries):
            try:
                request = GetImage.Request()
                future = self.get_image_client.call_async(request)
                
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
                
                if future.done():
                    response = future.result()
                    if response.image.data:
                        cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                        self.get_logger().info("✅ 서비스로부터 이미지 수신 완료!")
                        return cv_image
                    else:
                        self.get_logger().warn(f"⚠️ 빈 이미지 수신 (시도 {attempt+1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(1.0)  # 1초 대기 후 재시도
                            continue
                else:
                    self.get_logger().warn(f"⚠️ 서비스 타임아웃 (시도 {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(1.0)  # 1초 대기 후 재시도
                        continue
                        
            except Exception as e:
                self.get_logger().error(f"서비스 호출 중 에러 (시도 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)  # 1초 대기 후 재시도
                    continue
                    
        self.get_logger().error(f"❌ {max_retries}번 시도 후에도 이미지를 가져오지 못했습니다.")
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

    def collect_data_point_with_action(self, action_event_type: str, action: Dict[str, float], image: np.ndarray = None):
        """
        특정 액션과 이미지를 지정하여 데이터 포인트 수집
        """
        if image is None:
            current_image = self.get_latest_image_via_service()
            if current_image is None:
                self.get_logger().warn(f"⚠️ {action_event_type} - 서비스로부터 이미지를 가져오지 못해 데이터 포인트를 건너뜁니다.")
                return
        else:
            current_image = image
            
        frame_data = {
            "image": current_image.copy(),
            "action": action.copy(),
            "action_event_type": action_event_type
        }
        self.episode_data.append(frame_data)
        
        # 액션 내용 로깅
        action_desc = []
        if abs(action['linear_x']) > 0.1:
            action_desc.append(f"전진{action['linear_x']:+.1f}")
        if abs(action['linear_y']) > 0.1:
            action_desc.append(f"횡이동{action['linear_y']:+.1f}")
        if abs(action['angular_z']) > 0.1:
            action_desc.append(f"회전{action['angular_z']:+.1f}")
        if not action_desc:
            action_desc.append("정지")
            
        self.get_logger().info(f"💾 {action_event_type} 액션[{', '.join(action_desc)}] 데이터 수집: {len(self.episode_data)}개")

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
        
        self.get_logger().info("⏳ 에피소드 시작 전 준비 중...")
        
        # 🔄 카메라 스트림 완전 재시작 (가장 확실한 버퍼 초기화 방법)
        self.get_logger().info("🔄 카메라 스트림 재시작 중... (버퍼 완전 초기화)")
        try:
            reset_request = Empty.Request()
            reset_future = self.reset_camera_client.call_async(reset_request)
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=10.0)
            
            if reset_future.done():
                self.get_logger().info("✅ 카메라 스트림 재시작 완료!")
                # 재시작 후 안정화 대기
                time.sleep(1.0)
            else:
                self.get_logger().warn("⚠️ 카메라 재시작 타임아웃, 일반 플러시로 대체")
                # 기존 플러시 방식으로 대체
                for i in range(3):
                    flush_image = self.get_latest_image_via_service(max_retries=1)
                    time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f"❌ 카메라 재시작 실패: {e}, 일반 플러시로 대체")
            # 기존 플러시 방식으로 대체
            for i in range(3):
                flush_image = self.get_latest_image_via_service(max_retries=1)
                time.sleep(0.1)
        
        self.get_logger().info("📸 새로운 스트림에서 첫 이미지 요청 중...")
        initial_image = self.get_latest_image_via_service(max_retries=5)
        
        if initial_image is None:
            self.get_logger().error("❌ 에피소드 시작을 위한 첫 이미지를 가져오지 못했습니다. 서비스 서버(카메라 노드)를 확인하세요.")
            return

        self.collecting = True
        self.episode_start_time = time.time()
        
        self.get_logger().info(f"🎬 에피소드 시작: {self.episode_name}")
        self.get_logger().info(f"🔍 수집 상태: collecting={self.collecting}, 초기이미지크기={initial_image.shape}")
        
        # 에피소드 시작 시점의 이미지를 첫 번째 데이터 포인트로 수집
        self.collect_data_point_with_action("episode_start", self.STOP_ACTION, initial_image)

    def start_episode_with_strategy(self, strategy: str, message: str):
        """전략을 지정하여 에피소드 시작"""
        # 전략 정보를 에피소드명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategy_episode_name = f"episode_{timestamp}_{strategy}"
        
        self.get_logger().info(f"🎯 {message}")
        self.get_logger().info(f"📝 전략: {self.avoidance_targets[strategy]['description']}")
        
        # 현재 전략별 진행 상황 표시
        current = self.avoidance_stats[strategy]
        target = self.avoidance_targets[strategy]["target"]
        progress_bar = self.create_progress_bar(current, target)
        self.get_logger().info(f"📊 {strategy.upper()}: {progress_bar}")
        
        self.start_episode(strategy_episode_name)

    def stop_episode(self):
        """Ends episode collection and saves data"""
        if not self.collecting:
            self.get_logger().warn("수집 중이 아닙니다.")
            return

        if self.movement_timer and self.movement_timer.is_alive():
            self.movement_timer.cancel()
            self.stop_movement_internal(collect_data=False)  # 에피소드 종료시에는 데이터 수집 안함
        else:
            self.stop_movement_internal(collect_data=False)  # 에피소드 종료시에는 데이터 수집 안함

        self.collecting = False
        
        end_time = time.time()
        total_duration = end_time - self.episode_start_time
        
        save_path = self.save_episode_data(self.episode_data, self.episode_name, total_duration)
        
        # 프레임 수에 따른 분류 및 통계 업데이트
        num_frames = len(self.episode_data)
        category = self.classify_by_frames(num_frames)
        self.dataset_stats[category] += 1
        
        # 시나리오별 통계 업데이트 (에피소드명에서 시나리오 추출)
        scenario = self.extract_scenario_from_episode_name(self.episode_name)
        if scenario:
            self.scenario_stats[scenario] += 1
            self.save_scenario_progress()
        
        # 프레임 18개 데이터 특별 표시
        frame_18_indicator = "🎯 [18프레임!]" if num_frames == 18 else ""
        scenario_indicator = f" 🎯[{scenario}]" if scenario else ""
        
        self.get_logger().info(f"✅ 에피소드 완료: {total_duration:.1f}초, 총 프레임 수: {num_frames}개{frame_18_indicator}{scenario_indicator}")
        self.get_logger().info(f"📂 카테고리: {category} ({self.categories[category]['description']})")
        self.get_logger().info(f"💾 저장됨: {save_path}")
        
        # 현재 진행 상황 표시
        self.show_category_progress(category)
        if scenario:
            self.show_scenario_progress(scenario)

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

    def classify_by_frames(self, num_frames: int) -> str:
        """프레임 수에 따라 카테고리 분류"""
        for category, config in self.categories.items():
            if config["min"] <= num_frames <= config["max"]:
                return category
        return "unknown"
    
    def load_dataset_stats(self):
        """기존 데이터셋 통계 로드"""
        try:
            h5_files = list(self.data_dir.glob("*.h5"))
            self.dataset_stats = defaultdict(int)
            
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        num_frames = f.attrs.get('num_frames', 0)
                        if 'images' in f:
                            num_frames = f['images'].shape[0]
                        category = self.classify_by_frames(num_frames)
                        self.dataset_stats[category] += 1
                except Exception as e:
                    self.get_logger().warn(f"⚠️ 파일 읽기 실패 {h5_file.name}: {e}")
                    
        except Exception as e:
            self.get_logger().warn(f"⚠️ 데이터셋 통계 로드 실패: {e}")
            
    def create_progress_bar(self, current: int, target: int, width: int = 15) -> str:
        """진행률 바 생성"""
        if target == 0:
            return "█" * width + " (무제한)"
        percentage = min(current / target, 1.0)
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {current}/{target}"
        
    def show_category_progress(self, category: str):
        """특정 카테고리의 진행 상황 표시"""
        config = self.categories[category]
        current = self.dataset_stats[category]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        percentage = (current / target * 100) if target > 0 else 0
        
        status_emoji = "✅" if current >= target else "⏳"
        self.get_logger().info(f"{status_emoji} {category.upper()}: {progress_bar} ({percentage:.1f}%)")
        
    def show_progress_status(self):
        """전체 진행 상황 표시"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("📊 현재 데이터셋 진행 상황")
        self.get_logger().info("=" * 50)
        
        total_current = 0
        total_target = 0
        frame_18_count = 0
        
        # 프레임 18개 데이터 별도 카운트
        for h5_file in self.data_dir.glob("*.h5"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    num_frames = f.attrs.get('num_frames', 0)
                    if 'images' in f:
                        num_frames = f['images'].shape[0]
                    if num_frames == 18:
                        frame_18_count += 1
            except:
                pass
        
        for category, config in self.categories.items():
            current = self.dataset_stats[category]
            target = config["target"]
            percentage = (current / target * 100) if target > 0 else 0
            
            total_current += current
            total_target += target
            
            status_emoji = "✅" if current >= target else "⏳"
            progress_bar = self.create_progress_bar(current, target)
            
            self.get_logger().info(f"{status_emoji} {category.upper()}: {progress_bar} ({percentage:.1f}%)")
            self.get_logger().info(f"   {config['description']}")
            
        # 전체 진행률
        overall_percentage = (total_current / total_target * 100) if total_target > 0 else 0
        overall_progress = self.create_progress_bar(total_current, total_target, width=25)
        
        self.get_logger().info("-" * 50)
        self.get_logger().info(f"🎯 전체: {overall_progress} ({overall_percentage:.1f}%)")
        self.get_logger().info(f"🎯 프레임 18개 데이터: {frame_18_count}개 발견!")
        
        # 시나리오별 진행 상황도 표시
        self.get_logger().info("-" * 50)
        self.get_logger().info("🎯 컵 도달 시나리오별 진행 상황:")
        
        total_completed = 0
        total_target = 0
        
        for scenario, config in self.cup_scenarios.items():
            current = self.scenario_stats[scenario]
            target = config["target"]
            total_completed += current
            total_target += target
            percentage = (current / target * 100) if target > 0 else 0
            progress_bar = self.create_progress_bar(current, target)
            status_emoji = "✅" if current >= target else "⏳"
            
            self.get_logger().info(f"{status_emoji} {config['key']}키 {scenario}: {progress_bar} ({percentage:.1f}%)")
            self.get_logger().info(f"   {config['description']}")
            
        # 전체 진행률
        overall_percentage = (total_completed / total_target * 100) if total_target > 0 else 0
        overall_progress = self.create_progress_bar(total_completed, total_target, width=30)
        self.get_logger().info("-" * 50)
        self.get_logger().info(f"🏁 전체 진행률: {overall_progress} ({overall_percentage:.1f}%)")
        self.get_logger().info(f"📊 {total_completed}/{total_target}개 완료, {total_target - total_completed}개 남음")
        
        self.get_logger().info("=" * 50)
        
    def extract_strategy_from_episode_name(self, episode_name: str) -> str:
        """에피소드명에서 전략 추출"""
        for strategy in self.avoidance_targets.keys():
            if strategy in episode_name:
                return strategy
        return None
        
    def show_strategy_progress(self, strategy: str):
        """특정 전략의 진행 상황 표시"""
        if strategy not in self.avoidance_targets:
            return
            
        config = self.avoidance_targets[strategy]
        current = self.avoidance_stats[strategy]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        percentage = (current / target * 100) if target > 0 else 0
        
        status_emoji = "✅" if current >= target else "⏳"
        self.get_logger().info(f"{status_emoji} {strategy.upper()}: {progress_bar} ({percentage:.1f}%)")
        
    def start_episode_with_scenario(self, scenario_id: str):
        """시나리오를 지정하여 에피소드 시작"""
        config = self.cup_scenarios[scenario_id]
        
        # 시나리오 정보를 에피소드명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scenario_episode_name = f"episode_{timestamp}_{scenario_id}"
        
        self.get_logger().info(f"🎯 {config['description']} 시나리오 시작!")
        
        # 현재 시나리오별 진행 상황 표시
        current = self.scenario_stats[scenario_id]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        self.get_logger().info(f"📊 {scenario_id.upper()}: {progress_bar}")
        
        # WASD 경로 예시 표시
        wasd_examples = self.get_wasd_example(scenario_id)
        self.get_logger().info(f"🎮 예시 경로: {wasd_examples}")
        
        self.start_episode(scenario_episode_name)
        
    def get_wasd_example(self, scenario_id: str) -> str:
        """시나리오별 WASD 경로 예시"""
        examples = {
            "1box_vert_left": "W W W → A A → W W → D D",
            "1box_vert_right": "W W → D D → W W W → A A", 
            "1box_hori_left": "W → A A A → W W → D D D",
            "1box_hori_right": "W W → D → W W → A",
            "2box_vert_left": "W W → A A A → W W → D D D",
            "2box_vert_right": "W → D D D → W W W → A A A",
            "2box_hori_left": "W → A A A A → W W → D D D D",
            "2box_hori_right": "W W → D D → W W → A A"
        }
        return examples.get(scenario_id, "W → A/D → W → ...")
        
    def extract_scenario_from_episode_name(self, episode_name: str) -> str:
        """에피소드명에서 시나리오 추출"""
        for scenario in self.cup_scenarios.keys():
            if scenario in episode_name:
                return scenario
        return None
        
    def show_scenario_progress(self, scenario: str):
        """특정 시나리오의 진행 상황 표시"""
        if scenario not in self.cup_scenarios:
            return
            
        config = self.cup_scenarios[scenario]
        current = self.scenario_stats[scenario]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        percentage = (current / target * 100) if target > 0 else 0
        
        status_emoji = "✅" if current >= target else "⏳"
        remaining = max(0, target - current)
        self.get_logger().info(f"{status_emoji} {config['key']}키 {scenario}: {progress_bar} ({percentage:.1f}%) - {remaining}개 남음")
        
    def load_scenario_progress(self):
        """저장된 시나리오 진행상황 로드"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.scenario_stats = defaultdict(int, data.get('scenario_stats', {}))
                self.get_logger().info(f"📊 시나리오 진행상황 로드 완료: {dict(self.scenario_stats)}")
            else:
                self.scenario_stats = defaultdict(int)
                self.get_logger().info("📊 새로운 시나리오 진행상황 시작")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시나리오 진행상황 로드 실패: {e}")
            self.scenario_stats = defaultdict(int)
            
    def save_scenario_progress(self):
        """시나리오 진행상황 저장"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "scenario_stats": dict(self.scenario_stats),
                "total_completed": sum(self.scenario_stats.values()),
                "total_target": sum(config["target"] for config in self.cup_scenarios.values())
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시나리오 진행상황 저장 실패: {e}")
            
    def show_scenario_selection(self):
        """8가지 시나리오 선택 메뉴 표시"""
        self.scenario_selection_mode = True
        
        self.get_logger().info("🎯 컵 도달 시나리오 선택")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 환경을 설정한 후 원하는 시나리오 번호를 누르세요:")
        self.get_logger().info("")
        
        # 시나리오별 상세 정보 표시
        scenario_details = [
            {"key": "1", "id": "1box_vert_left", "env": "📦 박스 1개 세로 배치", "path": "W W W → A A → W W → D D"},
            {"key": "2", "id": "1box_vert_right", "env": "📦 박스 1개 세로 배치", "path": "W W → D D → W W W → A A"},
            {"key": "3", "id": "1box_hori_left", "env": "📦 박스 1개 가로 배치", "path": "W → A A A → W W → D D D"},
            {"key": "4", "id": "1box_hori_right", "env": "📦 박스 1개 가로 배치", "path": "W W → D → W W → A"},
            {"key": "5", "id": "2box_vert_left", "env": "📦📦 박스 2개 세로 배치", "path": "W W → A A A → W W → D D D"},
            {"key": "6", "id": "2box_vert_right", "env": "📦📦 박스 2개 세로 배치", "path": "W → D D D → W W W → A A A"},
            {"key": "7", "id": "2box_hori_left", "env": "📦📦 박스 2개 가로 배치", "path": "W → A A A A → W W → D D D D"},
            {"key": "8", "id": "2box_hori_right", "env": "📦📦 박스 2개 가로 배치", "path": "W W → D D → W W → A A"}
        ]
        
        for scenario in scenario_details:
            scenario_id = scenario["id"]
            description = self.cup_scenarios[scenario_id]["description"]
            current = self.scenario_stats[scenario_id]
            target = self.cup_scenarios[scenario_id]["target"]
            remaining = max(0, target - current)
            progress_bar = self.create_progress_bar(current, target, width=10)
            
            status_emoji = "✅" if current >= target else "⏳"
            
            self.get_logger().info(f"{status_emoji} {scenario['key']}키: {description}")
            self.get_logger().info(f"   🏗️ {scenario['env']}")
            self.get_logger().info(f"   🎮 {scenario['path']}")
            self.get_logger().info(f"   📊 {progress_bar} ({current}/{target}) - {remaining}개 남음")
            self.get_logger().info("")
        
        # 전체 진행률 요약
        total_completed = sum(self.scenario_stats.values())
        total_target = sum(config["target"] for config in self.cup_scenarios.values())
        overall_progress = self.create_progress_bar(total_completed, total_target, width=20)
        overall_percentage = (total_completed / total_target * 100) if total_target > 0 else 0
        
        self.get_logger().info("🏁 전체 진행률:")
        self.get_logger().info(f"   {overall_progress} ({total_completed}/{total_target}) {overall_percentage:.1f}%")
        self.get_logger().info("")
        self.get_logger().info("✨ 1-8번 중 원하는 시나리오를 선택하세요!")
        self.get_logger().info("💡 환경 설정 후 숫자키를 누르면 에피소드가 시작됩니다.")
        self.get_logger().info("🚫 취소하려면 다른 키를 누르세요.")

    def resync_scenario_progress(self):
        """실제 H5 파일들을 스캔하여 시나리오 진행률 재동기화"""
        self.get_logger().info("🔄 H5 파일 스캔하여 시나리오 진행률 동기화 중...")
        
        # 시나리오 통계 초기화
        self.scenario_stats = defaultdict(int)
        
        # 데이터 디렉토리에서 모든 H5 파일 스캔
        if self.data_dir.exists():
            h5_files = list(self.data_dir.glob("*.h5"))
            self.get_logger().info(f"📁 {len(h5_files)}개의 H5 파일을 발견했습니다.")
            
            scenario_matched = 0
            old_format_files = []
            
            for h5_file in h5_files:
                try:
                    # 파일명에서 시나리오 추출
                    scenario = self.extract_scenario_from_episode_name(h5_file.stem)
                    if scenario and scenario in self.cup_scenarios:
                        self.scenario_stats[scenario] += 1
                        scenario_matched += 1
                        self.get_logger().info(f"✅ {h5_file.name} → {scenario}")
                    else:
                        old_format_files.append(h5_file.name)
                        self.get_logger().info(f"⚠️ {h5_file.name} → 시나리오 이름 없음 (구형 파일)")
                except Exception as e:
                    self.get_logger().warning(f"⚠️ {h5_file.name} 분석 실패: {e}")
            
            # 구형 파일들 정보 출력
            if old_format_files:
                self.get_logger().info(f"📋 시나리오 이름이 없는 구형 파일 {len(old_format_files)}개:")
                for old_file in old_format_files[:5]:  # 최대 5개만 표시
                    self.get_logger().info(f"   • {old_file}")
                if len(old_format_files) > 5:
                    self.get_logger().info(f"   • ... 외 {len(old_format_files) - 5}개")
        else:
            self.get_logger().info("📁 데이터 디렉토리가 존재하지 않습니다.")
        
        # 새로운 진행상황 저장
        self.save_scenario_progress()
        
        # 동기화 결과 요약
        total_found = sum(self.scenario_stats.values())
        self.get_logger().info(f"✅ 동기화 완료! 총 {total_found}개의 시나리오 에피소드 발견")
        
        for scenario_id, count in self.scenario_stats.items():
            if count > 0:
                scenario_info = self.cup_scenarios[scenario_id]
                key = scenario_info["key"]
                desc = scenario_info["description"]
                self.get_logger().info(f"   {key}키 {scenario_id}: {count}개 → {desc}")
        
        if total_found == 0:
            self.get_logger().info("📝 시나리오 이름이 포함된 파일이 없습니다.")
            self.get_logger().info("💡 새로운 N-숫자키 시스템으로 수집한 파일만 카운트됩니다.")

    def resync_and_show_progress(self):
        """H5 파일 재스캔 후 진행률 표시"""
        self.resync_scenario_progress()
        self.load_dataset_stats()  # 전체 데이터셋 통계도 다시 로드
        self.show_progress_status()

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
