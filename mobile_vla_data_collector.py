#!/usr/bin/env python3
"""
Mobile VLA 데이터 수집 시스템
- WASD 키보드 입력을 2D 연속값으로 변환
- 실시간 이미지 + 액션 데이터 수집
- RoboVLMs Action Chunk 형식으로 저장
"""

import rclpy
from rclpy.node import Node
import sys, tty, termios
import time
import numpy as np
import cv2
import json
import h5py
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from collections import deque

# ROS2 메시지
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 기존 로봇 제어 (keyboard_control_pkg 참조)
try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available. Using simulation mode.")
    ROBOT_AVAILABLE = False

class MobileVLADataCollector(Node):
    """Mobile VLA용 데이터 수집 노드"""
    
    def __init__(self):
        super().__init__('mobile_vla_data_collector')
        
        # === 핵심 설정 ===
        self.WASD_TO_CONTINUOUS = {
            'w': {"linear_x": 0.5, "linear_y": 0.0, "angular_z": 0.0},   # 전진
            'a': {"linear_x": 0.0, "linear_y": 0.5, "angular_z": 0.0},   # 좌이동  
            's': {"linear_x": -0.5, "linear_y": 0.0, "angular_z": 0.0},  # 후진
            'd': {"linear_x": 0.0, "linear_y": -0.5, "angular_z": 0.0},  # 우이동
            'q': {"linear_x": 0.5, "linear_y": 0.5, "angular_z": 0.0},   # 전좌대각
            'e': {"linear_x": 0.5, "linear_y": -0.5, "angular_z": 0.0},  # 전우대각
            'z': {"linear_x": -0.5, "linear_y": 0.5, "angular_z": 0.0},  # 후좌대각
            'c': {"linear_x": -0.5, "linear_y": -0.5, "angular_z": 0.0}, # 후우대각
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.5},   # 좌회전
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -0.5},  # 우회전
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}    # 정지 (스페이스바)
        }
        
        # === RoboVLMs Action Chunk 설정 ===
        self.WINDOW_SIZE = 10      # 과거 프레임
        self.CHUNK_SIZE = 8        # 미래 프레임 (예측할 액션)
        self.TOTAL_FRAMES = self.WINDOW_SIZE + self.CHUNK_SIZE  # 18프레임
        
        # === 데이터 저장 ===
        self.episode_data = {
            "episode_name": "",
            "action_chunks": [],
            "total_duration": 0.0,
            "obstacle_config": {},
            "cup_position": {"x": 0.0, "y": 0.0}  # 고정 컵 위치
        }
        
        # === 상태 관리 ===
        self.collecting = False
        self.episode_start_time = None
        self.current_action = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        self.action_history = deque(maxlen=self.TOTAL_FRAMES)
        self.image_history = deque(maxlen=self.TOTAL_FRAMES)
        
        # === 로봇 제어 초기화 ===
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50  # 기본 속도
        else:
            self.driver = None
            
        # === ROS2 퍼블리셔/서브스크라이버 ===
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # QoS 설정 - 카메라와 호환성을 위해 BEST_EFFORT 사용
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 카메라 기본값
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_profile
        )
        self.cv_bridge = CvBridge()
        self.latest_image = None
        
        # === 데이터 저장 경로 ===
        self.data_dir = Path("mobile_vla_dataset")
        self.data_dir.mkdir(exist_ok=True)
        
        self.get_logger().info("🤖 Mobile VLA Data Collector 준비 완료!")
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("   W/A/S/D: 이동, Q/E/Z/C: 대각선")
        self.get_logger().info("   R/T: 회전, 스페이스바: 정지")
        self.get_logger().info("   F/G: 속도 조절, N: 새 에피소드 시작")
        self.get_logger().info("   M: 에피소드 종료, Ctrl+C: 프로그램 종료")

    def image_callback(self, msg: Image):
        """카메라 이미지 콜백"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
            
            # 🔍 디버깅: 이미지 콜백 호출 확인
            if len(self.image_history) % 10 == 0:  # 10프레임마다 로그
                self.get_logger().info(f"📸 이미지 콜백: {len(self.image_history)}프레임, 수집중={self.collecting}")
            
            # 수집 중이면 이미지 히스토리에 추가
            if self.collecting and cv_image is not None:
                timestamp = time.time()
                self.image_history.append({
                    "image": cv_image.copy(),
                    "timestamp": timestamp
                })
                
        except Exception as e:
            self.get_logger().error(f"이미지 처리 에러: {e}")

    def get_key(self) -> str:
        """키보드 입력 받기 (keyboard_control_pkg 방식)"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()

    def execute_action(self, action: Dict[str, float]):
        """액션 실행 (로봇 제어 + ROS2 퍼블리시) - 10cm씩 단발성 움직임"""
        self.current_action = action.copy()
        timestamp = time.time()
        
        # 이동 거리 설정 (약 10cm에 해당하는 시간)
        move_duration = 0.3  # 0.3초 (속도에 따라 약 10cm)
        
        # 1. ROS2 Twist 메시지 퍼블리시 (단발성)
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action["angular_z"])
        
        # 단발성 움직임: 짧은 시간 동안만 퍼블리시
        if abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1 or abs(action["angular_z"]) > 0.1:
            # 움직임 시작
            self.cmd_pub.publish(twist)
            self.get_logger().info(f"🚀 움직임 시작: {move_duration:.1f}초")
            
            # 지정된 시간 동안 움직임 유지
            start_time = time.time()
            while time.time() - start_time < move_duration:
                self.cmd_pub.publish(twist)
                time.sleep(0.05)  # 20Hz로 퍼블리시
            
            # 움직임 정지
            stop_twist = Twist()  # 모든 값이 0
            self.cmd_pub.publish(stop_twist)
            self.get_logger().info(f"🛑 움직임 완료")
        else:
            # 정지 명령
            self.cmd_pub.publish(twist)
        
        # 2. 실제 로봇 제어 (가능한 경우) - 단발성 움직임
        if self.driver:
            if abs(action["angular_z"]) > 0.1:  # 회전 명령
                spin_speed = int(action["angular_z"] * self.throttle)
                self.driver.spin(spin_speed)
                time.sleep(move_duration)  # 짧은 시간만 회전
                self.driver.stop()
            elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:  # 이동 명령
                # linear_x, linear_y를 각도로 변환
                angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                if angle < 0:
                    angle += 360
                self.driver.move(int(angle), self.throttle)
                time.sleep(move_duration)  # 짧은 시간만 이동
                self.driver.stop()
            else:  # 정지
                self.driver.stop()
        
        # 3. 수집 중이면 액션 히스토리에 추가
        if self.collecting:
            self.action_history.append({
                "action": action.copy(),
                "timestamp": timestamp
            })
            
        return timestamp

    def start_episode(self, episode_name: str = None):
        """새 에피소드 시작"""
        if episode_name is None:
            episode_name = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.episode_data = {
            "episode_name": episode_name,
            "action_chunks": [],
            "total_duration": 0.0,
            "obstacle_config": {},
            "cup_position": {"x": 0.0, "y": 1.0}  # 컵은 1m 앞에 고정
        }
        
        self.collecting = True
        self.episode_start_time = time.time()
        self.action_history.clear()
        self.image_history.clear()
        
        self.get_logger().info(f"🎬 에피소드 시작: {episode_name}")
        self.get_logger().info(f"🔍 수집 상태: collecting={self.collecting}, 최신이미지={self.latest_image is not None}")

    def stop_episode(self):
        """에피소드 종료 및 저장"""
        if not self.collecting:
            self.get_logger().warn("수집 중이 아닙니다.")
            return
            
        self.collecting = False
        end_time = time.time()
        total_duration = end_time - self.episode_start_time
        self.episode_data["total_duration"] = total_duration
        
        # Action chunks 생성
        self.create_action_chunks()
        
        # 데이터 저장
        save_path = self.save_episode_data()
        
        self.get_logger().info(f"✅ 에피소드 완료: {total_duration:.1f}초")
        self.get_logger().info(f"💾 저장됨: {save_path}")
        
        # 정지
        self.execute_action({"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0})

    def create_action_chunks(self):
        """RoboVLMs 방식의 Action Chunk 생성"""
        if len(self.action_history) < self.TOTAL_FRAMES:
            self.get_logger().warn(f"데이터 부족: {len(self.action_history)} < {self.TOTAL_FRAMES}")
            return
            
        chunks = []
        for i in range(len(self.action_history) - self.TOTAL_FRAMES + 1):
            # 과거 10프레임 + 미래 8프레임 추출
            chunk_actions = list(self.action_history)[i:i+self.TOTAL_FRAMES]
            chunk_images = list(self.image_history)[i:i+self.TOTAL_FRAMES] if len(self.image_history) >= self.TOTAL_FRAMES else []
            
            chunk = {
                "chunk_id": len(chunks),
                "timestamp": chunk_actions[self.WINDOW_SIZE]["timestamp"],  # 현재 시점
                "past_actions": [a["action"] for a in chunk_actions[:self.WINDOW_SIZE]],  # 과거 10개
                "future_actions": [a["action"] for a in chunk_actions[self.WINDOW_SIZE:]],  # 미래 8개
                "images": [img["image"] for img in chunk_images] if chunk_images else [],
                "window_size": self.WINDOW_SIZE,
                "chunk_size": self.CHUNK_SIZE
            }
            chunks.append(chunk)
            
        self.episode_data["action_chunks"] = chunks
        self.get_logger().info(f"📊 생성된 Action Chunks: {len(chunks)}개")

    def save_episode_data(self) -> Path:
        """에피소드 데이터를 HDF5 형식으로 저장"""
        episode_name = self.episode_data["episode_name"]
        save_path = self.data_dir / f"{episode_name}.h5"
        
        with h5py.File(save_path, 'w') as f:
            # 메타데이터
            f.attrs['episode_name'] = episode_name
            f.attrs['total_duration'] = self.episode_data["total_duration"]
            f.attrs['cup_position_x'] = self.episode_data["cup_position"]["x"]
            f.attrs['cup_position_y'] = self.episode_data["cup_position"]["y"]
            f.attrs['window_size'] = self.WINDOW_SIZE
            f.attrs['chunk_size'] = self.CHUNK_SIZE
            
            # Action chunks
            chunks_group = f.create_group('action_chunks')
            for i, chunk in enumerate(self.episode_data["action_chunks"]):
                chunk_group = chunks_group.create_group(f'chunk_{i}')
                chunk_group.attrs['chunk_id'] = chunk["chunk_id"]
                chunk_group.attrs['timestamp'] = chunk["timestamp"]
                
                # 과거/미래 액션 저장
                past_actions = np.array([[a["linear_x"], a["linear_y"], a["angular_z"]] 
                                       for a in chunk["past_actions"]])
                future_actions = np.array([[a["linear_x"], a["linear_y"], a["angular_z"]] 
                                         for a in chunk["future_actions"]])
                
                chunk_group.create_dataset('past_actions', data=past_actions)
                chunk_group.create_dataset('future_actions', data=future_actions)
                
                # 이미지 저장 (있는 경우)
                if chunk["images"]:
                    images = np.stack(chunk["images"])  # [frames, height, width, channels]
                    chunk_group.create_dataset('images', data=images, compression='gzip')
        
        return save_path

    def run(self):
        """메인 실행 루프"""
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == '\x03':  # Ctrl+C
                    if self.collecting:
                        self.stop_episode()
                    break
                    
                elif key == 'n':  # 새 에피소드 시작
                    if self.collecting:
                        self.stop_episode()
                    self.start_episode()
                    continue
                    
                elif key == 'm':  # 에피소드 종료
                    if self.collecting:
                        self.stop_episode()
                    continue
                    
                elif key == 'f':  # 속도 감소
                    if ROBOT_AVAILABLE:
                        self.throttle = max(10, self.throttle - 10)
                        self.get_logger().info(f'속도: {self.throttle}%')
                    continue
                    
                elif key == 'g':  # 속도 증가
                    if ROBOT_AVAILABLE:
                        self.throttle = min(100, self.throttle + 10)
                        self.get_logger().info(f'속도: {self.throttle}%')
                    continue
                
                # WASD 액션 실행
                if key in self.WASD_TO_CONTINUOUS:
                    action = self.WASD_TO_CONTINUOUS[key]
                    timestamp = self.execute_action(action)
                    
                    action_str = f"({action['linear_x']:+.1f}, {action['linear_y']:+.1f}, {action['angular_z']:+.1f})"
                    status = "🔴 수집중" if self.collecting else "⚪ 대기중"
                    self.get_logger().info(f"{status} | Key: {key.upper()} → Action: {action_str}")
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self.collecting:
                self.stop_episode()
            if self.driver:
                self.driver.stop()
            self.get_logger().info("🛑 Mobile VLA Data Collector 종료")

def main(args=None):
    rclpy.init(args=args)
    collector = MobileVLADataCollector()
    
    try:
        collector.run()
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()