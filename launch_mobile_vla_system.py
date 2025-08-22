#!/usr/bin/env python3
"""
Mobile VLA 시스템 Launch 파일
- 모든 ROS 노드들을 순차적으로 실행
- 시스템 상태 모니터링
- 안전한 종료 처리
"""

import rclpy
from rclpy.node import Node
import subprocess
import time
import signal
import sys
import threading
from typing import List, Dict
import os

class MobileVLASystemLauncher(Node):
    def __init__(self):
        super().__init__('mobile_vla_system_launcher')
        
        # 노드 프로세스 관리
        self.node_processes: Dict[str, subprocess.Popen] = {}
        self.node_status: Dict[str, bool] = {}
        
        # 노드 실행 순서 및 설정
        self.nodes_config = [
            {
                'name': 'camera_publisher',
                'package': 'camera_pub',
                'executable': 'camera_publisher_continuous',
                'delay': 2.0,
                'required': True
            },
            {
                'name': 'vla_inference',
                'package': 'vla_inference',
                'executable': 'vla_inference_node',
                'delay': 3.0,
                'required': True
            },
            {
                'name': 'robot_control',
                'package': 'robot_control',
                'executable': 'robot_control_node',
                'delay': 2.0,
                'required': True
            },
            {
                'name': 'data_collector',
                'package': 'mobile_vla_data_collector',
                'executable': 'mobile_vla_data_collector',
                'delay': 1.0,
                'required': False
            }
        ]
        
        # 시스템 상태
        self.system_running = True
        self.startup_complete = False
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.get_logger().info("🚀 Mobile VLA 시스템 Launcher 시작")
        
    def signal_handler(self, signum, frame):
        """시그널 핸들러 (Ctrl+C 등)"""
        self.get_logger().info(f"📡 시그널 {signum} 수신 - 시스템 종료 시작")
        self.shutdown_system()
        
    def launch_node(self, node_config: Dict) -> bool:
        """개별 노드 실행"""
        try:
            self.get_logger().info(f"🔄 {node_config['name']} 노드 시작 중...")
            
            # ROS2 실행 명령어
            cmd = [
                'ros2', 'run',
                node_config['package'],
                node_config['executable']
            ]
            
            # 프로세스 시작
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 프로세스 저장
            self.node_processes[node_config['name']] = process
            self.node_status[node_config['name']] = True
            
            # 시작 대기
            time.sleep(node_config['delay'])
            
            # 프로세스 상태 확인
            if process.poll() is None:
                self.get_logger().info(f"✅ {node_config['name']} 노드 시작 성공")
                return True
            else:
                stdout, stderr = process.communicate()
                self.get_logger().error(f"❌ {node_config['name']} 노드 시작 실패")
                self.get_logger().error(f"stdout: {stdout}")
                self.get_logger().error(f"stderr: {stderr}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ {node_config['name']} 노드 실행 중 오류: {e}")
            return False
            
    def monitor_nodes(self):
        """노드 상태 모니터링"""
        while self.system_running:
            try:
                # 각 노드 상태 확인
                for name, process in self.node_processes.items():
                    if process.poll() is not None:
                        # 노드가 종료됨
                        if self.node_status[name]:
                            self.get_logger().warn(f"⚠️ {name} 노드가 예기치 않게 종료됨")
                            self.node_status[name] = False
                            
                            # 필수 노드인 경우 재시작 시도
                            node_config = next(
                                (n for n in self.nodes_config if n['name'] == name), None
                            )
                            if node_config and node_config['required']:
                                self.get_logger().info(f"🔄 {name} 노드 재시작 시도...")
                                if self.launch_node(node_config):
                                    self.get_logger().info(f"✅ {name} 노드 재시작 성공")
                                else:
                                    self.get_logger().error(f"❌ {name} 노드 재시작 실패")
                                    
                time.sleep(5.0)  # 5초마다 체크
                
            except Exception as e:
                self.get_logger().error(f"❌ 노드 모니터링 오류: {e}")
                time.sleep(5.0)
                
    def launch_system(self):
        """전체 시스템 실행"""
        self.get_logger().info("🚀 Mobile VLA 시스템 시작")
        
        # 1. ROS2 환경 확인
        self.get_logger().info("🔍 ROS2 환경 확인 중...")
        try:
            result = subprocess.run(['ros2', 'node', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.get_logger().info("✅ ROS2 환경 정상")
            else:
                self.get_logger().error("❌ ROS2 환경 오류")
                return False
        except Exception as e:
            self.get_logger().error(f"❌ ROS2 환경 확인 실패: {e}")
            return False
            
        # 2. 노드 순차 실행
        for node_config in self.nodes_config:
            if not self.system_running:
                break
                
            success = self.launch_node(node_config)
            if not success and node_config['required']:
                self.get_logger().error(f"❌ 필수 노드 {node_config['name']} 시작 실패")
                self.shutdown_system()
                return False
                
        # 3. 모니터링 스레드 시작
        if self.system_running:
            self.startup_complete = True
            monitor_thread = threading.Thread(target=self.monitor_nodes)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            self.get_logger().info("🎉 Mobile VLA 시스템 시작 완료!")
            self.get_logger().info("📋 실행 중인 노드들:")
            for name, status in self.node_status.items():
                status_icon = "✅" if status else "❌"
                self.get_logger().info(f"   {status_icon} {name}")
                
            return True
            
        return False
        
    def shutdown_system(self):
        """시스템 종료"""
        self.get_logger().info("🛑 Mobile VLA 시스템 종료 시작")
        self.system_running = False
        
        # 각 노드 종료
        for name, process in self.node_processes.items():
            try:
                self.get_logger().info(f"🛑 {name} 노드 종료 중...")
                process.terminate()
                
                # 5초 대기 후 강제 종료
                try:
                    process.wait(timeout=5.0)
                    self.get_logger().info(f"✅ {name} 노드 정상 종료")
                except subprocess.TimeoutExpired:
                    self.get_logger().warn(f"⚠️ {name} 노드 강제 종료")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                self.get_logger().error(f"❌ {name} 노드 종료 중 오류: {e}")
                
        self.get_logger().info("🎉 Mobile VLA 시스템 종료 완료")
        
    def print_status(self):
        """시스템 상태 출력"""
        if not self.startup_complete:
            return
            
        self.get_logger().info("📊 Mobile VLA 시스템 상태:")
        for name, status in self.node_status.items():
            status_icon = "✅" if status else "❌"
            self.get_logger().info(f"   {status_icon} {name}")
            
        # ROS2 토픽 정보
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                self.get_logger().info(f"📡 활성 토픽 수: {len(topics)}")
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    launcher = MobileVLASystemLauncher()
    
    try:
        # 시스템 실행
        if launcher.launch_system():
            # 메인 루프
            while launcher.system_running:
                time.sleep(1.0)
                # 주기적 상태 출력 (선택사항)
                if time.time() % 30 < 1:  # 30초마다
                    launcher.print_status()
        else:
            launcher.get_logger().error("❌ 시스템 시작 실패")
            
    except KeyboardInterrupt:
        launcher.get_logger().info("📡 KeyboardInterrupt 수신")
    except Exception as e:
        launcher.get_logger().error(f"❌ 예상치 못한 오류: {e}")
    finally:
        launcher.shutdown_system()
        launcher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
