#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys, tty, termios
import os
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
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

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
        
        # RoboVLMs 일관성을 위한 18스텝 고정
        # RoboVLMs 설정: window_size=8, fwd_pred_next_n=10 → 총 18프레임 필요
        self.fixed_episode_length = 18   # RoboVLMs 표준 길이 (window_size + fwd_pred_next_n)
        
        # 데이터셋 분류 설정 (통계 모니터링용, 수집 목표와는 별개)
        # 실제 수집 목표는 18프레임 기준으로 설정됨
        self.categories = {
            "short": {"min": 1, "max": 10, "target": 100, "description": "짧은 에피소드"},
            "medium": {"min": 11, "max": 25, "target": 700, "description": "중간 에피소드"},  
            "long": {"min": 26, "max": 50, "target": 150, "description": "긴 에피소드"},
            "extra_long": {"min": 51, "max": float('inf'), "target": 50, "description": "매우 긴 에피소드"}
        }
        
        # 시간대별 수집 계획 (총 1000개 목표, 24시간 전체 커버)
        # 24시간을 4가지 시간대로 균등 분할
        self.time_period_targets = {
            "dawn": {"target": 200, "description": "새벽 (00:00-06:00)", "hour_range": (0, 6)},
            "morning": {"target": 200, "description": "아침 (06:00-12:00)", "hour_range": (6, 12)},
            "evening": {"target": 300, "description": "저녁 (12:00-18:00)", "hour_range": (12, 18)},
            "night": {"target": 300, "description": "밤 (18:00-24:00)", "hour_range": (18, 24)}
        }
        self.time_period_stats = defaultdict(int)  # 시간대별 통계
        
        # 4가지 탄산음료 페트병 도달 시나리오 목표 설정 (총 1000개 목표)
        # 시나리오별 목표: 각 250개 (1000개 ÷ 4개 시나리오)
        # 배치 타입(vert/hori)은 메타데이터로만 기록, 학습에는 영향 없음
        self.cup_scenarios = {
            "1box_left": {"target": 250, "description": "1박스-왼쪽경로", "key": "1"},
            "1box_right": {"target": 250, "description": "1박스-오른쪽경로", "key": "2"},
            "2box_left": {"target": 250, "description": "2박스-왼쪽경로", "key": "3"},
            "2box_right": {"target": 250, "description": "2박스-오른쪽경로", "key": "4"}
        }
        
        # 장애물 배치 타입 기본값 설정 (학습에 불필요하지만 호환성을 위해 기본값 사용)
        self.default_layout_type = "hori"  # 기본값: 가로 배치
        
        # 장애물(박스) 위치 단계 (3단계)
        # label: 안내용 텍스트, hint: 콘솔 힌트
        self.distance_levels = {
            "close":   {"label": "로봇과 가까운 위치",   "hint": "로봇 바로 앞에 가까운 장애물",
                         "samples_per_scenario": 3},
            "medium":  {"label": "중간 거리",          "hint": "장애물이 중간 거리에 배치",
                         "samples_per_scenario": 4},
            "far":     {"label": "로봇과 먼 위치",     "hint": "로봇에서 멀리 배치된 장애물",
                         "samples_per_scenario": 3}
        }
        
        self.dataset_stats = defaultdict(int)
        self.scenario_stats = defaultdict(int)
        # 패턴×거리(위치) 진행 통계: scenario -> pattern -> distance -> count
        self.pattern_distance_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # 시나리오당 목표(패턴/위치 분배) - 통합된 목표 (가로/세로 구분 없음)
        # 시나리오당 250개 목표를 패턴/거리별로 분배
        # Core: 150개 (60%), Variant: 100개 (40%)
        # 거리 분배: Core(50/75/25), Variant(25/25/50)
        self.pattern_targets = {"core": 150, "variant": 100}
        self.distance_targets_per_pattern = {
            "core": {"close": 50, "medium": 75, "far": 25},
            "variant": {"close": 25, "medium": 25, "far": 50},
        }
        
        # 시나리오/패턴/거리 선택 모드 및 상태 (배치 타입 제거로 단순화)
        self.scenario_selection_mode = False
        self.pattern_selection_mode = False
        self.distance_selection_mode = False
        self.repeat_count_mode = False  # 반복 횟수 입력 모드
        self.repeat_count_input = ""  # 입력 중인 숫자 문자열
        self.guide_edit_mode = False  # 가이드 편집 모드
        self.guide_edit_keys = []  # 편집 중인 가이드 키 시퀀스
        self.selected_scenario = None
        self.selected_pattern_type = None
        self.selected_distance_level = None
        self.current_repeat_index = 0  # 현재 반복 인덱스 (0이면 아직 시작 안함)
        self.target_repeat_count = 1  # 목표 반복 횟수
        self.is_repeat_measurement_active = False  # 반복 측정 활성 상태
        self.waiting_for_next_repeat = False  # 다음 반복 측정을 위한 시작 위치 세팅 대기 중

        # 핵심 패턴(표준) 관리
        # key: 시나리오 또는 시나리오__패턴__거리 (예: "1box_left__core__medium")
        self.core_patterns: Dict[str, List[str]] = {}
        self.core_guidance_active: bool = False
        self.core_guidance_index: int = 0
        self.current_episode_keys: List[str] = []
        self.record_core_pattern: bool = False
        self.overwrite_core: bool = False  # '핵심 표준 재등록' 토글 상태
        self.core_mismatch_count: int = 0  # 핵심 패턴 검증 불일치 카운트 (에피소드 단위)
        self.last_completed_episode_actions: List[str] = []  # 마지막 완료된 에피소드의 액션 시퀀스

        self.current_action = self.STOP_ACTION.copy()
        self.movement_timer = None
        self.movement_lock = threading.Lock()  # 타이머와 키 입력 동기화용 락
        # 명령 발행 추적용 변수
        self.command_counter = 0  # 명령 발행 카운터
        self.last_command_time = None  # 마지막 명령 발행 시간
        self.last_command_action = None  # 마지막 발행된 액션
        self.verbose_logging = False  # 상세 로깅 활성화 플래그
        # 자동 복귀 관련 변수
        self.auto_return_active = False  # 자동 복귀 모드 활성화 플래그
        self.return_thread = None  # 복귀 스레드
        # 자동 측정 관련 변수
        self.auto_measurement_active = False  # 자동 측정 모드 활성화 플래그
        self.auto_measurement_thread = None  # 자동 측정 스레드
        self.auto_measurement_queue = []  # 자동 측정할 태스크 큐
        self.auto_measurement_mode = False  # 자동 측정 모드 플래그 (선택 중)

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
        
        # 데이터 디렉토리 경로 설정 (환경변수 우선, 없으면 ROS_action 아래 별도 폴더 사용)
        # install/log/build 삭제 시 데이터가 보존되도록 ROS_action 바로 아래에 저장 (절대 경로 사용)
        # 현재 작업 디렉토리에 의존하지 않도록 절대 경로만 사용
        data_dir_env = os.environ.get('VLA_DATASET_DIR', None)
        if data_dir_env:
            # 환경변수 사용 시 절대 경로로 확실히 변환
            self.data_dir = Path(data_dir_env).expanduser().resolve()
        else:
            # ROS_action 바로 아래에 저장 (rm -rf install/log/build 시 데이터 보존)
            # 절대 경로로 ROS_action 찾기 (상대 경로 문제 방지, getcwd() 에러 방지)
            ros_action_dir = None
            
            # 방법 1: 현재 파일 위치에서 찾기 (절대 경로 사용, getcwd() 의존 없음)
            try:
                # __file__이 상대 경로일 수 있으므로 절대 경로로 확실히 변환
                current_file_abs = os.path.abspath(os.path.expanduser(__file__))
                current_file = Path(current_file_abs).resolve()
                # src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py
                # -> ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py
                candidate = current_file.parent.parent.parent.parent
                if candidate.exists() and candidate.is_absolute() and candidate.name == "ROS_action":
                    ros_action_dir = candidate
            except (OSError, ValueError, AttributeError) as e:
                self.get_logger().warn(f"⚠️ 현재 파일 위치에서 ROS_action 찾기 실패: {e}")
            
            # 방법 2: 홈 디렉토리 기준으로 찾기 (절대 경로)
            if ros_action_dir is None or not ros_action_dir.exists():
                try:
                    candidate = Path.home().resolve() / "vla" / "ROS_action"
                    if candidate.exists() and candidate.is_absolute():
                        ros_action_dir = candidate
                except (OSError, ValueError) as e:
                    self.get_logger().warn(f"⚠️ 홈 디렉토리 기준으로 ROS_action 찾기 실패: {e}")
            
            # 방법 3: 절대 경로 직접 지정 (getcwd() 의존 없음)
            if ros_action_dir is None or not ros_action_dir.exists():
                candidate = Path("/home/soda/vla/ROS_action")
                if candidate.exists() and candidate.is_absolute():
                    ros_action_dir = candidate
            
            if ros_action_dir is None or not ros_action_dir.exists():
                raise RuntimeError(f"❌ ROS_action 디렉토리를 찾을 수 없습니다. 환경변수 VLA_DATASET_DIR을 설정하거나, 올바른 위치에 설치하세요.")
            
            # ROS_action 바로 아래에 저장 (절대 경로 사용, resolve()로 확실히 절대 경로 보장)
            self.data_dir = ros_action_dir.resolve() / "mobile_vla_dataset"
            # 한 번 더 resolve()하여 절대 경로 확실히 보장
            self.data_dir = self.data_dir.resolve()
            
            # 기존 install/mobile_vla_dataset 경로 호환성: 데이터 마이그레이션 (절대 경로 사용)
            old_data_dir = ros_action_dir.resolve() / "install" / "mobile_vla_dataset"
            if old_data_dir.exists():
                if not self.data_dir.exists():
                    # 새 위치가 없으면 전체 폴더 이동
                    self.get_logger().info(f"🔄 기존 데이터 마이그레이션: {old_data_dir} → {self.data_dir}")
                    try:
                        import shutil
                        shutil.move(str(old_data_dir), str(self.data_dir))
                        self.get_logger().info(f"✅ 데이터 마이그레이션 완료: {self.data_dir}")
                    except Exception as e:
                        self.get_logger().warn(f"⚠️ 데이터 마이그레이션 실패: {e}. 새 위치를 사용합니다.")
                else:
                    # 둘 다 있으면 기존 위치의 파일들을 새 위치로 병합
                    old_h5_files = list(old_data_dir.glob("*.h5"))
                    old_json_files = list(old_data_dir.glob("*.json"))
                    if old_h5_files or old_json_files:
                        self.get_logger().info(f"🔄 기존 위치 데이터 병합: {old_data_dir} → {self.data_dir}")
                        try:
                            import shutil
                            moved_count = 0
                            for f in old_h5_files + old_json_files:
                                dest = self.data_dir / f.name
                                if not dest.exists():
                                    shutil.move(str(f), str(dest))
                                    moved_count += 1
                                else:
                                    self.get_logger().debug(f"   파일 건너뜀 (이미 존재): {f.name}")
                            if moved_count > 0:
                                self.get_logger().info(f"✅ {moved_count}개 파일 병합 완료")
                            # 병합 후 빈 폴더면 삭제 시도
                            try:
                                if not any(old_data_dir.iterdir()):
                                    old_data_dir.rmdir()
                                    self.get_logger().info(f"🗑️ 빈 기존 폴더 삭제: {old_data_dir}")
                            except:
                                pass
                        except Exception as e:
                            self.get_logger().warn(f"⚠️ 데이터 병합 실패: {e}. 기존 위치 파일은 그대로 유지됩니다.")
            
            # install 경로 사용 방지 확인 (절대 경로로 확인)
            if str(self.data_dir).endswith("/install/mobile_vla_dataset") or "install/mobile_vla_dataset" in str(self.data_dir):
                self.get_logger().error(f"❌ 잘못된 경로: install 안에 저장되지 않도록 설정되었습니다!")
                raise RuntimeError(f"❌ 데이터 디렉토리가 install 안에 있으면 안 됩니다: {self.data_dir}")
        
        # 부모 디렉토리까지 생성 (parents=True)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"📁 데이터 디렉토리: {self.data_dir}")
        except Exception as e:
            self.get_logger().error(f"❌ 데이터 디렉토리 생성 실패: {e}")
            raise
        
        # 진행상황 저장 파일 (data_dir 정의 후)
        self.progress_file = self.data_dir / "scenario_progress.json"
        self.time_period_file = self.data_dir / "time_period_stats.json"
        self.core_pattern_file = self.data_dir / "core_patterns.json"
        
        # 데이터셋 통계 로드
        self.load_dataset_stats()
        self.load_scenario_progress()
        self.load_time_period_stats()
        self.load_core_patterns()
        
        self.get_logger().info("🤖 Mobile VLA Data Collector 준비 완료!")
        self.get_logger().info("📋 조작 방법:")
        self.get_logger().info("   W/A/S/D: 이동, Q/E/Z/C: 대각선")
        self.get_logger().info("   R/T: 회전, 스페이스바: 정지")
        self.get_logger().info("   F/G: 속도 조절, N: 새 에피소드 시작")
        self.get_logger().info("   M: 에피소드 종료, P: 현재 진행 상황 확인")
        self.get_logger().info("   V: H5 파일 검증 및 추출 (최신 파일 또는 선택)")
        self.get_logger().info("   X: 리셋 (첫 화면으로 돌아가기, 수집 중에도 가능)")
        self.get_logger().info("   B: 자동 복귀 (에피소드 종료 후 시작 위치로 복귀)")
        self.get_logger().info("   A: 자동 측정 (가이드 기반 자동 측정)")
        self.get_logger().info("   T: 측정 태스크 표 보기")
        self.get_logger().info("🎯 수집 단계: N → 시나리오(1-4) → 패턴(C/V) → 장애물 위치(J/K/L)")
        self.get_logger().info("🎯 탄산음료 페트병 도달 시나리오 (총 1000개 목표):")
        self.get_logger().info("   📦 4개 시나리오 × 250개 샘플 × 18프레임 고정 (RoboVLMs 기준: window=8 + pred_next=10)")
        self.get_logger().info("   🎯 수집 목표: 18프레임 기준 (RoboVLMs 학습에 최적화)")
        self.get_logger().info("   💡 총 목표: 1000개 (시나리오당 250개)")
        self.get_logger().info("   🌅 시간대 분포: 새벽(200) + 아침(200) + 저녁(300) + 밤(300)")
        self.get_logger().info("   🔬 패턴 분포: Core(150) + Variant(100) / 시나리오")
        self.get_logger().info("   📊 카테고리 분류: 데이터셋 통계 모니터링용 (수집 목표와는 별개)")
        self.get_logger().info("   ✨ 단순화: 배치 타입 선택 단계 제거 (학습에 불필요)")
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
            # 반복 측정이 활성화되어 있고 다음 측정을 기다리는 중이면 다음 측정 시작
            elif self.is_repeat_measurement_active and self.waiting_for_next_repeat:
                if self.current_repeat_index < self.target_repeat_count:
                    self.waiting_for_next_repeat = False
                    if self.auto_measurement_mode:
                        # 자동 측정 모드: 다음 측정 시작
                        # 인덱스 증가 (execute_auto_measurement에서도 확인하지만 여기서 먼저 증가)
                        self.current_repeat_index += 1
                        scenario_id = self.selected_scenario
                        pattern_type = self.selected_pattern_type
                        distance_level = self.selected_distance_level
                        
                        # 자동 측정을 별도 스레드에서 실행
                        self.auto_measurement_active = True
                        self.auto_measurement_thread = threading.Thread(
                            target=self.execute_auto_measurement,
                            args=(scenario_id, pattern_type, distance_level)
                        )
                        self.auto_measurement_thread.daemon = True
                        self.auto_measurement_thread.start()
                    else:
                        # 일반 모드: 수동 측정 시작
                        self.start_next_repeat_measurement()
                else:
                    # 모든 반복 완료
                    self.get_logger().info(f"🎉 모든 반복 측정 완료! ({self.target_repeat_count}회)")
                    self.is_repeat_measurement_active = False
                    self.current_repeat_index = 0
                    self.waiting_for_next_repeat = False
                    self.auto_measurement_mode = False
                    self.show_scenario_selection()
            else:
                # 일반 모드: 시나리오 선택
                self.show_scenario_selection()
        elif key == 'm':
            if self.collecting:
                self.stop_episode()
        elif key == 'p':
            self.resync_and_show_progress()
        elif key == 'v':
            if self.collecting:
                self.get_logger().warn("⚠️ 수집 중에는 H5 파일 검증을 할 수 없습니다. 먼저 M키로 에피소드를 종료하세요.")
            else:
                self.show_h5_verification_menu()
        elif key == 'x':
            if self.guide_edit_mode:
                # 가이드 편집 취소
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.guide_edit_mode = False
                self.guide_edit_keys = []
                self.get_logger().info("🚫 가이드 편집이 취소되었습니다. 기존 가이드를 유지합니다.")
                # 반복 횟수 입력 모드로 돌아가기
                self.show_repeat_count_selection()
            else:
                # 리셋 기능: 모든 상태 초기화하고 첫 화면으로 돌아가기
                self.reset_to_initial_state()
        elif key == 'b':
            # 자동 복귀 기능: 에피소드 종료 후 시작 위치로 복귀
            if self.collecting:
                self.get_logger().warn("⚠️ 수집 중에는 자동 복귀를 할 수 없습니다. 먼저 M키로 에피소드를 종료하세요.")
            elif self.auto_return_active:
                # 복귀 중단
                self.get_logger().info("🛑 자동 복귀를 중단합니다...")
                self.auto_return_active = False
                # 정지 신호 전송
                self.current_action = self.STOP_ACTION.copy()
                for _ in range(3):
                    self.publish_cmd_vel(self.STOP_ACTION, source="auto_return_cancel")
                    time.sleep(0.02)
            elif len(self.episode_data) == 0:
                self.get_logger().warn("⚠️ 복귀할 경로가 없습니다. 먼저 에피소드를 수집하세요.")
            else:
                self.start_auto_return()
        elif key == 't':
            # 측정 태스크 표 보기
            self.show_measurement_task_table()
        elif key in ['1', '2', '3', '4']:
            if self.scenario_selection_mode:
                # 시나리오 선택 모드에서 숫자키 입력 (4개 시나리오로 축소)
                scenario_map = {
                    '1': "1box_left", '2': "1box_right",
                    '3': "2box_left", '4': "2box_right"
                }
                self.selected_scenario = scenario_map[key]
                self.scenario_selection_mode = False  # 시나리오 선택 모드 해제
                if self.auto_measurement_mode:
                    # 자동 측정 모드: 패턴 선택으로 바로 이동
                    self.show_pattern_selection()
                else:
                    # 일반 모드: 패턴 선택으로 전환
                    self.show_pattern_selection()
            else:
                self.get_logger().info("⚠️ 먼저 'N' 키를 눌러 에피소드 시작을 해주세요.")
        elif key in ['c', 'v']:
            if self.pattern_selection_mode:
                # 패턴 선택 모드에서 c/v 키 입력
                pattern_map = {
                    'c': "core",      # 핵심 패턴
                    'v': "variant"   # 변형 패턴  
                }
                pattern_type = pattern_map[key]
                self.pattern_selection_mode = False  # 패턴 선택 모드 해제
                self.selected_pattern_type = pattern_type
                
                if self.auto_measurement_mode:
                    # 자동 측정 모드: 핵심 패턴만 지원
                    if pattern_type == "variant":
                        self.get_logger().warn("⚠️ 자동 측정은 핵심 패턴(Core)만 지원합니다. 'C' 키를 눌러주세요.")
                        self.show_pattern_selection()  # 다시 패턴 선택
                    else:
                        # 핵심 패턴 선택 시 거리 선택으로 전환
                        self.show_distance_selection()
                else:
                    # 일반 모드: 거리 선택으로 전환
                    self.show_distance_selection()
            else:
                # 패턴 선택 모드가 아닐 때는 일반 대각선 움직임으로 처리
                if key == 'c':
                    # C키가 패턴 선택에 사용되지 않을 때만 움직임으로 처리
                    pass  # 아래 WASD 처리로 넘어감
        elif key in ['j', 'k', 'l']:
            if self.distance_selection_mode:
                # 거리 선택 모드: j=근거리, k=중거리, l=원거리
                distance_map = {'j': 'close', 'k': 'medium', 'l': 'far'}
                self.selected_distance_level = distance_map[key]
                self.distance_selection_mode = False
                
                if self.auto_measurement_mode:
                    # 자동 측정 모드: 반복 횟수 입력 모드로 전환
                    self.show_repeat_count_selection()
                else:
                    # 일반 모드: 반복 횟수 입력 모드로 전환
                    self.show_repeat_count_selection()
            elif self.repeat_count_mode:
                # 반복 횟수 입력 모드에서는 거리 선택 키는 무시
                pass
        elif key == 'f':
            if ROBOT_AVAILABLE:
                self.throttle = max(10, self.throttle - 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key == 'g':
            if self.guide_edit_mode:
                # 가이드 편집 모드에서는 G 키를 대각선 이동으로 처리하지 않음
                pass
            elif ROBOT_AVAILABLE:
                self.throttle = min(100, self.throttle + 10)
                self.get_logger().info(f'속도: {self.throttle}%')
        elif key == 'h':
            # 가이드 편집 모드 진입 (핵심 패턴이고 반복 횟수 입력 모드일 때만)
            if self.selected_pattern_type == "core" and self.repeat_count_mode:
                # 반복 횟수 입력 모드 취소
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.repeat_count_mode = False
                self.repeat_count_input = ""
                # 가이드 편집 모드 진입
                self.show_guide_edit_menu()
            elif self.guide_edit_mode:
                # 가이드 편집 모드에서는 H 키를 이동 키로 처리하지 않음
                pass
            else:
                # 다른 상황에서는 무시
                pass
        elif key == 'u':
            # 방금 수집한 키 입력을 가이드로 저장 (핵심 패턴이고 반복 횟수 입력 모드일 때만)
            if self.selected_pattern_type == "core" and self.repeat_count_mode and self.last_completed_episode_actions:
                # 반복 횟수 입력 모드 취소
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.repeat_count_mode = False
                self.repeat_count_input = ""
                
                # 마지막 완료된 에피소드의 액션을 가이드로 저장
                combo_key = f"{self.selected_scenario}__{self.selected_pattern_type}__{self.selected_distance_level}"
                # 18키로 정규화
                normalized_keys = self._normalize_to_18_keys(self.last_completed_episode_actions)
                self.core_patterns[combo_key] = normalized_keys
                self.save_core_patterns()
                
                guide_str = " ".join([k.upper() for k in normalized_keys])
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"✅ 가이드 갱신 완료: {combo_key}")
                self.get_logger().info(f"🎮 새 가이드: {guide_str}")
                self.get_logger().info("=" * 60)
                
                # 반복 횟수 입력 모드로 돌아가기
                self.show_repeat_count_selection()
            elif self.guide_edit_mode:
                # 가이드 편집 모드에서는 U 키를 이동 키로 처리하지 않음
                pass
            else:
                # 다른 상황에서는 무시
                pass
        elif key == '\r' or key == '\n':  # Enter 키
            if self.guide_edit_mode:
                # 가이드 편집 완료
                sys.stdout.write("\n")
                sys.stdout.flush()
                
                if self.save_edited_guide():
                    # 가이드 저장 성공 시 반복 횟수 입력 모드로 돌아가기
                    self.guide_edit_mode = False
                    self.guide_edit_keys = []
                    self.show_repeat_count_selection()
                else:
                    # 저장 실패 시 다시 편집 모드 유지
                    self.show_guide_edit_menu()
            elif self.repeat_count_mode:
                # 입력 줄 완료 표시
                sys.stdout.write("\n")
                sys.stdout.flush()
                
                # 반복 횟수 입력 완료
                if self.repeat_count_input == "":
                    # 빈 입력이면 1회로 설정
                    self.target_repeat_count = 1
                    self.get_logger().info("📝 입력된 횟수: 1 (기본값)")
                else:
                    try:
                        self.target_repeat_count = int(self.repeat_count_input)
                        if self.target_repeat_count <= 0:
                            self.get_logger().warn("⚠️ 반복 횟수는 1 이상이어야 합니다. 1회로 설정합니다.")
                            self.target_repeat_count = 1
                        elif self.target_repeat_count > 100:
                            self.get_logger().warn("⚠️ 반복 횟수는 100 이하여야 합니다. 100회로 제한합니다.")
                            self.target_repeat_count = 100
                        else:
                            self.get_logger().info(f"📝 입력된 횟수: {self.target_repeat_count}")
                    except ValueError:
                        self.get_logger().warn("⚠️ 잘못된 입력입니다. 1회로 설정합니다.")
                        self.target_repeat_count = 1
                
                self.repeat_count_mode = False
                self.repeat_count_input = ""
                self.current_repeat_index = 0
                self.is_repeat_measurement_active = True
                
                # 첫 번째 측정 시작
                if self.auto_measurement_mode:
                    # 자동 측정 모드: 첫 번째 측정 시작 (인덱스는 execute_auto_measurement에서 증가)
                    scenario_id = self.selected_scenario
                    pattern_type = self.selected_pattern_type
                    distance_level = self.selected_distance_level
                    
                    # 자동 측정을 별도 스레드에서 실행
                    self.auto_measurement_active = True
                    self.auto_measurement_thread = threading.Thread(
                        target=self.execute_auto_measurement,
                        args=(scenario_id, pattern_type, distance_level)
                    )
                    self.auto_measurement_thread.daemon = True
                    self.auto_measurement_thread.start()
                else:
                    # 일반 모드: 수동 측정 시작
                    self.start_next_repeat_measurement()
        elif key == '\x7f' or key == '\b' or key == '\x08':  # 백스페이스 키
            if self.guide_edit_mode:
                if len(self.guide_edit_keys) > 0:
                    # 마지막 키 삭제
                    self.guide_edit_keys.pop()
                    # 화면 업데이트
                    guide_str = " ".join([k.upper() for k in self.guide_edit_keys])
                    sys.stdout.write('\r' + ' ' * 80)  # 줄 지우기
                    sys.stdout.write(f'\r📝 가이드 입력: {guide_str}')
                    sys.stdout.flush()
            elif self.repeat_count_mode:
                if len(self.repeat_count_input) > 0:
                    # 마지막 문자 삭제
                    self.repeat_count_input = self.repeat_count_input[:-1]
                    # 화면 업데이트: 현재 줄을 지우고 다시 표시
                    sys.stdout.write('\r' + ' ' * 50)  # 줄 지우기
                    sys.stdout.write('\r📝 반복 횟수: ' + self.repeat_count_input)
                    sys.stdout.flush()
        elif key.isdigit():
            if self.guide_edit_mode:
                # 가이드 편집 모드에서는 숫자 입력 무시
                pass
            elif self.repeat_count_mode:
                # 숫자 입력 (최대 3자리)
                if len(self.repeat_count_input) < 3:
                    self.repeat_count_input += key
                    # 현재 줄을 업데이트 (커서가 깜빡이는 효과)
                    sys.stdout.write('\r📝 반복 횟수: ' + self.repeat_count_input)
                    sys.stdout.flush()
                else:
                    # 최대 자리수 초과 시 경고음 효과 (화면에 표시)
                    sys.stdout.write('\a')  # 벨 문자
                    sys.stdout.flush()
            elif self.scenario_selection_mode or self.pattern_selection_mode or self.distance_selection_mode:
                # 선택 모드 중에는 숫자 입력 무시
                pass
        elif key in self.WASD_TO_CONTINUOUS:
            # 이동 키 처리 (가이드 편집 모드, 선택 모드, 반복 횟수 입력 모드 우선 처리)
            if self.guide_edit_mode:
                # 가이드 편집 모드: 키를 가이드에 추가
                max_guide_actions = self.fixed_episode_length - 1  # 18 - 1 = 17 (초기 프레임 제외)
                if len(self.guide_edit_keys) < max_guide_actions:
                    # 키를 소문자로 변환하여 저장 (SPACE는 그대로)
                    if key == ' ':
                        guide_key = 'SPACE'
                    else:
                        guide_key = key.lower()
                    self.guide_edit_keys.append(guide_key)
                    # 화면 업데이트
                    guide_str = " ".join([k.upper() for k in self.guide_edit_keys])
                    sys.stdout.write('\r' + ' ' * 80)  # 줄 지우기
                    sys.stdout.write(f'\r📝 가이드 입력: {guide_str}')
                    sys.stdout.flush()
                else:
                    # 최대 길이 도달
                    sys.stdout.write('\a')  # 벨 문자
                    sys.stdout.flush()
                return
            elif self.scenario_selection_mode or self.pattern_selection_mode or self.distance_selection_mode:
                self.scenario_selection_mode = False
                self.pattern_selection_mode = False
                self.distance_selection_mode = False
                self.get_logger().info("🚫 선택이 취소되었습니다.")
                return
            elif self.repeat_count_mode:
                # 반복 횟수 입력 모드에서는 이동 키로 입력 취소
                sys.stdout.write("\n")  # 입력 줄 완료
                sys.stdout.flush()
                self.repeat_count_mode = False
                self.repeat_count_input = ""
                self.get_logger().info("🚫 반복 횟수 입력이 취소되었습니다.")
                return
            
            # 일반 수집(N 키 루프) 중일 때는 A 키를 이동 키로 처리
            # 가이드 편집 모드, 선택 모드, 반복 횟수 입력 모드가 아닐 때만 A 키 자동 측정 처리
            if key == 'a' and not (self.collecting and not self.auto_measurement_mode):
                # A 키이지만 일반 수집 중이 아닌 경우: 자동 측정 기능 처리
                if self.collecting:
                    self.get_logger().warn("⚠️ 수집 중에는 자동 측정을 시작할 수 없습니다. 먼저 M키로 에피소드를 종료하세요.")
                    return
                elif self.auto_measurement_active:
                    # 자동 측정 중단
                    self.get_logger().info("🛑 자동 측정을 중단합니다...")
                    self.auto_measurement_active = False
                    # 🔴 반복 측정 상태도 모두 리셋
                    self.is_repeat_measurement_active = False
                    self.waiting_for_next_repeat = False
                    self.current_repeat_index = 0
                    self.target_repeat_count = 1
                    self.auto_measurement_mode = False
                    # 정지 신호 전송
                    self.current_action = self.STOP_ACTION.copy()
                    for _ in range(3):
                        self.publish_cmd_vel(self.STOP_ACTION, source="auto_measurement_cancel")
                        time.sleep(0.02)
                    return
                else:
                    self.show_auto_measurement_menu()
                    return  # 자동 측정 메뉴를 표시했으므로 여기서 종료
                
            action = self.WASD_TO_CONTINUOUS[key]
            # 현재 에피소드 키 기록 (핵심 패턴 녹화/가이드 용)
            if self.collecting:
                self.current_episode_keys.append(key)
            
            # 🔴 이전 타이머 취소 및 강제 정지 처리 (ROS 버퍼 문제 방지)
            # 락을 사용하여 타이머와 키 입력 동기화
            timer_was_active = False
            timer_info = ""
            with self.movement_lock:
                if self.movement_timer is not None:
                    if self.movement_timer.is_alive():
                        timer_was_active = True
                        timer_info = f" | 기존 타이머 활성 상태: True (취소 예정)"
                        self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 상태 확인: is_alive()=True, 취소 시작...")
                        try:
                            cancel_result = self.movement_timer.cancel()
                            timer_info += f" | 취소 결과: {cancel_result}"
                            self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 취소 시도: cancel()={cancel_result}")
                        except Exception as e:
                            timer_info += f" | 취소 실패: {e}"
                            self.get_logger().error(f"❌ [키입력:{key.upper()}] 타이머 취소 중 오류: {e}")
                        self.movement_timer = None  # 참조 제거로 메모리 누수 방지
                    else:
                        timer_info = f" | 기존 타이머 활성 상태: False (이미 종료됨)"
                        self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 상태 확인: is_alive()=False (이미 종료됨)")
                        self.movement_timer = None
                else:
                    timer_info = f" | 기존 타이머: None (없음)"
                    self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 상태 확인: None (타이머 없음)")
                
                # 🔴 타이머가 실행 중이었으면 강제 정지 후 안정화 대기
                if timer_was_active:
                    self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머가 실행 중이었으므로 강제 정지 실행...")
                    self.stop_movement_internal(collect_data=False)
                    time.sleep(0.1)  # 안정화 대기
            
            if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50) or timer_was_active:
                self.get_logger().info(f"⏱️  기존 타이머 처리 완료 (키 입력: {key.upper()}){timer_info}")
            
            # 🔴 현재 액션 상태 확인 및 강제 정지 처리
            # 현재 정지 상태가 아니거나, 수집 중이 아니면 반드시 정지 상태로 만들어야 함
            was_moving = (self.current_action != self.STOP_ACTION)
            
            if was_moving:
                if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                    prev_action = self.current_action
                    self.get_logger().info(
                        f"🛑 강제 정지 시작 (이전 액션: lx={prev_action['linear_x']:.2f}, "
                        f"ly={prev_action['linear_y']:.2f}, az={prev_action['angular_z']:.2f})"
                    )
                
                self.current_action = self.STOP_ACTION.copy()
                # 여러 번 발행하여 ROS 버퍼와 하드웨어에 확실히 전달
                for i in range(3):
                    self.publish_cmd_vel(self.STOP_ACTION, source=f"key_input_stop_{i+1}")
                    time.sleep(0.02)  # 각 신호 사이 딜레이 (버퍼 플러시)
                
                # 🔴 추가 안정화 대기 (로봇이 완전히 정지할 시간 확보)
                # 첫 번째 키 입력 시 특히 중요 (에피소드 시작 직후)
                if self.collecting and len(self.episode_data) <= 1:
                    # 첫 번째 또는 두 번째 데이터 포인트일 때 더 긴 대기
                    time.sleep(0.08)  # 첫 동작 전 더 긴 안정화 시간
                else:
                    time.sleep(0.05)  # 일반적인 경우
                
                if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                    self.get_logger().info(f"✅ 강제 정지 완료 (3회 발행, 안정화 대기 완료)")
            else:
                # 이미 정지 상태여도 한 번 더 정지 신호 전송 (안전장치)
                if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                    self.get_logger().info(f"🛑 이미 정지 상태, 추가 정지 신호 전송 (안전장치)")
                self.publish_cmd_vel(self.STOP_ACTION, source="key_input_safety_stop")
                time.sleep(0.03)  # 짧은 안정화 대기

            # 🔴 새 액션 시작 (정지 상태 확인 후)
            if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                self.get_logger().info(
                    f"▶️  새 액션 시작 (키: {key.upper()}, "
                    f"lx={action['linear_x']:.2f}, ly={action['linear_y']:.2f}, az={action['angular_z']:.2f})"
                )
            
            self.current_action = action.copy()
            
            # 🔴 새 타이머 먼저 시작 (블로킹 전에 타이머 설정)
            # 타이머를 먼저 시작하여 이미지 수집 블로킹과 무관하게 정지 보장
            # 기존 타이머는 이미 취소되었으므로 새로 생성 (락 사용)
            timer_duration = 0.4
            self.get_logger().info(f"🔍 [키입력:{key.upper()}] 새 타이머 생성 시작: duration={timer_duration}초")
            try:
                with self.movement_lock:
                    self.movement_timer = threading.Timer(timer_duration, self.stop_movement_timed)
                    self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 객체 생성 완료: {self.movement_timer}")
                    self.movement_timer.start()
                    self.get_logger().info(f"🔍 [키입력:{key.upper()}] 타이머 start() 호출 완료, is_alive()={self.movement_timer.is_alive()}")
            except Exception as e:
                self.get_logger().error(f"❌ [키입력:{key.upper()}] 타이머 생성/시작 중 오류: {e}")
                import traceback
                self.get_logger().error(f"❌ 트레이스백:\n{traceback.format_exc()}")
            
            if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                self.get_logger().info(f"⏱️  타이머 시작: {timer_duration}초 후 자동 정지 예약 (타이머 객체: {self.movement_timer}, is_alive: {self.movement_timer.is_alive() if self.movement_timer else 'N/A'})")
            
            # 🔴 타이머 시작 후 이동 명령 발행 및 데이터 수집
            self.publish_cmd_vel(action, source=f"key_input_{key}")

            if self.collecting:
                self.collect_data_point_with_action("start_action", action)
            
        elif key == ' ':
            if self.verbose_logging or (self.collecting and len(self.episode_data) >= 50):
                self.get_logger().info("🛑 스페이스바: 수동 정지 명령")
            self.stop_movement_internal(collect_data=True) 
            self.get_logger().info("🛑 정지")

    def stop_movement_timed(self):
        """Stop function called by the timer - NO data collection for auto-stop"""
        import threading
        current_thread = threading.current_thread().name
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # 🔍 상세 디버깅 로그 (항상 출력)
        self.get_logger().info(f"🔍 [타이머콜백] {timestamp} | Thread: {current_thread} | stop_movement_timed() 호출됨")
        
        # 상세 로깅 활성화 여부 확인
        should_log_verbose = self.verbose_logging or (self.collecting and len(self.episode_data) >= 50)
        
        if should_log_verbose:
            self.get_logger().info(
                f"⏰ [TIMER] {timestamp} | Thread: {current_thread} | "
                f"타이머 콜백 실행됨"
            )
        
        # 🔴 타이머 콜백 실행 시 안전성 체크 강화
        # 타이머가 이미 취소되었거나 현재 정지 상태면 리턴 (중복 호출 방지)
        current_action_str = f"lx={self.current_action['linear_x']:.2f}, ly={self.current_action['linear_y']:.2f}, az={self.current_action['angular_z']:.2f}"
        self.get_logger().info(f"🔍 [타이머콜백] 현재 액션 상태 확인: {current_action_str}")
        
        if self.current_action == self.STOP_ACTION:
            self.get_logger().info(f"🔍 [타이머콜백] ⏭️  이미 정지 상태, 타이머 콜백 스킵")
            if should_log_verbose:
                self.get_logger().info(f"   ⏭️  이미 정지 상태, 타이머 콜백 스킵")
            return
        
        # 🔴 타이머가 취소되었는지 확인 (타이머 객체가 여전히 유효한지, 락 사용)
        timer_status = "None"
        with self.movement_lock:
            if self.movement_timer is not None:
                is_alive = self.movement_timer.is_alive()
                timer_status = f"is_alive()={is_alive}"
                self.get_logger().info(f"🔍 [타이머콜백] 타이머 상태 확인: movement_timer={self.movement_timer}, {timer_status}")
            else:
                self.get_logger().info(f"🔍 [타이머콜백] 타이머 상태 확인: movement_timer=None")
            
            if self.movement_timer and not self.movement_timer.is_alive():
                # 타이머가 이미 취소되었으면 리턴
                self.get_logger().info(f"🔍 [타이머콜백] ⏭️  타이머가 이미 취소됨, 콜백 스킵")
                if should_log_verbose:
                    self.get_logger().info(f"   ⏭️  타이머가 이미 취소됨, 콜백 스킵")
                return
        
        self.get_logger().info(f"🔍 [타이머콜백] stop_movement_internal() 호출 시작...")
        
        # 현재 액션 상태 로깅
        if should_log_verbose:
            current_action = self.current_action
            self.get_logger().info(
                f"   📊 현재 액션 상태: lx={current_action['linear_x']:.2f}, "
                f"ly={current_action['linear_y']:.2f}, az={current_action['angular_z']:.2f}"
            )
        
        # 🔴 ROS 버퍼 문제 방지를 위해 여러 번 정지 신호 발행
        self.stop_movement_internal(collect_data=False)
        self.get_logger().info(f"🔍 [타이머콜백] stop_movement_internal() 호출 완료, 추가 정지 신호 발행 시작...")
        
        # 추가로 여러 번 정지 신호 발행 (ROS 버퍼 보장, 2회 → 3회)
        for i in range(3):
            self.get_logger().info(f"🔍 [타이머콜백] 추가 정지 신호 {i+1}/3 발행 중...")
            self.publish_cmd_vel(self.STOP_ACTION, source=f"timer_extra_stop_{i+1}")
            time.sleep(0.05)  # 딜레이 증가 (0.01초 → 0.05초)
        
        self.get_logger().info(f"🔍 [타이머콜백] ✅ 타이머 기반 정지 완료 (총 8회 정지 명령 발행)")
        if should_log_verbose:
            self.get_logger().info(f"   ✅ 타이머 기반 정지 완료 (총 8회 정지 명령 발행)")

    def stop_movement_internal(self, collect_data: bool):
        """
        Internal function to stop robot movement and collect data if needed.
        collect_data: If True, collects data at the time of stopping.
        """
        import threading
        current_thread = threading.current_thread().name
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        should_log_verbose = self.verbose_logging or (self.collecting and len(self.episode_data) >= 50)
        
        # 🔍 상세 디버깅 로그 (항상 출력)
        prev_action_str = f"lx={self.current_action['linear_x']:.2f}, ly={self.current_action['linear_y']:.2f}, az={self.current_action['angular_z']:.2f}"
        self.get_logger().info(f"🔍 [STOP_INTERNAL] {timestamp} | Thread: {current_thread} | 호출됨 | collect_data={collect_data} | 이전 액션: {prev_action_str}")
        
        # 🔴 이미 정지 상태면 리턴 (중복 호출 방지)
        if self.current_action == self.STOP_ACTION:
            self.get_logger().info(f"🔍 [STOP_INTERNAL] ⏭️  이미 정지 상태, 스킵")
            if should_log_verbose:
                self.get_logger().info(f"   ⏭️  stop_movement_internal: 이미 정지 상태, 스킵")
            return

        prev_action = self.current_action.copy()
        if should_log_verbose:
            self.get_logger().info(
                f"🛑 [STOP_INTERNAL] {timestamp} | Thread: {current_thread} | "
                f"정지 시작 (이전: lx={prev_action['linear_x']:.2f}, "
                f"ly={prev_action['linear_y']:.2f}, az={prev_action['angular_z']:.2f})"
            )

        self.current_action = self.STOP_ACTION.copy()
        self.get_logger().info(f"🔍 [STOP_INTERNAL] 정지 명령 발행 시작 (5회)...")
        
        # 🔴 ROS 버퍼 문제 방지를 위해 여러 번 정지 신호 발행 (더 강화: 3회 → 5회)
        for i in range(5):
            self.get_logger().info(f"🔍 [STOP_INTERNAL] 정지 신호 {i+1}/5 발행 중...")
            self.publish_cmd_vel(self.STOP_ACTION, source=f"stop_internal_{i+1}")
            time.sleep(0.05)  # 각 신호 사이 딜레이 증가 (0.02초 → 0.05초)
        
        self.get_logger().info(f"🔍 [STOP_INTERNAL] 정지 명령 발행 완료 (5회)")
        
        # 🔴 추가 안정화 대기 (로봇이 완전히 정지할 시간 확보, 0.03초 → 0.1초)
        time.sleep(0.1)
        
        if should_log_verbose:
            self.get_logger().info(f"   ✅ stop_movement_internal 완료 (5회 발행, 안정화 대기 완료)")

        if self.collecting and collect_data:
            self.collect_data_point_with_action("stop_action", self.STOP_ACTION)

    def publish_cmd_vel(self, action: Dict[str, float], source: str = "unknown"):
        """
        Publishes Twist message and controls the actual robot
        
        Args:
            action: 액션 딕셔너리
            source: 명령 발행 소스 (디버깅용)
        """
        import threading
        current_thread = threading.current_thread().name
        
        # 명령 발행 추적
        current_time = time.time()
        self.command_counter += 1
        
        # 🔍 조건 1: 명령을 늦게 받았는지 확인
        time_since_last_command = None
        if self.last_command_time is not None:
            time_since_last_command = current_time - self.last_command_time
        
        # 🔍 조건 3: 다른 명령을 보고 멈췄는지 확인 (이전 명령과 비교)
        # 비교는 저장하기 전에 수행해야 함
        unexpected_command = False
        if self.last_command_action is not None:
            prev_action = self.last_command_action
            # 이전 명령이 STOP이 아니었는데, 현재 명령이 다른 액션이면 예상치 못한 명령
            prev_is_stop = (abs(prev_action["linear_x"]) < 0.01 and 
                           abs(prev_action["linear_y"]) < 0.01 and 
                           abs(prev_action["angular_z"]) < 0.01)
            curr_is_stop = (abs(action["linear_x"]) < 0.01 and 
                           abs(action["linear_y"]) < 0.01 and 
                           abs(action["angular_z"]) < 0.01)
            if not prev_is_stop and not curr_is_stop:
                # 이전 액션과 현재 액션이 다르면 예상치 못한 명령
                if (abs(prev_action["linear_x"] - action["linear_x"]) > 0.1 or
                    abs(prev_action["linear_y"] - action["linear_y"]) > 0.1 or
                    abs(prev_action["angular_z"] - action["angular_z"]) > 0.1):
                    unexpected_command = True
        
        # 이제 현재 명령을 저장
        self.last_command_time = current_time
        self.last_command_action = action.copy()
        
        # 액션 타입 판별
        is_stop = (abs(action["linear_x"]) < 0.01 and 
                  abs(action["linear_y"]) < 0.01 and 
                  abs(action["angular_z"]) < 0.01)
        action_type = "STOP" if is_stop else "MOVE"
        
        # 상세 로깅 (50개 이상 데이터 수집 시 또는 verbose_logging 활성화 시)
        should_log_verbose = self.verbose_logging or (self.collecting and len(self.episode_data) >= 50)
        
        if should_log_verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # 조건 1: 명령 간 시간 간격 로깅
            time_info = ""
            if time_since_last_command is not None:
                time_info = f" | 이전 명령으로부터: {time_since_last_command*1000:.1f}ms"
                if time_since_last_command > 0.5:  # 500ms 이상 지연
                    time_info += " ⚠️ 지연 감지!"
            
            # 조건 3: 예상치 못한 명령 경고
            unexpected_info = ""
            if unexpected_command:
                unexpected_info = " | ⚠️ 예상치 못한 명령 변경 감지!"
            
            self.get_logger().info(
                f"📤 [CMD#{self.command_counter}] {timestamp}{time_info}{unexpected_info} | "
                f"Source: {source} | Thread: {current_thread} | "
                f"Type: {action_type} | "
                f"Action: lx={action['linear_x']:.2f}, ly={action['linear_y']:.2f}, az={action['angular_z']:.2f}"
            )
        
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action["angular_z"])
        
        # ROS 발행
        ros_publish_success = False
        ros_publish_time = None
        try:
            ros_publish_start = time.time()
            self.cmd_pub.publish(twist)
            ros_publish_time = time.time() - ros_publish_start
            ros_publish_success = True
            
            if should_log_verbose:
                self.get_logger().info(f"   ✅ ROS publish 성공 (토픽: /cmd_vel, 소요시간: {ros_publish_time*1000:.2f}ms)")
        except Exception as e:
            # 🔍 조건 2: 명령을 아예 무시했는지 확인 (ROS publish 실패)
            self.get_logger().error(f"   ❌ ROS publish 실패: {e} | ⚠️ 명령 무시됨!")
            return

        # 하드웨어 제어 (ROBOT_AVAILABLE일 때)
        hardware_success = False
        if ROBOT_AVAILABLE and self.driver:
            try:
                hw_start_time = time.time()
                if any(abs(v) > 0.1 for v in action.values()):
                    if abs(action["angular_z"]) > 0.1:
                        spin_speed = int(action["angular_z"] * self.throttle)
                        self.driver.spin(spin_speed)
                        hardware_success = True
                        hw_time = (time.time() - hw_start_time) * 1000
                        if should_log_verbose:
                            self.get_logger().info(f"   ✅ Hardware: spin({spin_speed}) (소요시간: {hw_time:.2f}ms)")
                    elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                        angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                        if angle < 0:
                            angle += 360
                        self.driver.move(int(angle), self.throttle)
                        hardware_success = True
                        hw_time = (time.time() - hw_start_time) * 1000
                        if should_log_verbose:
                            self.get_logger().info(f"   ✅ Hardware: move(angle={int(angle)}, throttle={self.throttle}) (소요시간: {hw_time:.2f}ms)")
                else:
                    self.driver.stop()
                    hardware_success = True
                    hw_time = (time.time() - hw_start_time) * 1000
                    if should_log_verbose:
                        self.get_logger().info(f"   ✅ Hardware: stop() (소요시간: {hw_time:.2f}ms)")
            except Exception as e:
                # 🔍 조건 2: 명령을 아예 무시했는지 확인 (하드웨어 제어 실패)
                self.get_logger().error(f"   ❌ Hardware 제어 실패: {e} | ⚠️ 명령 무시됨!")
        
        # 🔍 조건 2: ROS publish는 성공했지만 하드웨어 제어가 없거나 실패한 경우
        if should_log_verbose and ros_publish_success and ROBOT_AVAILABLE and not hardware_success:
            self.get_logger().warn(f"   ⚠️ ROS publish는 성공했지만 하드웨어 제어가 실행되지 않음 (명령 무시 가능성)")

    def get_latest_image_via_service(self, max_retries: int = 3) -> np.ndarray | None:
        """
        GetImage 서비스를 호출하여 최신 이미지를 가져옵니다.
        서비스 호출에 실패하거나 타임아웃되면 재시도합니다.
        """
        for attempt in range(max_retries):
            try:
                request = GetImage.Request()
                future = self.get_image_client.call_async(request)
                
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)  # 10초 → 2초로 단축
                
                if future.done():
                    response = future.result()
                    if response.image.data:
                        cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                        # 로그 간소화: 이미지 수신 성공 로그 제거
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
        
        # 간소화된 로그: 현재 수집 개수와 남은 개수만 표시
        current_count = len(self.episode_data)
        total_target = self.fixed_episode_length
        remaining = max(0, total_target - current_count)
        
        # 핵심 패턴 불일치 감지 및 다음 키 가져오기
        next_key_hint = None
        if self.core_guidance_active and action_event_type == "start_action":
            scenario_for_guide = self.selected_scenario or self.extract_scenario_from_episode_name(self.episode_name)
            pattern_for_guide = self.selected_pattern_type or self.extract_pattern_from_episode_name(self.episode_name)
            distance_for_guide = self.selected_distance_level or self.extract_distance_from_episode_name(self.episode_name)
            planned_seq = self._get_planned_core_keys_18(scenario_for_guide, pattern_for_guide, distance_for_guide)
            current_key = self._infer_key_from_action(action)
            
            # 불일치 감지 (Core일 때만, 로그 없이 통계만 업데이트)
            if planned_seq and len(self.current_episode_keys) < self.fixed_episode_length:
                next_key = planned_seq[len(self.current_episode_keys)]
                if planned_seq and next_key and pattern_for_guide == 'core':
                    if current_key != next_key:
                        self.core_mismatch_count += 1
        
        # 핵심 패턴 가이드가 활성화되어 있으면 다음 키 표시 (마지막 키까지 포함)
        if self.core_guidance_active and current_count < total_target:
            scenario_for_guide = self.selected_scenario or self.extract_scenario_from_episode_name(self.episode_name)
            pattern_for_guide = self.selected_pattern_type or self.extract_pattern_from_episode_name(self.episode_name)
            distance_for_guide = self.selected_distance_level or self.extract_distance_from_episode_name(self.episode_name)
            planned_seq = self._get_planned_core_keys_18(scenario_for_guide, pattern_for_guide, distance_for_guide)
            
            # 다음 키 계산: 현재 수집된 프레임 수를 기준으로 (episode_start는 인덱스 0, 첫 start_action은 인덱스 1)
            # 다음에 눌러야 할 키는 current_count 번째 키 (0-based이므로 current_count - 1이 현재 완료된 것)
            if planned_seq and current_count > 0:
                next_key_index = current_count  # 다음에 눌러야 할 키 인덱스
                if next_key_index < len(planned_seq):
                    next_key = planned_seq[next_key_index]
                    # SPACE는 ' '로 표시, 나머지는 대문자로 표시
                    if next_key == 'SPACE':
                        next_key_hint = ' '
                    else:
                        next_key_hint = next_key.upper()
        
        # 50개 이상 데이터 수집 시 상세 로깅 자동 활성화
        if current_count >= 50 and not self.verbose_logging:
            self.verbose_logging = True
            self.get_logger().info("🔍 상세 로깅 모드 활성화 (50개 이상 데이터 수집 감지)")
        
        # 간소화된 로그 출력 (마지막 키 힌트도 포함)
        if remaining > 0:
            if next_key_hint:
                self.get_logger().info(f"📊 수집 진행: {current_count}/{total_target} (남은: {remaining}) (다음 키: {next_key_hint})")
            else:
                self.get_logger().info(f"📊 수집 진행: {current_count}/{total_target} (남은: {remaining})")
        else:
            self.get_logger().info(f"✅ 수집 완료: {current_count}/{total_target}")

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

    def _normalize_to_18_keys(self, keys: List[str]) -> List[str]:
        """핵심 패턴 키 시퀀스를 17 길이로 정규화 (초기 프레임 1개 + 17개 액션 = 18 프레임)"""
        action_count = self.fixed_episode_length - 1  # 18 - 1 = 17 (초기 프레임 제외)
        normalized = list(keys[:action_count])
        if len(normalized) < action_count:
            normalized += ['SPACE'] * (action_count - len(normalized))
        return normalized

    def _get_planned_core_keys_18(self, scenario_id: str, pattern_type: str | None, distance_level: str | None) -> List[str]:
        """조합별 핵심 패턴을 17 길이로 반환 (초기 프레임 제외)"""
        # 1) 조합 우선
        if pattern_type and distance_level:
            combo = self._combined_key(scenario_id, pattern_type, distance_level)
            if combo in self.core_patterns and self.core_patterns[combo]:
                return self._normalize_to_18_keys(self.core_patterns[combo])
        # 2) 시나리오 단독
        if scenario_id in self.core_patterns and self.core_patterns[scenario_id]:
            return self._normalize_to_18_keys(self.core_patterns[scenario_id])
        return []

    def _infer_key_from_action(self, action: Dict[str, float]) -> str:
        """현재 액션 벡터에서 대표 키 추정 (로깅용)"""
        # 단순 규칙: 회전 우선, 그 다음 전진/횡이동의 사분면
        if abs(action.get("angular_z", 0.0)) > 0.1:
            return 'R' if action["angular_z"] > 0 else 'T'
        lx, ly = action.get("linear_x", 0.0), action.get("linear_y", 0.0)
        if abs(lx) < 0.1 and abs(ly) < 0.1:
            return 'SPACE'
        # 사분면 매핑: W/A/S/D/Q/E/Z/C와 유사
        if lx > 0.1 and abs(ly) <= 0.1:
            return 'W'
        if lx < -0.1 and abs(ly) <= 0.1:
            return 'S'
        if ly > 0.1 and abs(lx) <= 0.1:
            return 'A'
        if ly < -0.1 and abs(lx) <= 0.1:
            return 'D'
        if lx > 0.1 and ly > 0.1:
            return 'Q'
        if lx > 0.1 and ly < -0.1:
            return 'E'
        if lx < -0.1 and ly > 0.1:
            return 'Z'
        if lx < -0.1 and ly < -0.1:
            return 'C'
        return 'UNK'

    def start_episode(self, episode_name: str = None):
        """Starts a new episode collection"""
        if episode_name is None:
            self.episode_name = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.episode_name = episode_name

        self.episode_data = []
        self.current_episode_keys = []
        self.core_mismatch_count = 0
        
        self.get_logger().info("⏳ 에피소드 시작 전 준비 중...")
        
        # 🔄 카메라 스트림 완전 재시작 (가장 확실한 버퍼 초기화 방법)
        self.get_logger().info("🔄 카메라 스트림 재시작 중... (버퍼 완전 초기화)")
        try:
            reset_request = Empty.Request()
            reset_future = self.reset_camera_client.call_async(reset_request)
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=2.0)  # 10초 → 2초로 단축
            
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

        # 🔴 에피소드 시작 전 로봇 완전 정지 보장 (중요!)
        # 이전에 움직이고 있었을 수 있으므로 반드시 정지 상태로 초기화
        self.get_logger().info("🛑 에피소드 시작 전 로봇 정지 상태 확인 중...")
        
        # 기존 타이머가 있으면 취소
        if self.movement_timer and self.movement_timer.is_alive():
            self.movement_timer.cancel()
            self.movement_timer = None
        
        # 현재 액션을 STOP_ACTION으로 설정하고 강제 정지 신호 전송
        self.current_action = self.STOP_ACTION.copy()
        # 여러 번 정지 신호 전송하여 ROS 버퍼와 하드웨어에 확실히 전달
        for _ in range(3):
            self.publish_cmd_vel(self.STOP_ACTION)
            time.sleep(0.02)  # 각 신호 사이 짧은 딜레이
        
        # 추가 안정화 대기 (로봇이 완전히 정지할 시간 확보)
        time.sleep(0.05)
        
        self.collecting = True
        self.episode_start_time = time.time()
        
        # 에피소드 시작 시 상세 로깅은 비활성화 (데이터 수집 중 자동 활성화됨)
        self.verbose_logging = False
        
        # 에피소드 시작 시점의 시간대 자동 분류
        start_timestamp = datetime.now()
        start_time_period = self.classify_time_period(start_timestamp)
        start_time_str = start_timestamp.strftime("%H:%M:%S")
        period_info = self.time_period_targets.get(start_time_period, {})
        period_desc = period_info.get('description', start_time_period)
        
        self.get_logger().info(f"🎬 에피소드 시작: {self.episode_name}")
        self.get_logger().info(f"⏰ 시작 시간: {start_time_str} → 시간대: {period_desc} ({start_time_period})")
        self.get_logger().info(f"🔍 수집 상태: collecting={self.collecting}, 초기이미지크기={initial_image.shape}")
        
        # 명령 카운터 초기화 (에피소드별로 추적)
        self.command_counter = 0
        self.get_logger().info(f"📊 명령 추적 시작 (카운터 초기화)")
        
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

        # 핵심 패턴 표준 저장/갱신
        scenario = self.extract_scenario_from_episode_name(self.episode_name)
        if scenario and ("_core_" in self.episode_name or self.episode_name.endswith("_core")):
            if self.record_core_pattern and len(self.current_episode_keys) > 0:
                # SPACE는 명시적 정지일 때만 기록. 자동 패딩은 저장 시 제거
                normalized = self._normalize_to_18_keys(self.current_episode_keys)
                # 끝에 SPACE만 남았을 경우 제거하여 불필요한 SPACE 표준 방지
                while normalized and normalized[-1] == 'SPACE':
                    normalized.pop()
                # 조합 키 우선 저장 (core + distance 있으면 조합으로 저장)
                pattern = self.extract_pattern_from_episode_name(self.episode_name) or self.selected_pattern_type
                distance = self.extract_distance_from_episode_name(self.episode_name) or self.selected_distance_level
                if pattern and distance:
                    combo = self._combined_key(scenario, pattern, distance)
                    self.core_patterns[combo] = normalized
                else:
                    self.core_patterns[scenario] = normalized
                self.save_core_patterns()
                self.get_logger().info(f"💾 핵심 패턴 표준 등록: {scenario} [{pattern or '-'}|{distance or '-'}]")
            self.core_guidance_active = False
            self.core_guidance_index = 0
            self.record_core_pattern = False
        
        # 프레임 수에 따른 분류 및 통계 업데이트
        num_frames = len(self.episode_data)
        category = self.classify_by_frames(num_frames)
        self.dataset_stats[category] += 1
        
        # 시간대별 통계 업데이트 (에피소드 종료 시점의 시간 기준)
        current_timestamp = datetime.now()
        current_time_period = self.classify_time_period(current_timestamp)
        self.time_period_stats[current_time_period] += 1
        self.save_time_period_stats()
        
        # 시간대 분류 정보 로그
        current_time_str = current_timestamp.strftime("%H:%M:%S")
        period_info = self.time_period_targets.get(current_time_period, {})
        period_desc = period_info.get('description', current_time_period)
        self.get_logger().info(f"⏰ 수집 시간: {current_time_str} → 시간대: {period_desc} ({current_time_period})")
        
        # 시나리오별 통계 업데이트 (에피소드명에서 시나리오 추출)
        if scenario:
            self.scenario_stats[scenario] += 1
            self.save_scenario_progress()
            # 패턴×거리 통계 업데이트 (에피소드명에서 추출)
            pattern = self.extract_pattern_from_episode_name(self.episode_name)
            distance = self.extract_distance_from_episode_name(self.episode_name)
            if pattern and distance:
                self.pattern_distance_stats[scenario][pattern][distance] += 1
        
        # 프레임 18개 데이터 특별 표시 (RoboVLMs 기준 목표)
        if num_frames == 18:
            frame_18_indicator = "🎯 [18프레임 목표 달성!]"
        else:
            frame_18_indicator = f"⚠️ [{num_frames}프레임] (목표: 18프레임)"
        scenario_indicator = f" 🎯[{scenario}]" if scenario else ""
        time_period_info = self.time_period_targets.get(current_time_period, {})
        time_period_desc = time_period_info.get('description', current_time_period) if current_time_period else ""
        time_period_indicator = f" 🌅[{time_period_desc}]" if time_period_desc else ""
        
        self.get_logger().info(f"✅ 에피소드 완료: {total_duration:.1f}초, 총 프레임 수: {num_frames}개 {frame_18_indicator}{scenario_indicator}{time_period_indicator}")
        self.get_logger().info(f"📂 카테고리: {category} ({self.categories[category]['description']})")
        if time_period_desc:
            time_period_current = self.time_period_stats[current_time_period]
            time_period_target = time_period_info.get('target', 0)
            time_period_progress = self.create_progress_bar(time_period_current, time_period_target, width=10)
            self.get_logger().info(f"🌅 시간대: {time_period_desc} {time_period_progress} ({time_period_current}/{time_period_target})")
        self.get_logger().info(f"💾 저장됨: {save_path}")
        if self.core_guidance_active:
            self.get_logger().info(f"🧪 핵심 가이드 일치 여부: 불일치 {self.core_mismatch_count}회")
        
        # 현재 진행 상황 표시
        self.show_category_progress(category)
        if scenario:
            self.show_scenario_progress(scenario)
            self.show_pattern_distance_table(scenario)

        # 반복 측정이 활성화되어 있으면 다음 측정 확인 및 진행
        self.check_and_continue_repeat_measurement()

        self.publish_cmd_vel(self.STOP_ACTION)
        
        # 자동 복귀 안내 메시지
        if len(self.episode_data) > 0:
            self.get_logger().info("")
            self.get_logger().info("🔄 'B' 키를 눌러 시작 위치로 자동 복귀할 수 있습니다.")

    def reset_to_initial_state(self):
        """모든 상태를 초기화하고 첫 화면으로 리셋"""
        self.get_logger().info("🔄 리셋 중...")
        
        # 수집 중이면 에피소드 취소 (저장하지 않음)
        if self.collecting:
            self.get_logger().info("⚠️ 수집 중인 에피소드를 취소합니다 (저장하지 않음)")
            if self.movement_timer and self.movement_timer.is_alive():
                self.movement_timer.cancel()
            self.stop_movement_internal(collect_data=False)
            self.collecting = False
            self.episode_data = []
            self.episode_name = ""
            self.episode_start_time = None
        
        # 반복 횟수 입력 모드 중이면 취소
        if self.repeat_count_mode:
            sys.stdout.write("\n")  # 입력 줄 완료
            sys.stdout.flush()
            self.repeat_count_mode = False
            self.repeat_count_input = ""
        
        # 가이드 편집 모드 중이면 취소
        if self.guide_edit_mode:
            sys.stdout.write("\n")  # 입력 줄 완료
            sys.stdout.flush()
            self.guide_edit_mode = False
            self.guide_edit_keys = []
        
        # 마지막 완료된 에피소드 액션 초기화
        self.last_completed_episode_actions = []
        
        # 모든 선택 상태 초기화
        self.scenario_selection_mode = False
        self.pattern_selection_mode = False
        self.distance_selection_mode = False
        self.selected_scenario = None
        self.selected_pattern_type = None
        self.selected_distance_level = None
        
        # 반복 측정 상태 초기화
        self.is_repeat_measurement_active = False
        self.waiting_for_next_repeat = False
        self.current_repeat_index = 0
        self.target_repeat_count = 1
        
        # 🔴 자동 측정 상태 초기화
        self.auto_measurement_mode = False
        self.auto_measurement_active = False
        self.auto_measurement_thread = None
        
        # 자동 복귀 상태 초기화
        if self.auto_return_active:
            self.auto_return_active = False
            self.get_logger().info("🛑 자동 복귀를 중단합니다...")
            self.current_action = self.STOP_ACTION.copy()
            for _ in range(3):
                self.publish_cmd_vel(self.STOP_ACTION, source="reset_cancel_return")
                time.sleep(0.02)
        
        # 핵심 패턴 가이드 상태 초기화
        self.core_guidance_active = False
        self.core_guidance_index = 0
        self.record_core_pattern = False
        self.current_episode_keys = []
        self.core_mismatch_count = 0
        
        # 로봇 정지
        self.stop_movement_internal(collect_data=False)
        self.publish_cmd_vel(self.STOP_ACTION)
        
        self.get_logger().info("✅ 리셋 완료! 첫 화면으로 돌아갑니다.")
        self.get_logger().info("")
        
        # 첫 화면(시나리오 선택 메뉴) 표시
        self.show_scenario_selection()

    def save_episode_data(self, episode_data: List[Dict], episode_name: str, total_duration: float) -> Path:
        """Saves collected episode data to an HDF5 file"""
        # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
        data_dir_abs = Path(self.data_dir).resolve()
        save_path = data_dir_abs / f"{episode_name}.h5"
        # 저장 경로도 절대 경로로 확실히 변환
        save_path = save_path.resolve()
        
        # 데이터 디렉토리 존재 확인 및 생성 (절대 경로 사용)
        try:
            data_dir_abs.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f"❌ 데이터 디렉토리 생성 실패 ({data_dir_abs}): {e}")
            raise
        
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

        # 현재 시간대 자동 분류 (에피소드 종료 시점 기준)
        current_timestamp = datetime.now()
        current_time_period = self.classify_time_period(current_timestamp)
        
        with h5py.File(save_path, 'w') as f:
            f.attrs['episode_name'] = episode_name
            f.attrs['total_duration'] = total_duration
            f.attrs['num_frames'] = images.shape[0]
            f.attrs['action_chunk_size'] = self.action_chunk_size
            # 배치 타입 메타데이터 추가 (기본값으로 자동 설정)
            f.attrs['obstacle_layout_type'] = self.default_layout_type  # 기본값: "hori"
            # 시간대 메타데이터 추가 (자동 분류된 시간대)
            f.attrs['time_period'] = current_time_period  # "day", "night", "dawn"
            f.attrs['collection_datetime'] = current_timestamp.isoformat()  # ISO 형식 저장
            f.attrs['collection_hour'] = current_timestamp.hour
            f.attrs['collection_minute'] = current_timestamp.minute

            f.create_dataset('images', data=images, compression='gzip')
            f.create_dataset('actions', data=actions, compression='gzip')
            f.create_dataset('action_event_types', data=action_event_types, compression='gzip')

        # 마지막 완료된 에피소드의 액션 시퀀스 저장 (가이드 갱신 옵션용)
        # "start_action" 제외하고 WASD/QEZC 키만 추출
        valid_keys = {'w', 'a', 's', 'd', 'q', 'e', 'z', 'c'}
        self.last_completed_episode_actions = [
            action_type for action_type in action_event_types
            if action_type.lower() in valid_keys
        ]
        
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
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            data_dir_abs = Path(self.data_dir).resolve()
            h5_files = list(data_dir_abs.glob("*.h5"))
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
        self.get_logger().info("📌 수집 목표: 18프레임 기준 (RoboVLMs: window=8 + pred_next=10)")
        self.get_logger().info("📊 카테고리 분류: 데이터셋 통계 모니터링용 (수집 목표와는 별개)")
        self.get_logger().info("=" * 50)
        
        total_current = 0
        total_target = 0
        frame_18_count = 0
        
        # 프레임 18개 데이터 별도 카운트 (절대 경로 사용)
        data_dir_abs = Path(self.data_dir).resolve()
        for h5_file in data_dir_abs.glob("*.h5"):
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
        self.get_logger().info(f"🎯 18프레임 목표 데이터: {frame_18_count}개 발견! (RoboVLMs 기준)")
        
        # 시간대별 진행 상황 표시
        self.get_logger().info("-" * 50)
        self.get_logger().info("🌅 시간대별 진행 상황:")
        
        total_time_period_current = 0
        total_time_period_target = 0
        
        for period_key, period_config in self.time_period_targets.items():
            current = self.time_period_stats[period_key]
            target = period_config["target"]
            total_time_period_current += current
            total_time_period_target += target
            percentage = (current / target * 100) if target > 0 else 0
            progress_bar = self.create_progress_bar(current, target)
            status_emoji = "✅" if current >= target else "⏳"
            
            self.get_logger().info(f"{status_emoji} {period_config['description']}: {progress_bar} ({percentage:.1f}%)")
        
        # 시간대별 전체 진행률
        time_period_overall_percentage = (total_time_period_current / total_time_period_target * 100) if total_time_period_target > 0 else 0
        time_period_overall_progress = self.create_progress_bar(total_time_period_current, total_time_period_target, width=25)
        self.get_logger().info(f"   🌅 시간대별 전체: {time_period_overall_progress} ({time_period_overall_percentage:.1f}%)")
        
        # 시나리오별 진행 상황도 표시
        self.get_logger().info("-" * 50)
        self.get_logger().info("🎯 탄산음료 페트병 도달 시나리오별 진행 상황:")
        
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
            # 각 시나리오 옆에 패턴×거리 표 간단 요약 출력
            self.show_pattern_distance_table(scenario, compact=True)
            
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
        """시나리오별 WASD 경로 예시 (4개 시나리오 통합)"""
        # 핵심 패턴 (4개 시나리오)
        core_patterns = {
            "1box_left": "W W W → A A → W W → D D",
            "1box_right": "W W → D D → W W W → A A", 
            "2box_left": "W W → A A A → W W → D D D",
            "2box_right": "W → D D D → W W W → A A A"
        }
        
        # 변형 패턴 예시
        variant_info = "변형: 타이밍 조정, 세분화된 움직임"
        
        core_pattern = core_patterns.get(scenario_id, "W → A/D → W → ...")
        return f"📍 핵심: {core_pattern}\n   🔄 {variant_info}"
        
    def extract_scenario_from_episode_name(self, episode_name: str) -> str:
        """에피소드명에서 시나리오 추출 (기존 형식 호환: vert/hori 포함된 형식도 처리)"""
        # 먼저 새로운 형식(4개 시나리오) 확인
        for scenario in self.cup_scenarios.keys():
            if scenario in episode_name:
                return scenario
        
        # 기존 형식 호환: 1box_vert_left → 1box_left로 변환
        old_to_new = {
            "1box_vert_left": "1box_left", "1box_hori_left": "1box_left",
            "1box_vert_right": "1box_right", "1box_hori_right": "1box_right",
            "2box_vert_left": "2box_left", "2box_hori_left": "2box_left",
            "2box_vert_right": "2box_right", "2box_hori_right": "2box_right"
        }
        for old_id, new_id in old_to_new.items():
            if old_id in episode_name:
                return new_id
        
        return None

    def extract_pattern_from_episode_name(self, episode_name: str) -> str:
        """에피소드명에서 패턴(core/variant) 추출"""
        for p in ["core", "variant"]:
            if f"_{p}_" in episode_name or episode_name.endswith(f"_{p}"):
                return p
        return None

    def extract_distance_from_episode_name(self, episode_name: str) -> str:
        """에피소드명에서 거리(close/medium/far) 추출"""
        for d in ["close", "medium", "far"]:
            if episode_name.endswith(f"_{d}") or f"_{d}." in episode_name:
                return d
        return None

    def show_pattern_distance_table(self, scenario: str, compact: bool = False):
        """특정 시나리오의 패턴×거리 진행 현황 표 출력"""
        if scenario not in self.cup_scenarios:
            return
        counts = self.pattern_distance_stats[scenario]
        # 통합된 목표 사용 (가로/세로 구분 없음)
        pattern_targets = self.pattern_targets
        dist_targets = self.distance_targets_per_pattern
        # 표 헤더
        header = "패턴/위치  Close  Medium  Far   소계 (목표)"
        rows = []
        total_close = total_medium = total_far = total_all = 0
        patterns = ["core", "variant"]
        for pattern in patterns:
            c_close = counts[pattern]["close"]
            c_medium = counts[pattern]["medium"]
            c_far = counts[pattern]["far"]
            subtotal = c_close + c_medium + c_far
            target_pd = dist_targets[pattern]
            row = f"{pattern.capitalize():<10}  {c_close:>5}/{target_pd['close']}  {c_medium:>6}/{target_pd['medium']}  {c_far:>4}/{target_pd['far']}   {subtotal:>3}/{pattern_targets[pattern]}"
            rows.append(row)
            total_close += c_close
            total_medium += c_medium
            total_far += c_far
            total_all += subtotal
        # 합계 행의 목표도 시나리오 유형에 따라 다르게 표기 (Exception 제거)
        total_close_target = dist_targets["core"]["close"] + dist_targets["variant"]["close"]
        total_medium_target = dist_targets["core"]["medium"] + dist_targets["variant"]["medium"]
        total_far_target = dist_targets["core"]["far"] + dist_targets["variant"]["far"]
        total_target_all = sum(pattern_targets.values())
        total_row = f"합계        {total_close:>5}/{total_close_target}  {total_medium:>6}/{total_medium_target}  {total_far:>4}/{total_far_target}   {total_all:>3}/{total_target_all}"
        if compact:
            self.get_logger().info("   ─ 패턴×위치 진행 요약")
            self.get_logger().info(f"   {header}")
            for r in rows:
                self.get_logger().info(f"   {r}")
            self.get_logger().info(f"   {total_row}")
        else:
            self.get_logger().info("📋 패턴×위치 진행 현황")
            self.get_logger().info(header)
            for r in rows:
                self.get_logger().info(r)
            self.get_logger().info(total_row)
        
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
        # 상세 표 즉시 제공
        self.show_pattern_distance_table(scenario)
        
    def load_scenario_progress(self):
        """저장된 시나리오 진행상황 로드"""
        try:
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            progress_file_abs = Path(self.progress_file).resolve()
            if progress_file_abs.exists():
                with open(progress_file_abs, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.scenario_stats = defaultdict(int, data.get('scenario_stats', {}))
                self.get_logger().info(f"📊 시나리오 진행상황 로드 완료: {dict(self.scenario_stats)}")
            else:
                self.scenario_stats = defaultdict(int)
                self.get_logger().info("📊 새로운 시나리오 진행상황 시작")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시나리오 진행상황 로드 실패: {e}")
            self.scenario_stats = defaultdict(int)

    def load_core_patterns(self):
        """핵심 패턴(표준) 파일 로드"""
        try:
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            core_pattern_file_abs = Path(self.core_pattern_file).resolve()
            if core_pattern_file_abs.exists():
                with open(core_pattern_file_abs, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 값은 키 시퀀스 리스트
                    loaded = {k: list(v) for k, v in data.items()}
                    # exception 키를 variant로 마이그레이션
                    migrated = {}
                    for k, seq in loaded.items():
                        new_k = k.replace('__exception__', '__variant__').replace('_exception__', '_variant__').replace('__exception', '__variant').replace('_exception', '_variant')
                        migrated[new_k] = seq
                    self.core_patterns = migrated
                self.get_logger().info(f"📘 핵심 패턴 로드: {list(self.core_patterns.keys())}")
            else:
                self.core_patterns = {}
        except Exception as e:
            self.get_logger().warn(f"⚠️ 핵심 패턴 로드 실패: {e}")
            self.core_patterns = {}

    def save_core_patterns(self):
        """핵심 패턴(표준) 파일 저장"""
        try:
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            core_pattern_file_abs = Path(self.core_pattern_file).resolve()
            # 부모 디렉토리 생성
            core_pattern_file_abs.parent.mkdir(parents=True, exist_ok=True)
            with open(core_pattern_file_abs, 'w', encoding='utf-8') as f:
                json.dump(self.core_patterns, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 핵심 패턴 저장 실패: {e}")
            
    def save_scenario_progress(self):
        """시나리오 진행상황 저장"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "scenario_stats": dict(self.scenario_stats),
                "total_completed": sum(self.scenario_stats.values()),
                "total_target": sum(config["target"] for config in self.cup_scenarios.values())
            }
            
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            progress_file_abs = Path(self.progress_file).resolve()
            # 부모 디렉토리 생성
            progress_file_abs.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file_abs, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시나리오 진행상황 저장 실패: {e}")
            
    def classify_time_period(self, timestamp: datetime = None) -> str:
        """
        현재 시간을 기반으로 시간대 자동 분류 (24시간 전체 커버)
        새벽/아침/저녁/밤 4가지로 분류
        
        Args:
            timestamp: 분류할 시간 (None이면 현재 시간 사용)
        
        Returns:
            "dawn", "morning", "evening", "night" 중 하나
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        current_hour = timestamp.hour
        current_time_str = timestamp.strftime("%H:%M")
        
        # 24시간을 4가지 시간대로 균등 분할
        # 새벽: 00:00-06:00 (0 <= hour < 6)
        if 0 <= current_hour < 6:
            self.get_logger().debug(f"⏰ 현재 시간: {current_time_str} → 시간대: 새벽 (dawn)")
            return "dawn"
        # 아침: 06:00-12:00 (6 <= hour < 12)
        elif 6 <= current_hour < 12:
            self.get_logger().debug(f"⏰ 현재 시간: {current_time_str} → 시간대: 아침 (morning)")
            return "morning"
        # 저녁: 12:00-18:00 (12 <= hour < 18)
        elif 12 <= current_hour < 18:
            self.get_logger().debug(f"⏰ 현재 시간: {current_time_str} → 시간대: 저녁 (evening)")
            return "evening"
        # 밤: 18:00-24:00 (18 <= hour < 24)
        else:  # 18 <= hour < 24
            self.get_logger().debug(f"⏰ 현재 시간: {current_time_str} → 시간대: 밤 (night)")
            return "night"
    
    def load_time_period_stats(self):
        """저장된 시간대별 통계 로드"""
        try:
            if self.time_period_file.exists():
                with open(self.time_period_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.time_period_stats = defaultdict(int, data.get('time_period_stats', {}))
                self.get_logger().info(f"📊 시간대별 통계 로드 완료: {dict(self.time_period_stats)}")
            else:
                self.time_period_stats = defaultdict(int)
                self.get_logger().info("📊 새로운 시간대별 통계 시작")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시간대별 통계 로드 실패: {e}")
            self.time_period_stats = defaultdict(int)
    
    def save_time_period_stats(self):
        """시간대별 통계 저장"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "time_period_stats": dict(self.time_period_stats),
                "total_completed": sum(self.time_period_stats.values()),
                "total_target": sum(config["target"] for config in self.time_period_targets.values())
            }
            
            # 절대 경로로 확실히 변환 (현재 작업 디렉토리 의존 제거)
            time_period_file_abs = Path(self.time_period_file).resolve()
            # 부모 디렉토리 생성
            time_period_file_abs.parent.mkdir(parents=True, exist_ok=True)
            with open(time_period_file_abs, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.get_logger().warn(f"⚠️ 시간대별 통계 저장 실패: {e}")
            
    def show_scenario_selection(self):
        """4가지 시나리오 선택 메뉴 표시 (가로/세로 통합)"""
        self.scenario_selection_mode = True
        
        self.get_logger().info("🎯 탄산음료 페트병 도달 시나리오 선택")
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 환경을 설정한 후 원하는 시나리오 번호를 누르세요:")
        self.get_logger().info("")
        
        # 시나리오별 상세 정보 표시 (4개로 축소)
        scenario_details = [
            {"key": "1", "id": "1box_left", "path": "W W W → A A → W W → D D"},
            {"key": "2", "id": "1box_right", "path": "W W → D D → W W W → A A"},
            {"key": "3", "id": "2box_left", "path": "W W → A A A → W W → D D D"},
            {"key": "4", "id": "2box_right", "path": "W → D D D → W W W → A A A"}
        ]
        
        for scenario in scenario_details:
            scenario_id = scenario["id"]
            description = self.cup_scenarios[scenario_id]["description"]
            # 기존 통계는 vert/hori 포함 형식도 집계하도록 호환 처리
            current = self.scenario_stats.get(scenario_id, 0)
            # 기존 형식(vert/hori 포함)도 카운트
            for layout in ["vert", "hori"]:
                old_id = f"{scenario_id.replace('_left', '_vert_left').replace('_right', '_vert_right')}"
                if "_left" in scenario_id:
                    old_id = old_id.replace("_vert_left", f"_{layout}_left")
                elif "_right" in scenario_id:
                    old_id = old_id.replace("_vert_right", f"_{layout}_right")
                if old_id in self.scenario_stats:
                    current += self.scenario_stats[old_id]
            
            target = self.cup_scenarios[scenario_id]["target"]
            remaining = max(0, target - current)
            progress_bar = self.create_progress_bar(current, target, width=10)
            
            status_emoji = "✅" if current >= target else "⏳"
            
            self.get_logger().info(f"{status_emoji} {scenario['key']}키: {description}")
            self.get_logger().info(f"   🎮 예시 경로: {scenario['path']}")
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
        self.get_logger().info("✨ 1-4번 중 원하는 시나리오를 선택하세요!")
        self.get_logger().info("💡 환경 설정 후 숫자키를 누르면 배치 타입 선택으로 넘어갑니다.")
        self.get_logger().info("🚫 취소하려면 다른 키를 누르세요.")

    def resync_scenario_progress(self):
        """실제 H5 파일들을 스캔하여 시나리오 진행률 재동기화"""
        self.get_logger().info("🔄 H5 파일 스캔하여 시나리오 진행률 동기화 중...")
        
        # 시나리오 통계 초기화
        self.scenario_stats = defaultdict(int)
        self.pattern_distance_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.time_period_stats = defaultdict(int)  # 시간대별 통계도 초기화
        combo_files = defaultdict(list)  # (scenario, pattern, distance) -> List[Path]
        
        # 데이터 디렉토리에서 모든 H5 파일 스캔 (절대 경로 사용)
        data_dir_abs = Path(self.data_dir).resolve()
        if data_dir_abs.exists():
            h5_files = list(data_dir_abs.glob("*.h5"))
            self.get_logger().info(f"📁 {len(h5_files)}개의 H5 파일을 발견했습니다.")
            
            scenario_matched = 0
            old_format_files = []
            
            for h5_file in h5_files:
                try:
                    # 파일명에서 시나리오 추출
                    stem = h5_file.stem
                    scenario = self.extract_scenario_from_episode_name(stem)
                    if scenario and scenario in self.cup_scenarios:
                        self.scenario_stats[scenario] += 1
                        # 패턴/거리도 함께 복원
                        pattern = self.extract_pattern_from_episode_name(stem)
                        distance = self.extract_distance_from_episode_name(stem)
                        if pattern and distance:
                            self.pattern_distance_stats[scenario][pattern][distance] += 1
                            combo_files[(scenario, pattern, distance)].append(h5_file)
                        # 시간대별 통계도 복원 (저장된 메타데이터에서)
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                time_period = f.attrs.get('time_period', None)
                                if time_period and time_period in self.time_period_targets:
                                    self.time_period_stats[time_period] += 1
                                elif not time_period:
                                    # 시간대 정보가 없으면 파일명의 타임스탬프에서 추정
                                    try:
                                        # episode_20251030_132725_... 형식에서 시간 추출
                                        datetime_str = stem.split('_')[1] + '_' + stem.split('_')[2]
                                        file_timestamp = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
                                        estimated_period = self.classify_time_period(file_timestamp)
                                        self.time_period_stats[estimated_period] += 1
                                        self.get_logger().debug(f"📅 {h5_file.name}: 시간대 추정 → {estimated_period}")
                                    except:
                                        pass  # 추정 실패해도 계속 진행
                        except:
                            pass  # 시간대 정보가 없어도 계속 진행
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
        self.save_time_period_stats()
        
        # 동기화 결과 요약
        total_found = sum(self.scenario_stats.values())
        self.get_logger().info(f"✅ 동기화 완료! 총 {total_found}개의 시나리오 에피소드 발견")
        
        # 시간대별 통계 요약
        if any(self.time_period_stats.values()):
            self.get_logger().info("🌅 시간대별 통계:")
            for period_key, period_config in self.time_period_targets.items():
                count = self.time_period_stats[period_key]
                if count > 0:
                    self.get_logger().info(f"   {period_config['description']}: {count}개")
        
        for scenario_id, count in self.scenario_stats.items():
            if count > 0:
                scenario_info = self.cup_scenarios[scenario_id]
                key = scenario_info["key"]
                desc = scenario_info["description"]
                self.get_logger().info(f"   {key}키 {scenario_id}: {count}개 → {desc}")
        
        if total_found == 0:
            self.get_logger().info("📝 시나리오 이름이 포함된 파일이 없습니다.")
            self.get_logger().info("💡 새로운 N-숫자키 시스템으로 수집한 파일만 카운트됩니다.")

        # === 가이드(핵심 표준) 동기화 규칙 ===
        # 1) 실제 존재 개수가 0이면 해당 조합(또는 시나리오 단독) 가이드를 초기화
        # 2) 현재 케이스(조합) 중 오직 1개가 존재한다면 그 파일의 키 시퀀스를 가이드로 설정
        changed = False
        # 시나리오 단독 핵심(과거 호환) 초기화 조건: core 전체가 0개일 때만
        for scenario in self.cup_scenarios.keys():
            core_total = sum(len(combo_files[(scenario, 'core', d)]) for d in ['close', 'medium', 'far'])
            if core_total == 0 and scenario in self.core_patterns:
                del self.core_patterns[scenario]
                changed = True
                self.get_logger().info(f"🧹 가이드 초기화(시나리오 핵심): {scenario} (파일 0개)")
        
        # 조합별(core만) 초기화/생성
        for scenario in self.cup_scenarios.keys():
            for d in ['close', 'medium', 'far']:
                combo = (scenario, 'core', d)
                files = combo_files.get(combo, [])
                combo_key = self._combined_key(scenario, 'core', d)
                if len(files) == 0:
                    if combo_key in self.core_patterns:
                        del self.core_patterns[combo_key]
                        changed = True
                        self.get_logger().info(f"🧹 가이드 초기화(조합 핵심): {combo_key} (파일 0개)")
                elif len(files) == 1:
                    # 단 1개의 데이터가 존재하면 그 시퀀스를 가이드로 설정
                    try:
                        with h5py.File(files[0], 'r') as f:
                            actions = np.array(f['actions']) if 'actions' in f else None
                            events = f['action_event_types'][:] if 'action_event_types' in f else None
                            if actions is not None and events is not None:
                                # 문자열 디코딩
                                if isinstance(events[0], bytes):
                                    events = [e.decode('utf-8') for e in events]
                                keys: List[str] = []
                                for idx, ev in enumerate(events):
                                    if ev == 'start_action':
                                        ax, ay, az = float(actions[idx][0]), float(actions[idx][1]), float(actions[idx][2])
                                        k_upper = self._infer_key_from_action({
                                            'linear_x': ax, 'linear_y': ay, 'angular_z': az
                                        })
                                        # 저장은 소문자 키 사용, SPACE는 그대로 유지
                                        k_store = k_upper.lower() if k_upper != 'SPACE' else 'SPACE'
                                        keys.append(k_store)
                                if keys:
                                    normalized = self._normalize_to_18_keys(keys)
                                    while normalized and normalized[-1] == 'SPACE':
                                        normalized.pop()
                                    self.core_patterns[combo_key] = normalized
                                    changed = True
                                    self.get_logger().info(f"📌 가이드 설정(조합 핵심): {combo_key} ← {files[0].name}")
                    except Exception as e:
                        self.get_logger().warn(f"⚠️ 가이드 복원 실패: {combo_key} → {files[0].name}: {e}")
        if changed:
            self.save_core_patterns()

    def resync_and_show_progress(self):
        """H5 파일 재스캔 후 진행률 표시"""
        self.resync_scenario_progress()
        self.load_dataset_stats()  # 전체 데이터셋 통계도 다시 로드
        self.load_time_period_stats()  # 시간대별 통계도 다시 로드
        self.show_progress_status()
        

    def show_pattern_selection(self):
        """패턴 타입 선택 메뉴 표시"""
        self.pattern_selection_mode = True
        
        config = self.cup_scenarios[self.selected_scenario]
        
        self.get_logger().info("🎯 패턴 타입 선택")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"📦 선택된 시나리오: {config['description']}")
        self.get_logger().info("")
        
        # 핵심 패턴 가이드 표시
        core_pattern = self.get_core_pattern_guide(self.selected_scenario)
        
        self.get_logger().info("📍 C키: 핵심 패턴 (Core) - 6개 수집 목표")
        self.get_logger().info(f"   🎮 가이드: {core_pattern}")
        self.get_logger().info("   💡 위 순서를 참고하여 정확히 따라하세요!")
        self.get_logger().info("")
        
        self.get_logger().info("🔄 V키: 변형 패턴 (Variant) - 4개 수집 목표")
        self.get_logger().info("   🎮 핵심 패턴의 타이밍이나 순서를 조금 변경")
        self.get_logger().info("   💡 창의적으로 변형하여 움직이세요!")
        self.get_logger().info("")
        
        self.get_logger().info("✨ C, V 중 원하는 패턴을 선택하세요!")
        self.get_logger().info("🚫 취소하려면 다른 키를 누르세요.")

    def show_distance_selection(self):
        """장애물 위치 선택 메뉴 표시 (근/중/원 개념을 위치로 안내)"""
        self.distance_selection_mode = True
        levels = self.distance_levels
        
        # 선택된 시나리오와 패턴 정보 표시
        scenario_config = self.cup_scenarios.get(self.selected_scenario, {})
        pattern_names = {
            "core": "핵심 패턴 (Core)",
            "variant": "변형 패턴 (Variant)"
        }
        pattern_display = pattern_names.get(self.selected_pattern_type, self.selected_pattern_type)
        
        self.get_logger().info("🎯 장애물 위치 선택")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"📦 선택된 시나리오: {scenario_config.get('description', self.selected_scenario)}")
        self.get_logger().info(f"📋 선택된 패턴: {pattern_display}")
        self.get_logger().info("")
        self.get_logger().info("J키: CLOSE")
        self.get_logger().info(f"   📍 {levels['close']['label']}")
        self.get_logger().info(f"   💡 {levels['close']['hint']}")
        self.get_logger().info("")
        self.get_logger().info("K키: MEDIUM")
        self.get_logger().info(f"   📍 {levels['medium']['label']}")
        self.get_logger().info(f"   💡 {levels['medium']['hint']}")
        self.get_logger().info("")
        self.get_logger().info("L키: FAR")
        self.get_logger().info(f"   📍 {levels['far']['label']}")
        self.get_logger().info(f"   💡 {levels['far']['hint']}")
        self.get_logger().info("")
        self.get_logger().info("✨ J/K/L 중 장애물 위치를 선택하세요!")
        self.get_logger().info("🚫 취소하려면 다른 키를 누르세요.")
        
    def show_repeat_count_selection(self):
        """반복 횟수 입력 메뉴 표시 (가이드 표시 및 편집 옵션 포함)"""
        # 선택된 정보 표시
        scenario_config = self.cup_scenarios.get(self.selected_scenario, {})
        pattern_names = {
            "core": "핵심 패턴 (Core)",
            "variant": "변형 패턴 (Variant)"
        }
        distance_names = {
            "close": "CLOSE (가까운 위치)",
            "medium": "MEDIUM (중간 거리)",
            "far": "FAR (먼 위치)"
        }
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📦 시나리오: {scenario_config.get('description', self.selected_scenario)}")
        self.get_logger().info(f"📋 패턴: {pattern_names.get(self.selected_pattern_type, self.selected_pattern_type)}")
        self.get_logger().info(f"📍 거리: {distance_names.get(self.selected_distance_level, self.selected_distance_level)}")
        self.get_logger().info("")
        
        # 핵심 패턴인 경우 가이드 표시
        if self.selected_pattern_type == "core":
            guide_keys = self.get_core_pattern_guide_keys(
                self.selected_scenario,
                self.selected_pattern_type,
                self.selected_distance_level
            )
            guide_str = " ".join([k.upper() for k in guide_keys])
            self.get_logger().info(f"🎮 현재 가이드: {guide_str}")
            self.get_logger().info("")
            self.get_logger().info("✨ 가이드 편집: H 키를 눌러 가이드를 수정하거나 새로 입력하세요")
            self.get_logger().info("   (가이드를 수정하면 해당 조합에 대해 저장됩니다)")
            # 마지막 완료된 에피소드가 있으면 U 키 옵션 표시
            if self.last_completed_episode_actions:
                last_actions_str = " ".join([k.upper() for k in self.last_completed_episode_actions])
                self.get_logger().info("")
                self.get_logger().info("🔄 가이드 갱신: U 키를 눌러 방금 수집한 키 입력을 가이드로 저장")
                self.get_logger().info(f"   방금 수집: {last_actions_str}")
            self.get_logger().info("")
        
        self.get_logger().info("🔄 반복 횟수 입력")
        self.get_logger().info("=" * 60)
        self.get_logger().info("✨ 반복 횟수를 입력하세요:")
        self.get_logger().info("   Enter: 1회 측정 (기본값)")
        self.get_logger().info("   숫자 입력 후 Enter: 해당 횟수만큼 반복 측정 (최대 100회)")
        self.get_logger().info("   예: '5' 입력 후 Enter → 5회 반복")
        self.get_logger().info("")
        self.get_logger().info("🚫 취소하려면 WASD 키를 누르세요.")
        
        self.repeat_count_mode = True
        self.repeat_count_input = ""
        # 입력 프롬프트 표시 (커서 깜빡임을 위한)
        sys.stdout.write("📝 반복 횟수: ")
        sys.stdout.flush()
    
    def show_guide_edit_menu(self):
        """가이드 편집 메뉴 표시"""
        self.guide_edit_mode = True
        self.guide_edit_keys = []
        
        # 현재 가이드 가져오기
        current_guide_keys = self.get_core_pattern_guide_keys(
            self.selected_scenario,
            self.selected_pattern_type,
            self.selected_distance_level
        )
        
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("✏️ 가이드 편집 모드")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📦 시나리오: {self.selected_scenario}")
        self.get_logger().info(f"📋 패턴: {self.selected_pattern_type}")
        self.get_logger().info(f"📍 거리: {self.selected_distance_level}")
        self.get_logger().info("")
        if current_guide_keys:
            current_guide_str = " ".join([k.upper() for k in current_guide_keys])
            self.get_logger().info(f"📋 현재 가이드: {current_guide_str}")
        else:
            self.get_logger().info("📋 현재 가이드: 없음 (새로 입력하세요)")
        self.get_logger().info("")
        self.get_logger().info("✨ 가이드 키를 입력하세요:")
        self.get_logger().info("   W/A/S/D: 이동, Q/E/Z/C: 대각선, R/T: 회전, SPACE: 정지")
        self.get_logger().info("   Enter: 가이드 저장 및 완료")
        self.get_logger().info("   백스페이스: 마지막 키 삭제")
        self.get_logger().info("   X: 취소 (기존 가이드 유지)")
        self.get_logger().info("")
        self.get_logger().info("💡 최대 17개 액션까지 입력 가능 (초기 프레임 1개 + 17개 액션 = 18프레임)")
        sys.stdout.write("📝 가이드 입력: ")
        sys.stdout.flush()
    
    def save_edited_guide(self):
        """편집된 가이드를 저장"""
        if not self.guide_edit_keys:
            self.get_logger().warn("⚠️ 가이드가 비어있습니다. 저장하지 않습니다.")
            return False
        
        # 가이드 키 정규화 (18개로)
        normalized_keys = self._normalize_to_18_keys(self.guide_edit_keys)
        
        # 조합 키 생성
        combo_key = self._combined_key(
            self.selected_scenario,
            self.selected_pattern_type,
            self.selected_distance_level
        )
        
        # 가이드 저장
        self.core_patterns[combo_key] = normalized_keys
        self.save_core_patterns()
        
        guide_str = " ".join([k.upper() for k in normalized_keys])
        self.get_logger().info(f"✅ 가이드 저장 완료: {guide_str}")
        self.get_logger().info(f"   키: {combo_key}")
        return True
        
    def start_next_repeat_measurement(self):
        """다음 반복 측정 시작 (상태 머신 방식)"""
        if not self.is_repeat_measurement_active:
            return
        
        self.current_repeat_index += 1
        self.get_logger().info(f"📊 [{self.current_repeat_index}/{self.target_repeat_count}] 측정 시작...")
        
        # 에피소드 시작
        self.start_episode_with_pattern_and_distance(
            self.selected_scenario,
            self.selected_pattern_type,
            self.selected_distance_level
        )
    
    def check_and_continue_repeat_measurement(self):
        """에피소드 완료 후 다음 반복 측정 확인 및 대기 상태로 전환"""
        if not self.is_repeat_measurement_active:
            return
        
        # 현재 반복이 완료되었는지 확인
        if self.current_repeat_index < self.target_repeat_count:
            # 다음 측정을 위한 시작 위치 세팅 대기 상태로 전환
            self.waiting_for_next_repeat = True
            remaining = self.target_repeat_count - self.current_repeat_index
            self.get_logger().info(f"✅ [{self.current_repeat_index}/{self.target_repeat_count}] 완료.")
            self.get_logger().info(f"📍 시작 위치로 로봇을 이동시킨 후 'N' 키를 눌러 다음 측정을 시작하세요. (남은: {remaining}회)")
        else:
            # 모든 반복 완료
            self.get_logger().info(f"🎉 모든 반복 측정 완료! ({self.target_repeat_count}회)")
            self.is_repeat_measurement_active = False
            self.current_repeat_index = 0
            self.waiting_for_next_repeat = False
            # 🔴 자동 측정 모드도 종료
            self.auto_measurement_mode = False
        
    def _combined_key(self, scenario_id: str, pattern_type: str | None, distance_level: str | None) -> str:
        parts = [scenario_id]
        if pattern_type:
            parts.append(pattern_type)
        if distance_level:
            parts.append(distance_level)
        return "__".join(parts)

    def get_core_pattern_guide(self, scenario_id: str, pattern_type: str | None = None, distance_level: str | None = None) -> str:
        """핵심 패턴 가이드 반환 (시나리오/패턴/거리 조합별로 분기, 없으면 시나리오 기본값 → 디폴트)"""
        # 1) 조합 키 우선
        if pattern_type and distance_level:
            combo = self._combined_key(scenario_id, pattern_type, distance_level)
            if combo in self.core_patterns and self.core_patterns[combo]:
                keys = self._normalize_to_18_keys(self.core_patterns[combo])
                return " ".join([k.upper() for k in keys])
        # 2) 시나리오 단독 키 (과거 호환)
        if scenario_id in self.core_patterns and self.core_patterns[scenario_id]:
            keys = self._normalize_to_18_keys(self.core_patterns[scenario_id])
            return " ".join([k.upper() for k in keys])
        # 3) 초기 기본 가이드(없을 때만 사용) - 4개 시나리오로 통합
        default_guides = {
            "1box_left": "W W W → A A → W W → D D",
            "1box_right": "W W → D D → W W W → A A", 
            "2box_left": "W W → A A A → W W → D D D",
            "2box_right": "W → D D D → W W W → A A A"
        }
        # 기존 형식 호환 (vert/hori 포함)
        old_format_guides = {
            "1box_vert_left": "W W W → A A → W W → D D",
            "1box_vert_right": "W W → D D → W W W → A A", 
            "1box_hori_left": "W → A A A → W W → D D D",
            "1box_hori_right": "W W → D → W W → A",
            "2box_vert_left": "W W → A A A → W W → D D D",
            "2box_vert_right": "W → D D D → W W W → A A A",
            "2box_hori_left": "W → A A A A → W W → D D D D",
            "2box_hori_right": "W W → D D → W W → A A"
        }
        return default_guides.get(scenario_id) or old_format_guides.get(scenario_id, "W → A/D → W → ...")
        
    def start_episode_with_pattern(self, scenario_id: str, pattern_type: str):
        """패턴 타입을 지정하여 에피소드 시작 (거리 선택 전)"""
        config = self.cup_scenarios[scenario_id]
        
        # 패턴 타입 정보를 에피소드명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pattern_episode_name = f"episode_{timestamp}_{scenario_id}_{pattern_type}"
        
        pattern_names = {
            "core": "핵심 패턴",
            "variant": "변형 패턴"
        }
        
        self.get_logger().info(f"🎯 {config['description']} - {pattern_names[pattern_type]} 시작!")
        
        if pattern_type == "core":
            # 핵심 패턴인 경우 가이드 다시 표시
            guide = self.get_core_pattern_guide(scenario_id, pattern_type="core", distance_level=None)
            self.get_logger().info(f"🎮 가이드 순서: {guide}")
            self.get_logger().info("💡 위 순서를 정확히 따라해주세요!")
            # 핵심 패턴 가이드/녹화 플래그 (거리 미선택 플로우에서도 활성화)
            self.core_guidance_active = True
            self.core_guidance_index = 0
            # 이미 표준이 있어도 재등록 가능하도록 토글 반영
            self.record_core_pattern = (scenario_id not in self.core_patterns) or self.overwrite_core
            if scenario_id in self.core_patterns:
                self.core_patterns[scenario_id] = self._normalize_to_18_keys(self.core_patterns[scenario_id])
        elif pattern_type == "variant":
            self.get_logger().info("🔄 핵심 패턴을 변형하여 움직여주세요!")
        
        # 현재 진행 상황 표시
        current = self.scenario_stats[scenario_id]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        self.get_logger().info(f"📊 {scenario_id.upper()}: {progress_bar}")
        
        self.start_episode(pattern_episode_name)

    def start_episode_with_pattern_and_distance(self, scenario_id: str, pattern_type: str, distance_level: str):
        """패턴 + 거리 정보를 포함하여 에피소드 시작 (배치 타입 제거로 단순화)"""
        config = self.cup_scenarios[scenario_id]
        levels = self.distance_levels
        if distance_level not in levels:
            self.get_logger().warn("⚠️ 알 수 없는 거리 레벨, 기본값 medium 사용")
            distance_level = 'medium'
        label = levels[distance_level]['label']
        
        # 에피소드명 생성 (배치 타입은 기본값으로 자동 설정)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 기존 형식 호환: episode_..._{num_box}_{layout_type}_{direction}_{pattern_type}_{distance_level}
        # 기본값으로 가로(hori) 배치 사용
        num_box = scenario_id.split("_")[0]  # "1box" or "2box"
        direction = scenario_id.split("_")[1]  # "left" or "right"
        layout_type = self.default_layout_type  # 기본값: "hori"
        episode_name = f"episode_{timestamp}_{num_box}_{layout_type}_{direction}_{pattern_type}_{distance_level}"
        
        # 현재 선택 상태를 저장해서 종료 시 통계 업데이트에 사용
        self.selected_scenario = scenario_id
        self.selected_pattern_type = pattern_type
        self.selected_distance_level = distance_level
        
        pattern_names = {
            "core": "핵심 패턴",
            "variant": "변형 패턴"
        }
        
        self.get_logger().info(f"🎯 {config['description']} - {pattern_names.get(pattern_type, pattern_type)} - {distance_level.upper()}({label}) 시작!")
        
        if pattern_type == "core":
            guide = self.get_core_pattern_guide(scenario_id, pattern_type="core", distance_level=distance_level)
            # 거리별로 W 길이 참고 안내
            self.get_logger().info(f"🎮 가이드 순서: {guide}")
            self.get_logger().info("💡 위치별 조정: 가까움/W 줄임, 멀음/W 늘림")
            # 핵심 패턴 가이드/녹화 플래그
            self.core_guidance_active = True
            self.core_guidance_index = 0
            # 이미 표준이 있어도 재등록 가능하도록 토글 반영 (조합 키 우선 확인)
            combo_key = self._combined_key(scenario_id, pattern_type, distance_level)
            has_combo = combo_key in self.core_patterns
            has_scenario_only = scenario_id in self.core_patterns
            self.record_core_pattern = (not has_combo and not has_scenario_only) or self.overwrite_core
            # 안내용 정규화 (조합 키 또는 시나리오 키)
            if has_combo:
                self.core_patterns[combo_key] = self._normalize_to_18_keys(self.core_patterns[combo_key])
            elif has_scenario_only:
                self.core_patterns[scenario_id] = self._normalize_to_18_keys(self.core_patterns[scenario_id])
        elif pattern_type == "variant":
            self.get_logger().info("🔄 핵심 패턴을 변형하여 움직여주세요! (타이밍/순서 변경)")
        
        current = self.scenario_stats[scenario_id]
        target = config["target"]
        progress_bar = self.create_progress_bar(current, target)
        self.get_logger().info(f"📊 {scenario_id.upper()}: {progress_bar}")
        
        self.start_episode(episode_name)
        
    def show_h5_verification_menu(self):
        """H5 파일 검증 및 추출 메뉴 표시"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("📋 H5 파일 검증 및 추출")
        self.get_logger().info("=" * 60)
        
        # 최신 파일 목록 표시 (절대 경로 사용)
        data_dir_abs = Path(self.data_dir).resolve()
        h5_files = sorted(data_dir_abs.glob("*.h5"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not h5_files:
            self.get_logger().info("❌ H5 파일이 없습니다.")
            return
        
        self.get_logger().info(f"📁 최근 수집된 파일 (최대 10개):")
        for i, h5_file in enumerate(h5_files[:10], 1):
            file_size_mb = h5_file.stat().st_size / (1024*1024)
            self.get_logger().info(f"   {i}. {h5_file.name} ({file_size_mb:.2f} MB)")
        
        self.get_logger().info("")
        self.get_logger().info("✨ 최신 파일 검증: Enter 키")
        self.get_logger().info("✨ 파일 번호 선택: 1-10 숫자 키")
        self.get_logger().info("🚫 취소: 다른 키")
        
        # 키 입력 대기
        key = self.get_key()
        
        if key == '\r' or key == '\n':
            # 최신 파일 검증
            target_file = h5_files[0]
            self.verify_and_extract_h5_file(target_file)
        elif key.isdigit() and 1 <= int(key) <= min(10, len(h5_files)):
            # 선택한 파일 검증
            file_index = int(key) - 1
            target_file = h5_files[file_index]
            self.verify_and_extract_h5_file(target_file)
        else:
            self.get_logger().info("🚫 취소되었습니다.")
    
    def verify_and_extract_h5_file(self, file_path: Path):
        """H5 파일 검증 및 추출 옵션 제공"""
        self.get_logger().info(f"📁 선택된 파일: {file_path.name}")
        self.get_logger().info("=" * 60)
        
        # 파일 정보 확인
        self.check_h5_file(file_path)
        
        self.get_logger().info("")
        self.get_logger().info("✨ 추출 옵션:")
        self.get_logger().info("   1: 이미지 추출 (PNG)")
        self.get_logger().info("   2: CSV 추출 (액션 데이터)")
        self.get_logger().info("   3: JSON 추출 (전체 데이터)")
        self.get_logger().info("   4: 모든 추출 (이미지 + CSV + JSON)")
        self.get_logger().info("   Enter: 정보만 확인 (추출 없음)")
        self.get_logger().info("🚫 취소: 다른 키")
        
        key = self.get_key()
        
        if key == '1':
            output_dir = file_path.parent / file_path.stem
            self.extract_images_from_h5(file_path, output_dir)
        elif key == '2':
            if not PANDAS_AVAILABLE:
                self.get_logger().warn("⚠️ pandas가 설치되어 있지 않습니다. CSV 추출을 사용할 수 없습니다.")
            else:
                self.export_h5_to_csv(file_path)
        elif key == '3':
            self.export_h5_to_json(file_path)
        elif key == '4':
            output_dir = file_path.parent / file_path.stem
            self.extract_images_from_h5(file_path, output_dir)
            if PANDAS_AVAILABLE:
                self.export_h5_to_csv(file_path)
            else:
                self.get_logger().warn("⚠️ pandas가 설치되어 있지 않아 CSV 추출을 건너뜁니다.")
            self.export_h5_to_json(file_path)
            self.get_logger().info("✅ 모든 추출 완료!")
        elif key == '\r' or key == '\n':
            self.get_logger().info("✅ 파일 정보 확인 완료.")
        else:
            self.get_logger().info("🚫 취소되었습니다.")
    
    def check_h5_file(self, file_path: Path):
        """HDF5 파일의 메타데이터와 데이터 구조를 출력합니다."""
        if not file_path.is_file():
            self.get_logger().error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        try:
            with h5py.File(file_path, 'r') as f:
                file_size_mb = file_path.stat().st_size / (1024*1024)
                self.get_logger().info(f"📁 파일: {file_path.name}")
                self.get_logger().info(f"💾 크기: {file_size_mb:.2f} MB")
                self.get_logger().info("=" * 60)
                
                self.get_logger().info("📋 메타데이터:")
                for key, value in f.attrs.items():
                    if isinstance(value, (np.integer, np.floating)):
                        self.get_logger().info(f"   {key}: {value}")
                    elif isinstance(value, bytes):
                        try:
                            self.get_logger().info(f"   {key}: {value.decode('utf-8')}")
                        except:
                            self.get_logger().info(f"   {key}: {value}")
                    else:
                        self.get_logger().info(f"   {key}: {value}")
                
                self.get_logger().info("")
                self.get_logger().info("📦 데이터 구조:")
                for name, dset in f.items():
                    self.get_logger().info(f"   📄 {name}: {dset.shape} {dset.dtype}")
                
                if 'action_chunks' not in f:
                    self.get_logger().info("💡 정보: Action Chunks 데이터가 없습니다 (이미지 추출에는 영향 없음)")
        
        except Exception as e:
            self.get_logger().error(f"❌ HDF5 파일을 읽는 중 오류 발생: {e}")
    
    def extract_images_from_h5(self, file_path: Path, output_dir: Path):
        """HDF5 파일에서 이미지를 추출하여 PNG 파일로 저장합니다."""
        if not file_path.is_file():
            self.get_logger().error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        try:
            output_dir.mkdir(exist_ok=True)
            self.get_logger().info(f"🖼️  'images' 데이터셋을 '{output_dir}' 폴더에 추출합니다...")
            
            with h5py.File(file_path, 'r') as f:
                if 'images' not in f:
                    self.get_logger().error("'images' 데이터셋을 찾을 수 없습니다.")
                    return
                
                images = f['images']
                num_images = images.shape[0]
                
                for i in range(num_images):
                    img_bgr = images[i]
                    save_path = output_dir / f"frame_{i:04d}.png"
                    cv2.imwrite(str(save_path), img_bgr)
                    if (i + 1) % 5 == 0 or i == num_images - 1:
                        self.get_logger().info(f"   -> 저장 중... {i+1}/{num_images}")
                
                self.get_logger().info(f"✅ 이미지 추출 완료! {num_images}개 프레임 저장됨")
        
        except Exception as e:
            self.get_logger().error(f"❌ 이미지 추출 중 오류 발생: {e}")
    
    def save_single_image_from_h5(self, file_path: Path, index: int):
        """HDF5 파일에서 특정 인덱스의 이미지를 파일로 저장합니다."""
        if not file_path.is_file():
            self.get_logger().error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        try:
            with h5py.File(file_path, 'r') as f:
                if 'images' not in f:
                    self.get_logger().error("'images' 데이터셋을 찾을 수 없습니다.")
                    return
                
                images = f['images']
                if not (0 <= index < images.shape[0]):
                    self.get_logger().error(f"❌ 인덱스 오류: 0에서 {images.shape[0]-1} 사이의 값을 입력하세요.")
                    return
                
                img_bgr = images[index]
                save_path = file_path.parent / f"viewed_{file_path.stem}_frame_{index}.png"
                cv2.imwrite(str(save_path), img_bgr)
                self.get_logger().info(f"🖼️  프레임 {index}번 이미지를 '{save_path}' 파일로 저장했습니다.")
        
        except Exception as e:
            self.get_logger().error(f"❌ 이미지를 저장하는 중 오류 발생: {e}")
    
    def export_h5_to_csv(self, file_path: Path, output_path: Path = None):
        """HDF5 데이터를 CSV 파일로 추출합니다."""
        if not PANDAS_AVAILABLE:
            self.get_logger().error("❌ pandas가 설치되어 있지 않습니다.")
            return
        
        if not file_path.is_file():
            self.get_logger().error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        try:
            with h5py.File(file_path, 'r') as f:
                metadata = dict(f.attrs)
                actions = f['actions'][:]
                action_event_types = f['action_event_types'][:]
                
                data = []
                for i in range(len(actions)):
                    row = {
                        'frame_index': i,
                        'action_x': actions[i][0],
                        'action_y': actions[i][1], 
                        'action_z': actions[i][2],
                        'event_type': action_event_types[i].decode('utf-8') if isinstance(action_event_types[i], bytes) else str(action_event_types[i]),
                        'episode_name': metadata.get('episode_name', ''),
                        'total_duration': metadata.get('total_duration', 0),
                        'action_chunk_size': metadata.get('action_chunk_size', 0)
                    }
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                if output_path is None:
                    # H5 파일의 time_period 메타데이터를 읽어서 파일명에 추가
                    time_period = metadata.get('time_period', None)
                    stem = file_path.stem
                    
                    # stem에서 "medium" 뒤에 시간대 정보 추가
                    if time_period and 'medium' in stem:
                        # medium 뒤에 시간대 추가
                        stem = stem.replace('medium', f'medium_{time_period}')
                    elif time_period:
                        # medium이 없으면 그냥 끝에 추가
                        stem = f"{stem}_{time_period}"
                    
                    output_path = file_path.parent / f"{stem}_data.csv"
                
                df.to_csv(output_path, index=False)
                self.get_logger().info(f"📊 CSV 파일 저장 완료: {output_path}")
                self.get_logger().info(f"   총 {len(data)}개 프레임 데이터 추출")
        
        except Exception as e:
            self.get_logger().error(f"❌ CSV 추출 중 오류 발생: {e}")
    
    def export_h5_to_json(self, file_path: Path, output_path: Path = None):
        """HDF5 데이터를 JSON 파일로 추출합니다."""
        if not file_path.is_file():
            self.get_logger().error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        try:
            with h5py.File(file_path, 'r') as f:
                metadata = {}
                for key, value in f.attrs.items():
                    if isinstance(value, (np.integer, np.floating)):
                        metadata[key] = value.item()
                    elif isinstance(value, bytes):
                        try:
                            metadata[key] = value.decode('utf-8')
                        except:
                            metadata[key] = str(value)
                    else:
                        metadata[key] = value
                
                data = {
                    "file_name": file_path.name,
                    "file_size_mb": float(file_path.stat().st_size / (1024*1024)),
                    "metadata": metadata,
                    "frames": []
                }
                
                actions = f['actions'][:]
                action_event_types = f['action_event_types'][:]
                
                for i in range(len(actions)):
                    frame_data = {
                        "frame_index": i,
                        "action": {
                            "x": float(actions[i][0]),
                            "y": float(actions[i][1]), 
                            "z": float(actions[i][2])
                        },
                        "event_type": action_event_types[i].decode('utf-8') if isinstance(action_event_types[i], bytes) else str(action_event_types[i]),
                        "image_file": f"frame_{i:04d}.png"
                    }
                    data["frames"].append(frame_data)
                
                if output_path is None:
                    output_path = file_path.parent / f"{file_path.stem}_data.json"
                
                with open(output_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, indent=2, ensure_ascii=False)
                
                self.get_logger().info(f"📄 JSON 파일 저장 완료: {output_path}")
                self.get_logger().info(f"   총 {len(data['frames'])}개 프레임 데이터 추출")
        
        except Exception as e:
            self.get_logger().error(f"❌ JSON 추출 중 오류 발생: {e}")
    
    def get_reverse_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """
        액션의 반대 방향을 반환합니다.
        
        Args:
            action: 원본 액션 딕셔너리
            
        Returns:
            반대 방향 액션 딕셔너리
        """
        return {
            "linear_x": -action["linear_x"],
            "linear_y": -action["linear_y"],
            "angular_z": -action["angular_z"]
        }
    
    def start_auto_return(self):
        """에피소드 종료 후 시작 위치로 자동 복귀 시작"""
        if self.auto_return_active:
            self.get_logger().warn("⚠️ 이미 자동 복귀가 진행 중입니다.")
            return
        
        if len(self.episode_data) == 0:
            self.get_logger().warn("⚠️ 복귀할 경로가 없습니다.")
            return
        
        # 복귀할 액션 리스트 생성 (역순, 반대 방향)
        return_actions = []
        # episode_start는 제외하고, start_action만 추출
        for data in self.episode_data:
            if data.get('action_event_type') == 'start_action':
                # 반대 방향 액션 생성
                reverse_action = self.get_reverse_action(data['action'])
                return_actions.append(reverse_action)
        
        if len(return_actions) == 0:
            self.get_logger().warn("⚠️ 복귀할 액션이 없습니다.")
            return
        
        # 역순으로 변경 (마지막 액션부터 첫 액션까지)
        return_actions.reverse()
        
        # 🔴 17개 액션으로 정규화 (초기 프레임 1개 + 17개 액션 = 18프레임)
        target_action_count = self.fixed_episode_length - 1  # 18 - 1 = 17
        if len(return_actions) < target_action_count:
            padding_count = target_action_count - len(return_actions)
            self.get_logger().info(f"📏 복귀 액션 정규화: {len(return_actions)}개 → {target_action_count}개 (STOP {padding_count}개 추가)")
            return_actions.extend([self.STOP_ACTION.copy() for _ in range(padding_count)])
        elif len(return_actions) > target_action_count:
            self.get_logger().warn(f"⚠️ 복귀 액션이 {target_action_count}개를 초과합니다 ({len(return_actions)}개). 첫 {target_action_count}개만 사용합니다.")
            return_actions = return_actions[:target_action_count]
        
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔄 자동 복귀 시작")
        self.get_logger().info(f"   📍 복귀할 액션 수: {len(return_actions)}개 ({target_action_count}개 액션)")
        self.get_logger().info(f"   ⏱️  예상 소요 시간: {len(return_actions) * 0.4:.1f}초 (연속 실행)")
        self.get_logger().info("   💡 각 액션을 0.4초 동안 실행합니다.")
        self.get_logger().info("   🛑 중단하려면 'B' 키를 다시 누르세요.")
        self.get_logger().info("=" * 60)
        
        # 자동 복귀를 별도 스레드에서 실행
        self.auto_return_active = True
        self.return_thread = threading.Thread(target=self.execute_auto_return, args=(return_actions,))
        self.return_thread.daemon = True
        self.return_thread.start()
    
    def execute_auto_return(self, return_actions: List[Dict[str, float]]):
        """
        자동 복귀 실행 (별도 스레드에서 실행)
        자동 연속 실행을 위해 정지 신호 최소화
        
        Args:
            return_actions: 복귀할 액션 리스트 (역순, 반대 방향, 17개 액션으로 정규화됨)
        """
        try:
            # 먼저 정지 상태로 초기화 (간단하게)
            self.current_action = self.STOP_ACTION.copy()
            self.publish_cmd_vel(self.STOP_ACTION, source="auto_return_init")
            time.sleep(0.1)
            
            # 각 액션을 0.4초 동안 실행 (연속 실행)
            for i, action in enumerate(return_actions):
                if not self.auto_return_active:
                    self.get_logger().info("🛑 자동 복귀가 중단되었습니다.")
                    break
                
                self.get_logger().info(f"🔄 복귀 진행: {i+1}/{len(return_actions)} (액션: lx={action['linear_x']:.2f}, ly={action['linear_y']:.2f}, az={action['angular_z']:.2f})")
                
                # 액션 실행 (타이머 의존 제거)
                self.current_action = action.copy()
                self.publish_cmd_vel(action, source=f"auto_return_{i+1}")
                
                # 0.4초 동안 유지 후 다음 액션으로
                time.sleep(0.4)
            
            # 최종 정지 (간단하게)
            if self.auto_return_active:
                self.current_action = self.STOP_ACTION.copy()
                self.publish_cmd_vel(self.STOP_ACTION, source="auto_return_final")
                time.sleep(0.1)
                
                self.get_logger().info("")
                self.get_logger().info("=" * 60)
                self.get_logger().info("✅ 자동 복귀 완료!")
                self.get_logger().info("=" * 60)
                self.get_logger().info("")
        
        except Exception as e:
            self.get_logger().error(f"❌ 자동 복귀 중 오류 발생: {e}")
            import traceback
            self.get_logger().error(f"❌ 트레이스백:\n{traceback.format_exc()}")
        finally:
            self.auto_return_active = False
            self.return_thread = None
    
    def show_measurement_task_table(self):
        """측정 가능한 태스크와 종류를 표로 정리하여 표시"""
        # 조합별 통계를 최신 상태로 업데이트
        self.resync_scenario_progress()
        
        self.get_logger().info("")
        self.get_logger().info("=" * 80)
        self.get_logger().info("📊 측정 태스크 표")
        self.get_logger().info("=" * 80)
        self.get_logger().info("")
        
        # 시나리오별 목표와 설명
        self.get_logger().info("📋 시나리오 (4개):")
        for key, scenario in self.cup_scenarios.items():
            desc = scenario["description"]
            target = scenario["target"]
            current = self.scenario_stats.get(key, 0)
            progress = self.create_progress_bar(current, target)
            self.get_logger().info(f"   {scenario['key']}: {desc} - 목표: {target}개 | {progress}")
        self.get_logger().info("")
        
        # 패턴 타입별 목표
        self.get_logger().info("🎯 패턴 타입 (2개):")
        for pattern, target in self.pattern_targets.items():
            pattern_name = "핵심 패턴 (Core)" if pattern == "core" else "변형 패턴 (Variant)"
            self.get_logger().info(f"   {pattern.upper()}: {pattern_name} - 목표: {target}개")
        self.get_logger().info("")
        
        # 거리 레벨별 목표
        self.get_logger().info("📍 거리 레벨 (3개):")
        for distance, config in self.distance_levels.items():
            label = config["label"]
            samples = config["samples_per_scenario"]
            self.get_logger().info(f"   {distance.upper()}: {label} - 샘플/시나리오: {samples}개")
        self.get_logger().info("")
        
        # 조합별 통계
        self.get_logger().info("📈 조합별 통계:")
        self.get_logger().info("   시나리오 × 패턴 × 거리 = 총 조합")
        total_combinations = 0
        for scenario in self.cup_scenarios.keys():
            for pattern in self.pattern_targets.keys():
                for distance in self.distance_levels.keys():
                    combo = (scenario, pattern, distance)
                    current = self.pattern_distance_stats.get(scenario, {}).get(pattern, {}).get(distance, 0)
                    target = self.distance_targets_per_pattern[pattern][distance]
                    progress = self.create_progress_bar(current, target)
                    self.get_logger().info(f"   {scenario} × {pattern} × {distance}: {progress}")
                    total_combinations += 1
        self.get_logger().info("")
        self.get_logger().info(f"   총 조합 수: {total_combinations}개 (4 시나리오 × 2 패턴 × 3 거리)")
        self.get_logger().info("")
        self.get_logger().info("=" * 80)
        self.get_logger().info("")
    
    def show_auto_measurement_menu(self):
        """자동 측정 메뉴 표시"""
        self.get_logger().info("")
        self.get_logger().info("=" * 80)
        self.get_logger().info("🤖 자동 측정 메뉴")
        self.get_logger().info("=" * 80)
        self.get_logger().info("")
        self.get_logger().info("📋 측정할 태스크를 선택하세요:")
        self.get_logger().info("")
        
        # 시나리오 선택
        self.get_logger().info("1️⃣ 시나리오 선택:")
        for key, scenario in self.cup_scenarios.items():
            desc = scenario["description"]
            current = self.scenario_stats.get(key, 0)
            target = scenario["target"]
            progress = self.create_progress_bar(current, target)
            self.get_logger().info(f"   {scenario['key']}: {desc} | {progress}")
        self.get_logger().info("")
        
        # 패턴 선택
        self.get_logger().info("2️⃣ 패턴 타입 선택:")
        self.get_logger().info("   C: 핵심 패턴 (Core) - 가이드 기반 자동 측정")
        self.get_logger().info("   V: 변형 패턴 (Variant) - 수동 측정 필요")
        self.get_logger().info("")
        
        # 거리 선택
        self.get_logger().info("3️⃣ 거리 레벨 선택:")
        for key, config in self.distance_levels.items():
            label = config["label"]
            key_map = {"close": "J", "medium": "K", "far": "L"}
            self.get_logger().info(f"   {key_map[key]}: {label} ({key})")
        self.get_logger().info("")
        
        self.get_logger().info("💡 자동 측정은 핵심 패턴(Core)만 지원합니다.")
        self.get_logger().info("   핵심 패턴 가이드가 있는 경우에만 자동 측정이 가능합니다.")
        self.get_logger().info("")
        self.get_logger().info("🚫 취소하려면 다른 키를 누르세요.")
        self.get_logger().info("=" * 80)
        self.get_logger().info("")
        
        # 자동 측정 모드 활성화 (시나리오 선택 대기)
        self.scenario_selection_mode = True
        self.auto_measurement_mode = True  # 자동 측정 모드 플래그
    
    def execute_auto_measurement(self, scenario_id: str, pattern_type: str, distance_level: str):
        """자동 측정 실행 (핵심 패턴 가이드 기반)"""
        try:
            # 핵심 패턴 가이드 가져오기
            guide_keys = self.get_core_pattern_guide_keys(scenario_id, pattern_type, distance_level)
            
            if not guide_keys:
                self.get_logger().warn(f"⚠️ {scenario_id}의 핵심 패턴 가이드가 없습니다. 자동 측정을 시작할 수 없습니다.")
                return
            
            self.get_logger().info("")
            self.get_logger().info("=" * 80)
            self.get_logger().info("🤖 자동 측정 시작")
            self.get_logger().info(f"   시나리오: {scenario_id}")
            self.get_logger().info(f"   패턴: {pattern_type}")
            self.get_logger().info(f"   거리: {distance_level}")
            self.get_logger().info(f"   가이드: {' '.join([k.upper() for k in guide_keys])}")
            self.get_logger().info(f"   총 액션 수: {len(guide_keys)}개 (초기 프레임 1개 + 액션 {len(guide_keys)}개 = 총 {len(guide_keys)+1}프레임)")
            self.get_logger().info(f"   예상 소요 시간: {len(guide_keys) * (0.4 + 0.3):.1f}초 (액션 0.4초 + 안정화 0.3초)")
            self.get_logger().info("   🛑 중단하려면 'A' 키를 다시 누르세요.")
            self.get_logger().info("=" * 80)
            self.get_logger().info("")
            
            # 반복 측정 인덱스 업데이트
            if self.is_repeat_measurement_active:
                # 첫 번째 측정일 때만 인덱스 증가 (N 키를 눌렀을 때는 이미 증가되어 있음)
                if self.current_repeat_index == 0:
                    self.current_repeat_index = 1
                self.get_logger().info(f"📊 [{self.current_repeat_index}/{self.target_repeat_count}] 측정 시작...")
            
            # 에피소드 시작
            self.start_episode_with_pattern_and_distance(scenario_id, pattern_type, distance_level)
            
            # 각 키를 순차적으로 실행
            for idx, key in enumerate(guide_keys):
                if not self.auto_measurement_active:
                    self.get_logger().info("🛑 자동 측정이 중단되었습니다.")
                    break
                
                # 키를 액션으로 변환하여 실행
                if key.lower() in self.WASD_TO_CONTINUOUS:
                    action = self.WASD_TO_CONTINUOUS[key.lower()]
                    self.get_logger().info(f"🔄 자동 측정 진행: {idx+1}/{len(guide_keys)} (키: {key.upper()})")
                    
                    # 🔴 N 키와 동일한 메커니즘: 기존 타이머 취소 및 강제 정지
                    timer_was_active = False
                    with self.movement_lock:
                        if self.movement_timer and self.movement_timer.is_alive():
                            self.movement_timer.cancel()
                            timer_was_active = True
                    
                    if timer_was_active:
                        # 강제 정지 (N 키와 동일)
                        self.current_action = self.STOP_ACTION.copy()
                        for i in range(3):
                            self.publish_cmd_vel(self.STOP_ACTION, source=f"auto_measurement_stop_prev_{i+1}")
                            time.sleep(0.02)
                        time.sleep(0.05)
                    
                    # 🔴 N 키와 동일한 메커니즘: 타이머 먼저 시작
                    timer_duration = 0.4
                    with self.movement_lock:
                        self.movement_timer = threading.Timer(timer_duration, self.stop_movement_timed)
                        self.movement_timer.start()
                    
                    # 액션 실행
                    self.current_action = action.copy()
                    self.publish_cmd_vel(action, source=f"auto_measurement_{idx+1}")
                    
                    # 각 액션마다 데이터 수집
                    if self.collecting:
                        self.collect_data_point_with_action("start_action", action)
                    
                    # 타이머가 실행될 때까지 대기 (0.4초)
                    time.sleep(timer_duration)
                    
                    # 타이머가 정지 명령을 발행했는지 확인하고, 추가 대기
                    time.sleep(0.3)  # 정지 신호가 완전히 처리될 시간 확보
                    
                elif key.upper() == 'SPACE':
                    # 정지 명령 (N 키 스페이스바와 동일)
                    self.stop_movement_internal(collect_data=True)
                    time.sleep(0.3)
                else:
                    self.get_logger().warn(f"⚠️ 알 수 없는 키: {key}")
            
            # 에피소드 종료
            if self.auto_measurement_active:
                self.get_logger().info("")
                self.get_logger().info("✅ 자동 측정 완료! 에피소드를 종료합니다...")
                self.stop_episode()
                
                # 반복 측정 확인
                if self.is_repeat_measurement_active:
                    self.check_and_continue_repeat_measurement()
            
        except Exception as e:
            self.get_logger().error(f"❌ 자동 측정 중 오류 발생: {e}")
        finally:
            # 🔴 auto_measurement_active만 False로 설정 (스레드 완료 표시)
            # auto_measurement_mode는 모든 반복 측정이 완료될 때까지 유지
            self.auto_measurement_active = False
            self.auto_measurement_thread = None
            # 🔴 auto_measurement_mode는 여기서 False로 설정하지 않음
            # (반복 측정이 모두 완료되었을 때만 check_and_continue_repeat_measurement에서 False로 설정)
    
    def get_core_pattern_guide_keys(self, scenario_id: str, pattern_type: str, distance_level: str) -> List[str]:
        """핵심 패턴 가이드 키 리스트 반환"""
        # 1) 조합 키 우선
        if pattern_type and distance_level:
            combo = self._combined_key(scenario_id, pattern_type, distance_level)
            if combo in self.core_patterns and self.core_patterns[combo]:
                return self._normalize_to_18_keys(self.core_patterns[combo])
        # 2) 시나리오 단독 키 (과거 호환)
        if scenario_id in self.core_patterns and self.core_patterns[scenario_id]:
            return self._normalize_to_18_keys(self.core_patterns[scenario_id])
        # 3) 기본 가이드 (없을 때만 사용)
        default_guides = {
            "1box_left": ["w", "w", "w", "a", "a", "w", "w", "d", "d"],
            "1box_right": ["w", "w", "d", "d", "w", "w", "w", "a", "a"],
            "2box_left": ["w", "w", "a", "a", "a", "w", "w", "d", "d", "d"],
            "2box_right": ["w", "d", "d", "d", "w", "w", "w", "a", "a", "a"]
        }
        return default_guides.get(scenario_id, [])


def main(args=None):
    # ROS2 초기화
    try:
        rclpy.init(args=args)
    except Exception as e:
        # 이미 초기화되었거나 다른 문제
        print(f"⚠️ ROS2 초기화 경고: {e}")
    
    collector = None
    try:
        collector = MobileVLADataCollector()
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
    finally:
        # 정리 작업
        try:
            if collector:
                collector.stop_episode()
                collector.destroy_node()
        except Exception as e:
            print(f"⚠️ 노드 정리 중 경고: {e}")
        
        # ROS2 종료 (중복 호출 방지)
        try:
            rclpy.shutdown()
        except Exception as e:
            # 이미 종료되었거나 다른 문제 (무시)
            pass

if __name__ == '__main__':
    main()
