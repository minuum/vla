#!/usr/bin/env python3
"""
WASD 수동 로봇 컨트롤러 (단독 실행 버전)
키보드로 직접 로봇을 조작할 수 있습니다.
"""
import sys
import tty
import termios
import time
import threading
import numpy as np

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
    print("✅ 실제 로봇 제어 모드")
except ImportError:
    ROBOT_AVAILABLE = False
    print("🎮 시뮬레이션 모드 (pop.driving 없음)")

class ManualRobotController:
    def __init__(self):
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50  # 속도 설정
        else:
            self.driver = None
        
        # WASD 방식 액션들 (데이터 수집기와 동일)
        self.actions = {
            'w': {"linear_x": 1.25, "linear_y": 0.0, "angular_z": 0.0},      # 앞으로
            'a': {"linear_x": 0.0, "linear_y": 1.25, "angular_z": 0.0},      # 왼쪽으로
            's': {"linear_x": -1.25, "linear_y": 0.0, "angular_z": 0.0},     # 뒤로
            'd': {"linear_x": 0.0, "linear_y": -1.25, "angular_z": 0.0},     # 오른쪽으로
            'q': {"linear_x": 1.25, "linear_y": 1.25, "angular_z": 0.0},     # 대각선 앞왼쪽
            'e': {"linear_x": 1.25, "linear_y": -1.25, "angular_z": 0.0},    # 대각선 앞오른쪽
            'z': {"linear_x": -1.25, "linear_y": 1.25, "angular_z": 0.0},    # 대각선 뒤왼쪽
            'c': {"linear_x": -1.25, "linear_y": -1.25, "angular_z": 0.0},   # 대각선 뒤오른쪽
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 1.25},      # 왼쪽 회전
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -1.25},     # 오른쪽 회전
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}        # 정지 (스페이스바)
        }
        
        self.running = True
        
        print("🎮 Manual Robot Controller 시작!")
        print("📋 조작 방법:")
        print("   W/A/S/D: 이동, Q/E/Z/C: 대각선")
        print("   R/T: 회전, 스페이스바: 정지")
        print("   F/G: 속도 조절, X: 종료")
        print("⌨️  키보드 입력 대기 중...")
        print("=" * 40)

    def get_key(self) -> str:
        """터미널에서 키 입력을 읽습니다"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()

    def handle_key_input(self, key: str):
        """키보드 입력에 따른 로직 실행"""
        if key == '\x03' or key == 'x':  # Ctrl+C 또는 X
            print("🛑 종료 중...")
            self.stop_robot()
            self.running = False
            return
        elif key == 'f':
            if ROBOT_AVAILABLE:
                self.throttle = max(10, self.throttle - 10)
                print(f'🔽 속도: {self.throttle}%')
        elif key == 'g':
            if ROBOT_AVAILABLE:
                self.throttle = min(100, self.throttle + 10)
                print(f'🔼 속도: {self.throttle}%')
        elif key in self.actions:
            action = self.actions[key]
            self.execute_action(action, key)

    def execute_action(self, action: dict, key: str):
        """액션을 실제 로봇으로 실행 (단발성 움직임)"""
        move_duration = 0.3  # 0.3초간 움직이고 정지

        if ROBOT_AVAILABLE and self.driver:
            # WASD 액션을 실제 로봇 명령으로 변환
            if abs(action["angular_z"]) > 0.1:
                # 회전 명령
                spin_speed = int(action["angular_z"] * self.throttle)
                self.driver.spin(spin_speed)
                time.sleep(move_duration)
                self.driver.stop()
            elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                # 이동 명령
                angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                if angle < 0:
                    angle += 360
                self.driver.move(int(angle), self.throttle)
                time.sleep(move_duration)
                self.driver.stop()
            else:
                self.driver.stop()
        
        # 로그 출력
        if key == ' ':
            print("🛑 정지")
        else:
            action_names = {
                'w': '앞으로', 'a': '왼쪽', 's': '뒤로', 'd': '오른쪽',
                'q': '앞왼쪽', 'e': '앞오른쪽', 'z': '뒤왼쪽', 'c': '뒤오른쪽',
                'r': '왼쪽회전', 't': '오른쪽회전'
            }
            name = action_names.get(key, key.upper())
            print(f"🚀 {name}: ({action['linear_x']:+.1f}, {action['linear_y']:+.1f}, {action['angular_z']:+.1f}) - {move_duration}초간 실행")

    def stop_robot(self):
        """로봇 정지"""
        if ROBOT_AVAILABLE and self.driver:
            self.driver.stop()

    def run(self):
        """메인 컨트롤 루프"""
        try:
            while self.running:
                key = self.get_key()
                self.handle_key_input(key)
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C로 중단됨")
        finally:
            self.stop_robot()
            print("✅ 프로그램 종료")

def main():
    controller = ManualRobotController()
    controller.run()

if __name__ == '__main__':
    main()