#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class ActionExecutor(Node):
    def __init__(self):
        super().__init__('action_executor')
        self.get_logger().info('ActionExecutor 초기화 완료, /selected_action 구독 대기 중…')

        # 모델에서 선택된 동작을 수신
        self.subscription = self.create_subscription(
            String,
            '/selected_action',
            self.action_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('ActionExecutor 노드 시작됨')

    def action_callback(self, msg):
        raw_action = msg.data.strip().lower()
        action = self.normalize_command(raw_action)
        self.get_logger().info(f"수신된 동작 명령: {raw_action} → 해석된 명령: {action}")
        if action == '전진':
            self.move(linear_x=0.2)
        elif action == '후진':
            self.move(linear_x=-0.2)
        elif action == '좌측이동':
            self.move(linear_y=0.2)
        elif action == '우측이동':
            self.move(linear_y=-0.2)
        elif action == '제자리회전':
            self.move(angular_z=0.5)
        elif action == '대각선이동':
            self.move(linear_x=0.2, linear_y=0.2)
        elif action == '정지':
            self.stop()
        else:
            self.get_logger().warn(f"알 수 없는 명령: {raw_action}")

    # 명령어를 정규화하는 메서드
    def normalize_command(self, text):
        mapping = {
            '전진': ['전진', '앞으로', '앞으로 가', '앞으로 이동', 'forward', 'go forward'],
            '후진': ['후진', '뒤로', '뒤로 가', '뒤로 이동', 'backward', 'go backward'],
            '좌측이동': ['왼쪽', '왼쪽으로', '좌측이동', 'left', 'move left'],
            '우측이동': ['오른쪽', '오른쪽으로', '우측이동', 'right', 'move right'],
            '제자리회전': ['회전', '제자리 회전', '제자리에서 회전', 'rotate', 'spin', 'turn'],
            '대각선이동': ['대각선', '대각선으로 이동', 'diagonal', 'move diagonally'],
            '정지': ['정지', '멈춰', '멈추기', 'stop', 'halt', 'freeze']
        }
        for command, keywords in mapping.items():
            if text in keywords:
                return command
        return "알수없음"
    
    # duration 동안 명령을 수행
    def move(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=1.5): 
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_pub.publish(twist)
            time.sleep(0.1)
        self.stop()

    # 정지
    def stop(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.get_logger().info("정지 명령 실행됨")

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
