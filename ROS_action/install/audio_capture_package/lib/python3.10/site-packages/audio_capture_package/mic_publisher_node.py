#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
from std_msgs.msg import UInt8MultiArray

class MicPublisher(Node):
    def __init__(self):
        super().__init__('mic_publisher')
        self.declare_parameter('duration', 5)
        self.pub = self.create_publisher(UInt8MultiArray, '/audio/raw', 10)
        self.record_and_publish()

    def record_and_publish(self):
        duration = self.get_parameter('duration').value
        samplerate = 16000
        self.get_logger().info(f"🎙️ 녹음 시작: {duration}s (16kHz, mono)")

        # arecord로 raw PCM을 stdout에 출력
        cmd = (
            f"arecord -q -r{samplerate} -f S16_LE "
            f"-c1 -d{duration} -t raw"
        )
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, check=True
            )
            audio_bytes = result.stdout
            self.get_logger().info("✅ 녹음 완료: raw PCM 데이터 수신")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"🔴 녹음 오류: {e}")
            rclpy.shutdown()
            return

        msg = UInt8MultiArray(data=list(audio_bytes))
        self.pub.publish(msg)
        self.get_logger().info("✅ /audio/raw 토픽 전송 성공")

        # 퍼블리시 보장
        rclpy.spin_once(self, timeout_sec=0.1)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    MicPublisher()

if __name__ == '__main__':
    main()