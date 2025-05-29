#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
import numpy as np
from std_msgs.msg import Int16MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

class MicPublisher(Node):
    def __init__(self):
        super().__init__('mic_publisher')
        self.declare_parameter('duration', 5)
        self.pub = self.create_publisher(Int16MultiArray, '/audio/raw', qos_profile=qos)
        # 타이머를 걸어서 노드를 종료하지 않고 계속 떠 있도록 함
        self.timer = self.create_timer(
            self.get_parameter('duration').value,
            self.record_and_publish
        )

    def record_and_publish(self):
        duration = self.get_parameter('duration').value
        samplerate = 16000
        self.get_logger().info(f"🎙️ 녹음 시작: {duration}s (16kHz, mono)")

        cmd = f"arecord -q -r{samplerate} -f S16_LE -c1 -d{duration} -t raw"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, check=True
            )
            audio_bytes = result.stdout
            self.get_logger().info("✅ 녹음 완료: raw PCM 데이터 수신")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"🔴 녹음 오류: {e}")
            return

        arr16 = np.frombuffer(audio_bytes, dtype=np.int16)
        msg = Int16MultiArray(data=arr16.tolist())
        self.pub.publish(msg)
        self.get_logger().info("✅ /audio/raw 토픽 전송 성공")

def main(args=None):
    rclpy.init(args=args)
    node = MicPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
