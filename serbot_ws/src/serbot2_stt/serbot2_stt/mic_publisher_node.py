#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import subprocess
import numpy as np
import threading
import time

class MicPublisherNode(Node):
    def __init__(self):
        super().__init__('mic_publisher_node')
        
        # Publisher 설정
        self.publisher_ = self.create_publisher(Int16MultiArray, '/audio/raw', 10)
        
        # Audio 설정
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 1.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        self.get_logger().info('Mic Publisher Node started')
        
        # Audio recording thread 시작
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def record_audio(self):
        """arecord를 사용하여 오디오를 녹음하고 퍼블리시"""
        while rclpy.ok():
            try:
                # arecord 명령어로 1초간 녹음
                cmd = [
                    'arecord',
                    '-D', 'plughw:0,0',  # 기본 오디오 장치
                    '-r', str(self.sample_rate),
                    '-c', str(self.channels),
                    '-f', 'S16_LE',
                    '-d', str(self.chunk_duration),
                    '-t', 'raw'
                ]
                
                # subprocess로 arecord 실행
                result = subprocess.run(cmd, capture_output=True, check=True)
                
                # 바이너리 데이터를 numpy array로 변환
                audio_data = np.frombuffer(result.stdout, dtype=np.int16)
                
                # ROS 메시지로 변환하여 퍼블리시
                msg = Int16MultiArray()
                msg.data = audio_data.tolist()
                
                self.publisher_.publish(msg)
                self.get_logger().debug(f'Published audio chunk: {len(audio_data)} samples')
                
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f'Audio recording failed: {e}')
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Unexpected error: {e}')
                time.sleep(1.0)

def main(args=None):
    rclpy.init(args=args)
    
    mic_publisher = MicPublisherNode()
    
    try:
        rclpy.spin(mic_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        mic_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 