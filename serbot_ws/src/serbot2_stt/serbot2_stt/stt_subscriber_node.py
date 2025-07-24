#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
import numpy as np
import os
import tempfile
import threading
from queue import Queue

class STTSubscriberNode(Node):
    def __init__(self):
        super().__init__('stt_subscriber_node')
        
        # Subscriber 설정
        self.subscription = self.create_subscription(
            Int16MultiArray,
            '/audio/raw',
            self.audio_callback,
            10)
        
        # Publisher 설정
        self.publisher_ = self.create_publisher(String, '/command/text', 10)
        
        # Audio buffer 설정
        self.audio_buffer = Queue()
        self.sample_rate = 16000
        self.buffer_duration = 3.0  # seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Whisper 모델 경로 설정
        self.whisper_model_path = os.path.expanduser('~/jetson-containers/data/models/whisper')
        
        self.get_logger().info('STT Subscriber Node started')
        self.get_logger().info(f'Whisper model path: {self.whisper_model_path}')
        
        # STT processing thread 시작
        self.processing_thread = threading.Thread(target=self.process_stt)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Whisper 모델 초기화 (간단한 버전)
        self.init_whisper()
    
    def init_whisper(self):
        """Whisper 모델 초기화"""
        try:
            # 여기서는 간단하게 whisper 명령어를 사용
            # 실제로는 transformers 라이브러리를 사용할 수 있음
            import subprocess
            result = subprocess.run(['which', 'whisper'], capture_output=True, text=True)
            if result.returncode == 0:
                self.get_logger().info('Whisper CLI found')
                self.use_whisper_cli = True
            else:
                self.get_logger().warning('Whisper CLI not found, using mock STT')
                self.use_whisper_cli = False
        except Exception as e:
            self.get_logger().error(f'Failed to initialize Whisper: {e}')
            self.use_whisper_cli = False
    
    def audio_callback(self, msg):
        """오디오 데이터 수신 콜백"""
        audio_data = np.array(msg.data, dtype=np.int16)
        self.audio_buffer.put(audio_data)
        self.get_logger().debug(f'Received audio chunk: {len(audio_data)} samples')
    
    def process_stt(self):
        """STT 처리 스레드"""
        accumulated_audio = np.array([], dtype=np.int16)
        
        while rclpy.ok():
            try:
                # 오디오 버퍼에서 데이터 가져오기
                if not self.audio_buffer.empty():
                    audio_chunk = self.audio_buffer.get()
                    accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])
                    
                    # 충분한 오디오가 쌓이면 STT 처리
                    if len(accumulated_audio) >= self.buffer_size:
                        text = self.transcribe_audio(accumulated_audio)
                        if text.strip():
                            # 결과 퍼블리시
                            msg = String()
                            msg.data = text
                            self.publisher_.publish(msg)
                            self.get_logger().info(f'Transcribed: "{text}"')
                        
                        # 버퍼 리셋 (50% 오버랩)
                        overlap_size = self.buffer_size // 2
                        accumulated_audio = accumulated_audio[-overlap_size:]
                
                # 짧은 대기
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f'STT processing error: {e}')
                import time
                time.sleep(1.0)
    
    def transcribe_audio(self, audio_data):
        """오디오 데이터를 텍스트로 변환"""
        try:
            if self.use_whisper_cli:
                return self.transcribe_with_whisper_cli(audio_data)
            else:
                return self.mock_transcribe(audio_data)
        except Exception as e:
            self.get_logger().error(f'Transcription failed: {e}')
            return ""
    
    def transcribe_with_whisper_cli(self, audio_data):
        """Whisper CLI를 사용한 음성 인식"""
        import subprocess
        import tempfile
        
        # 임시 파일에 오디오 저장
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # WAV 파일 헤더 생성
            import wave
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Whisper 실행
            cmd = [
                'whisper',
                tmp_file.name,
                '--model', 'base',
                '--language', 'ko',
                '--output_format', 'txt'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # 임시 파일 삭제
            os.unlink(tmp_file.name)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.get_logger().error(f'Whisper CLI error: {result.stderr}')
                return ""
                
    def mock_transcribe(self, audio_data):
        """모의 음성 인식 (개발용)"""
        # 간단한 음성 활동 감지
        volume = np.sqrt(np.mean(audio_data ** 2))
        if volume > 1000:  # 임계값
            return f"음성이 감지되었습니다 (볼륨: {volume:.0f})"
        return ""

def main(args=None):
    rclpy.init(args=args)
    
    stt_subscriber = STTSubscriberNode()
    
    try:
        rclpy.spin(stt_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        stt_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 