#!/usr/bin/env python3
import os
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from whisper_live.run_whisper_live import load_whisper_model, transcribe_raw_audio

class STTSubscriber(Node):
    def __init__(self):
        super().__init__('stt_subscriber')
        model_path = os.path.expanduser('~/jetson-containers/data/models/whisper')
        self.device = torch.device('cpu')
        self.processor, self.model = load_whisper_model(model_path, device=self.device)

        self.pub = self.create_publisher(String, '/command/text', 10)
        self.sub = self.create_subscription(
            AudioData,
            '/audio/raw',
            self.audio_callback,
            10)
        self.get_logger().info('STTSubscriber: Waiting for /audio/raw')

    def audio_callback(self, msg: AudioData):
        self.get_logger().info('STTSubscriber: Received raw audio, processing STT')
        import numpy as np
        pcm = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

        text = transcribe_raw_audio(pcm, self.processor, self.model, device=self.device)
        self.get_logger().info(f'STTSubscriber: Recognized -> {text}')

        msg_out = String(data=text)
        self.pub.publish(msg_out)
        self.get_logger().info('STTSubscriber: Published /command/text')

def main(args=None):
    rclpy.init(args=args)
    node = STTSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()