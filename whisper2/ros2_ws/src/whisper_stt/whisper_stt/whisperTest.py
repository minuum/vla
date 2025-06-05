#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, String
import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000   # 모델 샘플링 레이트
MODEL_NAME = "TheoJo/whisper-tiny-ko"

qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

class AudioTranscriber(Node):

    def __init__(self):
        super().__init__('audio_transcriber')

        # Whisper 모델 로드
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(
            MODEL_NAME, language="korean", task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME
        ).to(device)
        self.device = device

        # /audio/raw 토픽 구독 (5초 단위 Int16MultiArray)
        self.sub = self.create_subscription(
            Int16MultiArray,
            '/audio/raw',
            self.audio_callback,
            qos_profile=qos
        )

        # 전사 결과 퍼블리셔
        self.text_pub = self.create_publisher(String, '/stt/text', 10)
        self.transcript_pub = self.create_publisher(String, '/audio/transcript', 10)

        self.get_logger().info("Listening to /audio/raw, ready to transcribe")

    def audio_callback(self, msg: Int16MultiArray):
        # Int16 배열 → float32 정규화 [-1, 1]
        audio_data = np.array(msg.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Whisper 입력 생성
        inputs = self.processor(
            audio_float,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        input_feats = inputs.input_features.to(self.device)

        # 전사 생성 및 디코딩
        output_ids = self.model.generate(input_feats)
        transcription = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # 결과 발행
        out_msg = String()
        out_msg.data = transcription
        self.transcript_pub.publish(out_msg)
        self.text_pub.publish(out_msg)

        # 로그
        self.get_logger().info(f"[Whisper] {transcription}")

def main(args=None):
    rclpy.init(args=args)
    node = AudioTranscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
