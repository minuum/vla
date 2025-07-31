#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import String
import numpy as np
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000   # 모델 샘플링 레이트
DURATION = 5          # 전사 단위(초)
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

        # 버퍼 초기화
        self.target_samples = SAMPLE_RATE * DURATION
        self.buffer = []
        self.accum_samples = 0

        # /audio/raw 토픽 구독 (QoS 프로파일은 필요에 따라 조정)
        self.sub = self.create_subscription(
            Int16MultiArray,
            '/audio/raw',
            self.audio_callback,
            qos_profile=qos
        )

        self.text_pub = self.create_publisher(
            String,
            '/stt/text',
            10
        )

        self.get_logger().info(f"Listening to /audio/raw, "
                               f"transcribing every {DURATION}s")

        self.pub = self.create_publisher(String, '/audio/transcript', 10)

    def audio_callback(self, msg: Int16MultiArray):
        # msg.data가 bytes 배열이라면 int16로 해석
        chunk = np.array(msg.data, dtype=np.int16)
        self.buffer.append(chunk)
        self.accum_samples += chunk.shape[0]

        # 충분히 모였으면 전사
        if self.accum_samples >= self.target_samples:
            # 정확히 target_samples만큼 자름
            audio_data = np.concatenate(self.buffer)[:self.target_samples]
            # float32로 정규화 [-1,1]
            audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max

            # Whisper 입력 생성
            inputs = self.processor(
                audio_float,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            )
            input_feats = inputs.input_features.to(self.device)

            # 생성
            output_ids = self.model.generate(input_feats)
            transcription = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            # 로그 출력
            self.pub.publish(String(data=transcription))
            self.get_logger().info(f"[Whisper] {transcription}")
<<<<<<< HEAD
=======


>>>>>>> 460f1aa (feat: VLA 노드 Model_ws 추가 및 핵심 ROS 노드들 개선)
            txt_msg = String()
            txt_msg.data = transcription
            self.text_pub.publish(txt_msg)

            # 버퍼 초기화
            self.buffer = []
            self.accum_samples = 0


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
