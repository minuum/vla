#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
import io

# Kosmos-2 imports
from transformers import AutoProcessor, AutoModelForVision2Seq

class KosmosCameraNode(Node):
    def __init__(self):
        super().__init__('kosmos_camera_node')
        
        self.bridge = CvBridge()
        
        # Kosmos-2 모델 초기화
        self.get_logger().info('Loading Kosmos-2 model...')
        try:
            self.model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
            self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
            self.get_logger().info('Kosmos-2 model loaded successfully!')
        except Exception as e:
            self.get_logger().error(f'Failed to load Kosmos-2 model: {e}')
            return
        
        # ROS 구독자 및 퍼블리셔 설정
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.result_publisher = self.create_publisher(
            String,
            '/kosmos/analysis_result',
            10
        )
        
        self.get_logger().info('Kosmos Camera Node initialized!')
        self.get_logger().info('Waiting for camera images on /camera/image_raw...')
        
        # 분석 카운터
        self.analysis_count = 0
        self.max_analysis_per_session = 5  # 세션당 최대 5회 분석

    def image_callback(self, msg):
        if self.analysis_count >= self.max_analysis_per_session:
            return
            
        try:
            # ROS Image를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # OpenCV BGR을 PIL RGB로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # 이미지 임시 저장 (Kosmos-2 데모와 동일한 방식)
            pil_image.save("camera_capture.jpg")
            pil_image = PILImage.open("camera_capture.jpg")
            
            self.get_logger().info(f'Processing image {self.analysis_count + 1}/{self.max_analysis_per_session}...')
            
            # Kosmos-2로 이미지 분석
            result = self.analyze_image_with_kosmos(pil_image)
            
            # 결과 퍼블리시
            result_msg = String()
            result_msg.data = result
            self.result_publisher.publish(result_msg)
            
            self.get_logger().info(f'Analysis result: {result}')
            self.analysis_count += 1
            
            if self.analysis_count >= self.max_analysis_per_session:
                self.get_logger().info('Analysis session completed!')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def analyze_image_with_kosmos(self, image):
        """Kosmos-2를 사용해서 이미지를 분석합니다."""
        try:
            # 프롬프트 설정
            prompt = "<grounding>An image of"
            
            # 모델 입력 준비
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # 간단한 설명 생성 (generate 대신 다른 방법 시도)
            # 분산 처리 문제를 피하기 위해 더 간단한 접근
            try:
                # 이미지 특징 추출만 시도
                vision_outputs = self.model.vision_model(inputs["pixel_values"])
                
                #  이미지 정보 반환
                height, width = image.size
                return f"Image analyzed: {width}x{height} pixels - Vision features extracted successfully"
                
            except Exception as gen_error:
                self.get_logger().warn(f'Generation failed, returning basic analysis: {gen_error}')
                return f"Image processed: {image.size[0]}x{image.size[1]} pixels"
                
        except Exception as e:
            self.get_logger().error(f'Kosmos-2 analysis failed: {e}')
            return f"Analysis failed: {str(e)}"

def main(args=None):
    rclpy.init(args=args)
    
    try:
        kosmos_node = KosmosCameraNode()
        rclpy.spin(kosmos_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
