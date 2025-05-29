#!/usr/bin/env python3
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String as RosString
from geometry_msgs.msg import Twist

import torch
from PIL import Image as PilImage
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import sys
import re
from pathlib import Path

# --- 전역 변수 및 헬퍼 함수 (simple_inference.py에서 가져옴) ---
big_vision_available = False
reconstruct_masks_fn = None

def setup_big_vision(custom_big_vision_path, logger):
    global big_vision_available, reconstruct_masks_fn
    if not custom_big_vision_path or not os.path.isdir(custom_big_vision_path):
        if custom_big_vision_path:
            logger.warn(f"제공된 big_vision 경로를 찾을 수 없습니다: {custom_big_vision_path}")
        logger.warn("big_vision 경로가 제공되지 않았거나 유효하지 않아 세그멘테이션 마스크 고급 재구성이 불가능합니다.")
        big_vision_available = False
        reconstruct_masks_fn = None
        return

    abs_bv_path = os.path.abspath(custom_big_vision_path)
    if abs_bv_path not in [os.path.abspath(p) for p in sys.path]:
        sys.path.append(abs_bv_path)
        logger.info(f"'{abs_bv_path}' 경로를 sys.path에 추가했습니다.")

    try:
        import big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
        reconstruct_masks_fn = segeval.get_reconstruct_masks('oi')
        logger.info("big_vision의 segeval 모듈 및 reconstruct_masks 함수 로드 성공!")
        big_vision_available = True
    except ImportError as e:
        logger.error(f"big_vision 모듈 임포트 실패: {e}")
        logger.error(f"big_vision_path 변수('{abs_bv_path}')가 정확한지, 해당 경로에 big_vision 리포지토리가 올바르게 클론되었는지,")
        logger.error("그리고 필요한 의존성 (예: absl-py, ml_collections, einops)이 설치되었는지 확인해주세요.")
        big_vision_available = False
    except Exception as e:
        logger.error(f"big_vision 관련 로드 중 예상치 못한 오류: {e}")
        big_vision_available = False

    if not big_vision_available:
        logger.warn("big_vision 모듈을 로드할 수 없어 세그멘테이션 마스크 재구성이 불가능합니다.")

def parse_segmentation_output(text_output, img_width, img_height, current_prompt_text):
    detections = []
    text_to_parse = text_output
    prompt_cleaned = current_prompt_text.strip()
    if text_to_parse.startswith(prompt_cleaned):
            text_to_parse = text_to_parse[len(prompt_cleaned):].strip()
    
    segments_data = text_to_parse.split(';')
    norm_factor = 1023.0

    for segment_data_str in segments_data:
        segment_data_str = segment_data_str.strip()
        if not segment_data_str: continue
        loc_tokens_match = re.findall(r"<loc(\d{4})>", segment_data_str)
        seg_tokens_match = re.findall(r"<seg(\d{3})>", segment_data_str)
        label_text_candidate = segment_data_str
        for loc_token_str in re.findall(r"<loc\d{4}>", label_text_candidate): label_text_candidate = label_text_candidate.replace(loc_token_str, "")
        for seg_token_str in re.findall(r"<seg\d{3}>", label_text_candidate): label_text_candidate = label_text_candidate.replace(seg_token_str, "")
        label = label_text_candidate.replace("segment", "").strip()
        if not label: label = "unknown"
        if len(loc_tokens_match) == 4:
            y_min_token,x_min_token,y_max_token,x_max_token = map(int,loc_tokens_match)
            norm_y_min,norm_x_min,norm_y_max,norm_x_max = y_min_token/norm_factor,x_min_token/norm_factor,y_max_token/norm_factor,x_max_token/norm_factor
            box_scaled_pixels = [round(norm_x_min*img_width,2),round(norm_y_min*img_height,2),round(norm_x_max*img_width,2),round(norm_y_max*img_height,2)]
            current_detection = {"label":label,"original_loc_tokens":[y_min_token,x_min_token,y_max_token,x_max_token],"scaled_box_pixels":box_scaled_pixels}
            if big_vision_available and reconstruct_masks_fn and len(seg_tokens_match)==16:
                seg_token_values = np.array([int(st) for st in seg_tokens_match],dtype=np.int32)
                try:
                    reconstructed_mask_array = reconstruct_masks_fn(seg_token_values[np.newaxis,:])
                    current_detection["segmentation_mask_tokens"] = seg_token_values.tolist()
                    current_detection["reconstructed_mask"] = reconstructed_mask_array[0].tolist()
                except Exception as e_mask: 
                    print(f"마스크 재구성 중 오류(Label:{label}):{e_mask}") 
                    current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match]
            elif len(seg_tokens_match)>0: current_detection["segmentation_mask_tokens"] = [int(st) for st in seg_tokens_match]
            detections.append(current_detection)
    return detections

def calculate_move_command_for_twist(bbox_pixels, image_width, image_height, logger): 
    if bbox_pixels is None or len(bbox_pixels) != 4:
        logger.warn("유효한 바운딩 박스 정보가 없어 이동 명령을 계산할 수 없습니다.")
        return 0.0, 0.0, 0.0

    x_min, y_min, x_max, y_max = bbox_pixels
    KP_ALIGN_ANGULAR_Z = 0.005
    ALIGNMENT_TOLERANCE_RATIO = 0.05
    FORWARD_SPEED_WHEN_ALIGNED = 0.1
    MAX_ANGULAR_Z_SPEED = 0.5
    OBJECT_REACHED_HEIGHT_RATIO = 0.6 
    
    target_cx = (x_min + x_max) / 2
    image_center_x = image_width / 2
    error_x = target_cx - image_center_x
    bbox_height = y_max - y_min

    linear_x_cmd, linear_y_cmd, angular_z_cmd = 0.0, 0.0, 0.0

    if (bbox_height / image_height) >= OBJECT_REACHED_HEIGHT_RATIO:
        logger.info(f"객체({(bbox_height / image_height)*100:.1f}% 차지)에 충분히 접근함. 정지.")
        return 0.0, 0.0, 0.0
    if abs(error_x) > (image_width * ALIGNMENT_TOLERANCE_RATIO):
        angular_z_cmd = -KP_ALIGN_ANGULAR_Z * error_x
        angular_z_cmd = max(-MAX_ANGULAR_Z_SPEED, min(MAX_ANGULAR_Z_SPEED, angular_z_cmd))
        logger.info(f"정렬 시도: error_x={error_x:.2f}, 계산된 angular_z_cmd={angular_z_cmd:.3f}")
    else:
        logger.info("정렬 완료. 전진 시도.")
        linear_x_cmd = FORWARD_SPEED_WHEN_ALIGNED
    return linear_x_cmd, linear_y_cmd, angular_z_cmd
# --- 헬퍼 함수 끝 ---

class PaliGemmaROSNode(Node):
    def __init__(self):
        super().__init__('paligemma_ros_node')
        self.get_logger().info("PaliGemma ROS Node initializing...")

        # --- 설정값 (ROS 파라미터로 변경 권장) ---
        self.declare_parameter('model_id', "google/paligemma-3b-mix-224")
        self.declare_parameter('tokenizer_path', self.get_parameter('model_id').get_parameter_value().string_value)
        self.declare_parameter('model_cache_dir', ".paligemma_ros_cache")
        self.declare_parameter('max_new_tokens', 256)
        self.declare_parameter('big_vision_path', "")
        self.declare_parameter('device_preference', "cuda")

        self.model_id = self.get_parameter('model_id').get_parameter_value().string_value
        self.tokenizer_path = self.get_parameter('tokenizer_path').get_parameter_value().string_value
        self.model_cache_dir = self.get_parameter('model_cache_dir').get_parameter_value().string_value
        self.max_new_tokens = self.get_parameter('max_new_tokens').get_parameter_value().integer_value
        self.big_vision_path_param = self.get_parameter('big_vision_path').get_parameter_value().string_value
        self.device_preference = self.get_parameter('device_preference').get_parameter_value().string_value
        
        if self.big_vision_path_param:
            setup_big_vision(self.big_vision_path_param, self.get_logger())
        else:
            self.get_logger().warn("ROS 파라미터 'big_vision_path'가 제공되지 않아 고급 마스크 재구성이 불가능합니다.")
            global big_vision_available
            big_vision_available = False
        
        if self.device_preference == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # 모델 및 프로세서 초기화
        self.model = None
        self.processor = None 
        try:
            self.get_logger().info(f"Loading model '{self.model_id}' and processor...")
            model_save_path = Path(self.model_cache_dir) / self.model_id.split('/')[-1]
            model_save_path.mkdir(parents=True, exist_ok=True)

            self.processor = AutoProcessor.from_pretrained(self.tokenizer_path, cache_dir=model_save_path)
            model_kwargs = {"cache_dir": model_save_path, "low_cpu_mem_usage": True}
            
            if self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto" 
            else:
                model_kwargs["torch_dtype"] = torch.float32
            
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            if self.device.type != "cuda": 
                 self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("Model and processor loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"모델 또는 프로세서 로드 중 심각한 오류 발생: {e}")
            self.get_logger().warn("모델 및 프로세서가 로드되지 않았습니다. 노드는 계속 실행되지만 추론은 불가능합니다.")
            # self.model과 self.processor는 이미 위에서 None으로 초기화되었거나, 
            # 로드 중 실패하면 None 상태를 유지하게 됩니다. 여기서 추가로 None을 할당해도 괜찮습니다.
            self.model = None 
            self.processor = None
            # rclpy.shutdown() 호출 제거
            # return도 제거하여 나머지 subscriber/publisher가 생성되도록 함

        self.bridge = CvBridge()
        self.current_image_cv = None
        self.current_text_prompt = None
        self.inference_in_progress = False 

        image_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # 또는 BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.image_subscription = self.create_subscription(
            RosImage,
            '/camera/image_raw',
            self.image_callback,
            image_qos_profile
        )

        self.text_subscription = self.create_subscription(RosString, '/stt/text', self.text_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vqa_answer_publisher = self.create_publisher(RosString, '/paligemma/vqa_answer', 10)
        
        self.get_logger().info("PaliGemma ROS Node initialized and ready.")

    def image_callback(self, msg):
        if self.inference_in_progress: 
            return 
        try:
            self.current_image_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info('Image received.') 
            self.try_inference()
        except Exception as e:
            self.get_logger().error(f"이미지 콜백 처리 중 오류: {e}")

    def text_callback(self, msg):
        if self.inference_in_progress: 
            self.get_logger().warn("Inference already in progress, new text prompt ignored.")
            return
        self.current_text_prompt = msg.data
        self.get_logger().info(f"텍스트 프롬프트 수신: '{self.current_text_prompt}'")
        self.try_inference()

    def try_inference(self):
        if self.current_image_cv is not None and self.current_text_prompt is not None:
            if self.model is None or self.processor is None: 
                self.get_logger().error("모델 또는 프로세서가 로드되지 않아 추론을 수행할 수 없습니다.")
                self.current_image_cv = None # 다음 처리를 위해 초기화
                self.current_text_prompt = None # 다음 처리를 위해 초기화
                return

            self.inference_in_progress = True 
            self.get_logger().info(f"프롬프트로 추론 수행 중: '{self.current_text_prompt}'")
            
            image_to_process = self.current_image_cv
            prompt_to_process = self.current_text_prompt

            self.current_image_cv = None
            self.current_text_prompt = None
            
            rgb_frame_cv = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
            pil_image = PilImage.fromarray(rgb_frame_cv)
            img_width, img_height = pil_image.size

            try:
                inputs_data = self.processor(text=prompt_to_process, images=pil_image, return_tensors="pt").to(self.device)
                if self.device.type != "cuda" and 'pixel_values' in inputs_data and inputs_data['pixel_values'].dtype != self.model.dtype:
                    inputs_data['pixel_values'] = inputs_data['pixel_values'].to(self.model.dtype)
                elif 'pixel_values' in inputs_data and self.device.type == "mps": 
                     inputs_data['pixel_values'] = inputs_data['pixel_values'].to(torch.float32)

                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs_data, max_new_tokens=self.max_new_tokens, do_sample=False)
                    generated_text_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
                
                self.get_logger().info(f"모델 원본 결과: {generated_text_output.strip()}")
                parsed_detections = parse_segmentation_output(generated_text_output, img_width, img_height, prompt_to_process)
                
                twist_msg = Twist()

                if parsed_detections:
                    target_detection = parsed_detections[0] 
                    if target_detection.get('scaled_box_pixels'):
                        self.get_logger().info(f"레이블 '{target_detection.get('label', 'N/A')}'에 대한 이동 명령 계산 중...")
                        lx, ly, az = calculate_move_command_for_twist(target_detection['scaled_box_pixels'], img_width, img_height, self.get_logger())
                        self.get_logger().info(f"제안된 Twist 명령: linear.x={lx:.3f}, linear.y={ly:.3f}, angular.z={az:.3f}")
                        twist_msg.linear.x = lx
                        twist_msg.linear.y = ly
                        twist_msg.angular.z = az
                    else:
                        self.get_logger().warn("첫 번째 감지에서 유효한 바운딩 박스가 없어 이동 명령을 계산할 수 없습니다.")
                else: 
                    self.get_logger().info("위치 토큰을 포함한 객체가 파싱되지 않았습니다.")
                    if not "<loc" in generated_text_output: 
                        vqa_answer = generated_text_output.replace(prompt_to_process.strip(), '').strip()
                        if vqa_answer: 
                            self.get_logger().info(f"VQA 답변: {vqa_answer}")
                            self.vqa_answer_publisher.publish(RosString(data=vqa_answer))
                
                self.cmd_vel_publisher.publish(twist_msg)

            except Exception as e_gen:
                self.get_logger().error(f"추론 또는 결과 처리 중 오류: {e_gen}")
            finally:
                self.inference_in_progress = False 

def main(args=None):
    rclpy.init(args=args)
    paligemma_ros_node = PaliGemmaROSNode()
    
    # 모델 로드 성공 여부와 관계없이 spin을 시도하여 ROS 콜백 처리
    # (만약 __init__에서 모델 로드 실패 시 노드 자체를 종료하고 싶다면 다른 방식 필요)
    if paligemma_ros_node.model is None or paligemma_ros_node.processor is None:
        paligemma_ros_node.get_logger().error("모델 또는 프로세서가 성공적으로 로드되지 않아 스핀을 시작하지만, 추론은 불가능합니다.")
        # 여기서 프로그램을 종료하거나, 혹은 계속 실행하여 다른 ROS 메시지라도 처리하게 둘 수 있습니다.
        # 만약 여기서 종료하고 싶다면:
        # paligemma_ros_node.destroy_node()
        # rclpy.shutdown()
        # return # 또는 sys.exit(1)

    try:
        rclpy.spin(paligemma_ros_node)
    except KeyboardInterrupt:
        paligemma_ros_node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        paligemma_ros_node.get_logger().info('Destroying node...') # 종료 로그 추가
        paligemma_ros_node.destroy_node()
        if rclpy.ok(): # 이미 shutdown()이 호출되지 않았는지 확인
            paligemma_ros_node.get_logger().info('Shutting down rclpy...') # 종료 로그 추가
            rclpy.shutdown()

if __name__ == '__main__':
    main()