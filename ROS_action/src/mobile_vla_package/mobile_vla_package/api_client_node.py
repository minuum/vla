#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from mobile_vla_interfaces.srv import VLACommand
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import threading
import time
import os
import sys

# Add RoboVLMs to path for pipeline import
ROBOVLMS_PATH = "/home/soda/vla/RoboVLMs"
sys.path.append(ROBOVLMS_PATH)

try:
    from inference_pipeline import MobileVLAInferencePipeline
except ImportError as e:
    print(f"Error importing inference_pipeline: {e}")
    # We will crash later if we can't import, but let's try to proceed 
    # so we can see logs if it's just a path issue.

class MobileVLAAPIClient(Node):
    def __init__(self):
        super().__init__('mobile_vla_api_client')
        
        # --- Configuration ---
        self.ckpt_path = self.declare_parameter("ckpt_path", "/home/soda/vla/ROS_action/last.ckpt").value
        # Use Mock mode by default for now to avoid loading issues in ROS environment without explicit configuration
        self.mock_mode = self.declare_parameter("mock", True).value 
        
        self.log_info(f"Initializing On-Device Inference Node (Mock={self.mock_mode})...")

        # Initialize Inference Pipeline
        try:
             self.inference_pipeline = MobileVLAInferencePipeline(
                checkpoint_path=self.ckpt_path,
                mock_mode=self.mock_mode
            )
             self.log_info("Inference Pipeline initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize pipeline: {e}")
            # If pipeline fails, we can't really function, but maybe we stay alive to report error?
            # For now, let's exit.
            sys.exit(1)

        # Service
        self.srv = self.create_service(
            VLACommand, 
            '/execute_vla_command', 
            self.handle_vla_command,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup()
        )
        
        # Image Subscriber
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.cv_bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1, # Keep queue size small
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup()
        )
        self.log_info("Node initialized and waiting for commands...")

    def log_info(self, msg):
        self.get_logger().info(msg)

    def image_callback(self, msg):
        """Update latest image"""
        try:
            # self.log_info("Received image")
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert to PIL Image for pipeline
            from PIL import Image as PILImage
            import cv2
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            with self.image_lock:
                self.latest_image = pil_image
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def handle_vla_command(self, request, response):
        """Handle Service Request: Execute Action Inference"""
        start_time = time.time()
        instruction = request.instruction
        rel_state = np.array(request.rel_state) 
        
        self.log_info(f"Received command: '{instruction}'")

        # 1. Get Image
        current_image = None
        with self.image_lock:
            if self.latest_image is not None:
                current_image = self.latest_image.copy()
        
        if current_image is None:
            self.get_logger().error("No image available!")
            
            # If in mock mode, maybe we can fake an image?
            if self.mock_mode:
                self.get_logger().warn("Mock mode: Creating dummy image")
                from PIL import Image as PILImage
                current_image = PILImage.new('RGB', (224, 224), color='blue')
            else:
                response.success = False
                response.message = "No image received from camera"
                return response

        # 2. Run Inference
        try:
            self.log_info("Running inference...")
            inference_start = time.time()
            
            result = self.inference_pipeline.predict(
                image=current_image,
                instruction=instruction,
                rel_state=rel_state
            )
            
            inference_time = time.time() - inference_start
            self.log_info(f"Inference completed in {inference_time:.3f}s")
            
            # 3. Parse Output
            pred_linear_y = result.get("linear_y", np.zeros(10))
            pred_gripper = result.get("gripper", np.zeros(10))
            
            # Convert to list for ROS msg
            response.action_linear_y = pred_linear_y.tolist()
            response.action_gripper = pred_gripper.tolist()
            response.success = True
            response.message = f"Inference successful (Time: {inference_time:.3f}s)"
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            response.success = False
            response.message = f"Inference failed: {str(e)}"

        return response

def main(args=None):
    rclpy.init(args=args)
    node = MobileVLAAPIClient()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down API Client Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
