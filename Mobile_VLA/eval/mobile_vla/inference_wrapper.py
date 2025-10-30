"""
Mobile VLA Inference Wrapper with ROS2 Integration
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json

from PIL import Image
from transformers import AutoProcessor

# RoboVLMs imports
from robovlms.model.backbone.robokosmos import RoboKosMos
from robovlms.model.policy_head.base_policy import LSTMDecoder

logger = logging.getLogger(__name__)


class MobileVLAInference:
    """
    Mobile VLA Inference Wrapper
    
    Loads trained Mobile VLA model and provides inference interface
    compatible with ROS2 integration.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda',
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to model config JSON
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup model
        self.model = self._load_model()
        self.processor = self._load_processor()
        
        # Action bounds for denormalization
        self.action_bounds = self.config.get('action_bounds', {
            'linear_x': (-2.0, 2.0),
            'linear_y': (-1.15, 1.15),
            'angular_z': (-3.14, 3.14),
            'action_type': (0, 3)
        })
        
        # History buffer
        self.window_size = self.config['window_size']
        self.image_history = []
        self.action_history = []
        
        logger.info(f"Mobile VLA Inference initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Window size: {self.window_size}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        # Create model
        model = RoboKosMos(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {self.checkpoint_path}")
        
        return model
    
    def _load_processor(self):
        """Load image processor"""
        processor = AutoProcessor.from_pretrained(
            self.config['vlm']['pretrained_model_name_or_path']
        )
        return processor
    
    def reset(self):
        """Reset history buffers"""
        self.image_history = []
        self.action_history = []
        logger.info("History buffers reset")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: (H, W, 3) numpy array in RGB
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Process with VLM processor
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Denormalize action from [-1, 1] to original range
        
        Args:
            action: (4,) tensor with normalized actions
            
        Returns:
            Denormalized action array
        """
        action = action.cpu().numpy()
        denormalized = np.zeros(4)
        
        bounds = [
            self.action_bounds['linear_x'],
            self.action_bounds['linear_y'],
            self.action_bounds['angular_z'],
            self.action_bounds['action_type'],
        ]
        
        for i, (min_val, max_val) in enumerate(bounds):
            denormalized[i] = (action[i] + 1) / 2 * (max_val - min_val) + min_val
        
        return denormalized
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict action from current observation
        
        Args:
            image: Current RGB image (H, W, 3)
            instruction: Language instruction
            return_normalized: Return normalized actions (for debugging)
            
        Returns:
            dict with:
                - action: (4,) array with [linear_x, linear_y, angular_z, action_type]
                - action_chunk: (fwd_pred_next_n, 4) predicted action sequence
        """
        # Preprocess image
        pixel_values = self.preprocess_image(image)
        
        # Add to history
        self.image_history.append(pixel_values)
        
        # Maintain window size
        if len(self.image_history) > self.window_size:
            self.image_history = self.image_history[-self.window_size:]
        
        # Pad history if needed
        while len(self.image_history) < self.window_size:
            self.image_history.insert(0, self.image_history[0])
        
        # Stack images
        images = torch.cat(self.image_history, dim=0)  # (window_size, C, H, W)
        images = images.unsqueeze(0)  # (1, window_size, C, H, W)
        
        # Prepare input
        inputs = {
            'images': images,
            'text': [instruction],
        }
        
        # Forward pass
        outputs = self.model.inference(inputs)
        
        # Extract actions
        predicted_actions = outputs['actions']  # (1, fwd_pred_next_n, 4)
        
        # Get first action (for MPC-style execution)
        action = predicted_actions[0, 0]  # (4,)
        action_chunk = predicted_actions[0]  # (fwd_pred_next_n, 4)
        
        # Denormalize if needed
        if not return_normalized:
            action = self.denormalize_action(action)
            action_chunk = np.array([self.denormalize_action(a) for a in action_chunk])
        else:
            action = action.cpu().numpy()
            action_chunk = action_chunk.cpu().numpy()
        
        return {
            'action': action,
            'action_chunk': action_chunk,
        }
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        instructions: List[str],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Batch prediction (for evaluation)
        
        Args:
            images: List of RGB images
            instructions: List of language instructions
            
        Returns:
            List of prediction dicts
        """
        results = []
        
        for image, instruction in zip(images, instructions):
            result = self.predict(image, instruction)
            results.append(result)
        
        return results


class MobileVLAROS2Node:
    """
    ROS2 Node wrapper for Mobile VLA Inference
    
    Subscribes to camera and text command topics,
    publishes cmd_vel actions.
    """
    
    def __init__(
        self,
        inference_wrapper: MobileVLAInference,
        camera_topic: str = '/camera/image_raw',
        text_topic: str = '/vla_text_command',
        cmd_vel_topic: str = '/cmd_vel',
    ):
        """
        Args:
            inference_wrapper: MobileVLAInference instance
            camera_topic: Camera image topic
            text_topic: Text command topic
            cmd_vel_topic: Velocity command topic
        """
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
            from geometry_msgs.msg import Twist
            from cv_bridge import CvBridge
        except ImportError:
            raise ImportError("ROS2 not available. Install ros-humble-desktop")
        
        self.inference = inference_wrapper
        
        # Initialize ROS2
        rclpy.init()
        
        class VLANode(Node):
            def __init__(node_self):
                super().__init__('mobile_vla_node')
                
                # Subscribers
                node_self.image_sub = node_self.create_subscription(
                    Image,
                    camera_topic,
                    node_self.image_callback,
                    10
                )
                
                node_self.text_sub = node_self.create_subscription(
                    String,
                    text_topic,
                    node_self.text_callback,
                    10
                )
                
                # Publisher
                node_self.cmd_vel_pub = node_self.create_publisher(
                    Twist,
                    cmd_vel_topic,
                    10
                )
                
                # State
                node_self.bridge = CvBridge()
                node_self.current_instruction = "Navigate to the target"
                node_self.latest_image = None
                
                # Timer for inference (10 Hz)
                node_self.timer = node_self.create_timer(0.1, node_self.inference_callback)
                
                logger.info("Mobile VLA ROS2 Node initialized")
            
            def image_callback(node_self, msg):
                """Receive camera image"""
                node_self.latest_image = node_self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            
            def text_callback(node_self, msg):
                """Receive text command"""
                node_self.current_instruction = msg.data
                logger.info(f"Received instruction: {msg.data}")
            
            def inference_callback(node_self):
                """Run inference and publish action"""
                if node_self.latest_image is None:
                    return
                
                # Predict action
                result = self.inference.predict(
                    node_self.latest_image,
                    node_self.current_instruction
                )
                
                action = result['action']
                
                # Create Twist message
                twist = Twist()
                twist.linear.x = float(action[0])
                twist.linear.y = float(action[1])
                twist.angular.z = float(action[2])
                
                # Publish
                node_self.cmd_vel_pub.publish(twist)
                
                logger.debug(f"Published action: linear_x={action[0]:.3f}, "
                           f"linear_y={action[1]:.3f}, angular_z={action[2]:.3f}")
        
        self.node = VLANode()
    
    def spin(self):
        """Run ROS2 node"""
        import rclpy
        rclpy.spin(self.node)
    
    def shutdown(self):
        """Shutdown node"""
        import rclpy
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ros2', action='store_true', help='Run as ROS2 node')
    args = parser.parse_args()
    
    # Create inference wrapper
    inference = MobileVLAInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    
    if args.ros2:
        # Run as ROS2 node
        node = MobileVLAROS2Node(inference)
        try:
            node.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.shutdown()
    else:
        # Test mode
        logger.info("Running in test mode (no ROS2)")
        
        # Create dummy input
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        instruction = "Navigate around obstacles"
        
        result = inference.predict(dummy_image, instruction)
        
        logger.info(f"Predicted action: {result['action']}")
        logger.info(f"Action chunk shape: {result['action_chunk'].shape}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

