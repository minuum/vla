import numpy as np
import time
import json
import logging

# Optional dependencies
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not installed. YOLO detection will be mocked.")

class ObstacleDetectionMCPServer:
    """
    MCP Server for Context-Aware Obstacle Detection.
    
    Tools:
    - detect_obstacles(image): Returns structured JSON of detected objects.
    - assess_path_risk(image, action): Evaluates risk of proposed action.
    
    Resources:
    - sensor_data: Returns stream of sensor metadata.
    """
    
    def __init__(self, model_path='yolov8n.pt', focal_length=500):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.focal_length = focal_length # Heuristic for distance estimation
        
        if ULTRALYTICS_AVAILABLE:
            try:
                # Force CPU if not on Jetson for now, or allow auto
                self.model = YOLO(model_path)
                self.logger.info(f"Loaded YOLO model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
        
        # Priority mapping for navigation
        self.target_labels = ['ball', 'red_ball', 'goal', 'target']
        self.obstacle_labels = ['bucket', 'box', 'chair', 'cone', 'person', 'bottle']

    def detect_obstacles(self, image: np.ndarray) -> dict:
        """
        Detects obstacles and targets, mapping them to local spatial coordinates.
        Includes a 'Vision Blindness' check to detect sensor/VLM issues.
        """
        # 1. Blindness Check (Diagnostic Insight)
        blindness_status = self._check_blindness(image)
        
        if self.model and image is not None and not blindness_status["blind"]:
            results = self.model(image, verbose=False)
            context = self._process_results(results[0])
        else:
            context = self._mock_response()
            
        context["diagnostics"] = blindness_status
        return context

    def _check_blindness(self, image):
        """
        Checks if the image is too dark or washed out, 
        potentially causing 'Vision Blindness' in INT8 models.
        """
        if image is None: return {"blind": True, "reason": "No Image"}
        
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        is_blind = mean_val < 15 or mean_val > 240 or std_val < 5
        
        return {
            "blind": is_blind,
            "mean": round(float(mean_val), 2),
            "std": round(float(std_val), 2),
            "reason": "OK" if not is_blind else "Exposure/Contrast Issue"
        }

    def _process_results(self, result) -> dict:
        """
        Maps YOLO detections to a structured context including X,Y coordinates.
        X+ = Forward, Y+ = Left (Standard ROS convention)
        """
        context = {
            "obstacles": [],
            "target": None,
            "scene_summary": ""
        }
        
        img_h, img_w = result.orig_shape
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id].lower()
            conf = float(box.conf[0])
            
            if conf < 0.4: continue # Detection threshold
            
            bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            # 1. Spatial Estimation
            dist_x, dist_y = self._estimate_spatial_coords(bbox, img_w, img_h)
            
            obj_data = {
                "type": cls_name,
                "bbox": [round(c, 1) for c in bbox],
                "confidence": round(conf, 2),
                "x_forward": round(dist_x, 2),
                "y_lateral": round(dist_y, 2),
                "position_relative": self._get_relative_tag(dist_y),
                "blocking_path": self._check_blocking(dist_x, dist_y)
            }
            
            # 2. Categorization
            if any(label in cls_name for label in self.target_labels):
                context["target"] = obj_data
            elif any(label in cls_name for label in self.obstacle_labels) or conf > 0.6:
                context["obstacles"].append(obj_data)
        
        # 3. Dynamic Summary
        self._generate_summary(context)
        
        return context

    def _estimate_spatial_coords(self, bbox, img_w, img_h):
        """
        Rough heuristic to map BBox to local X (forward) and Y (lateral) in meters.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        # Assume standard target height (e.g., 0.2m) to estimate distance via focal length
        # dist = (real_height * focal_length) / pixel_height
        pixel_h = max(y2 - y1, 1)
        real_h = 0.25 # Typical obstacle/ball size assumption (25cm)
        dist_x = (real_h * self.focal_length) / pixel_h
        
        # Lateral offset: (center_x - img_center) * dist_x / focal_length
        dist_y = - (center_x - img_w / 2) * dist_x / self.focal_length
        
        return dist_x, dist_y

    def _get_relative_tag(self, dist_y):
        if dist_y > 0.3: return "left"
        if dist_y < -0.3: return "right"
        return "center"

    def _check_blocking(self, dist_x, dist_y):
        # Blocking if within 0.4m laterally and less than 2.0m ahead
        return abs(dist_y) < 0.4 and dist_x < 2.0

    def _generate_summary(self, context):
        msg = []
        if context["target"]:
            t = context["target"]
            msg.append(f"Target ({t['type']}) at {t['x_forward']}m {t['position_relative']}.")
        
        blockers = [o for o in context["obstacles"] if o["blocking_path"]]
        if blockers:
            msg.append(f"Path BLOCKED by {len(blockers)} obstacles.")
        else:
            msg.append(f"Found {len(context['obstacles'])} obstacles, path clear.")
            
        context["scene_summary"] = " ".join(msg)

    def _mock_response(self) -> dict:
        return {
            "obstacles": [
                {"type": "box", "x_forward": 1.2, "y_lateral": 0.1, "position_relative": "center", "blocking_path": True}
            ],
            "target": {"type": "red_ball", "x_forward": 3.0, "y_lateral": 0.8, "position_relative": "left"},
            "scene_summary": "Mock: Path blocked by box at 1.2m center. Target ball on left."
        }

    def get_sensor_data(self) -> dict:
        return {
            "timestamp": time.time(),
            "status": "ready",
            "perception_engine": "yolo" if self.model else "mock",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

    def get_sensor_data(self) -> dict:
        """Resource: Returns current sensor status."""
        return {
            "timestamp": time.time(),
            "status": "active",
            "model": "yolov8n" if self.model else "mock"
        }

if __name__ == "__main__":
    # Test the server
    server = ObstacleDetectionMCPServer()
    # Mock image (random noise)
    mock_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(json.dumps(server.detect_obstacles(mock_img), indent=2))
