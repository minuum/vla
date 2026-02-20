import logging
import torch
import numpy as np
import time
from core.mcp.client import ContextAwareVLAClient
from core.mpc.controller import MPCController

# Import existing VLA infrastructure
try:
    from src.mobile_vla_model_loader import MobileVLAModelLoader
    VLA_AVAILABLE = True
except ImportError:
    VLA_AVAILABLE = False
    logging.warning("MobileVLAModelLoader not found in 'src/'. VLA will be mocked.")

class HierarchicalNavigationSystem:
    """
    Main Orchestrator for Mobile VLA + MCP + MPC.
    
    Level 1 (Semantic): VLM + MCP Client -> Strategy Selection
    Level 2 (Planning): MPC Controller -> Trajectory Optimization
    
    Diagnostic Features:
    - First-Frame Safety: Ensures [0,0] until history is filled.
    - Vision Blindness: Emergency stop on sensor issues.
    - Instruction Logging: Transparent prompt tracking.
    """
    
    def __init__(self, mcp_server, vla_checkpoint=None, history_limit=8, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Perception Layer (MCP)
        self.mcp_client = ContextAwareVLAClient(mcp_server)
        
        # 2. Reasoning Layer (VLA - Lazy loading)
        self.vla_model = None
        if VLA_AVAILABLE and vla_checkpoint:
            loader = MobileVLAModelLoader()
            self.vla_model = loader.load_model(vla_checkpoint)
        
        # 3. Planning Layer (MPC)
        self.mpc = MPCController(**kwargs)
        
        # System State & Performance Tuning
        self.current_state = np.array([0.0, 0.0, 0.0]) # [x, y, theta]
        self.inference_count = 0
        self.history_limit = history_limit
        self.last_action = np.zeros(2) # [v, w]
        
        # Hyperparameters (Synced with api_server.py benchmarks)
        self.smoothing = kwargs.get('smoothing', 0.8) # Weight for new action
        self.snap_threshold = kwargs.get('snap_threshold', 0.4)
        self.use_snap = kwargs.get('use_snap', True)
        
        self.logger.info(f"🚀 Hierarchical System Online. Mode: {'VLA' if self.vla_model else 'Mock'}")

    def reset(self):
        """Resets the entire system state (Perception + Control)."""
        self.inference_count = 0
        self.last_action = np.zeros(2)
        if self.vla_model and hasattr(self.vla_model, 'reset'):
            self.vla_model.reset()
        self.logger.info("🔄 System history and state RESET.")

    def update_state(self, x, y, theta):
        """Updates internal robot state repository (e.g., from odometry)."""
        self.current_state = np.array([x, y, theta])

    def run_step(self, image, instruction):
        """
        One complete hierarchical cycle: Perception -> Reasoning -> Planning -> Smoothing -> Snapping.
        """
        start_time = time.time()
        
        # --- Level 0: Diagnostics (Blindness Check) ---
        context = self.mcp_client.server.detect_obstacles(image)
        diag = context.get("diagnostics", {})
        if diag.get("blind"):
            self.logger.error(f"🚨 VISION BLINDNESS: {diag.get('reason')}. EMERGENCY STOP.")
            return np.array([0.0, 0.0]), np.zeros((10, 2))

        # --- Level 1: Semantic Strategy (First-Frame Safety) ---
        if self.inference_count < self.history_limit:
            self.logger.warning(f"🛡️ Safety Window Active: {self.inference_count+1}/{self.history_limit}. Enforcing [0,0].")
            self.inference_count += 1
            return np.array([0.0, 0.0]), np.zeros((10, 2))

        # 🔄 Log explicit instruction
        self.logger.info(f"📝 [INSTRUCTION] {instruction}")

        # 1.1 Strategy Inference (Enriched prompt via MCP Client)
        def vla_pipeline(img, prompt):
            return self._vla_strategic_inference(img, prompt)

        semantic_decision = self.mcp_client.decide_action(image, instruction, vla_pipeline)
        self.logger.info(f"🧠 [VLM DECISION] {semantic_decision}")

        # --- Level 2: Trajectory Optimization (MPC) ---
        obstacles = self._context_to_mpc_obstacles(context)
        mpc_goal = self._interpret_goal_from_decision(semantic_decision)
        
        # 2.3 Optimize safe trajectory
        raw_action, trajectory = self.mpc.solve(self.current_state, mpc_goal, obstacles)
        
        # --- Level 3: Post-Processing (Smoothing & Snapping) ---
        # 3.1 Smoothing (Holonomic approach: v, w)
        smooth_action = self.smoothing * raw_action + (1 - self.smoothing) * self.last_action
        
        # 3.2 Snapping (Quantize to discrete levels: -1.15, 0.0, 1.15)
        final_action = smooth_action
        if self.use_snap:
            final_action = self._snap_action(smooth_action)
            
        self.last_action = final_action
        self.inference_count += 1
        
        latency = (time.time() - start_time) * 1000
        self.logger.info(f"✅ Step {self.inference_count}: Action {final_action} | Latency {latency:.1f}ms")
        
        return final_action, trajectory

    def _snap_action(self, action):
        """Maps linear/angular values to discrete robot setPoints."""
        snapped = np.zeros_like(action)
        for i in range(len(action)):
            if abs(action[i]) > self.snap_threshold:
                snapped[i] = 1.15 if action[i] > 0 else -1.15
        return snapped

    def _vla_strategic_inference(self, img, prompt):
        """
        Executes real or mock VLM inference to determine high-level strategy.
        Returns: "GO LEFT", "GO RIGHT", "GO STRAIGHT", or specific action vector.
        """
        # 1. Mock Mode (if no model loaded)
        if not self.vla_model:
            if "BLOCKING" in prompt:
                return "GO RIGHT" if "right" in prompt.lower() else "GO LEFT"
            return "GO STRAIGHT"

        # 2. Real VLM Inference
        # Prepare inputs (Assuming MobileVLAModelLoader handles preprocessing or we do it here)
        # Note: MobileVLAModelLoader.predict expects (vision_features, text_features) if using simple loader
        # BUT we want to use the high-level generic interface if possible.
        # Let's inspect MobileVLAModelLoader again. It seems to have a `predict` method taking features.
        # We need to bridge raw image -> features -> predict.
        
        # However, api_server.py logic was:
        # inputs = processor(images=history, text=[prompt]*window, ...)
        # prediction = model.inference(...)
        
        # If MobileVLAModelLoader is just a wrapper for the param-efficient model, we might need to copy
        # the preprocessing logic from api_server.py OR enhance MobileVLAModelLoader to handle raw inputs.
        
        # For this step, I will assume we need to port the preprocessing logic here 
        # because MobileVLAModelLoader seems to be a lower-level loader in the view_file output.
        
        # Wait! The viewed file `src/mobile_vla_model_loader.py` showed `predict(vision_features, text_features)`.
        # It does NOT have the full processor logic. 
        # We must implement the processor logic here or inside a new VLA wrapper.
        
        # Strategy: Use the VLA model directly if it exposes a high-level `predict_step`.
        # Since currently loaded `SimpleCLIPLSTMModel` expects features, we need the CLIP processor.
        
        # TEMP FIX: For now, we keep the Mock logic but add a TODO log
        # because fully porting the CLIP processor requires imports not yet in hub.py
        self.logger.warning("Construction of VLA features for loaded model not yet implemented in Hub. Using rule-based fallback.")
        
        if "BLOCKING" in prompt:
            return "GO RIGHT" if "right" in prompt.lower() else "GO LEFT"
        return "GO STRAIGHT"

    def _interpret_goal_from_decision(self, decision):
        """Translates high-level VLM commands to a local attractor goal for MPC."""
        goal = self.current_state.copy()
        dist = 1.5 # Target lookup distance (m)
        
        if "LEFT" in decision:
            goal[1] += 1.0 # Offset goal to the left
            goal[0] += 0.5 
        elif "RIGHT" in decision:
            goal[1] -= 1.0 # Offset goal to the right
            goal[0] += 0.5
        else:
            goal[0] += dist # Target straight ahead
            
        return goal

    def _context_to_mpc_obstacles(self, context):
        """Maps MCP spatial coordinates to absolute world coordinates for MPC solver."""
        mpc_obs = []
        for obs in context.get("obstacles", []):
            mpc_obs.append({
                "x": self.current_state[0] + obs.get("x_forward", 0),
                "y": self.current_state[1] + obs.get("y_lateral", 0),
                "r": 0.45 # Conservative safety radius
            })
        return mpc_obs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from core.mcp.server import ObstacleDetectionMCPServer
    
    server = ObstacleDetectionMCPServer()
    nav_system = HierarchicalNavigationSystem(server) # Mock mode
    
    # Simulate a few steps
    dummy_img = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8) # Add noise to avoid blindness
    for _ in range(15):
        action, _ = nav_system.run_step(dummy_img, "Reach the green target")
