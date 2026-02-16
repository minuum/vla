import logging
import torch
import numpy as np
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
    - Instruction Logging: Transparent prompt tracking.
    """
    
    def __init__(self, mcp_server, vla_checkpoint=None, history_limit=8, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Perception Layer (MCP)
        self.mcp_client = ContextAwareVLAClient(mcp_server)
        
        # 2. Reasoning Layer (VLA)
        self.vla_loader = None
        self.vla_model = None
        if VLA_AVAILABLE:
            self.vla_loader = MobileVLAModelLoader()
            self.vla_model = self.vla_loader.load_model(vla_checkpoint)
        
        # 3. Planning Layer (MPC)
        self.mpc = MPCController(**kwargs)
        
        # System State & Diagnostics
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.inference_count = 0
        self.history_limit = history_limit
        self.last_semantic_goal = np.array([1.0, 0.0, 0.0])
        
        self.logger.info(f"Hierarchical System initialized. Safety Window: {history_limit}")

    def reset(self):
        """Resets the entire system state (Perception + Control)."""
        self.inference_count = 0
        if self.vla_model and hasattr(self.vla_model, 'reset'):
            self.vla_model.reset()
        self.logger.info("🔄 System history and state RESET.")

    def update_state(self, x, y, theta):
        """Updates internal robot state repository."""
        self.current_state = np.array([x, y, theta])

    def run_step(self, image, instruction):
        """
        One complete hierarchical cycle with First-Frame Safety and Diagnostics.
        """
        start_time = time.time()
        
        # --- Level 1: Semantic Strategy (MCP + VLM) ---
        
        # First-Frame Safety Check (Insight from @Integrate Inference Diagnostics)
        if self.inference_count < self.history_limit:
            self.logger.warning(f"🛡️ Safety Window Active: {self.inference_count+1}/{self.history_limit}. Enforcing [0,0].")
            # Fill history with mock forward pass
            _ = self.mcp_client.server.detect_obstacles(image)
            self.inference_count += 1
            return np.array([0.0, 0.0]), np.zeros((10, 2))

        # 🔄 Log explicit instruction
        self.logger.info(f"📝 [INSTRUCTION] {instruction}")

        # 1.1 MCP Client generates enriched prompt
        def vla_pipeline(img, prompt):
            # Simulated or real VLM Reasoning
            return self._vla_strategic_inference(img, prompt)

        semantic_decision = self.mcp_client.decide_action(image, instruction, vla_pipeline)
        
        # --- Level 2: Trajectory Optimization (MPC) ---
        
        # 2.1 Extract obstacles from MCP
        context = self.mcp_client.server.detect_obstacles(image)
        obstacles = self._context_to_mpc_obstacles(context)
        
        # 2.2 Translate decision to spatial goal
        mpc_goal = self._interpret_goal_from_decision(semantic_decision)
        
        # 2.3 Optimize safe trajectory
        action, trajectory = self.mpc.solve(self.current_state, mpc_goal, obstacles)
        
        self.inference_count += 1
        latency = (time.time() - start_time) * 1000
        self.logger.info(f"✅ Step {self.inference_count}: [VLM] {semantic_decision} | [MPC Action] {action} | Latency: {latency:.1f}ms")
        
        return action, trajectory

    def _vla_strategic_inference(self, img, prompt):
        # Local logic reflecting VLA diagnostics
        if "BLOCKING" in prompt:
            return "GO RIGHT" if "right" in prompt.lower() else "GO LEFT"
        return "GO STRAIGHT"

    def _interpret_goal_from_decision(self, decision):
        goal = self.current_state.copy()
        look_ahead = 1.2 # 1.2m goal (slightly reduced for precision)
        
        if "LEFT" in decision:
            goal[1] += 0.8
        elif "RIGHT" in decision:
            goal[1] -= 0.8
        else:
            goal[0] += look_ahead
            
        return goal

    def _context_to_mpc_obstacles(self, context):
        mpc_obs = []
        for obs in context.get("obstacles", []):
            mpc_obs.append({
                "x": self.current_state[0] + obs.get("x_forward", 0),
                "y": self.current_state[1] + obs.get("y_lateral", 0),
                "r": 0.4 # Slightly larger safety radius
            })
        return mpc_obs

if __name__ == "__main__":
    # Integration Mock Test
    from core.mcp.server import ObstacleDetectionMCPServer
    
    server = ObstacleDetectionMCPServer()
    def mock_vlm(img, prompt):
        return "GO LEFT" # Example VLM decision
        
    nav_system = HierarchicalNavigationSystem(server, mock_vlm)
    action, traj = nav_system.run_step(np.zeros((480, 640, 3)), "Avoid obstacles and go to ball")
    
    print("Action:", action)
    print("Path Shape:", traj.shape)
