import logging
import json
# In a real networked scenario, we would import mcp.client
# For local integration, we might import the server class directly or use an interface.

class ContextAwareVLAClient:
    """
    MCP Client that orchestrates VLM reasoning by injecting
    real-time environmental context from the MCP Server.
    """
    
    def __init__(self, mcp_server_instance=None):
        self.logger = logging.getLogger(__name__)
        self.server = mcp_server_instance
        self.history = []

    def connect(self, server_instance):
        """Connects to a local MCP Server instance."""
        self.server = server_instance
        self.logger.info("Connected to MCP Server.")

    def decide_action(self, image, instruction, vlm_pipeline_func):
        """
        Orchestrates the decision process:
        1. Query MCP Server for context.
        2. Construct informed prompt.
        3. Call VLM pipeline with context.
        """
        if not self.server:
            self.logger.warning("MCP Server not connected. Using raw prompt.")
            return vlm_pipeline_func(image, instruction)

        # 1. Perception: Get context from MCP Server
        # In a real MCP, this would be: client.call_tool("detect_obstacles", ...)
        context = self.server.detect_obstacles(image)
        
        # 2. Reasoning: Build context-aware prompt
        enriched_prompt = self._build_context_prompt(instruction, context)
        self.logger.info(f"Generated Prompt: {enriched_prompt}")
        
        # 3. Decision: detailed VLM inference
        # We pass the enriched prompt to the VLM
        # vlm_pipeline_func should accept (image, prompt)
        action = vlm_pipeline_func(image, enriched_prompt)
        
        # Optional: Log interaction
        self.history.append({
            "instruction": instruction,
            "context": context,
            "prompt": enriched_prompt,
            "action": action
        })
        
        return action

    def _build_context_prompt(self, instruction, context):
        """
        Constructs the structured prompt with grounding and obstacle info.
        """
        prompt = f"Task: {instruction}\n"
        
        # Add Context Section
        if context.get("obstacles") or context.get("target"):
            prompt += "\n[REAL-TIME SCENE CONTEXT]\n"
            
            if context.get("target"):
                t = context["target"]
                prompt += f"- Target detected: {t['type']} at {t['x_forward']}m forward, {t['y_lateral']}m lateral ({t['position_relative']})\n"
            
            if context.get("obstacles"):
                prompt += f"- Obstacles detected: {len(context['obstacles'])}\n"
                for i, obs in enumerate(context["obstacles"]):
                    blocking_status = "BLOCKING PATH" if obs.get("blocking_path") else "clear"
                    prompt += (f"  {i+1}. {obs['type']} at {obs['x_forward']}m forward, "
                               f"{obs['y_lateral']}m lateral ({obs['position_relative']}) - [{blocking_status}]\n")
            
            # Diagnostic Summary from Server
            if context.get("scene_summary"):
                prompt += f"\nSummary: {context['scene_summary']}\n"
            
            prompt += "\nDecision Guide: If path is blocked, suggest a steering detour (Left/Right).\n"
        
        else:
            prompt += "\n[CONTEXT] No significant obstacles detected. Path is clear.\n"

        return prompt

# Example usage for testing
if __name__ == "__main__":
    # Mock VLM function
    def mock_vlm(img, prompt):
        return f"Action generated based on: {prompt[:50]}..."

    # Mock Server (if not importing the real one)
    class MockServer:
        def detect_obstacles(self, img):
            return {
                "obstacles": [{"type": "box", "position_relative": "center", "distance_estimate": 1.0, "blocking_path": True}],
                "target": {"type": "red_ball", "position_relative": "left", "distance_estimate": 3.0}
            }
    
    client = ContextAwareVLAClient(MockServer())
    action = client.decide_action(None, "Go to red ball", mock_vlm)
    print("Action Result:", action)
