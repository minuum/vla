import json
import os
import time
from datetime import datetime
import numpy as np

class InferenceLogger:
    def __init__(self, log_dir="/home/soda/vla/docs/inference_reports"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        self.data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": "unknown",
            "instruction": "unknown",
            "history": []
        }
            
    def start_session(self, model_name, instruction):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        self.data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "instruction": instruction,
            "history": []
        }
        print(f"📝 Logging session started: {self.log_file}")

    def update_instruction(self, instruction):
        if hasattr(self, "data"):
            self.data["instruction"] = instruction
            print(f"📝 Instruction updated: {instruction}")
        
    def log_step(self, step_idx, action, latency, chunk=None):
        if not hasattr(self, "data") or self.data is None:
            self.data = {"history": []}

        step_data = {
            "step": step_idx,
            "timestamp": datetime.now().isoformat(),
            "action": action.tolist() if isinstance(action, np.ndarray) else action,
            "latency_ms": latency,
        }
        if chunk is not None:
             step_data["chunk_preview"] = chunk.tolist() if isinstance(chunk, np.ndarray) else chunk
             
        self.data["history"].append(step_data)
        
    def end_session(self, status="completed"):
        if not hasattr(self, "data") or self.data is None:
             print("⚠️ No session data to save.")
             return None
             
        self.data["status"] = status
        self.data["end_time"] = datetime.now().isoformat()
        
        # Calculate summary statistics
        if self.data["history"]:
            # Filter for numeric latencies only
            latencies = [
                h["latency_ms"] for h in self.data["history"] 
                if isinstance(h["latency_ms"], (int, float))
            ]
            
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            
            self.data["summary"] = {
                "avg_latency": avg_lat,
                "total_steps": len(self.data["history"]),
                "last_action": self.data["history"][-1]["action"]
            }
        
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=4)
        print(f"✅ Session log saved: {self.log_file}")
        return self.log_file

# Singleton instance
_logger = InferenceLogger()

def get_logger():
    return _logger
