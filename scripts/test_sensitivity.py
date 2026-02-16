import os
import torch
import numpy as np
from Mobile_VLA.inference_server import MobileVLAInference
import logging

# Disable logging noise
logging.basicConfig(level=logging.INFO)

def test_instruction_sensitivity():
    checkpoint = "/home/soda/vla/runs/unified_regression_win12/kosmos/epoch=epoch=09-val_loss=val_loss=0.0013.ckpt"
    config = "/home/soda/vla/Mobile_VLA/configs/mobile_vla_exp17_win8_k1.json"
    
    print("🚀 Loading model in INT8...")
    os.environ["VLA_QUANTIZE"] = "true"
    model = MobileVLAInference(checkpoint, config)
    
    # 더미 이미지 (Gray)
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    instructions = [
        "Navigate to the brown pot on the left",
        "Navigate to the basket on the right",
        "Stop moving",
        "Go backward"
    ]
    
    print("\n🧪 Testing sensitivity to instructions:")
    for inst in instructions:
        action, latency = model.predict(dummy_img, inst)
        print(f"Instruction: {inst:40s} -> Action: {action}")

if __name__ == "__main__":
    test_instruction_sensitivity()
