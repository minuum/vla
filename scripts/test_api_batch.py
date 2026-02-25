import os
import glob
import base64
import json
import time
import requests

API_URL = "http://127.0.0.1:8000"
API_KEY = "test_key_1234"

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_inference(img_path, instruction):
    headers = {"X-API-Key": API_KEY}
    img_b64 = image_to_base64(img_path)
    
    payload = {
        "image": img_b64,
        "instruction": instruction,
        "strategy": "receding_horizon"
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{API_URL}/predict", headers=headers, json=payload, timeout=60)
        latency = (time.time() - start_time) * 1000
        if response.status_code == 200:
            result = response.json()
            return result, latency
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}, latency
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return {"error": str(e)}, latency

def main():
    img_dir = "/home/billy/25-1kp/vla/docs/object_test_images"
    test_cases = [
        {"img": "test_apple_floor_1768456959811.png", "instruction": "Navigate to the apple"},
        {"img": "test_blue_mug_floor_1768456939932.png", "instruction": "Navigate to the blue mug"},
        {"img": "test_chair_obstacle_1768456981699.png", "instruction": "Navigate around the chair"},
        {"img": "test_coke_can_floor_1768456916967.png", "instruction": "Navigate to the red coke can"},
        {"img": "test_cone_obstacle_1768457003835.png", "instruction": "Navigate around the cone"}
    ]
    
    print("Resetting API server history...")
    requests.post(f"{API_URL}/reset", headers={"X-API-Key": API_KEY})
    print("History reset successfully.\n")

    results = []
    
    for case in test_cases:
        img_path = os.path.join(img_dir, case["img"])
        print(f"Testing image: {case['img']}")
        print(f"Instruction: {case['instruction']}")
        
        result, latency = test_inference(img_path, case['instruction'])
        
        if "error" in result:
            print(f"❌ Error: {result['error']} (Latency: {latency:.1f}ms)")
            status_emoji = "❌"
            action_str = "Error"
        else:
            action = result['action']
            print(f"✅ Predicted Action: [{action[0]:.3f}, {action[1]:.3f}] (Latency: {latency:.1f}ms, API reported latency: {result.get('latency_ms', -1):.1f}ms)")
            status_emoji = "✅"
            action_str = f"[{action[0]:.3f}, {action[1]:.3f}]"
            
        print("-" * 60)
        
        results.append({
            "image": case["img"],
            "instruction": case["instruction"],
            "action": action_str,
            "latency": f"{latency:.1f}ms",
            "status": status_emoji
        })
        
    print("\n--- Summary ---")
    for r in results:
        print(f"{r['status']} {r['image']} | {r['instruction']} | Action: {r['action']} | Latency: {r['latency']}")

if __name__ == '__main__':
    main()
