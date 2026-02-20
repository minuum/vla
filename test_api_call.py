import requests
import base64
import json

# Load image
with open("docs/object_test_images/test_blue_mug_floor_1768456939932.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# Call API
url = "http://localhost:8000/predict"
data = {
    "image": img_base64,
    "instruction": "Go to the blue mug",
    "snap_to_grid": True
}
headers = {"X-API-Key": "test_key"} # Matching VLA_API_KEY env var

try:
    for i in range(10):
        response = requests.post(url, json=data, headers=headers, timeout=60)
        res_json = response.json()
        print(f"Step {i+1}: Action {res_json.get('action')} | Latency {res_json.get('latency_ms')}ms")
        if response.status_code != 200:
            print(f"Error: {response.text}")
            break
except Exception as e:
    print(f"Error: {e}")
