#!/usr/bin/env python3
import requests, h5py, base64, io
from PIL import Image
import numpy as np
import time
import sys
from pathlib import Path

# Add path for instruction mapping
sys.path.insert(0, str(Path(__file__).parent))
from Mobile_VLA.instruction_mapping import get_instruction_for_scenario

# Load image
with h5py.File('ROS_action/mobile_vla_dataset/episode_20251203_042905_1box_hori_left_core_medium.h5', 'r') as f:
    img_np = f['images'][()][0]

# Convert to base64
img_pil = Image.fromarray(img_np.astype(np.uint8))
buf = io.BytesIO()
img_pil.save(buf, 'PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

print('Image loaded')

# Switch model
response = requests.post(
    'http://localhost:8000/model/switch',
    json={'model_name': 'chunk5_epoch6'},
    headers={'X-API-Key': 'qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU'},
    timeout=60
)
print(f'Switch: {response.status_code}')

time.sleep(5)

# Predict - 학습 시 사용된 한국어 instruction 사용
instruction = get_instruction_for_scenario('left')
print(f'Using instruction: {instruction}')

response = requests.post(
    'http://localhost:8000/predict',  
    json={'image': img_b64, 'instruction': instruction},
    headers={'X-API-Key': 'qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU'},
    timeout=60
)

print(f'Predict: {response.status_code}')
if response.status_code == 200:
    r = response.json()
    print(f'✅ SUCCESS! Action: {r["action"]}, Latency: {r["latency_ms"]}ms')
else:
    print(f'❌ Error: {response.text[:500]}')
