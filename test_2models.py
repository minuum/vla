#!/usr/bin/env python3
"""
간단한 2개 모델 비교 테스트
Chunk5 vs Chunk10
"""

import requests
import h5py
import base64
import io
from PIL import Image
import numpy as np
import time

# Load image
print("Loading test image...")
with h5py.File('ROS_action/mobile_vla_dataset/episode_20251203_042905_1box_hori_left_core_medium.h5', 'r') as f:
    img_np = f['images'][()][0]

img_pil = Image.fromarray(img_np.astype(np.uint8))
buf = io.BytesIO()
img_pil.save(buf, 'PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"

API_KEY = "qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU"
headers = {'X-API-Key': API_KEY}

print("="*70)
print("🚀 2-Model Comparison Test (Chunk5 vs Chunk10)")
print("="*70)

results = []

for model_name in ["chunk5_epoch6", "chunk10_epoch8"]:
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    # Switch model
    response = requests.post(
        'http://localhost:8000/model/switch',
        json={'model_name': model_name},
        headers=headers,
        timeout=60
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to switch to {model_name}")
        continue
    
    print(f"✅ Switched to {model_name}")
    time.sleep(5)
    
    # Run predictions
    latencies = []
    actions = []
    
    for i in range(3):
        response = requests.post(
            'http://localhost:8000/predict',
            json={'image': img_b64, 'instruction': instruction},
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            latencies.append(result['latency_ms'])
            actions.append(result['action'])
            print(f"  Run {i+1}: {result['latency_ms']:.1f}ms, Action: {result['action']}")
        else:
            print(f"  Run {i+1}: ❌ Failed")
    
    if latencies:
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        action_std = np.std(actions, axis=0)
        
        results.append({
            'model': model_name,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'action_std': action_std
        })
        
        print(f"\n📊 Summary:")
        print(f"  Avg Latency: {avg_latency:.1f} ± {std_latency:.1f} ms")
        print(f"  Action Std: [{action_std[0]:.4f}, {action_std[1]:.4f}]")

# Final comparison
print(f"\n{'='*70}")
print("📊 FINAL COMPARISON")
print(f"{'='*70}")

if len(results) == 2:
    print(f"\n{'Model':<20} {'Avg Latency':<20} {'Action Consistency'}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['avg_latency']:>10.1f} ± {r['std_latency']:>5.1f} ms   "
              f"[{r['action_std'][0]:.4f}, {r['action_std'][1]:.4f}]")
    
    # Winner
    best = min(results, key=lambda x: x['avg_latency'])
    print(f"\n🏆 Fastest: {best['model']} ({best['avg_latency']:.1f}ms)")

print(f"\n{'='*70}")
print("✅ Test completed!")
print(f"{'='*70}")
