#!/usr/bin/env python3
"""
API Server Inference Test
BitsAndBytes INT8 모델 로딩 및 추론 테스트
"""

import requests
import base64
import json
import time
from PIL import Image
import numpy as np
import io

# API 설정
API_URL = "http://localhost:8000"
API_KEY = "qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU"

print("="*70)
print("API Server Inference Test")
print("="*70)

# 1. Health Check
print("\n1. Health Check...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    health = response.json()
    print(f"   Response: {json.dumps(health, indent=2)}")
    print(f"   ✅ Health check passed")
except Exception as e:
    print(f"   ❌ Health check failed: {e}")
    exit(1)

# 2. 더미 이미지 생성
print("\n2. Creating dummy image...")
try:
    # 480x640 RGB 이미지 생성
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    
    # Base64 인코딩
    buffered = io.BytesIO()
    dummy_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    print(f"   Image size: 480x640")
    print(f"   Base64 length: {len(img_b64)} chars")
    print(f"   ✅ Dummy image created")
except Exception as e:
    print(f"   ❌ Image creation failed: {e}")
    exit(1)

# 3. Inference 요청
print("\n3. Sending inference request...")
print("   (First request will load model - may take ~30 seconds)")

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "image": img_b64,
    "instruction": "Move forward to the target"
}

try:
    start = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        headers=headers,
        json=payload,
        timeout=120  # 2분 타임아웃 (첫 로딩 시간 고려)
    )
    latency = (time.time() - start) * 1000
    
    print(f"   Status Code: {response.status_code}")
    print(f"   Total Latency: {latency:.1f} ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n   ✅ Inference successful!")
        print(f"   Response JSON:")
        print(json.dumps(result, indent=4))
        
        # 결과 검증
        print(f"\n4. Validating response...")
        if "action" in result:
            action = result["action"]
            print(f"   ✅ Action: {action}")
            if isinstance(action, list) and len(action) == 2:
                print(f"   ✅ Action format correct: [linear_x, linear_y]")
                print(f"      linear_x: {action[0]:.4f}")
                print(f"      linear_y: {action[1]:.4f}")
            else:
                print(f"   ❌ Invalid action format")
        else:
            print(f"   ❌ No 'action' in response")
        
        if "latency_ms" in result:
            print(f"   ✅ Inference latency: {result['latency_ms']:.1f} ms")
        
        if "model_name" in result:
            print(f"   ✅ Model: {result['model_name']}")
            
        if "quantization" in result:
            print(f"   ✅ Quantization: {result['quantization']}")
        
    else:
        print(f"   ❌ Request failed")
        print(f"   Response: {response.text}")
        
except requests.exceptions.Timeout:
    print(f"   ❌ Request timeout (>120s)")
except Exception as e:
    print(f"   ❌ Request failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 두 번째 요청 (모델 이미 로딩됨)
print("\n5. Second inference request (model already loaded)...")
try:
    start = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        headers=headers,
        json=payload,
        timeout=30
    )
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Second request successful")
        print(f"   Total Latency: {latency:.1f} ms")
        print(f"   Inference Latency: {result.get('latency_ms', 'N/A')} ms")
        print(f"   Action: {result.get('action', 'N/A')}")
    else:
        print(f"   ❌ Second request failed: {response.status_code}")
        
except Exception as e:
    print(f"   ❌ Second request error: {e}")

# 6. Health Check (모델 로딩 후)
print("\n6. Health check after model loading...")
try:
    response = requests.get(f"{API_URL}/health")
    health = response.json()
    print(f"   Model loaded: {health.get('model_loaded', 'N/A')}")
    if 'gpu_memory' in health:
        print(f"   GPU Memory: {health['gpu_memory'].get('allocated_gb', 0):.2f} GB")
    print(f"   ✅ Post-loading health check passed")
except Exception as e:
    print(f"   ❌ Health check failed: {e}")

print("\n" + "="*70)
print("✅ API Server Test Complete!")
print("="*70)
