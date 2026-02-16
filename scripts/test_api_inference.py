#!/usr/bin/env python3
"""
API 서버를 통한 추론 테스트
이미 실행 중인 API 서버 활용
"""

import requests
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import time
import os

def image_to_base64(image):
    """PIL Image를 Base64로 인코딩"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def test_api_inference():
    """API 서버 추론 테스트"""
    
    print("🧪 API Server Inference Test")
    print("="*60 + "\n")
    
    # API 서버 설정
    api_server = os.getenv("VLA_API_SERVER", "http://localhost:8000")
    api_key = os.getenv("VLA_API_KEY")
    
    if not api_key:
        print("⚠️  VLA_API_KEY not set. Trying without authentication...")
    
    print(f"📡 API Server: {api_server}")
    print(f"🔑 API Key: {api_key[:10] if api_key else 'None'}...\n")
    
    # 1. Health Check
    print("1️⃣  Health Check...")
    try:
        response = requests.get(f"{api_server}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   ✅ Model Loaded: {data['model_loaded']}")
            print(f"   ✅ Device: {data['device']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 2. 더미 이미지 생성
    print("\n2️⃣  Creating test images...")
    
    # 파란색 이미지 (왼쪽 테스트)
    left_image = Image.new('RGB', (224, 224), color='blue')
    # 빨간색 이미지 (오른쪽 테스트)
    right_image = Image.new('RGB', (224, 224), color='red')
    
    print("   ✅ Created blue image (for left navigation)")
    print("   ✅ Created red image (for right navigation)")
    
    # 3. 추론 테스트
    print("\n3️⃣  Running inference tests...\n")
    
    test_cases = [
        {
            "name": "Left Navigation",
            "image": left_image,
            "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
        },
        {
            "name": "Right Navigation",
            "image": right_image,
            "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the right"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Instruction: {test_case['instruction'][:50]}...")
        
        # Base64 인코딩
        img_base64 = image_to_base64(test_case['image'])
        
        # API 요청
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{api_server}/predict",
                json={
                    "image": img_base64,
                    "instruction": test_case['instruction']
                },
                headers=headers,
                timeout=10
            )
            
            total_latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                action = data['action']
                model_latency = data.get('latency_ms', 0)
                
                print(f"  ✅ Success!")
                print(f"     Action: [{action[0]:.4f}, {action[1]:.4f}] m/s")
                print(f"     Linear X: {action[0]:.4f} m/s (forward/backward)")
                print(f"     Linear Y: {action[1]:.4f} m/s (left/right)")
                print(f"     Model Latency: {model_latency:.1f} ms")
                print(f"     Total Latency: {total_latency:.1f} ms")
                
                results.append({
                    "test": test_case['name'],
                    "success": True,
                    "action": action,
                    "latency": total_latency
                })
            else:
                print(f"  ❌ Failed: Status {response.status_code}")
                print(f"     {response.text}")
                results.append({
                    "test": test_case['name'],
                    "success": False,
                    "error": response.text
                })
                
        except requests.exceptions.Timeout:
            print(f"  ❌ Timeout (>10 seconds)")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": "Timeout"
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
        
        print()
    
    # 4. 결과 요약
    print("="*60)
    print("📊 Test Results Summary")
    print("="*60 + "\n")
    
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"Total Tests: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    if success_count > 0:
        avg_latency = np.mean([r['latency'] for r in results if r.get('success', False)])
        print(f"Average Latency: {avg_latency:.1f} ms")
        
        print("\n📈 Action Analysis:")
        for r in results:
            if r.get('success', False):
                action = r['action']
                print(f"  {r['test']}:")
                print(f"    Linear X: {action[0]:.4f} m/s")
                print(f"    Linear Y: {action[1]:.4f} m/s")
    
    print("\n" + "="*60)
    
    if success_count == len(results):
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check logs above.")
    
    print("="*60)

if __name__ == "__main__":
    test_api_inference()
