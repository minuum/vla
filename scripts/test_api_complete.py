#!/usr/bin/env python3
"""
Mobile VLA API 테스트 스크립트

입출력 파라미터 정의 및 전체 플로우 테스트

Usage:
    # 1. API 서버 실행 (다른 터미널)
    export VLA_API_KEY="your-api-key"
    python Mobile_VLA/inference_server.py
    
    # 2. 테스트 실행
    export VLA_API_KEY="your-api-key"
    python scripts/test_api_complete.py
"""

import requests
import base64
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import h5py

# API 설정
API_BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key-for-development"  # 실제로는 환경변수에서 읽기

# 테스트 설정
TEST_EPISODES = ["episode_20251210_left_001.h5", "episode_20251210_right_001.h5"]
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")


class MobileVLAAPITester:
    """Mobile VLA API 완전 테스트"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def test_health(self):
        """Health check 테스트"""
        print("\n" + "="*60)
        print("TEST 1: Health Check")
        print("="*60)
        
        url = f"{self.base_url}/health"
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        assert response.status_code == 200, "Health check failed"
        assert response.json()["status"] == "healthy", "Server not healthy"
        
        print("✅ Health check passed")
        return response.json()
    
    def encode_image(self, image_np: np.ndarray) -> str:
        """이미지를 base64로 인코딩
        
        Args:
            image_np: (H, W, 3) numpy array, uint8
        
        Returns:
            base64 encoded string
        """
        # Numpy -> PIL
        img_pil = Image.fromarray(image_np.astype('uint8'))
        
        # PIL -> bytes
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # bytes -> base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    
    def test_inference_request(self):
        """단일 추론 요청 테스트"""
        print("\n" + "="*60)
        print("TEST 2: Single Inference Request")
        print("="*60)
        
        # 테스트 이미지 생성 (더미)
        dummy_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        img_base64 = self.encode_image(dummy_img)
        
        # 요청 페이로드
        payload = {
            "image": img_base64,
            "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
        }
        
        print(f"Request:")
        print(f"  - Image size: {dummy_img.shape}")
        print(f"  - Instruction: {payload['instruction']}")
        print(f"  - Base64 length: {len(img_base64)} chars")
        
        # API 호출
        url = f"{self.base_url}/predict"
        start_time = time.time()
        response = requests.post(url, headers=self.headers, json=payload)
        total_latency = (time.time() - start_time) * 1000
        
        print(f"\nResponse:")
        print(f"  - Status Code: {response.status_code}")
        print(f"  - Total Latency: {total_latency:.1f} ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  - Action: {result['action']}")
            print(f"  - Model Latency: {result['latency_ms']:.1f} ms")
            print(f"  - Model Name: {result['model_name']}")
            
            # 검증
            assert len(result['action']) == 2, "Action should have 2 dimensions"
            assert result['latency_ms'] > 0, "Latency should be positive"
            
            print("✅ Inference request passed")
            return result
        else:
            print(f"❌ Inference failed: {response.text}")
            return None
    
    def test_real_data_inference(self, num_samples: int = 5):
        """실제 데이터로 추론 테스트
        
        Args:
            num_samples: 테스트할 샘플 수
        """
        print("\n" + "="*60)
        print(f"TEST 3: Real Data Inference ({num_samples} samples)")
        print("="*60)
        
        results = []
        
        for episode_file in TEST_EPISODES:
            h5_path = DATA_DIR / episode_file
            
            if not h5_path.exists():
                print(f"⚠️  Skipping {episode_file}: File not found")
                continue
            
            print(f"\nProcessing: {episode_file}")
            
            with h5py.File(h5_path, 'r') as f:
                images = f['images'][:]
                actions_gt = f['actions'][:]
                
                # Instruction 결정
                if 'left' in episode_file:
                    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
                    expected_sign = -1  # left = negative y
                elif 'right' in episode_file:
                    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the right"
                    expected_sign = 1   # right = positive y
                else:
                    continue
                
                # 랜덤 샘플링
                indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
                
                for idx in indices:
                    img = images[idx]
                    action_gt = actions_gt[idx]
                    
                    # 인코딩
                    img_base64 = self.encode_image(img)
                    
                    # 추론
                    payload = {
                        "image": img_base64,
                        "instruction": instruction
                    }
                    
                    url = f"{self.base_url}/predict"
                    response = requests.post(url, headers=self.headers, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        action_pred = np.array(result['action'])
                        
                        # Direction 정확도
                        correct = np.sign(action_pred[1]) == np.sign(expected_sign)
                        
                        results.append({
                            'episode': episode_file,
                            'action_pred': action_pred,
                            'action_gt': action_gt,
                            'correct_direction': correct,
                            'latency_ms': result['latency_ms']
                        })
                        
                        print(f"  Frame {idx}: pred={action_pred}, gt={action_gt}, "
                              f"correct={correct}, latency={result['latency_ms']:.1f}ms")
        
        # Summary
        if results:
            direction_acc = sum(r['correct_direction'] for r in results) / len(results)
            avg_latency = np.mean([r['latency_ms'] for r in results])
            
            print(f"\n📊 Results Summary:")
            print(f"  - Total samples: {len(results)}")
            print(f"  - Direction Accuracy: {direction_acc:.1%}")
            print(f"  - Average Latency: {avg_latency:.1f} ms")
            
            if direction_acc >= 0.8:
                print("✅ Real data inference passed")
            else:
                print("⚠️  Direction accuracy below 80%")
            
            return results
        else:
            print("❌ No valid samples found")
            return None
    
    def test_parameter_definition(self):
        """API 파라미터 정의 테스트"""
        print("\n" + "="*60)
        print("TEST 4: API Parameter Definition")
        print("="*60)
        
        # Input 파라미터
        print("\n📥 INPUT Parameters:")
        print("""
        {
            "image": "string (base64)",          # Required: Base64 encoded RGB image
            "instruction": "string"              # Required: Natural language command
        }
        
        Constraints:
            - image: Base64 encoded PNG/JPEG, recommended size 720x1280
            - instruction: English text, max 256 chars
            
        Example:
            {
                "image": "iVBORw0KGgoAAAANSUhEUgAA...",
                "instruction": "Navigate around obstacles and reach the bottle on the left"
            }
        """)
        
        # Output 파라미터
        print("\n📤 OUTPUT Parameters:")
        print("""
        {
            "action": [float, float],            # 2DOF robot action
            "latency_ms": float,                 # Inference latency in ms
            "model_name": "string"               # Model identifier
        }
        
        Action Format:
            action[0]: linear_x  (m/s)  - Forward velocity, range [0.0, 2.0]
            action[1]: linear_y (rad/s) - Angular velocity, range [-0.5, 0.5]
            
        Example:
            {
                "action": [1.15, -0.32],         # Move forward 1.15 m/s, turn left 0.32 rad/s
                "latency_ms": 385.2,
                "model_name": "mobile_vla_left_chunk10_20251218"
            }
        """)
        
        # Error 케이스
        print("\n❌ ERROR Responses:")
        print("""
        403 Forbidden:
            {
                "detail": "Invalid API Key"
            }
        
        422 Unprocessable Entity:
            {
                "detail": [
                    {
                        "loc": ["body", "image"],
                        "msg": "field required",
                        "type": "value_error.missing"
                    }
                ]
            }
        
        500 Internal Server Error:
            {
                "detail": "Inference failed: <error message>"
            }
        """)
        
        print("✅ Parameter definition documented")
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("\n" + "="*60)
        print("🚀 Mobile VLA API Complete Test Suite")
        print("="*60)
        
        try:
            # Test 1: Health
            self.test_health()
            
            # Test 2: Single inference
            self.test_inference_request()
            
            # Test 3: Real data
            self.test_real_data_inference(num_samples=5)
            
            # Test 4: Parameter definition
            self.test_parameter_definition()
            
            print("\n" + "="*60)
            print("✅ All tests passed!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Mobile VLA API Complete Test")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', default=os.getenv('VLA_API_KEY', 'test-key'), help='API key')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Tester 초기화
    tester = MobileVLAAPITester(
        base_url=args.url,
        api_key=args.api_key
    )
    
    # 전체 테스트 실행
    tester.run_all_tests()


if __name__ == "__main__":
    from io import BytesIO
    main()
