#!/usr/bin/env python3
"""
빠른 API 키 테스트 - 현재 API 서버의 실제 키 찾기
"""
import requests

API_BASE = "http://localhost:8000"

# 가능한 모든 키 시도
POSSIBLE_KEYS = [
    "nozI8wvqUV1i_4iESNwn-2D5x-Mf-JUJjvXKwOy0C00",
    "vla-mobile-fixed-key-20260205",
    "0Yf5D1z0AfjMY5aqtBVBvskHVeNjIo5uKYwH7bGfkKE",
    "35bnTxV_Wa7zD8WhmWb-lUMCjt0EAqT8gV6H8lpt7LA",
    "vla_mobile_robot_2025",
    "mobile-vla-key-2024",
]

print("🔐 API 키 테스트 중...")
print(f"API 서버: {API_BASE}")
print()

import base64
from PIL import Image
import numpy as np
import io

# 더미 이미지 생성
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

for i, key in enumerate(POSSIBLE_KEYS):
    try:
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"X-API-Key": key},
            json={"image": img_b64, "instruction": "test"},
            timeout=3
        )
        
        if response.status_code == 200:
            print(f"✅ 성공! 키: {key[:20]}...")
            print(f"   인덱스: {i}")
            print(f"   Response: {response.json()}")
            break
        else:
            print(f"❌ 실패 ({response.status_code}): {key[:20]}...")
    except Exception as e:
        print(f"⚠️  에러: {key[:20]}... - {e}")
