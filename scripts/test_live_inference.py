#!/usr/bin/env python3
"""
실제 추론 테스트 스크립트
"""

import requests
import base64
import json
import sys
import os
from pathlib import Path
from PIL import Image
import io
import numpy as np
import time

# 색상
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_color(text, color):
    print(f"{color}{text}{NC}")

def create_test_image(width=640, height=480, pattern="gradient"):
    """테스트 이미지 생성"""
    if pattern == "gradient":
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        for i in range(width):
            for j in range(height):
                # 그라데이션 패턴
                r = int(255 * i / width)
                g = int(255 * j / height)
                b = int(255 * (i + j) / (width + height))
                pixels[i, j] = (r, g, b)
    elif pattern == "checkerboard":
        # 체커보드 패턴
        img = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 40
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    img[i:i+square_size, j:j+square_size] = [100, 100, 100]
        img = Image.fromarray(img)
    else:
        # 단색
        img = Image.new('RGB', (width, height), color=(128, 128, 255))
    
    return img

def image_to_base64(image):
    """PIL Image를 base64로 변환"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_inference(server_url, api_key, instruction, image=None, save_image=False):
    """추론 테스트 실행"""
    print_color("\n" + "="*60, BLUE)
    print_color("  실제 추론 테스트", BLUE)
    print_color("="*60, BLUE)
    
    # 이미지 준비
    if image is None:
        print("테스트 이미지 생성 중...")
        image = create_test_image(pattern="gradient")
        if save_image:
            save_path = "test_image_temp.png"
            image.save(save_path)
            print(f"테스트 이미지 저장: {save_path}")
    
    print(f"이미지 크기: {image.size}")
    print(f"지시문: {instruction}")
    
    # Base64 인코딩
    print("\n이미지 인코딩 중...")
    image_b64 = image_to_base64(image)
    print(f"인코딩 완료 (크기: {len(image_b64)} bytes)")
    
    # 추론 요청
    print("\n추론 요청 전송 중...")
    print(f"서버: {server_url}")
    
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "image": image_b64,
        "instruction": instruction,
        "use_abs_action": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/predict",
            headers=headers,
            json=payload,
            timeout=60  # Jetson은 느릴 수 있으므로 60초 타임아웃
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_color("\n✅ 추론 성공!", GREEN)
            print_color("="*60, GREEN)
            print(f"  예측 액션: {data['action']}")
            print(f"  - linear_x (전진): {data['action'][0]:.4f} m/s")
            print(f"  - linear_y (좌우): {data['action'][1]:.4f} m/s")
            print()
            print(f"  모델 지연: {data['latency_ms']:.1f} ms")
            print(f"  전체 요청 시간: {request_time*1000:.1f} ms")
            print(f"  모델: {data['model_name']}")
            print(f"  Chunk size: {data['chunk_size']}")
            
            if 'direction' in data and data['direction'] is not None:
                direction_str = "왼쪽" if data['direction'] > 0 else "오른쪽" if data['direction'] < 0 else "직진"
                print(f"  방향: {direction_str} ({data['direction']})")
            
            if 'full_chunk' in data and data['full_chunk']:
                print(f"\n  전체 action chunk ({len(data['full_chunk'])} steps):")
                for i, action in enumerate(data['full_chunk'][:3]):  # 처음 3개만 표시
                    print(f"    [{i}] linear_x={action[0]:.4f}, linear_y={action[1]:.4f}")
                if len(data['full_chunk']) > 3:
                    print(f"    ... (총 {len(data['full_chunk'])} actions)")
            
            print_color("="*60, GREEN)
            return True, data
            
        else:
            print_color(f"\n❌ 추론 실패 (HTTP {response.status_code})", RED)
            print(f"응답: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print_color("\n❌ 타임아웃! 서버가 60초 내에 응답하지 않았습니다", RED)
        print("Jetson 환경에서 모델 로딩이 오래 걸릴 수 있습니다.")
        return False, None
    except Exception as e:
        print_color(f"\n❌ 에러 발생: {e}", RED)
        import traceback
        traceback.print_exc()
        return False, None

def multiple_inference_test(server_url, api_key, num_tests=3):
    """여러 번 추론 테스트"""
    print_color("\n" + "="*60, BLUE)
    print_color(f"  연속 추론 테스트 ({num_tests}회)", BLUE)
    print_color("="*60, BLUE)
    
    instructions = [
        "Navigate to the left bottle",
        "Navigate to the right box",
        "Move forward to the target"
    ]
    
    results = []
    for i in range(num_tests):
        instruction = instructions[i % len(instructions)]
        print_color(f"\n[테스트 {i+1}/{num_tests}]", YELLOW)
        
        success, data = test_inference(
            server_url, 
            api_key, 
            instruction,
            image=create_test_image(pattern=["gradient", "checkerboard", "solid"][i % 3])
        )
        
        if success:
            results.append({
                "test_num": i+1,
                "instruction": instruction,
                "action": data['action'],
                "latency_ms": data['latency_ms']
            })
        
        time.sleep(0.5)  # 각 테스트 사이 대기
    
    # 결과 요약
    if results:
        print_color("\n" + "="*60, BLUE)
        print_color("  테스트 결과 요약", BLUE)
        print_color("="*60, BLUE)
        
        latencies = [r['latency_ms'] for r in results]
        print(f"총 성공: {len(results)}/{num_tests}")
        print(f"평균 지연: {np.mean(latencies):.1f} ms")
        print(f"최소 지연: {np.min(latencies):.1f} ms")
        print(f"최대 지연: {np.max(latencies):.1f} ms")
        
        print("\n개별 결과:")
        for r in results:
            print(f"  [{r['test_num']}] {r['instruction']}")
            print(f"      액션: [{r['action'][0]:.4f}, {r['action'][1]:.4f}], 지연: {r['latency_ms']:.1f}ms")

def main():
    server_url = os.getenv("VLA_API_SERVER", "http://localhost:8000")
    api_key = os.getenv("VLA_API_KEY")
    
    print_color("="*60, BLUE)
    print_color("  Mobile VLA 실제 추론 테스트", BLUE)
    print_color("="*60, BLUE)
    print(f"서버: {server_url}")
    
    if not api_key:
        print_color("\n⚠️  VLA_API_KEY 환경변수가 설정되지 않았습니다", YELLOW)
        api_key = input("API Key를 입력하세요: ").strip()
        if not api_key:
            print_color("❌ API Key가 필요합니다", RED)
            sys.exit(1)
    
    print(f"API Key: {api_key[:10]}...")
    
    # 1. 단일 추론 테스트
    success, _ = test_inference(
        server_url,
        api_key,
        "Navigate to the left bottle",
        save_image=True
    )
    
    if not success:
        print_color("\n❌ 추론 테스트 실패", RED)
        sys.exit(1)
    
    # 2. 연속 추론 테스트
    response = input("\n연속 추론 테스트를 진행하시겠습니까? (y/n): ").strip().lower()
    if response == 'y':
        multiple_inference_test(server_url, api_key, num_tests=3)
    
    print_color("\n✅ 모든 테스트 완료!", GREEN)

if __name__ == "__main__":
    main()
