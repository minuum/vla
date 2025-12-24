#!/usr/bin/env python3
"""
Mobile VLA 추론 서버 테스트 클라이언트
"""

import requests
import base64
import json
import sys
from pathlib import Path
import argparse
from PIL import Image
import io
import numpy as np

# 색상
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_color(text, color):
    print(f"{color}{text}{NC}")

def test_health(server_url):
    """Health check 테스트"""
    print_color("🔍 Health Check...", BLUE)
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print_color("✅ Health Check 성공", GREEN)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print_color(f"❌ Health Check 실패: {e}", RED)
        return False

def test_model_info(server_url, api_key):
    """모델 정보 조회 테스트"""
    print_color("\n📊 모델 정보 조회...", BLUE)
    try:
        headers = {"X-API-Key": api_key}
        response = requests.get(f"{server_url}/model/info", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        print_color("✅ 모델 정보 조회 성공", GREEN)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print_color(f"❌ 모델 정보 조회 실패: {e}", RED)
        return False

def create_test_image():
    """테스트용 더미 이미지 생성"""
    # 간단한 그라데이션 이미지
    img = Image.new('RGB', (640, 480))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
    return img

def test_predict(server_url, api_key, image_path=None, instruction=None):
    """추론 테스트"""
    print_color("\n🎯 추론 요청...", BLUE)
    
    # 이미지 준비
    if image_path and Path(image_path).exists():
        print(f"이미지 로드: {image_path}")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    else:
        print("더미 이미지 생성...")
        img = create_test_image()
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # 지시문 준비
    if instruction is None:
        instruction = "Navigate to the left bottle"
    
    print(f"지시문: {instruction}")
    
    # 추론 요청
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "image": image_b64,
            "instruction": instruction,
            "use_abs_action": True
        }
        
        print("추론 실행 중...")
        response = requests.post(
            f"{server_url}/predict",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        print_color("✅ 추론 성공", GREEN)
        print(f"  액션: {data['action']}")
        print(f"  지연 시간: {data['latency_ms']:.1f} ms")
        print(f"  모델: {data['model_name']}")
        print(f"  Chunk size: {data['chunk_size']}")
        if 'direction' in data and data['direction'] is not None:
            print(f"  방향: {data['direction']}")
        if 'full_chunk' in data:
            print(f"  전체 chunk: {len(data['full_chunk'])} actions")
        
        return True
        
    except Exception as e:
        print_color(f"❌ 추론 실패: {e}", RED)
        if hasattr(e, 'response') and e.response is not None:
            print(f"응답: {e.response.text}")
        return False

def test_benchmark(server_url, api_key, iterations=50):
    """벤치마크 테스트"""
    print_color(f"\n⏱️ 벤치마크 ({iterations}회)...", BLUE)
    try:
        headers = {"X-API-Key": api_key}
        response = requests.get(
            f"{server_url}/benchmark?iterations={iterations}",
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        
        print_color("✅ 벤치마크 완료", GREEN)
        print(f"  평균 지연: {data['avg_latency_ms']:.2f} ms")
        print(f"  최대 지연: {data['max_latency_ms']:.2f} ms")
        print(f"  최소 지연: {data['min_latency_ms']:.2f} ms")
        print(f"  FPS: {data['fps']:.2f}")
        
        return True
        
    except Exception as e:
        print_color(f"❌ 벤치마크 실패: {e}", RED)
        return False

def main():
    parser = argparse.ArgumentParser(description="Mobile VLA 추론 서버 테스트")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="서버 URL (기본값: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key (환경변수 VLA_API_KEY 사용 가능)"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="테스트 이미지 경로 (선택적, 없으면 더미 생성)"
    )
    parser.add_argument(
        "--instruction",
        default="Navigate to the left bottle",
        help="언어 지시문"
    )
    parser.add_argument(
        "--test",
        choices=["all", "health", "info", "predict", "benchmark"],
        default="all",
        help="실행할 테스트 (기본값: all)"
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="벤치마크 반복 횟수 (기본값: 50)"
    )
    
    args = parser.parse_args()
    
    # API Key 확인
    import os
    api_key = args.api_key or os.getenv("VLA_API_KEY")
    if not api_key:
        print_color("❌ API Key가 필요합니다", RED)
        print("  --api-key 옵션 또는 VLA_API_KEY 환경변수를 설정하세요")
        sys.exit(1)
    
    server_url = args.server
    
    print_color("════════════════════════════════════════", BLUE)
    print_color("  Mobile VLA 추론 서버 테스트", BLUE)
    print_color("════════════════════════════════════════", BLUE)
    print(f"서버: {server_url}")
    print(f"API Key: {api_key[:10]}...")
    print("")
    
    results = {}
    
    # 테스트 실행
    if args.test in ["all", "health"]:
        results["health"] = test_health(server_url)
    
    if args.test in ["all", "info"]:
        results["info"] = test_model_info(server_url, api_key)
    
    if args.test in ["all", "predict"]:
        results["predict"] = test_predict(
            server_url, 
            api_key, 
            args.image, 
            args.instruction
        )
    
    if args.test in ["all", "benchmark"]:
        results["benchmark"] = test_benchmark(
            server_url, 
            api_key, 
            args.benchmark_iterations
        )
    
    # 결과 요약
    print_color("\n════════════════════════════════════════", BLUE)
    print_color("  테스트 결과 요약", BLUE)
    print_color("════════════════════════════════════════", BLUE)
    
    for test_name, success in results.items():
        status = f"{GREEN}✅ PASS{NC}" if success else f"{RED}❌ FAIL{NC}"
        print(f"{test_name:15s}: {status}")
    
    # 전체 성공 여부
    all_passed = all(results.values())
    if all_passed:
        print_color("\n✅ 모든 테스트 통과!", GREEN)
        sys.exit(0)
    else:
        print_color("\n❌ 일부 테스트 실패", RED)
        sys.exit(1)

if __name__ == "__main__":
    main()
