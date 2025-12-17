#!/usr/bin/env python3
"""
실제 데이터셋 이미지로 API 서버 테스트 및 모델 비교

3개 모델 (Chunk5, Chunk10, No Chunk)의:
- Latency 비교
- Action 출력 비교
- GPU 메모리 사용량 비교
"""

import requests
import base64
from PIL import Image
import io
import h5py
import numpy as np
import time
from pathlib import Path
import json


def load_real_image_from_h5(h5_path: str, frame_idx: int = 0) -> np.ndarray:
    """HDF5에서 실제 이미지 로드"""
    with h5py.File(h5_path, 'r') as f:
        # ROS_action dataset format: /images (not /observations/images)
        images = f['images'][()]
        image = images[frame_idx]  # (H, W, 3)
        return image


def image_to_base64(image_np: np.ndarray) -> str:
    """numpy 이미지를 base64로 변환"""
    # Convert to PIL Image
    image_pil = Image.fromarray(image_np.astype(np.uint8))
    
    # Encode to base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    return img_b64


def test_model_with_real_data(
    host: str,
    port: int,
    api_key: str,
    model_name: str,
    image_b64: str,
    instruction: str,
    num_runs: int = 5
):
    """특정 모델로 여러 번 추론하여 평균 latency 측정"""
    
    base_url = f"http://{host}:{port}"
    headers = {"X-API-Key": api_key}
    
    print(f"\n{'='*70}")
    print(f"🎯 Testing Model: {model_name}")
    print(f"{'='*70}")
    
    # 1. Switch to target model
    print(f"\n1️⃣ Switching to {model_name}...")
    switch_response = requests.post(
        f"{base_url}/model/switch",
        json={"model_name": model_name},
        headers=headers,
        timeout=30
    )
    
    if switch_response.status_code != 200:
        print(f"❌ Failed to switch model: {switch_response.text}")
        return None
    
    switch_result = switch_response.json()
    print(f"✅ Model switched successfully")
    print(f"   Current model: {switch_result['current_model']}")
    print(f"   Chunk size: {switch_result['model_info']['fwd_pred_next_n']}")
    print(f"   Device: {switch_result['model_info']['device']}")
    
    # Wait for model to load
    time.sleep(2)
    
    # 2. Get model info
    print(f"\n2️⃣ Getting model info...")
    info_response = requests.get(f"{base_url}/model/info", headers=headers)
    if info_response.status_code == 200:
        model_info = info_response.json()
        print(f"   Model name: {model_info['model_name']}")
        print(f"   Chunk size (fwd_pred_next_n): {model_info['fwd_pred_next_n']}")
        print(f"   Action dim: {model_info['action_dim']}")
        print(f"   Device: {model_info['device']}")
    
    # 3. Check GPU memory
    print(f"\n3️⃣ Checking GPU memory...")
    health_response = requests.get(f"{base_url}/health")
    if health_response.status_code == 200:
        health = health_response.json()
        if 'gpu_memory' in health and health['gpu_memory']:
            print(f"   GPU allocated: {health['gpu_memory'].get('allocated_gb', 0):.2f} GB")
            print(f"   GPU reserved: {health['gpu_memory'].get('reserved_gb', 0):.2f} GB")
    
    # 4. Run predictions
    print(f"\n4️⃣ Running predictions ({num_runs} times)...")
    latencies = []
    actions = []
    
    payload = {
        "image": image_b64,
        "instruction": instruction
    }
    
    for run_idx in range(num_runs):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                headers=headers,
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                client_latency = (end_time - start_time) * 1000  # ms
                server_latency = result.get('latency_ms', 0)
                
                latencies.append(client_latency)
                actions.append(result['action'])
                
                print(f"   Run {run_idx+1}: Client={client_latency:.1f}ms, "
                      f"Server={server_latency:.1f}ms, "
                      f"Action=[{result['action'][0]:.3f}, {result['action'][1]:.3f}]")
            else:
                print(f"   Run {run_idx+1}: ❌ Failed - {response.status_code}")
                
        except Exception as e:
            print(f"   Run {run_idx+1}: ❌ Error - {e}")
    
    # 5. Calculate statistics
    if latencies:
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print(f"\n📊 Latency Statistics (Client-side, {len(latencies)} runs):")
        print(f"   Average: {avg_latency:.1f} ms")
        print(f"   Std Dev: {std_latency:.1f} ms")
        print(f"   Min: {min_latency:.1f} ms")
        print(f"   Max: {max_latency:.1f} ms")
        
        # Action consistency
        if len(actions) > 1:
            actions_np = np.array(actions)
            action_std = np.std(actions_np, axis=0)
            print(f"\n📊 Action Consistency:")
            print(f"   Std Dev: [{action_std[0]:.4f}, {action_std[1]:.4f}]")
            print(f"   (Lower is more consistent)")
        
        return {
            "model_name": model_name,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "num_runs": len(latencies),
            "actions": actions,
            "action_std": action_std.tolist() if len(actions) > 1 else None
        }
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="API Server Real Data Test")
    parser.add_argument("--host", type=str, default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--api-key", type=str, default="qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU", help="API key")
    parser.add_argument("--h5-file", type=str, 
                       default="ROS_action/mobile_vla_dataset/episode_20251203_042905_1box_hori_left_core_medium.h5",
                       help="HDF5 file path")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to test")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of prediction runs per model")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 Mobile VLA API Server - Real Data Test & Model Comparison")
    print("="*70)
    
    # Load real image
    print(f"\n📸 Loading image from: {args.h5_file}")
    print(f"   Frame index: {args.frame_idx}")
    
    image_np = load_real_image_from_h5(args.h5_file, args.frame_idx)
    print(f"   Image shape: {image_np.shape}")
    print(f"   Image dtype: {image_np.dtype}")
    print(f"   Image range: [{image_np.min()}, {image_np.max()}]")
    
    # Convert to base64
    image_b64 = image_to_base64(image_np)
    print(f"   Base64 length: {len(image_b64)} chars")
    
    # Test instruction
    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
    print(f"\n💬 Instruction: {instruction}")
    
    # Test all 3 models
    models = [
        "chunk5_epoch6",   # Recommended
        "chunk10_epoch8",
        "no_chunk_epoch4"
    ]
    
    results = []
    for model_name in models:
        result = test_model_with_real_data(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            model_name=model_name,
            image_b64=image_b64,
            instruction=instruction,
            num_runs=args.num_runs
        )
        if result:
            results.append(result)
        
        # Small break between models
        time.sleep(1)
    
    # Final comparison
    print(f"\n{'='*70}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*70}")
    
    if results:
        print(f"\n{'Model':<20} {'Avg Latency':<15} {'Std Dev':<15} {'Action Std'}")
        print("-" * 70)
        for result in results:
            action_std_str = f"[{result['action_std'][0]:.4f}, {result['action_std'][1]:.4f}]" if result['action_std'] else "N/A"
            print(f"{result['model_name']:<20} "
                  f"{result['avg_latency_ms']:>10.1f} ms   "
                  f"{result['std_latency_ms']:>10.1f} ms   "
                  f"{action_std_str}")
        
        # Find best model
        best_model = min(results, key=lambda x: x['avg_latency_ms'])
        print(f"\n🏆 Fastest Model: {best_model['model_name']} "
              f"({best_model['avg_latency_ms']:.1f} ms)")
        
        # Save results
        output_file = "api_test_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "test_params": {
                    "h5_file": args.h5_file,
                    "frame_idx": args.frame_idx,
                    "instruction": instruction,
                    "num_runs": args.num_runs
                },
                "results": results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("✅ Test completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
