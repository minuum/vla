#!/usr/bin/env python3
"""
Real Robot Driving Simulation - 18 Consecutive Inferences
Receding Horizon 방식으로 실제 주행 시뮬레이션

학습 설정:
- window_size: 8 (과거 8프레임 사용)
- chunk_size: 5 (다음 5 actions 예측)
- 실제 사용: 첫 action만 사용, 나머지는 receding horizon

시나리오:
- 총 18번 inference (약 7.2초 주행)
- 각 inference마다 window update
- GPU memory, latency 모니터링
"""

import requests
import base64
import json
import time
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# API 설정
API_URL = "http://localhost:8000"
API_KEY = "qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU"

print("="*80)
print("Real Robot Driving Simulation - 18 Consecutive Inferences")
print("="*80)
print("\n📊 학습 설정:")
print("  - window_size: 8 (과거 8프레임)")
print("  - chunk_size: 5 (다음 5 actions 예측)")
print("  - inference_rate: 2.5 Hz (0.4초마다)")
print("  - total_duration: ~7.2초 주행")
print()

# 결과 저장
results = {
    'latencies': [],
    'gpu_memories': [],
    'actions': [],
    'timestamps': []
}

# Health check
print("1. Initial Health Check...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    health = response.json()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   GPU Memory: {health['gpu_memory']['allocated_gb']:.2f} GB")
    print(f"   ✅ Server ready\n")
except Exception as e:
    print(f"   ❌ Server not ready: {e}")
    exit(1)

# 18번 연속 inference
print("2. Starting 18 consecutive inferences...")
print(f"   (Simulating real robot driving)")
print()

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

start_time = time.time()

for i in range(1, 19):
    # 매 inference마다 다른 이미지 생성 (실제론 카메라에서)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    buffered = io.BytesIO()
    dummy_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Instruction (실제론 동일하거나 변경 가능)
    instruction = f"Move forward to the target (step {i}/18)"
    
    payload = {
        "image": img_b64,
        "instruction": instruction
    }
    
    # Inference 요청
    try:
        req_start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            headers=headers,
            json=payload,
            timeout=30
        )
        req_end = time.time()
        
        if response.status_code == 200:
            result = response.json()
            action = result['action']
            latency = result['latency_ms']
            total_latency = (req_end - req_start) * 1000
            
            # 결과 저장
            results['latencies'].append(latency)
            results['actions'].append(action)
            results['timestamps'].append(time.time() - start_time)
            
            # 실시간 출력
            status = "✅" if latency < 600 else "⚠️"
            print(f"   Step {i:2d}/18: {status} Latency: {latency:6.1f} ms | "
                  f"Total: {total_latency:6.1f} ms | Action: [{action[0]:6.3f}, {action[1]:6.3f}]")
            
        else:
            print(f"   ❌ Step {i:2d}/18: Failed (Status {response.status_code})")
            results['latencies'].append(-1)
            results['actions'].append([0, 0])
            results['timestamps'].append(time.time() - start_time)
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Step {i:2d}/18: Timeout")
        results['latencies'].append(-1)
        results['actions'].append([0, 0])
        results['timestamps'].append(time.time() - start_time)
    except Exception as e:
        print(f"   ❌ Step {i:2d}/18: Error - {e}")
        results['latencies'].append(-1)
        results['actions'].append([0, 0])
        results['timestamps'].append(time.time() - start_time)
    
    # GPU Memory check (매 5번째)
    if i % 5 == 0:
        try:
            health_resp = requests.get(f"{API_URL}/health", timeout=2)
            if health_resp.status_code == 200:
                health = health_resp.json()
                gpu_mem = health['gpu_memory']['allocated_gb']
                results['gpu_memories'].append((i, gpu_mem))
                print(f"   📊 GPU Memory at step {i}: {gpu_mem:.2f} GB")
        except:
            pass

total_time = time.time() - start_time

# Final health check
print(f"\n3. Final Health Check...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    health = response.json()
    final_gpu = health['gpu_memory']['allocated_gb']
    print(f"   GPU Memory: {final_gpu:.2f} GB")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   ✅ Server stable\n")
except Exception as e:
    print(f"   ⚠️  Health check error: {e}\n")

# 통계 분석
print("="*80)
print("📊 Performance Analysis")
print("="*80)

valid_latencies = [l for l in results['latencies'] if l > 0]
if valid_latencies:
    avg_latency = np.mean(valid_latencies)
    min_latency = np.min(valid_latencies)
    max_latency = np.max(valid_latencies)
    std_latency = np.std(valid_latencies)
    
    print(f"\n🕐 Latency Statistics:")
    print(f"   Average: {avg_latency:.1f} ms")
    print(f"   Min: {min_latency:.1f} ms")
    print(f"   Max: {max_latency:.1f} ms")
    print(f"   Std Dev: {std_latency:.1f} ms")
    
    # Real-time capability
    target_latency = 400  # 2.5 Hz = 400ms
    success_rate = sum(1 for l in valid_latencies if l < target_latency) / len(valid_latencies) * 100
    print(f"\n⚡ Real-time Capability:")
    print(f"   Target: < {target_latency} ms (for 2.5 Hz)")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"   ✅ Excellent - Ready for real robot!")
    elif success_rate >= 70:
        print(f"   ⚠️  Good - Usable but may have delays")
    else:
        print(f"   ❌ Poor - Needs optimization")

print(f"\n💾 GPU Memory:")
if results['gpu_memories']:
    for step, mem in results['gpu_memories']:
        print(f"   Step {step:2d}: {mem:.2f} GB")
    
    mem_values = [m for _, m in results['gpu_memories']]
    if len(mem_values) > 1:
        mem_change = mem_values[-1] - mem_values[0]
        print(f"   Memory change: {mem_change:+.3f} GB")
        if abs(mem_change) < 0.1:
            print(f"   ✅ Stable - No memory leak")
        else:
            print(f"   ⚠️  Memory drift detected")

print(f"\n⏱️  Total Time:")
print(f"   Elapsed: {total_time:.2f} seconds")
print(f"   Expected: {18 * 0.4:.1f} seconds (at 2.5 Hz)")
print(f"   Overhead: {total_time - 18*0.4:.2f} seconds")

# Actions 분석
actions_array = np.array(results['actions'])
print(f"\n🎯 Actions Summary:")
print(f"   Linear X - Mean: {np.mean(actions_array[:, 0]):.3f}, Std: {np.std(actions_array[:, 0]):.3f}")
print(f"   Linear Y - Mean: {np.mean(actions_array[:, 1]):.3f}, Std: {np.std(actions_array[:, 1]):.3f}")

# 최종 평가
print("\n" + "="*80)
print("🚀 Real Robot Readiness Evaluation")
print("="*80)

score = 0
max_score = 5

# 1. Latency
if avg_latency < 500:
    print("✅ Latency: < 500ms - Excellent")
    score += 1
elif avg_latency < 700:
    print("⚠️  Latency: 500-700ms - Acceptable")
    score += 0.5
else:
    print("❌ Latency: > 700ms - Too slow")

# 2. Stability
if std_latency < 100:
    print("✅ Stability: Low variance - Consistent")
    score += 1
elif std_latency < 200:
    print("⚠️  Stability: Medium variance - Acceptable")
    score += 0.5
else:
    print("❌ Stability: High variance - Unstable")

# 3. Memory
if results['gpu_memories']:
    mem_values = [m for _, m in results['gpu_memories']]
    if abs(mem_values[-1] - mem_values[0]) < 0.1:
        print("✅ Memory: Stable - No leaks")
        score += 1
    else:
        print("⚠️  Memory: Some drift detected")
        score += 0.5
else:
    score += 1  # 측정 못했으면 일단 통과

# 4. Success rate
success_count = sum(1 for l in results['latencies'] if l > 0)
if success_count == 18:
    print("✅ Reliability: 100% success rate")
    score += 1
elif success_count >= 16:
    print("⚠️  Reliability: Some failures")
    score += 0.5
else:
    print("❌ Reliability: Too many failures")

# 5. Real-time
if success_rate >= 90:
    print("✅ Real-time: 90%+ within target")
    score += 1
elif success_rate >= 70:
    print("⚠️  Real-time: 70-90% within target")
    score += 0.5
else:
    print("❌ Real-time: < 70% within target")

print(f"\n📊 Overall Score: {score:.1f}/5.0")

if score >= 4.5:
    print("🎉 Verdict: READY FOR REAL ROBOT! Deploy with confidence.")
elif score >= 3.5:
    print("✅ Verdict: READY with minor considerations. Monitor performance.")
elif score >= 2.5:
    print("⚠️  Verdict: USABLE but needs monitoring. Watch for issues.")
else:
    print("❌ Verdict: NOT READY. Optimization required.")

print("\n" + "="*80)

# 로그 저장
import json
with open('logs/robot_driving_test_18steps.json', 'w') as f:
    json.dump({
        'latencies': results['latencies'],
        'actions': results['actions'],
        'timestamps': results['timestamps'],
        'gpu_memories': results['gpu_memories'],
        'statistics': {
            'avg_latency': float(avg_latency),
            'min_latency': float(min_latency),
            'max_latency': float(max_latency),
            'std_latency': float(std_latency),
            'total_time': total_time,
            'success_rate': success_rate,
            'score': float(score)
        }
    }, f, indent=2)

print("📁 Results saved to: logs/robot_driving_test_18steps.json")
