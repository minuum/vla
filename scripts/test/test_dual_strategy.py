"""
Dual Strategy Inference API 테스트

두 가지 추론 전략을 비교 테스트합니다:
1. Chunk Reuse: 빠름 (9x)
2. Receding Horizon: 정확함

Usage:
    # Start server first
    python Mobile_VLA/inference_server_dual.py
    
    # Then run test
    export VLA_API_KEY="your-key"
    python scripts/test_dual_strategy.py
"""

import requests
import base64
import time
import numpy as np
from PIL import Image
from io import BytesIO
import os


class DualStrategyTester:
    """Dual Strategy API Tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("VLA_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key required! Set VLA_API_KEY environment variable")
        
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def create_test_image(self) -> str:
        """Create dummy test image"""
        img = Image.new('RGB', (1280, 720), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_health(self):
        """Test health endpoint"""
        print("\n" + "="*80)
        print("1. Testing Health Endpoint")
        print("="*80)
        
        response = requests.get(f"{self.base_url}/health")
        data = response.json()
        
        print(f"Status: {response.status_code}")
        print(f"Model Loaded: {data['model_loaded']}")
        print(f"Device: {data['device']}")
        print(f"Strategies: {data['strategies']}")
        
        return response.status_code == 200
    
    def test_single_prediction(self, strategy: str):
        """Test single prediction"""
        image_b64 = self.create_test_image()
        instruction = "Navigate to the left target"
        
        payload = {
            "image": image_b64,
            "instruction": instruction,
            "strategy": strategy
        }
        
        start = time.time()
        response = requests.post(
            f"{self.base_url}/predict",
            headers=self.headers,
            json=payload
        )
        total_time = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "action": data["action"],
                "latency_ms": data["latency_ms"],
                "total_time_ms": total_time,
                "source": data["source"],
                "buffer_status": data["buffer_status"],
                "strategy": data["strategy"]
            }
        else:
            return {
                "success": False,
                "error": response.text
            }
    
    def test_chunk_reuse_sequence(self, num_frames: int = 18):
        """
        Test Chunk Reuse strategy with sequential frames
        
        Expected:
        - Frame 0: Infer (chunk of 10)
        - Frame 1-9: Reuse
        - Frame 10: Infer (new chunk)
        - Frame 11-17: Reuse
        
        Total: 2 inferences for 18 frames
        """
        print("\n" + "="*80)
        print(f"2. Testing Chunk Reuse Strategy ({num_frames} frames)")
        print("="*80)
        
        results = []
        total_start = time.time()
        
        for i in range(num_frames):
            result = self.test_single_prediction("chunk_reuse")
            results.append(result)
            
            if result["success"]:
                source_icon = "🔄" if result["source"] == "inferred" else "📦"
                print(f"Frame {i:2d}: {source_icon} {result['source']:<8} " + 
                      f"Action={result['action']}, Latency={result['latency_ms']:.1f}ms")
        
        total_time = (time.time() - total_start) * 1000
        
        # Analysis
        inferred_count = sum(1 for r in results if r["source"] == "inferred")
        reused_count = sum(1 for r in results if r["source"] == "reused")
        
        print(f"\n📊 Summary:")
        print(f"  Total frames: {num_frames}")
        print(f"  Inferences: {inferred_count}")
        print(f"  Reuses: {reused_count}")
        print(f"  Total time: {total_time/1000:.2f}s")
        print(f"  FPS: {num_frames/(total_time/1000):.2f}")
        print(f"  Reuse ratio: {reused_count/num_frames*100:.1f}%")
        
        return results
    
    def test_receding_horizon_sequence(self, num_frames: int = 18):
        """
        Test Receding Horizon strategy
        
        Expected:
        - Every frame: Infer
        
        Total: 18 inferences for 18 frames
        """
        print("\n" + "="*80)
        print(f"3. Testing Receding Horizon Strategy ({num_frames} frames)")
        print("="*80)
        
        results = []
        total_start = time.time()
        
        for i in range(num_frames):
            result = self.test_single_prediction("receding_horizon")
            results.append(result)
            
            if result["success"]:
                print(f"Frame {i:2d}: 🎯 inferred  " +
                      f"Action={result['action']}, Latency={result['latency_ms']:.1f}ms")
        
        total_time = (time.time() - total_start) * 1000
        
        # Analysis
        inferred_count = sum(1 for r in results if r["source"] == "inferred")
        
        print(f"\n📊 Summary:")
        print(f"  Total frames: {num_frames}")
        print(f"  Inferences: {inferred_count}")
        print(f"  Total time: {total_time/1000:.2f}s")
        print(f"  FPS: {num_frames/(total_time/1000):.2f}")
        
        return results
    
    def compare_strategies(self, chunk_results, receding_results):
        """Compare two strategies"""
        print("\n" + "="*80)
        print("4. Strategy Comparison")
        print("="*80)
        
        # Chunk Reuse stats
        chunk_infer = sum(1 for r in chunk_results if r["source"] == "inferred")
        chunk_total_latency = sum(r["latency_ms"] for r in chunk_results if r["source"] == "inferred")
        chunk_total_time = sum(r["total_time_ms"] for r in chunk_results)
        chunk_fps = len(chunk_results) / (chunk_total_time / 1000)
        
        # Receding Horizon stats
        receding_infer = sum(1 for r in receding_results if r["source"] == "inferred")
        receding_total_latency = sum(r["latency_ms"] for r in receding_results)
        receding_total_time = sum(r["total_time_ms"] for r in receding_results)
        receding_fps = len(receding_results) / (receding_total_time / 1000)
        
        # Comparison table
        print(f"\n{'Metric':<30} {'Chunk Reuse':<20} {'Receding Horizon':<20} {'Diff':<15}")
        print("-"*85)
        print(f"{'Inference count':<30} {chunk_infer:<20} {receding_infer:<20} {receding_infer/chunk_infer:.1f}x")
        print(f"{'Total inference latency':<30} {chunk_total_latency:<20.1f} {receding_total_latency:<20.1f} {receding_total_latency/chunk_total_latency:.1f}x")
        print(f"{'Total time (s)':<30} {chunk_total_time/1000:<20.2f} {receding_total_time/1000:<20.2f} {receding_total_time/chunk_total_time:.1f}x")
        print(f"{'FPS':<30} {chunk_fps:<20.2f} {receding_fps:<20.2f} {chunk_fps/receding_fps:.1f}x")
        
        speedup = receding_total_time / chunk_total_time
        
        print(f"\n💡 Results:")
        print(f"  • Chunk Reuse is {speedup:.1f}x FASTER")
        print(f"  • Chunk Reuse: {chunk_fps:.1f} FPS (Real-time capable ✅)")
        print(f"  • Receding Horizon: {receding_fps:.1f} FPS (Accurate but slow)")


def main():
    """Main test"""
    print("="*80)
    print("🧪 Dual Strategy Inference API Test Suite")
    print("="*80)
    
    tester = DualStrategyTester()
    
    # Test health
    if not tester.test_health():
        print("❌ Health check failed!")
        return
    
    # Wait for model loading
    print("\n⏳ Waiting for model to load...")
    time.sleep(5)
    
    # Test Chunk Reuse (18 frames)
    chunk_results = tester.test_chunk_reuse_sequence(num_frames=18)
    
    # Test Receding Horizon (18 frames)
    receding_results = tester.test_receding_horizon_sequence(num_frames=18)
    
    # Compare
    tester.compare_strategies(chunk_results, receding_results)
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()
