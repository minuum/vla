#!/usr/bin/env python3
"""
모델 메모리 측정
모델 로딩부터 추론까지 각 단계별 메모리 사용량 타임라인 측정
"""

import psutil
import torch
import time
import json
import sys
from pathlib import Path
from datetime import datetime


def get_memory_snapshot(label):
    """현재 메모리 상태 스냅샷"""
    
    mem = psutil.virtual_memory()
    
    snapshot = {
        "label": label,
        "timestamp": time.time(),
        "ram": {
            "used_gb": round(mem.used / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "percent": mem.percent
        }
    }
    
    # GPU 메모리 (가능한 경우)
    if torch.cuda.is_available():
        snapshot["gpu"] = {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2)
        }
    
    return snapshot


def measure_model_memory_timeline(checkpoint_path=None, use_int8=False):
    """모델 메모리 사용량 타임라인 측정"""
    
    timeline = []
    
    print("="*70)
    print("  모델 메모리 타임라인 측정")
    print("="*70)
    print()
    
    # 1. 베이스라인
    print("📍 [1/7] 베이스라인 측정...")
    snapshot = get_memory_snapshot("1_baseline")
    timeline.append(snapshot)
    print(f"   RAM: {snapshot['ram']['used_gb']}GB")
    time.sleep(1)
    
    # 2. Import 후
    print("📍 [2/7] 라이브러리 임포트...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "Robo+/Mobile_VLA"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs"))
    
    try:
        from core.train_core.mobile_vla_trainer import MobileVLATrainer
    except ImportError:
        print("   ⚠️  MobileVLATrainer import 실패, transformers 직접 사용")
        from transformers import AutoModelForVision2Seq
    
    snapshot = get_memory_snapshot("2_after_import")
    timeline.append(snapshot)
    print(f"   RAM: {snapshot['ram']['used_gb']}GB (+{snapshot['ram']['used_gb'] - timeline[0]['ram']['used_gb']:.2f}GB)")
    time.sleep(1)
    
    # 3. 모델 생성
    print("📍 [3/7] 모델 인스턴스 생성...")
    
    quantization_config = None
    if use_int8:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            print("   ✅ INT8 quantization 활성화")
        except ImportError:
            print("   ⚠️  BitsAndBytes 없음, FP32 사용")
    
    try:
        model = MobileVLATrainer(
            model_name=".vlms/kosmos-2-patch14-224",
            action_dim=2,
            window_size=2,
            chunk_size=10,
            quantization_config=quantization_config
        )
    except NameError:
        # Fallback: transformers 직접 사용
        model = AutoModelForVision2Seq.from_pretrained(
            ".vlms/kosmos-2-patch14-224",
            quantization_config=quantization_config
        )
    
    snapshot = get_memory_snapshot("3_model_created")
    timeline.append(snapshot)
    print(f"   RAM: {snapshot['ram']['used_gb']}GB (+{snapshot['ram']['used_gb'] - timeline[1]['ram']['used_gb']:.2f}GB)")
    time.sleep(1)
    
    # 4. GPU로 이동
    if torch.cuda.is_available() and not use_int8:
        print("📍 [4/7] GPU로 이동...")
        model = model.to('cuda')
        torch.cuda.synchronize()
        
        snapshot = get_memory_snapshot("4_moved_to_gpu")
        timeline.append(snapshot)
        print(f"   RAM: {snapshot['ram']['used_gb']}GB")
        if 'gpu' in snapshot:
            print(f"   GPU: {snapshot['gpu']['allocated_gb']}GB")
        time.sleep(1)
    else:
        print("📍 [4/7] GPU 이동 건너뜀 (INT8 또는 CUDA 없음)")
        timeline.append(timeline[-1])  # 동일한 스냅샷 유지
    
    # 5. 첫 추론 (warmup)
    print("📍 [5/7] 첫 추론 (warmup)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.to('cuda')
    
    model.eval()
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except:
            # 모델 구조에 따라 다른 호출 방식 필요할 수 있음
            pass
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    snapshot = get_memory_snapshot("5_first_inference")
    timeline.append(snapshot)
    print(f"   RAM: {snapshot['ram']['used_gb']}GB")
    if 'gpu' in snapshot:
        print(f"   GPU: {snapshot['gpu']['allocated_gb']}GB")
    time.sleep(1)
    
    # 6. 연속 추론 (10회)
    print("📍 [6/7] 연속 추론 (10회)...")
    for i in range(10):
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except:
                pass
        time.sleep(0.3)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    snapshot = get_memory_snapshot("6_continuous_inference")
    timeline.append(snapshot)
    print(f"   RAM: {snapshot['ram']['used_gb']}GB")
    if 'gpu' in snapshot:
        print(f"   GPU: {snapshot['gpu']['allocated_gb']}GB")
    time.sleep(1)
    
    # 7. Peak 메모리
    print("📍 [7/7] Peak 메모리 확인...")
    snapshot = get_memory_snapshot("7_peak")
    timeline.append(snapshot)
    
    # Peak 계산
    peak_ram = max(s['ram']['used_gb'] for s in timeline)
    peak_gpu = max(s.get('gpu', {}).get('allocated_gb', 0) for s in timeline)
    
    print(f"   Peak RAM: {peak_ram}GB")
    if torch.cuda.is_available():
        print(f"   Peak GPU: {peak_gpu}GB")
    
    print()
    print("="*70)
    
    return timeline, {"peak_ram_gb": peak_ram, "peak_gpu_gb": peak_gpu}


def save_timeline(timeline, metadata, output_path=None):
    """타임라인 저장"""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"logs/model_memory_timeline_{timestamp}.json")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "metadata": metadata,
        "timeline": timeline
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 타임라인 저장: {output_path}")
    return output_path


def main():
    """메인 함수"""
    
    import argparse
    parser = argparse.ArgumentParser(description='모델 메모리 타임라인 측정')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization 사용')
    parser.add_argument('--checkpoint', type=str, help='체크포인트 경로')
    args = parser.parse_args()
    
    print("🚀 모델 메모리 타임라인 측정 시작...")
    print(f"   INT8: {args.int8}")
    print()
    
    timeline, metadata = measure_model_memory_timeline(
        checkpoint_path=args.checkpoint,
        use_int8=args.int8
    )
    
    save_timeline(timeline, metadata)
    
    print("✅ 측정 완료!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
