#!/usr/bin/env python3
"""
베이스라인 메모리 측정
서버 부팅 후 기본 상태의 메모리 사용량 측정
"""

import psutil
import json
import sys
from datetime import datetime
from pathlib import Path


def measure_baseline():
    """베이스라인 메모리 측정"""
    
    # 전체 메모리 정보
    mem = psutil.virtual_memory()
    
    # 주요 프로세스별 메모리 사용량
    processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            info = proc.info
            mem_mb = info['memory_info'].rss / 1024 / 1024
            
            # 50MB 이상 사용하는 프로세스만 수집
            if mem_mb > 50:
                cmdline = ' '.join(info['cmdline'][:3]) if info['cmdline'] else ''
                processes.append({
                    "name": info['name'],
                    "pid": info['pid'],
                    "memory_mb": round(mem_mb, 1),
                    "cmdline": cmdline[:100]  # 최대 100자
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # 메모리 사용량 기준으로 정렬
    processes.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_memory": {
            "total_gb": round(mem.total / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "percent": mem.percent
        },
        "top_processes": processes[:15],  # 상위 15개
        "summary": {
            "os_baseline": f"{processes[0]['memory_mb'] if processes else 0:.1f} MB",
            "total_processes": len(processes)
        }
    }
    
    return result


def print_report(result):
    """측정 결과 출력"""
    
    print("="*70)
    print("  베이스라인 메모리 측정 결과")
    print("="*70)
    print()
    
    mem = result['total_memory']
    print(f"📊 전체 메모리:")
    print(f"   Total:     {mem['total_gb']} GB")
    print(f"   Used:      {mem['used_gb']} GB ({mem['percent']:.1f}%)")
    print(f"   Available: {mem['available_gb']} GB")
    print()
    
    print(f"🔝 상위 메모리 사용 프로세스:")
    print(f"{'Process':<20} {'PID':<8} {'Memory':<12} {'Command'}")
    print("-"*70)
    
    for proc in result['top_processes'][:10]:
        print(f"{proc['name']:<20} {proc['pid']:<8} {proc['memory_mb']:>8.1f} MB   {proc['cmdline'][:30]}")
    
    print()
    print(f"✅ 총 {result['summary']['total_processes']}개 프로세스 (50MB 이상)")
    print()


def save_result(result, output_path=None):
    """결과를 JSON으로 저장"""
    
    if output_path is None:
        output_path = Path("logs/baseline_memory.json")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 결과 저장: {output_path}")
    return output_path


def main():
    """메인 함수"""
    
    print("🚀 베이스라인 메모리 측정 시작...")
    print()
    
    # 측정
    result = measure_baseline()
    
    # 출력
    print_report(result)
    
    # 저장
    save_result(result)
    
    print("="*70)
    print("✅ 측정 완료!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
