#!/usr/bin/env python3
"""
데이터셋 Core/Variant 분석 도구
"""

import os
import h5py
import numpy as np
from pathlib import Path
import json

def analyze_dataset():
    """간단한 데이터셋 분석"""
    data_path = Path("../../../../ROS_action/mobile_vla_dataset/")
    
    print(f"🔍 데이터 경로: {data_path}")
    print(f"📁 경로 존재: {data_path.exists()}")
    
    if not data_path.exists():
        print("❌ 데이터 경로가 존재하지 않습니다.")
        return
    
    # H5 파일 찾기
    h5_files = list(data_path.glob("*.h5"))
    print(f"📊 H5 파일 수: {len(h5_files)}")
    
    if len(h5_files) == 0:
        print("❌ H5 파일이 없습니다.")
        return
    
    # 첫 번째 파일 구조 분석
    first_file = h5_files[0]
    print(f"\n🔍 {first_file.name} 구조 분석:")
    
    try:
        with h5py.File(first_file, 'r') as f:
            print("📋 최상위 키:")
            for key in f.keys():
                print(f"   - {key}: {type(f[key])}")
                if hasattr(f[key], 'shape'):
                    print(f"     Shape: {f[key].shape}")
                elif hasattr(f[key], 'keys'):
                    print(f"     Sub-keys: {list(f[key].keys())[:5]}...")
            
            print("\n📋 속성 (Attributes):")
            for attr_name in f.attrs:
                print(f"   - {attr_name}: {f.attrs[attr_name]}")
    
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")

if __name__ == "__main__":
    analyze_dataset()