#!/usr/bin/env python3
"""
🔍 데이터 구조 디버깅 스크립트
H5 파일의 실제 구조를 확인
"""

import h5py
import numpy as np
from pathlib import Path

def debug_data_structure():
    """데이터 구조 디버깅"""
    
    data_path = "../../ROS_action/mobile_vla_dataset"
    
    # H5 파일들 찾기
    h5_files = list(Path(data_path).glob("*.h5"))
    print(f"📁 발견된 H5 파일 수: {len(h5_files)}")
    
    for h5_file in h5_files[:2]:  # 처음 2개만 확인
        print(f"\n🔍 {h5_file.name} 분석:")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                print(f"   📊 키 목록: {list(f.keys())}")
                
                for key in f.keys():
                    dataset = f[key]
                    print(f"   📋 {key}:")
                    print(f"      - Shape: {dataset.shape}")
                    print(f"      - Dtype: {dataset.dtype}")
                    
                    # 처음 몇 개 샘플 확인
                    if len(dataset.shape) > 0:
                        try:
                            sample = dataset[0]
                            print(f"      - Sample[0] shape: {sample.shape}")
                            print(f"      - Sample[0] dtype: {sample.dtype}")
                            
                            if len(sample.shape) == 3:  # [H, W, C] 또는 [T, H, W]
                                print(f"      - Sample[0, 0] shape: {sample[0].shape}")
                                print(f"      - Sample[0, 0] dtype: {sample[0].dtype}")
                            
                            if len(sample.shape) == 4:  # [T, H, W, C]
                                print(f"      - Sample[0, 0] shape: {sample[0, 0].shape}")
                                print(f"      - Sample[0, 0] dtype: {sample[0, 0].dtype}")
                            
                            # actions 특별 분석
                            if key == 'actions':
                                print(f"      - Actions 전체 shape: {dataset.shape}")
                                print(f"      - Actions[0] shape: {dataset[0].shape}")
                                print(f"      - Actions[0, 0] shape: {dataset[0, 0].shape}")
                                print(f"      - Actions[0, 0] 값: {dataset[0, 0]}")
                            
                            # images 특별 분석
                            if key == 'images':
                                print(f"      - Images 전체 shape: {dataset.shape}")
                                print(f"      - Images[0] shape: {dataset[0].shape}")
                                print(f"      - Images[0, 0] shape: {dataset[0, 0].shape}")
                                print(f"      - Images[0, 0] dtype: {dataset[0, 0].dtype}")
                                print(f"      - Images[0, 0] 값 범위: {dataset[0, 0].min()} ~ {dataset[0, 0].max()}")
                                
                        except Exception as e:
                            print(f"      - Sample 분석 실패: {e}")
                
        except Exception as e:
            print(f"   ❌ 오류: {e}")

if __name__ == "__main__":
    debug_data_structure()
