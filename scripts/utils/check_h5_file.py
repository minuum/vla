#!/usr/bin/env python3
"""
HDF5 파일 내용 확인 스크립트
"""
import h5py
import numpy as np
import sys
from pathlib import Path

def check_h5_file(file_path):
    """HDF5 파일 내용 분석"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return False
        
    print(f"📁 파일: {file_path.name}")
    print(f"💾 크기: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 메타데이터 확인
            print("📋 메타데이터:")
            for key in f.attrs.keys():
                value = f.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                print(f"   {key}: {value}")
            
            print("\n📦 데이터 구조:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"   📁 {name}/ (그룹)")
                elif isinstance(obj, h5py.Dataset):
                    print(f"   📄 {name}: {obj.shape} {obj.dtype}")
                    
            f.visititems(print_structure)
            
            # 2. Action Chunks 상세 분석
            if 'action_chunks' in f:
                chunks_group = f['action_chunks']
                print(f"\n🔍 Action Chunks 분석:")
                print(f"   총 청크 수: {len(chunks_group.keys())}")
                
                if 'chunk_0' in chunks_group:
                    chunk_0 = chunks_group['chunk_0']
                    print(f"   첫 번째 청크:")
                    
                    for attr_key in chunk_0.attrs.keys():
                        print(f"     • {attr_key}: {chunk_0.attrs[attr_key]}")
                    
                    if 'past_actions' in chunk_0:
                        past_actions = chunk_0['past_actions'][:]
                        print(f"     • 과거 액션: {past_actions.shape}")
                        print(f"       평균값: {np.mean(past_actions, axis=0)}")
                        
                    if 'future_actions' in chunk_0:
                        future_actions = chunk_0['future_actions'][:]
                        print(f"     • 미래 액션: {future_actions.shape}")
                        print(f"       평균값: {np.mean(future_actions, axis=0)}")
                        
                    if 'images' in chunk_0:
                        images = chunk_0['images']
                        print(f"     • 이미지: {images.shape}")
                        print(f"       메모리 사용량: {images.nbytes / (1024*1024):.1f} MB")
                        
                        # 첫 번째 이미지 통계
                        first_image = images[0]
                        print(f"       첫 이미지 통계: min={first_image.min()}, max={first_image.max()}, mean={first_image.mean():.1f}")
                    else:
                        print("     ❌ 이미지 데이터 없음")
            else:
                print("\n❌ Action Chunks 없음")
                
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return False
        
    return True

def main():
    if len(sys.argv) != 2:
        print("사용법: python check_h5_file.py <파일경로>")
        print("예시: python check_h5_file.py mobile_vla_dataset/episode_20250805_000248.h5")
        return
        
    file_path = sys.argv[1]
    check_h5_file(file_path)

if __name__ == "__main__":
    main()