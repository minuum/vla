#!/usr/bin/env python3
"""
HDF5 파일 내용 확인 및 딕셔너리 변환 스크립트
"""
import h5py
import numpy as np
import sys
from pathlib import Path
import json

def convert_h5_to_dict(file_path):
    """
    HDF5 파일의 내용을 딕셔너리로 변환하여 반환.
    
    Args:
        file_path (str): HDF5 파일 경로.
        
    Returns:
        dict: 파일 내용을 담은 딕셔너리. 파일이 없거나 읽기 오류 발생 시 None 반환.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return None
    
    file_info = {
        "file_name": file_path.name,
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "metadata": {},
        "data_structure": {},
        "action_chunks": {}
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 메타데이터 확인
            for key in f.attrs.keys():
                value = f.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                elif isinstance(value, (np.int64, np.float64, np.bool_)):
                    value = value.item()
                file_info["metadata"][key] = value

            # 2. 데이터 구조 확인
            def fill_structure(name, obj):
                path_parts = name.split('/')
                current_dict = file_info["data_structure"]
                
                for part in path_parts[:-1]:
                    current_dict = current_dict.setdefault(part, {})
                
                last_part = path_parts[-1]
                
                if isinstance(obj, h5py.Group):
                    current_dict.setdefault(last_part, {})
                elif isinstance(obj, h5py.Dataset):
                    current_dict[last_part] = {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype)
                    }

            f.visititems(fill_structure)
            
            # 3. Action Chunks 상세 분석
            if 'action_chunks' in f:
                chunks_group = f['action_chunks']
                file_info["action_chunks"]["total_chunks"] = len(chunks_group.keys())
                
                chunks_data = {}
                for chunk_name, chunk_obj in chunks_group.items():
                    chunk_info = {}
                    
                    for attr_key in chunk_obj.attrs.keys():
                        value = chunk_obj.attrs[attr_key]
                        if isinstance(value, (np.int64, np.float64, np.bool_)):
                            value = value.item()
                        chunk_info[attr_key] = value
                        
                    # 데이터셋 정보 - 평균 대신 모든 값들을 리스트로 저장
                    if 'past_actions' in chunk_obj:
                        past_actions = chunk_obj['past_actions']
                        chunk_info["past_actions"] = {
                            "shape": past_actions.shape,
                            "data": past_actions[:].tolist()  # 모든 값들을 리스트로 변환
                        }
                    
                    if 'future_actions' in chunk_obj:
                        future_actions = chunk_obj['future_actions']
                        chunk_info["future_actions"] = {
                            "shape": future_actions.shape,
                            "data": future_actions[:].tolist() # 모든 값들을 리스트로 변환
                        }
                        
                    if 'images' in chunk_obj:
                        images = chunk_obj['images']
                        first_image = images[0]
                        chunk_info["images"] = {
                            "shape": images.shape,
                            "memory_mb": images.nbytes / (1024 * 1024),
                            "first_image_stats": {
                                "min": float(first_image.min()),
                                "max": float(first_image.max()),
                                "mean": float(first_image.mean())
                            }
                        }
                    
                    chunks_data[chunk_name] = chunk_info
                
                file_info["action_chunks"]["chunks"] = chunks_data
            
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None
        
    return file_info

def main():
    if len(sys.argv) != 2:
        print("사용법: python check_h5_file_dict.py <파일경로>")
        print("예시: python check_h5_file_dict.py mobile_vla_dataset/episode_20250805_000248.h5")
        return
        
    file_path = sys.argv[1]
    
    h5_dict = convert_h5_to_dict(file_path)
    
    if h5_dict:
        print(json.dumps(h5_dict, indent=4))
        
if __name__ == "__main__":
    main()