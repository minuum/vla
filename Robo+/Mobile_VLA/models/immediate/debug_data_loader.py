#!/usr/bin/env python3
"""
데이터 로더 디버깅 스크립트
"""

import h5py
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_data_loading():
    """데이터 로딩 디버깅"""
    
    data_path = "../../../../ROS_action/mobile_vla_dataset/"
    logger.info(f"🔍 데이터 경로: {data_path}")
    
    # H5 파일들 찾기
    h5_files = list(Path(data_path).glob("*.h5"))
    logger.info(f"📁 H5 파일 수: {len(h5_files)}")
    
    total_samples = 0
    
    for h5_file in h5_files[:3]:  # 처음 3개만 테스트
        logger.info(f"📂 파일: {h5_file.name}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # 키 확인
                logger.info(f"   - 키들: {list(f.keys())}")
                
                if 'images' in f and 'actions' in f:
                    images = f['images'][:]
                    actions = f['actions'][:]
                    
                    logger.info(f"   - 이미지 shape: {images.shape}")
                    logger.info(f"   - 액션 shape: {actions.shape}")
                    
                    # 첫 번째 에피소드 확인
                    if len(images) > 0:
                        logger.info(f"   - 첫 번째 에피소드 이미지 shape: {images[0].shape}")
                        logger.info(f"   - 첫 번째 에피소드 액션 shape: {actions[0].shape}")
                        logger.info(f"   - 첫 번째 액션: {actions[0][0]}")
                        
                        total_samples += len(images)
                else:
                    logger.warning(f"   - 'images' 또는 'actions' 키가 없음")
                    
        except Exception as e:
            logger.error(f"   - 오류: {e}")
    
    logger.info(f"📊 총 샘플 수: {total_samples}")
    
    # 폴더 구조 확인
    folders = [f for f in Path(data_path).iterdir() if f.is_dir()]
    logger.info(f"📁 폴더 수: {len(folders)}")
    
    for folder in folders[:3]:  # 처음 3개만 테스트
        logger.info(f"📂 폴더: {folder.name}")
        
        # 이미지 파일들
        image_files = list(folder.glob("*.png"))
        logger.info(f"   - 이미지 파일 수: {len(image_files)}")
        
        # 액션 파일
        action_file = folder / "actions.npy"
        if action_file.exists():
            try:
                actions = np.load(action_file)
                logger.info(f"   - 액션 shape: {actions.shape}")
                logger.info(f"   - 첫 번째 액션: {actions[0]}")
            except Exception as e:
                logger.error(f"   - 액션 로드 오류: {e}")
        else:
            logger.warning(f"   - actions.npy 파일이 없음")

if __name__ == "__main__":
    debug_data_loading()
