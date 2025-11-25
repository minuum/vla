#!/usr/bin/env python3
"""
체크포인트 파일 복구 및 검증 스크립트
"""

import os
import torch
import shutil
from pathlib import Path

def check_checkpoint_integrity(file_path):
    """체크포인트 파일 무결성 검사"""
    print(f"🔍 체크포인트 무결성 검사: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return False
    
    # 파일 크기 확인
    file_size = os.path.getsize(file_path)
    print(f"📁 파일 크기: {file_size / (1024**3):.2f} GB")
    
    if file_size < 1024 * 1024:  # 1MB 미만
        print("❌ 파일이 너무 작습니다 (손상 가능성)")
        return False
    
    # 파일 확장자 확인
    if not file_path.endswith('.pth'):
        print("❌ .pth 파일이 아닙니다")
        return False
    
    return True

def try_load_checkpoint(file_path):
    """체크포인트 로드 시도"""
    print(f"📦 체크포인트 로드 시도: {file_path}")
    
    try:
        # CPU에서 로드 시도
        checkpoint = torch.load(file_path, map_location='cpu')
        print(f"✅ 로드 성공!")
        
        if isinstance(checkpoint, dict):
            print(f"📋 체크포인트 키들: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                print(f"🧠 모델 상태 키 개수: {len(model_state)}")
                print(f"🔑 첫 5개 키들: {list(model_state.keys())[:5]}")
                
                # 파라미터 수 계산
                total_params = sum(p.numel() for p in model_state.values())
                print(f"📊 총 파라미터 수: {total_params:,}")
            
            if 'epoch' in checkpoint:
                print(f"🎯 훈련 에포크: {checkpoint['epoch']}")
            
            if 'val_mae' in checkpoint:
                print(f"📉 검증 MAE: {checkpoint['val_mae']:.4f}")
        
        return True, checkpoint
        
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return False, None

def find_working_checkpoints():
    """작동하는 체크포인트 파일 찾기"""
    print("🔍 작동하는 체크포인트 파일 찾기...")
    
    # 검색할 디렉토리들
    search_dirs = [
        "./mobile-vla-omniwheel",
        "./Robo+/Mobile_VLA/results",
        "./",
        "./checkpoints",
        "./models"
    ]
    
    working_checkpoints = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"📂 검색 중: {search_dir}")
            
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.pth'):
                        full_path = os.path.join(root, file)
                        
                        print(f"\n📦 체크포인트 발견: {full_path}")
                        
                        # 무결성 검사
                        if check_checkpoint_integrity(full_path):
                            # 로드 시도
                            success, checkpoint = try_load_checkpoint(full_path)
                            if success:
                                working_checkpoints.append({
                                    'path': full_path,
                                    'checkpoint': checkpoint,
                                    'size': os.path.getsize(full_path)
                                })
                                print(f"✅ 작동하는 체크포인트: {full_path}")
                            else:
                                print(f"❌ 손상된 체크포인트: {full_path}")
    
    return working_checkpoints

def create_backup_and_fix(checkpoint_path):
    """체크포인트 백업 및 복구 시도"""
    print(f"🔧 체크포인트 복구 시도: {checkpoint_path}")
    
    # 백업 생성
    backup_path = checkpoint_path + ".backup"
    try:
        shutil.copy2(checkpoint_path, backup_path)
        print(f"✅ 백업 생성: {backup_path}")
    except Exception as e:
        print(f"❌ 백업 생성 실패: {e}")
        return False
    
    # 파일 크기 확인
    original_size = os.path.getsize(checkpoint_path)
    print(f"📁 원본 파일 크기: {original_size / (1024**3):.2f} GB")
    
    # 파일 끝 부분 확인 (손상된 경우)
    try:
        with open(checkpoint_path, 'rb') as f:
            f.seek(-1024, 2)  # 파일 끝에서 1KB 앞으로
            end_data = f.read()
            print(f"📄 파일 끝 데이터 크기: {len(end_data)} bytes")
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🔧 체크포인트 복구 및 검증 도구")
    print("=" * 50)
    
    # 1. 작동하는 체크포인트 찾기
    working_checkpoints = find_working_checkpoints()
    
    print(f"\n📊 검색 결과:")
    print(f"   - 총 발견된 체크포인트: {len(working_checkpoints)}개")
    
    if working_checkpoints:
        print("\n✅ 작동하는 체크포인트 목록:")
        for i, cp in enumerate(working_checkpoints, 1):
            size_gb = cp['size'] / (1024**3)
            print(f"   {i}. {cp['path']} ({size_gb:.2f} GB)")
            
            # 체크포인트 정보 출력
            checkpoint = cp['checkpoint']
            if isinstance(checkpoint, dict):
                if 'val_mae' in checkpoint:
                    print(f"      MAE: {checkpoint['val_mae']:.4f}")
                if 'epoch' in checkpoint:
                    print(f"      Epoch: {checkpoint['epoch']}")
    else:
        print("\n❌ 작동하는 체크포인트가 없습니다.")
        
        # 손상된 체크포인트 복구 시도
        print("\n🔧 손상된 체크포인트 복구 시도...")
        damaged_checkpoints = [
            "./mobile-vla-omniwheel/best_simple_lstm_model.pth"
        ]
        
        for cp_path in damaged_checkpoints:
            if os.path.exists(cp_path):
                create_backup_and_fix(cp_path)
    
    print("\n✅ 검증 완료!")

if __name__ == "__main__":
    main()
