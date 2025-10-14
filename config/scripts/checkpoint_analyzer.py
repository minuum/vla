#!/usr/bin/env python3
"""
체크포인트 파일 분석 스크립트
"""

import torch
import os
import sys

def analyze_checkpoint(checkpoint_path):
    """체크포인트 파일 분석"""
    print(f"🔍 체크포인트 분석 중: {checkpoint_path}")
    
    # 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ 파일이 존재하지 않습니다: {checkpoint_path}")
        return
    
    # 파일 크기 확인
    file_size = os.path.getsize(checkpoint_path)
    print(f"📁 파일 크기: {file_size / (1024**3):.2f} GB")
    
    try:
        # 체크포인트 로드
        print("📦 체크포인트 로딩 중...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 체크포인트 구조 분석
        print(f"📋 체크포인트 키들: {list(checkpoint.keys())}")
        
        # 모델 상태 딕셔너리 분석
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"🧠 모델 상태 딕셔너리 키 개수: {len(model_state)}")
            print(f"🔑 첫 10개 키들: {list(model_state.keys())[:10]}")
            
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in model_state.values())
            print(f"📊 총 파라미터 수: {total_params:,}")
        
        # 훈련 정보 확인
        if 'epoch' in checkpoint:
            print(f"🎯 훈련 에포크: {checkpoint['epoch']}")
        
        if 'loss' in checkpoint:
            print(f"📉 손실값: {checkpoint['loss']:.4f}")
        
        if 'optimizer_state_dict' in checkpoint:
            print("⚙️ 옵티마이저 상태 포함됨")
        
        print("✅ 체크포인트 분석 완료!")
        
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")

def main():
    checkpoint_path = './mobile-vla-omniwheel/best_simple_lstm_model.pth'
    analyze_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
