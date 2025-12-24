#!/usr/bin/env python3
"""
Jetson 로컬 온디바이스 추론 테스트
Billy 서버 없이 Jetson에서 직접 실행
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time

# 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "Robo+/Mobile_VLA"))
sys.path.insert(0, str(project_root / "RoboVLMs"))

print("="*60)
print("  Jetson 로컬 온디바이스 추론 테스트")
print("="*60)
print()

# Config 임포트
try:
    from core.train_core.mobile_vla_trainer import MobileVLATrainer
    print("✅ MobileVLATrainer 임포트 성공")
except Exception as e:
    print(f"❌ MobileVLATrainer 임포트 실패: {e}")
    print("\n대안: BitsAndBytes 직접 로딩 시도...")
    
    # 대안: transformers + BitsAndBytes 직접 사용
    try:
        from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
        print("✅ Transformers 임포트 성공")
        
        # BitsAndBytes config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        print("\n🔧 Kosmos-2 모델 로딩 (INT8)...")
        model = AutoModelForVision2Seq.from_pretrained(
            ".vlms/kosmos-2-patch14-224",
            quantization_config=bnb_config,
            device_map="auto"
        )
        print("✅ 모델 로드 성공!")
        print(f"   Device: {model.device}")
        
        # GPU 메모리 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU 메모리: {allocated:.2f} GB")
        
        print("\n✅ Jetson 로컬 추론 준비 완료!")
        print("   - 모델: Kosmos-2 (INT8 Quantized)")
        print("   - 위치: Jetson 로컬")
        print("   - Billy 서버: 사용 안 함 ✅")
        
    except Exception as e2:
        print(f"❌ 대안도 실패: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("  테스트 완료!")
print("="*60)
