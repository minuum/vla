#!/usr/bin/env python3
"""
20251106 에피소드 데이터셋 테스트 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.mobile_vla_h5_dataset import test_mobile_vla_h5_dataset

if __name__ == "__main__":
    print("🧪 20251106 에피소드 데이터셋 테스트 시작\n")
    
    success = test_mobile_vla_h5_dataset()
    
    if success:
        print("\n✅ 데이터셋 테스트 성공!")
        print("📋 다음 단계: LoRA Fine-tuning 실행")
        print("   bash Mobile_VLA/scripts/run_lora_finetune_20251106.sh")
    else:
        print("\n❌ 데이터셋 테스트 실패")
        sys.exit(1)

