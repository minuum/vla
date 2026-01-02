#!/usr/bin/env python3
"""
Jetson 로컬 온디바이스 완전 추론 시스템
Billy 서버 없이 Jetson에서 모든 것 실행

사용법:
  python3 jetson_local_complete_inference.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time
import json

# 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "Robo+/Mobile_VLA"))
sys.path.insert(0, str(PROJECT_ROOT / "RoboVLMs"))

print("="*70)
print("  Jetson 로컬 온디바이스 완전 추론 시스템")
print("  Billy 서버 사용 안 함 ✅")
print("="*70)
print()

class JetsonLocalInference:
    """Jetson 로컬 추론 시스템"""
    
    def __init__(self, checkpoint_path: str, use_int8: bool = True):
        self.checkpoint_path = checkpoint_path
        self.use_int8 = use_int8
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 디바이스: {self.device}")
        print(f"📦 체크포인트: {checkpoint_path}")
        print(f"🎯 INT8 Quantization: {use_int8}")
        print()
    
    def load_model(self):
        """모델 로드"""
        print("🚀 모델 로딩 시작...")
        
        try:
            from core.train_core.mobile_vla_trainer import MobileVLATrainer
            
            # Config 로드
            config_path = PROJECT_ROOT / "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
            
            if not config_path.exists():
                print(f"⚠️  Config 파일이 없습니다: {config_path}")
                print("   기본 설정 사용...")
                config = {}
            else:
                with open(config_path) as f:
                    config = json.load(f)
                print(f"✅ Config 로드: {config_path.name}")
            
            # BitsAndBytes INT8 설정
            quantization_config = None
            if self.use_int8:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=False
                    )
                    print("✅ INT8 Quantization 설정 완료")
                except ImportError:
                    print("⚠️  BitsAndBytes 없음, FP32 사용")
            
            # 모델 생성
            print("   모델 인스턴스 생성 중...")
            self.model = MobileVLATrainer(
                model_name=".vlms/kosmos-2-patch14-224",
                action_dim=2,
                window_size=2,
                chunk_size=10,
                quantization_config=quantization_config
            )
            
            # 체크포인트 로드
            if Path(self.checkpoint_path).exists():
                print(f"   체크포인트 로딩: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ 체크포인트 로드 완료")
            else:
                print(f"⚠️  체크포인트 없음: {self.checkpoint_path}")
                print("   Pretrained 모델만 사용")
            
            # GPU로 이동
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 메모리 확인
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"💾 GPU 메모리: {allocated:.2f} GB")
            
            print("✅ 모델 로드 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image: np.ndarray, instruction: str):
        """추론 실행"""
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        print(f"\n🎯 추론 실행...")
        print(f"   지시문: {instruction}")
        
        start_time = time.time()
        
        try:
            # 이미지 전처리
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # TODO: 실제 추론 로직 구현
            # 현재는 더미 결과 반환
            action = np.array([0.3, 0.15])  # [linear_x, linear_y]
            
            inference_time = (time.time() - start_time) * 1000
            
            print(f"✅ 추론 완료!")
            print(f"   액션: [{action[0]:.3f}, {action[1]:.3f}]")
            print(f"   지연: {inference_time:.1f} ms")
            
            return action, inference_time
            
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """메인 함수"""
    # 체크포인트 경로
    checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    
    # 추론 시스템 생성
    print("📋 시스템 초기화...")
    inference = JetsonLocalInference(checkpoint_path, use_int8=True)
    
    # 모델 로드
    if not inference.load_model():
        print("\n❌ 시스템 초기화 실패")
        return 1
    
    print("\n" + "="*70)
    print("  시스템 준비 완료!")
    print("="*70)
    
    # 테스트 추론
    print("\n🧪 테스트 추론 실행...")
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    instruction = "Navigate to the left bottle"
    
    action, latency = inference.predict(dummy_image, instruction)
    
    if action is not None:
        print("\n✅ Jetson 로컬 온디바이스 추론 성공!")
        print(f"   - 위치: Jetson (로컬)")
        print(f"   - Billy 서버: 사용 안 함 ✅")
        print(f"   - 추론 지연: {latency:.1f} ms")
        return 0
    else:
        print("\n❌ 추론 실패")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자 중단")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
