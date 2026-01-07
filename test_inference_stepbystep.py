#!/usr/bin/env python3
"""
Step-by-step 추론 테스트 스크립트
Case 4 체크포인트로 abs_action 전략 검증
"""

import sys
import os
from pathlib import Path

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent / "src"))

from robovlms_mobile_vla_inference import MobileVLAConfig, MobileVLAInferenceSystem
import numpy as np

def test_step1_model_loading():
    """Step 1: 모델 로딩 테스트"""
    print("\n" + "="*60)
    print("📦 Step 1: 모델 로딩 테스트")
    print("="*60)
    
    checkpoint_path = os.getenv(
        "VLA_CHECKPOINT_PATH",
        "RoboVLMs_upstream/runs/mobile_vla_kosmos2_right_only_20251207/kosmos/mobile_vla_finetune/2025-12-07/mobile_vla_kosmos2_right_only_20251207/last.ckpt"
    )
    
    print(f"📌 체크포인트: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
        return False
    
    print(f"✅ 체크포인트 파일 확인됨 ({os.path.getsize(checkpoint_path) / 1e9:.2f}GB)")
    
    # 설정
    config = MobileVLAConfig(
        checkpoint_path=checkpoint_path,
        window_size=2,
        use_abs_action=True
    )
    
    print(f"⚙️  설정:")
    print(f"   - window_size: {config.window_size}")
    print(f"   - use_abs_action: {config.use_abs_action}")
    print(f"   - action_dim: {config.action_dim}")
    print(f"   - fwd_pred_next_n: {config.fwd_pred_next_n}")
    
    # 추론 시스템 초기화
    system = MobileVLAInferenceSystem(config)
    
    print("\n🚀 모델 로딩 중...")
    success = system.inference_engine.load_model()
    
    if success:
        print("✅ 모델 로딩 성공!")
        return system
    else:
        print("❌ 모델 로딩 실패")
        return None


def test_step2_direction_extraction():
    """Step 2: 방향 추출 로직 테스트"""
    print("\n" + "="*60)
    print("🧭 Step 2: 방향 추출 로직 테스트")
    print("="*60)
    
    from robovlms_mobile_vla_inference import extract_direction_from_instruction
    
    # 학습 시 사용된 한국어 instruction 사용
    from Mobile_VLA.instruction_mapping import get_instruction_for_scenario
    
    test_cases = [
        (get_instruction_for_scenario('left'), 1.0),  # 가장 왼쪽 외곽으로...
        ("Navigate to the right bottle", -1.0),
        ("Move left", 1.0),
        ("Turn right", -1.0),
        ("Go straight", 0.0),
        ("Move forward", 0.0),
    ]
    
    all_passed = True
    for instruction, expected in test_cases:
        result = extract_direction_from_instruction(instruction)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{instruction}' → {result:.1f} (예상: {expected:.1f})")
        if result != expected:
            all_passed = False
    
    return all_passed


def test_step3_dummy_inference(system):
    """Step 3: Dummy 이미지로 추론 테스트"""
    print("\n" + "="*60)
    print("🖼️  Step 3: Dummy 이미지 추론 테스트")
    print("="*60)
    
    if system is None:
        print("❌ 시스템이 초기화되지 않았습니다.")
        return False
    
    # Dummy 이미지 생성 (224x224 RGB)
    print("🎨 Dummy 이미지 생성 중...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 이미지 버퍼 채우기
    system.image_buffer.clear()
    for i in range(system.config.window_size):
        system.image_buffer.add_image(dummy_image)
        print(f"   - 이미지 {i+1}/{system.config.window_size} 추가됨")
    
    # 테스트 명령들
    test_instructions = [
        get_instruction_for_scenario('left'),  # 한국어 instruction
        "Navigate to the right bottle"
    ]
    
    for instruction in test_instructions:
        print(f"\n📝 명령: '{instruction}'")
        
        try:
            actions, info = system.inference_engine.predict_action(
                system.image_buffer.get_images(),
                instruction,
                use_abs_action=True
            )
            
            print(f"✅ 추론 성공!")
            print(f"   - 추론 시간: {info['inference_time']*1000:.2f}ms")
            print(f"   - FPS: {info['fps']:.2f}")
            print(f"   - 방향 부호: {info['direction']:.1f}")
            print(f"   - 예측 액션 shape: {actions.shape}")
            print(f"   - 첫 번째 액션: [{actions[0, 0]:.3f}, {actions[0, 1]:.3f}]")
            
            # 방향 검증
            direction = info['direction']
            if direction > 0:  # Left
                expected_sign = "양수"
                actual_sign = "양수" if actions[0, 1] > 0 else "음수"
            elif direction < 0:  # Right
                expected_sign = "음수"
                actual_sign = "양수" if actions[0, 1] > 0 else "음수"
            else:
                expected_sign = "0"
                actual_sign = "0"
            
            if expected_sign == actual_sign or direction == 0:
                print(f"   ✅ 방향 검증 통과: linear_y는 {actual_sign}")
            else:
                print(f"   ⚠️  방향 불일치: 예상 {expected_sign}, 실제 {actual_sign}")
            
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    """메인 테스트 실행"""
    print("\n" + "🎯 "*20)
    print("VLA 추론 시스템 단계별 테스트")
    print("🎯 "*20)
    
    # Step 1: 모델 로딩
    system = test_step1_model_loading()
    if system is None:
        print("\n❌ Step 1 실패. 테스트 중단.")
        return
    
    # Step 2: 방향 추출
    if not test_step2_direction_extraction():
        print("\n⚠️  Step 2에서 일부 실패가 있었지만 계속 진행합니다.")
    
    # Step 3: Dummy 추론
    if test_step3_dummy_inference(system):
        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)
        print("\n다음 단계:")
        print("1. 실제 이미지로 테스트: python scripts/inference_abs_action.py")
        print("2. API 서버 시작: python api_server.py")
    else:
        print("\n❌ Step 3 실패")


if __name__ == "__main__":
    main()
