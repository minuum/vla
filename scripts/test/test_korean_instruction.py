#!/usr/bin/env python3
"""
Korean Instruction Inference Test Suite
학습 시 사용된 한국어 instruction으로 모델 추론이 올바르게 작동하는지 검증

Tests:
1. DataLoader consistency (학습 데이터와 동일한 instruction 사용 여부)
2. Inference with Korean instruction (Left/Right 시나리오)
3. Direction detection accuracy
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))


def test_dataloader_instruction():
    """Test 1: 데이터로더가 한국어 instruction을 사용하는지 확인"""
    print("\n" + "=" * 60)
    print("TEST 1: DataLoader Instruction Language")
    print("=" * 60)
    
    from robovlms.data.mobile_vla_action_dataset import MobileVLAActionDataset
    
    dataset = MobileVLAActionDataset(
        data_dir='/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset',
        model_name='kosmos',
        mode='train'
    )
    
    # 첫 번째 샘플 가져오기
    sample = dataset[0]
    
    # task_description이 있는지 확인 (batch_transform 후)
    if 'text' in sample:
        instruction = sample['text']
    elif hasattr(sample, 'get'):
        instruction = sample.get('task_description', 'NOT FOUND')
    else:
        instruction = 'NOT FOUND'
    
    print(f"Sample Instruction: {instruction}")
    
    # 한국어 확인
    is_korean = '가장' in str(instruction) or '컵까지' in str(instruction)
    
    if is_korean:
        print("✓ PASS: DataLoader uses Korean instructions")
        return True
    else:
        print("✗ FAIL: DataLoader does NOT use Korean instructions")
        return False


def test_inference_with_korean():
    """Test 2: 한국어 instruction으로 추론 실행"""
    print("\n" + "=" * 60)
    print("TEST 2: Inference with Korean Instructions")
    print("=" * 60)
    
    from Mobile_VLA.instruction_mapping import get_instruction_for_scenario
    from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline
    from PIL import Image
    
    # Setup
    checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    
    if not Path(checkpoint_path).exists():
        print(f"⚠ SKIP: Checkpoint not found")
        return None
    
    # Load model
    print("Loading model...")
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cuda"
    )
    
    # Test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Test scenarios
    results = {}
    for scenario, expected_sign in [('left', 1), ('right', -1)]:
        instruction = get_instruction_for_scenario(scenario)
        print(f"\n[{scenario.upper()}] Instruction: {instruction}")
        
        result = pipeline.predict(test_image, instruction)
        action = result['action']
        
        # Chunking이므로 첫 번째 액션만 확인
        if len(action.shape) > 1:
            action = action[0]
        
        linear_y = action[1] if len(action) > 1 else 0
        
        print(f"  Action: {action}")
        print(f"  linear_y: {linear_y:.4f}")
        
        # 방향 검증
        actual_sign = 1 if linear_y > 0 else -1
        is_correct = (actual_sign == expected_sign)
        
        if is_correct:
            print(f"  ✓ PASS: Correct direction")
        else:
            print(f"  ✗ FAIL: Expected {expected_sign}, got {actual_sign}")
        
        results[scenario] = is_correct
    
    # Overall result
    all_pass = all(results.values())
    if all_pass:
        print(f"\n✓ PASS: All scenarios correct")
    else:
        print(f"\n✗ FAIL: Some scenarios incorrect")
    
    return all_pass


def test_instruction_mapping():
    """Test 3: instruction_mapping 모듈 동작 확인"""
    print("\n" + "=" * 60)
    print("TEST 3: Instruction Mapping Module")
    print("=" * 60)
    
    from Mobile_VLA.instruction_mapping import (
        get_instruction_for_scenario,
        get_instruction_for_robot_id,
        SCENARIO_INSTRUCTIONS_KO
    )
    
    # Test cases
    test_cases = [
        ('left', '가장 왼쪽'),
        ('right', '가장 오른쪽'),
        ('1', '가장 왼쪽'),
        ('2', '가장 오른쪽'),
        ('1box_hori_left', '가장 왼쪽'),
    ]
    
    all_pass = True
    for input_val, expected_substring in test_cases:
        result = get_instruction_for_scenario(input_val)
        is_correct = expected_substring in result
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {input_val:20s} -> {result}")
        
        if not is_correct:
            all_pass = False
    
    if all_pass:
        print("\n✓ PASS: All mappings correct")
    else:
        print("\n✗ FAIL: Some mappings incorrect")
    
    return all_pass


def main():
    print("\n" + "=" * 70)
    print("Korean Instruction Inference Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    try:
        results['mapping'] = test_instruction_mapping()
    except Exception as e:
        print(f"\n✗ ERROR in Test 3: {e}")
        results['mapping'] = False
    
    try:
        results['dataloader'] = test_dataloader_instruction()
    except Exception as e:
        print(f"\n✗ ERROR in Test 1: {e}")
        results['dataloader'] = False
    
    try:
        results['inference'] = test_inference_with_korean()
    except Exception as e:
        print(f"\n✗ ERROR in Test 2: {e}")
        import traceback
        traceback.print_exc()
        results['inference'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⚠ SKIP"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{status}: {test_name}")
    
    # Overall
    passed_tests = sum(1 for v in results.values() if v is True)
    total_tests = len([v for v in results.values() if v is not None])
    
    print(f"\nPassed: {passed_tests}/{total_tests}")
    
    if all(v for v in results.values() if v is not None):
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
