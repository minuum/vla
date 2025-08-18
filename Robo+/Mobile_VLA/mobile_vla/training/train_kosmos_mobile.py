#!/usr/bin/env python3
"""
Mobile VLA + Kosmos 학습 런처
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
import importlib.util
from pathlib import Path


def _load_attr_from_file(module_name: str, file_path: str, attr: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return getattr(module, attr)


def parse_args():
    parser = argparse.ArgumentParser(description="Mobile VLA + Kosmos 학습")
    parser.add_argument('--data_dir', type=str, 
                       default=os.getenv('MOBILE_VLA_DATA_DIR', '/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/'))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=18)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--max_steps', type=int, default=10, help='스모크 테스트용 스텝 수')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--freeze_kosmos', action='store_true', default=True, 
                       help='Kosmos 가중치 고정 (정책 헤드만 학습)')
    parser.add_argument('--kosmos_model', type=str, default="microsoft/kosmos-2-patch14-224")
    parser.add_argument('--scenario_filter', nargs='*', 
                       help='학습할 시나리오 필터 (예: 1box_vert_left 2box_hori_right)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🤖 Mobile VLA + Kosmos 학습 설정:")
    print(f"   데이터: {args.data_dir}")
    print(f"   Kosmos 모델: {args.kosmos_model}")
    print(f"   Kosmos 고정: {args.freeze_kosmos}")
    print(f"   배치 크기: {args.batch_size}")
    print(f"   최대 스텝: {args.max_steps}")
    
    configs = {
        'kosmos_model_name': args.kosmos_model,
        'hidden_size': args.hidden_size,
        'freeze_kosmos': args.freeze_kosmos,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'max_epochs': 1,
    }
    
    # RoboVLMs 어댑터 로드 (동적 임포트)
    root = Path(__file__).resolve().parents[1]
    adapter_cls = _load_attr_from_file(
        'robovlms_adapter',
        str(root / 'data' / 'robovlms_adapter.py'),
        'MobileVLAToRoboVLMsAdapter'
    )
    
    # Kosmos processor 로드
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.kosmos_model)
    
    # 어댑터 초기화
    adapter = adapter_cls(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        scenario_filter=args.scenario_filter,
        image_processor=processor.image_processor  # Kosmos processor 전달
    )
    
    if len(adapter) == 0:
        print('⚠️ HDF5 데이터가 없습니다. data_dir를 확인하세요:', args.data_dir)
        return
    
    print(f"📊 데이터셋: {len(adapter)}개 에피소드")
    print(f"🎯 시나리오 분포: {adapter.get_scenario_statistics()}")
    
    # DataLoader 생성
    def collate_fn(batch):
        # 단일 배치 처리
        return batch[0] if len(batch) == 1 else batch[0]
    
    loader = DataLoader(
        adapter,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 다중 프로세싱 비활성화
        collate_fn=collate_fn
    )
    
    # Kosmos 트레이너 로드 (동적 임포트)
    trainer_cls = _load_attr_from_file(
        'kosmos_trainer',
        str(root / 'training' / 'kosmos_trainer.py'),
        'MobileKosmosTrainer'
    )
    trainer = trainer_cls(configs)
    
    # 스모크 학습 루프
    print("\n🚀 학습 시작...")
    steps = 0
    
    try:
        for batch_idx, batch in enumerate(loader):
            # 학습 스텝 실행
            result = trainer.train_step(batch)
            
            # 결과 출력
            result_str = json.dumps({
                'step': steps,
                'scenario': batch.get('scenario', 'unknown'),
                **{k: float(v) if isinstance(v, (int, float)) else v for k, v in result.items()}
            }, ensure_ascii=False, indent=2)
            
            print(f"\n📈 Step {steps}:")
            print(result_str)
            
            steps += 1
            if steps >= args.max_steps:
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f'\n✅ 스모크 학습 완료: {steps}개 스텝')
    
    # 모델 저장 (옵션)
    save_path = root / "experiments" / f"kosmos_mobile_vla_steps_{steps}.pt"
    save_path.parent.mkdir(exist_ok=True)
    trainer.save_model(str(save_path))
    print(f"💾 모델 저장: {save_path}")


if __name__ == '__main__':
    main()
