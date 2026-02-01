#!/usr/bin/env python3
"""
모든 학습된 모델에 대한 인퍼런스 테스트 및 비교 분석
- Chunk10 모델 (Epoch 5, 7, 8, 9)
- 결과 시각화 및 성능 비교
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer


class ModelInferenceTester:
    """모델 인퍼런스 테스트 및 비교"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = []
        
    def load_model(self, checkpoint_path, config_path):
        """모델 로딩"""
        print(f"Loading: {Path(checkpoint_path).name}")
        
        try:
            trainer = MobileVLATrainer.load_from_checkpoint(
                checkpoint_path,
                config_path=config_path,
                map_location=self.device
            )
            trainer.model.to(self.device)
            trainer.model.eval()
            return trainer
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return None
    
    @torch.no_grad()
    def test_model(self, trainer, model_name, test_cases):
        """모델 테스트"""
        results = []
        
        for case in tqdm(test_cases, desc=f"Testing {model_name}"):
            try:
                # Get image and instruction
                images = case['image'].to(self.device)  # (1, 8, 3, 224, 224)
                instruction = case['instruction']
                
                # Tokenize instruction
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained('.vlms/kosmos-2-patch14-224')
                
                # Process with tokenizer
                text_inputs = processor(
                    text=[instruction],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Access backbone model
                backbone = trainer.model.model  # RoboKosMos.model property
                
                # Forward through VLM
                outputs = backbone(
                    pixel_values=images.flatten(0, 1),  # (8, 3, 224, 224)
                    input_ids=text_inputs['input_ids'].expand(8, -1),
                    attention_mask=text_inputs['attention_mask'].expand(8, -1),
                    return_dict=True
                )
                
                # Get hidden states
                hidden_states = outputs.last_hidden_state  # (8, seq_len, hidden_dim)
                
                # Extract action token (마지막 토큰)
                action_token_idx = text_inputs['attention_mask'].sum(1) - 1  # Last valid token
                action_hs = hidden_states[torch.arange(8), action_token_idx, :]  # (8, hidden_dim)
                action_hs = action_hs.unsqueeze(0).unsqueeze(2)  # (1, 8, 1, hidden_dim)
                
                # Predict action through action head
                action_pred = trainer.model.act_head(action_hs, None)  # (1, fwd_pred_next_n, 2)
                
                # Take first action prediction
                if action_pred.dim() == 3:
                    action_pred = action_pred[:, 0, :]  # (1, 2)
                
                action = action_pred.cpu().numpy()[0]  # (2,)
                
                results.append({
                    'model': model_name,
                    'instruction': case['instruction'],
                    'expected_direction': case['expected_direction'],
                    'linear_x': float(action[0]),
                    'linear_y': float(action[1]),
                    'action': action.tolist()
                })
            except Exception as e:
                print(f"  Error in test case '{case['instruction'][:30]}...': {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'model': model_name,
                    'instruction': case['instruction'],
                    'expected_direction': case['expected_direction'],
                    'linear_x': np.nan,
                    'linear_y': np.nan,
                    'action': [np.nan, np.nan]
                })
        
        return results
    
    def create_test_cases(self):
        """테스트 케이스 생성"""
        # Dummy image tensor (1, 8, 3, 224, 224)
        image_tensor = torch.randn(1, 8, 3, 224, 224)
        
        test_cases = [
            {
                'instruction': 'Navigate around obstacles and reach the front of the beverage bottle on the left',
                'expected_direction': 'left',
                'image': image_tensor
            },
            {
                'instruction': 'Navigate around obstacles and reach the front of the beverage bottle on the right',
                'expected_direction': 'right',
                'image': image_tensor
            },
            {
                'instruction': 'Move forward to the left bottle',
                'expected_direction': 'left',
                'image': image_tensor
            },
            {
                'instruction': 'Move forward to the right bottle',
                'expected_direction': 'right',
                'image': image_tensor
            },
        ]
        
        return test_cases
    
    def analyze_results(self, results):
        """결과 분석"""
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 70)
        print("📊 인퍼런스 결과 분석")
        print("=" * 70)
        
        # Model별 통계
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            print(f"\n🔹 {model}")
            print(f"  Linear X: mean={model_df['linear_x'].mean():.4f}, std={model_df['linear_x'].std():.4f}")
            print(f"  Linear Y: mean={model_df['linear_y'].mean():.4f}, std={model_df['linear_y'].std():.4f}")
            
            # Direction별 분석
            left_df = model_df[model_df['expected_direction'] == 'left']
            right_df = model_df[model_df['expected_direction'] == 'right']
            
            if len(left_df) > 0:
                print(f"  Left:  linear_y mean={left_df['linear_y'].mean():.4f}")
            if len(right_df) > 0:
                print(f"  Right: linear_y mean={right_df['linear_y'].mean():.4f}")
        
        return df
    
    def plot_results(self, df, save_dir="docs/model_comparison"):
        """결과 시각화"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Action 분포 (scatter plot)
        plt.figure(figsize=(12, 6))
        
        models = df['model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_df = df[df['model'] == model]
            
            left_df = model_df[model_df['expected_direction'] == 'left']
            right_df = model_df[model_df['expected_direction'] == 'right']
            
            if len(left_df) > 0:
                plt.scatter(left_df['linear_x'], left_df['linear_y'], 
                          marker='o', s=150, alpha=0.7, c=[colors[i]], 
                          label=f'{model} (Left)', edgecolors='black', linewidths=1.5)
            
            if len(right_df) > 0:
                plt.scatter(right_df['linear_x'], right_df['linear_y'], 
                          marker='^', s=150, alpha=0.7, c=[colors[i]], 
                          label=f'{model} (Right)', edgecolors='black', linewidths=1.5)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.xlabel('Linear X (Forward)', fontsize=12, fontweight='bold')
        plt.ylabel('Linear Y (Lateral)', fontsize=12, fontweight='bold')
        plt.title('Action Predictions Comparison (All Models)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'action_distribution.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'action_distribution.png'}")
        plt.close()
        
        # 2. Model별 Linear Y 비교 (boxplot)
        plt.figure(figsize=(12, 6))
        
        plot_data = []
        labels = []
        
        for model in models:
            model_df = df[df['model'] == model]
            plot_data.append(model_df['linear_y'].values)
            labels.append(model.replace('mobile_vla_chunk10_20251217_', ''))
        
        bp = plt.boxplot(plot_data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # 색상 설정
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.ylabel('Linear Y (Lateral)', fontsize=12, fontweight='bold')
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.title('Linear Y Distribution Across Models', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(save_dir / 'linear_y_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'linear_y_comparison.png'}")
        plt.close()
        
        # 3. Direction 별 평균 비교
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left direction
        left_means = []
        right_means = []
        model_names = []
        
        for model in models:
            model_df = df[df['model'] == model]
            left_df = model_df[model_df['expected_direction'] == 'left']
            right_df = model_df[model_df['expected_direction'] == 'right']
            
            if len(left_df) > 0:
                left_means.append(left_df['linear_y'].mean())
            else:
                left_means.append(0)
                
            if len(right_df) > 0:
                right_means.append(right_df['linear_y'].mean())
            else:
                right_means.append(0)
            
            model_names.append(model.replace('mobile_vla_chunk10_20251217_', ''))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, left_means, width, label='Left', alpha=0.8, color='skyblue')
        axes[0].bar(x + width/2, right_means, width, label='Right', alpha=0.8, color='salmon')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Linear Y (Mean)', fontweight='bold')
        axes[0].set_xlabel('Model', fontweight='bold')
        axes[0].set_title('Average Linear Y by Direction', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Separation metric (abs diff)
        separation = [abs(l - r) for l, r in zip(left_means, right_means)]
        axes[1].bar(model_names, separation, alpha=0.8, color='green')
        axes[1].set_ylabel('Separation (|Left - Right|)', fontweight='bold')
        axes[1].set_xlabel('Model', fontweight='bold')
        axes[1].set_title('Direction Separation Quality', fontweight='bold')
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'direction_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_dir / 'direction_analysis.png'}")
        plt.close()
        
        print(f"\n✅ 모든 시각화 완료: {save_dir}/")


def main():
    """메인 실행"""
    print("=" * 70)
    print("🚀 모든 모델 인퍼런스 테스트")
    print("=" * 70)
    
    # 테스터 초기화
    tester = ModelInferenceTester(device="cuda")
    
    # 모델들 정의
    models_to_test = [
        {
            'name': 'mobile_vla_chunk10_20251217_epoch05_best',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=05-val_loss=val_loss=0.284.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'mobile_vla_chunk10_20251217_epoch07',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=07-val_loss=val_loss=0.317.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'mobile_vla_chunk10_20251217_epoch08',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=08-val_loss=val_loss=0.312.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        },
        {
            'name': 'mobile_vla_chunk10_20251217_last',
            'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/last.ckpt',
            'config': 'Mobile_VLA/configs/mobile_vla_chunk10_20251217.json'
        }
    ]
    
    # 테스트 케이스 생성
    test_cases = tester.create_test_cases()
    print(f"\n📝 테스트 케이스: {len(test_cases)}개")
    
    # 모든 모델 테스트
    all_results = []
    
    for model_info in models_to_test:
        if not Path(model_info['checkpoint']).exists():
            print(f"⚠️  Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        # 모델 로딩
        trainer = tester.load_model(model_info['checkpoint'], model_info['config'])
        if trainer is None:
            continue
        
        # 테스트 실행
        results = tester.test_model(trainer, model_info['name'], test_cases)
        all_results.extend(results)
        
        # 메모리 정리
        del trainer
        torch.cuda.empty_cache()
    
    if not all_results:
        print("\n❌ 테스트 결과가 없습니다.")
        return
    
    # 결과 분석
    df = tester.analyze_results(all_results)
    
    # CSV 저장
    output_dir = Path("docs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "inference_results.csv", index=False)
    print(f"\n✅ Results saved: {output_dir / 'inference_results.csv'}")
    
    # 시각화
    tester.plot_results(df, save_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
