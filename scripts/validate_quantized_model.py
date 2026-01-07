#!/usr/bin/env python3
"""
양자화 모델 검증 스크립트

양자화 후 Direction Accuracy 및 Latency 검증

Usage:
    python scripts/validate_quantized_model.py \
        --original runs/.../chunk5_epoch6.ckpt \
        --quantized quantized_models/full_quant/model_quantized.ckpt \
        --config Mobile_VLA/configs/mobile_vla_chunk5_20251217.json \
        --val-data /home/soda/25-1kp/vla/ROS_action/mobile_vla_dataset

검증 기준:
    - Direction Accuracy ≥ 95% (원본 100% 대비)
    - Latency ≤ 500ms (원본 ~385ms 대비)
    - Memory < 6GB
"""

import torch
import sys
import argparse
import time
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer


class ModelValidator:
    """양자화 모델 검증 클래스"""
    
    def __init__(
        self,
        original_ckpt: str,
        quantized_ckpt: str,
        config_path: str,
        val_data_dir: str
    ):
        self.config_path = config_path
        self.val_data_dir = Path(val_data_dir)
        
        print("📦 Loading original model...")
        self.model_orig = MobileVLATrainer.load_from_checkpoint(
            original_ckpt,
            config_path=config_path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model_orig.eval()
        
        print("📦 Loading quantized model...")
        self.model_quant = MobileVLATrainer.load_from_checkpoint(
            quantized_ckpt,
            config_path=config_path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model_quant.eval()
        
        print("✅ Models loaded")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_validation_data(self, num_samples=100):
        """검증 데이터 로드"""
        print(f"\n🔧 Loading validation data ({num_samples} samples)...")
        
        val_samples = []
        
        # H5 파일에서 샘플 추출
        h5_files = sorted(self.val_data_dir.glob("episode_*.h5"))
        
        count = 0
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Mobile VLA 데이터셋 구조: images, actions, language_instruction
                    images = f['images'][:]  # (T, H, W, C)
                    actions = f['actions'][:]  # (T, action_dim)
                    
                    # Language instruction 추출 (파일명 기반)
                    if 'left' in h5_file.name:
                        instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
                        expected_direction = -1  # left = negative y
                    elif 'right' in h5_file.name:
                        instruction = "Navigate around obstacles and reach the front of the beverage bottle on the right"
                        expected_direction = 1  # right = positive y
                    else:
                        # 파일명에서 방향 추출 못하면 스킵
                        continue
                    
                    # 랜덤 샘플링
                    num_frames = min(len(images), 10)
                    if num_frames == 0:
                        continue
                        
                    indices = np.random.choice(len(images), num_frames, replace=False)
                    
                    for idx in indices:
                        if count >= num_samples:
                            break
                        
                        img = images[idx]  # (720, 1280, 3)
                        action = actions[idx]  # (action_dim,)
                        
                        val_samples.append({
                            'image': img,
                            'instruction': instruction,
                            'action': action,
                            'expected_direction': expected_direction
                        })
                        count += 1
                    
                    if count >= num_samples:
                        break
            except Exception as e:
                print(f"⚠️  Skipping {h5_file.name}: {e}")
                continue
        
        self.val_samples = val_samples[:num_samples]
        print(f"✅ Loaded {len(self.val_samples)} validation samples")
        
        return self.val_samples
    
    def preprocess_image(self, image_np):
        """이미지 전처리"""
        # Numpy -> PIL -> Tensor
        img_pil = Image.fromarray(image_np.astype('uint8'))
        
        # Resize and normalize (Kosmos-2 기준)
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        img_tensor = transform(img_pil)
        
        # Add batch and window dimensions: (3, 224, 224) -> (1, 1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def evaluate_direction_accuracy(self):
        """Direction Accuracy 평가"""
        print("\n📊 Evaluating Direction Accuracy...")
        
        correct_orig = 0
        correct_quant = 0
        total = len(self.val_samples)
        
        latency_orig = []
        latency_quant = []
        
        with torch.no_grad():
            for sample in tqdm(self.val_samples, desc="Validating"):
                img_tensor = self.preprocess_image(sample['image'])
                instruction = sample['instruction']
                expected_dir = sample['expected_direction']
                
                # Original model prediction
                start = time.time()
                pred_orig = self.model_orig.model.inference(
                    vision_x=img_tensor,
                    lang_x=instruction
                )
                latency_orig.append((time.time() - start) * 1000)
                
                # Quantized model prediction
                start = time.time()
                pred_quant = self.model_quant.model.inference(
                    vision_x=img_tensor,
                    lang_x=instruction
                )
                latency_quant.append((time.time() - start) * 1000)
                
                # Extract actions
                if isinstance(pred_orig, tuple):
                    action_orig = pred_orig[0].cpu().numpy()[0]
                else:
                    action_orig = pred_orig.cpu().numpy()[0]
                
                if isinstance(pred_quant, tuple):
                    action_quant = pred_quant[0].cpu().numpy()[0]
                else:
                    action_quant = pred_quant.cpu().numpy()[0]
                
                # Check direction (linear_y sign vs expected_direction)
                # action[1] is linear_y (left/right)
                if np.sign(action_orig[1]) == np.sign(expected_dir):
                    correct_orig += 1
                
                if np.sign(action_quant[1]) == np.sign(expected_dir):
                    correct_quant += 1
        
        accuracy_orig = correct_orig / total if total > 0 else 0
        accuracy_quant = correct_quant / total if total > 0 else 0
        accuracy_drop = accuracy_orig - accuracy_quant
        
        avg_latency_orig = np.mean(latency_orig) if latency_orig else 0
        avg_latency_quant = np.mean(latency_quant) if latency_quant else 0
        
        results = {
            'direction_accuracy': {
                'original': accuracy_orig,
                'quantized': accuracy_quant,
                'drop': accuracy_drop
            },
            'latency_ms': {
                'original': avg_latency_orig,
                'quantized': avg_latency_quant,
                'speedup': avg_latency_orig / avg_latency_quant if avg_latency_quant > 0 else 0
            }
        }
        
        return results
    
    def measure_memory_usage(self):
        """메모리 사용량 측정"""
        print("\n💾 Measuring GPU Memory Usage...")
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping memory measurement")
            return None
        
        # Original model
        torch.cuda.reset_peak_memory_stats()
        sample = self.val_samples[0]
        img_tensor = self.preprocess_image(sample['image'])
        
        with torch.no_grad():
            _ = self.model_orig.model.inference(
                vision_x=img_tensor,
                lang_x=sample['instruction']
            )
        
        mem_orig = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Quantized model
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = self.model_quant.model.inference(
                vision_x=img_tensor,
                lang_x=sample['instruction']
            )
        
        mem_quant = torch.cuda.max_memory_allocated() / (1024**3)
        
        memory_results = {
            'original_gb': mem_orig,
            'quantized_gb': mem_quant,
            'reduction_gb': mem_orig - mem_quant,
            'reduction_percent': (mem_orig - mem_quant) / mem_orig * 100
        }
        
        return memory_results
    
    def print_results(self, results, memory_results):
        """결과 출력"""
        print("\n" + "=" * 60)
        print("📊 Validation Results")
        print("=" * 60)
        
        # Direction Accuracy
        print("\n🎯 Direction Accuracy:")
        print(f"  Original:  {results['direction_accuracy']['original']:.2%}")
        print(f"  Quantized: {results['direction_accuracy']['quantized']:.2%}")
        print(f"  Drop:      {results['direction_accuracy']['drop']:.2%}")
        
        if results['direction_accuracy']['quantized'] >= 0.95:
            print("  ✅ PASS (≥ 95%)")
        else:
            print("  ❌ FAIL (< 95%)")
        
        # Latency
        print("\n⏱️  Latency:")
        print(f"  Original:  {results['latency_ms']['original']:.1f} ms")
        print(f"  Quantized: {results['latency_ms']['quantized']:.1f} ms")
        print(f"  Speedup:   {results['latency_ms']['speedup']:.2f}x")
        
        if results['latency_ms']['quantized'] <= 500:
            print("  ✅ PASS (≤ 500ms)")
        else:
            print("  ❌ FAIL (> 500ms)")
        
        # Memory
        if memory_results:
            print("\n💾 Memory Usage:")
            print(f"  Original:  {memory_results['original_gb']:.2f} GB")
            print(f"  Quantized: {memory_results['quantized_gb']:.2f} GB")
            print(f"  Reduction: {memory_results['reduction_gb']:.2f} GB ({memory_results['reduction_percent']:.1f}%)")
            
            if memory_results['quantized_gb'] < 6.0:
                print("  ✅ PASS (< 6GB)")
            else:
                print("  ⚠️  WARNING (≥ 6GB)")
        
        print("\n" + "=" * 60)
    
    def save_results(self, results, memory_results, output_path):
        """결과 저장"""
        output = {
            'validation_results': results,
            'memory_results': memory_results,
            'pass_criteria': {
                'direction_accuracy': results['direction_accuracy']['quantized'] >= 0.95,
                'latency': results['latency_ms']['quantized'] <= 500,
                'memory': memory_results['quantized_gb'] < 6.0 if memory_results else None
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate quantized Mobile VLA model")
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original checkpoint'
    )
    parser.add_argument(
        '--quantized',
        type=str,
        required=True,
        help='Path to quantized checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        required=True,
        help='Path to validation dataset'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results.json',
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Quantized Model Validation")
    print("=" * 60)
    
    # Validator 초기화
    validator = ModelValidator(
        original_ckpt=args.original,
        quantized_ckpt=args.quantized,
        config_path=args.config,
        val_data_dir=args.val_data
    )
    
    # 검증 데이터 로드
    validator.load_validation_data(num_samples=args.num_samples)
    
    # Direction Accuracy 평가
    results = validator.evaluate_direction_accuracy()
    
    # 메모리 사용량 측정
    memory_results = validator.measure_memory_usage()
    
    # 결과 출력
    validator.print_results(results, memory_results)
    
    # 결과 저장
    validator.save_results(results, memory_results, args.output)
    
    print("\n✅ Validation completed!")


if __name__ == "__main__":
    main()
