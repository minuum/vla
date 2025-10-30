#!/usr/bin/env python3
"""
두 모델 양자화 성능 비교 테스트
MAE 0.222 (Kosmos2) vs MAE 0.212 (CLIP) 양자화 비교
"""

import torch
import torch.nn as nn
import time
import json
import logging
from transformers import CLIPProcessor, CLIPModel
import os
import gc
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantizationComparisonTest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = 100  # 더 정확한 측정을 위해 증가
        
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _create_kosmos2_model(self):
        """MAE 0.222 모델 구조 (Kosmos2 Vision 기반)"""
        class Kosmos2VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Kosmos2 Vision 모델만 사용 (이미지 인코딩)
                from transformers import AutoProcessor, AutoModel
                self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
                self.kosmos2 = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
                self.kosmos2.eval()
                for param in self.kosmos2.parameters():
                    param.requires_grad = False
                
                # 실제 MAE 0.222 모델의 Action Head 구조
                self.rnn = nn.RNN(
                    input_size=2048,  # Kosmos2 출력 크기
                    hidden_size=4096,
                    num_layers=4,
                    batch_first=True,
                    dropout=0.1
                )
                self.actions = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                )
            
            def forward(self, x):
                batch_size = x.size(0)
                with torch.no_grad():
                    # Kosmos2 Vision 모델로 이미지 인코딩
                    # 더미 텍스트 입력 생성 (Kosmos2 요구사항)
                    dummy_text = torch.zeros(batch_size, 1, dtype=torch.long).to(x.device)
                    
                    outputs = self.kosmos2(
                        pixel_values=x,
                        input_ids=dummy_text,
                        attention_mask=torch.ones_like(dummy_text)
                    )
                    
                    # Vision features 추출 (마지막 hidden state의 첫 번째 토큰)
                    image_features = outputs.last_hidden_state[:, 0, :]
                    
                sequence_features = image_features.unsqueeze(1)
                rnn_out, _ = self.rnn(sequence_features)
                actions = self.actions(rnn_out.squeeze(1))
                return actions
        
        return Kosmos2VisionModel()
    
    def _create_clip_model(self):
        """MAE 0.212 모델 구조 (CLIP 기반)"""
        class CLIPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip.eval()
                for param in self.clip.parameters():
                    param.requires_grad = False
                
                # CLIP 기반 Action Head 구조
                self.rnn = nn.RNN(
                    input_size=512,  # CLIP 출력 크기
                    hidden_size=4096,
                    num_layers=4,
                    batch_first=True,
                    dropout=0.1
                )
                self.actions = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                )
            
            def forward(self, x):
                batch_size = x.size(0)
                with torch.no_grad():
                    # CLIP 이미지 인코딩
                    image_features = self.clip.get_image_features(pixel_values=x)
                    
                sequence_features = image_features.unsqueeze(1)
                rnn_out, _ = self.rnn(sequence_features)
                actions = self.actions(rnn_out.squeeze(1))
                return actions
        
        return CLIPModel()
    
    def _create_quantized_model(self, base_model, quantization_type="fp16"):
        """양자화된 모델 생성"""
        class QuantizedModel(nn.Module):
            def __init__(self, base_model, quantization_type):
                super().__init__()
                self.base_model = base_model
                self.quantization_type = quantization_type
                
                # 양자화 적용
                if quantization_type == "fp16":
                    self.base_model = self.base_model.half()
                elif quantization_type == "int8":
                    # 동적 양자화
                    self.base_model = torch.quantization.quantize_dynamic(
                        self.base_model, {nn.Linear, nn.RNN}, dtype=torch.qint8
                    )
            
            def forward(self, x):
                if self.quantization_type == "fp16":
                    x = x.half()
                return self.base_model(x)
        
        return QuantizedModel(base_model, quantization_type)
    
    def _benchmark_model(self, model, name, input_data):
        """모델 벤치마크 수행"""
        logger.info(f"📊 {name} 벤치마크 시작...")
        
        model = model.to(self.device)
        model.eval()
        
        # 워밍업
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_data)
        
        # 정확한 메모리 측정
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        # 추론 시간 측정
        times = []
        outputs = []
        
        for _ in range(self.num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
            outputs.append(output.detach().cpu())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_used = (end_memory - start_memory) / (1024 ** 2)  # MB
            peak_memory_used = peak_memory / (1024 ** 2)  # MB
        else:
            memory_used = 0
            peak_memory_used = 0
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        logger.info(f"   추론 시간: {avg_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"   메모리 사용량: {memory_used:.2f} MB")
        logger.info(f"   최대 메모리: {peak_memory_used:.2f} MB")
        logger.info(f"   FPS: {fps:.2f}")
        
        return {
            'inference_time_ms': avg_time,
            'inference_time_std': std_time,
            'memory_usage_mb': memory_used,
            'peak_memory_mb': peak_memory_used,
            'fps': fps,
            'outputs': outputs
        }
    
    def _compare_outputs(self, original_outputs, quantized_outputs):
        """출력 정확도 비교"""
        if not original_outputs or not quantized_outputs:
            return "비교 불가"
        
        try:
            original_tensor = torch.stack(original_outputs)
            quantized_tensor = torch.stack(quantized_outputs)
            
            mae = torch.mean(torch.abs(original_tensor - quantized_tensor)).item()
            correlation = torch.corrcoef(torch.stack([original_tensor.flatten(), quantized_tensor.flatten()]))[0, 1].item()
            accuracy_01 = torch.mean((torch.abs(original_tensor - quantized_tensor) < 0.1).float()).item()
            accuracy_001 = torch.mean((torch.abs(original_tensor - quantized_tensor) < 0.01).float()).item()
            
            return {
                'mae': mae,
                'correlation': correlation,
                'accuracy_01': accuracy_01,
                'accuracy_001': accuracy_001
            }
            
        except Exception as e:
            return f"비교 오류: {e}"
    
    def run_comparison(self):
        """두 모델 양자화 성능 비교 실행"""
        logger.info("🚀 두 모델 양자화 성능 비교 시작")
        
        # 입력 데이터 생성
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        results = {}
        
        # 1. Kosmos2 모델 (MAE 0.222) 테스트
        logger.info("\n" + "="*60)
        logger.info("🎯 MAE 0.222 모델 (Kosmos2) 양자화 테스트")
        logger.info("="*60)
        
        kosmos2_original = self._create_kosmos2_model()
        kosmos2_fp16 = self._create_quantized_model(self._create_kosmos2_model(), "fp16")
        
        # 원본 모델 벤치마크
        kosmos2_original_results = self._benchmark_model(kosmos2_original, "Kosmos2 원본", input_data)
        
        # 메모리 정리
        del kosmos2_original
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # FP16 양자화 모델 벤치마크
        kosmos2_fp16_results = self._benchmark_model(kosmos2_fp16, "Kosmos2 FP16", input_data)
        
        # 메모리 정리
        del kosmos2_fp16
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Kosmos2 결과 저장
        results['kosmos2'] = {
            'original': kosmos2_original_results,
            'fp16': kosmos2_fp16_results,
            'improvement': {
                'speedup': kosmos2_original_results['inference_time_ms'] / kosmos2_fp16_results['inference_time_ms'],
                'memory_save': 0 if kosmos2_original_results['memory_usage_mb'] == 0 else 
                    (kosmos2_original_results['memory_usage_mb'] - kosmos2_fp16_results['memory_usage_mb']) / kosmos2_original_results['memory_usage_mb'] * 100,
                'fps_improvement': kosmos2_fp16_results['fps'] / kosmos2_original_results['fps']
            }
        }
        
        # 2. CLIP 모델 (MAE 0.212) 테스트
        logger.info("\n" + "="*60)
        logger.info("🎯 MAE 0.212 모델 (CLIP) 양자화 테스트")
        logger.info("="*60)
        
        clip_original = self._create_clip_model()
        clip_fp16 = self._create_quantized_model(self._create_clip_model(), "fp16")
        
        # 원본 모델 벤치마크
        clip_original_results = self._benchmark_model(clip_original, "CLIP 원본", input_data)
        
        # 메모리 정리
        del clip_original
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # FP16 양자화 모델 벤치마크
        clip_fp16_results = self._benchmark_model(clip_fp16, "CLIP FP16", input_data)
        
        # 메모리 정리
        del clip_fp16
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # CLIP 결과 저장
        results['clip'] = {
            'original': clip_original_results,
            'fp16': clip_fp16_results,
            'improvement': {
                'speedup': clip_original_results['inference_time_ms'] / clip_fp16_results['inference_time_ms'],
                'memory_save': 0 if clip_original_results['memory_usage_mb'] == 0 else 
                    (clip_original_results['memory_usage_mb'] - clip_fp16_results['memory_usage_mb']) / clip_original_results['memory_usage_mb'] * 100,
                'fps_improvement': clip_fp16_results['fps'] / clip_original_results['fps']
            }
        }
        
        # 3. 결과 출력
        logger.info("\n" + "="*80)
        logger.info("📊 양자화 성능 비교 결과")
        logger.info("="*80)
        
        # Kosmos2 결과
        logger.info(f"\n🥇 MAE 0.222 모델 (Kosmos2):")
        logger.info(f"   원본: {kosmos2_original_results['inference_time_ms']:.2f}ms, {kosmos2_original_results['fps']:.1f} FPS")
        logger.info(f"   FP16: {kosmos2_fp16_results['inference_time_ms']:.2f}ms, {kosmos2_fp16_results['fps']:.1f} FPS")
        logger.info(f"   속도 향상: {results['kosmos2']['improvement']['speedup']:.2f}x")
        logger.info(f"   메모리 절약: {results['kosmos2']['improvement']['memory_save']:.1f}%")
        
        # CLIP 결과
        logger.info(f"\n🥈 MAE 0.212 모델 (CLIP):")
        logger.info(f"   원본: {clip_original_results['inference_time_ms']:.2f}ms, {clip_original_results['fps']:.1f} FPS")
        logger.info(f"   FP16: {clip_fp16_results['inference_time_ms']:.2f}ms, {clip_fp16_results['fps']:.1f} FPS")
        logger.info(f"   속도 향상: {results['clip']['improvement']['speedup']:.2f}x")
        logger.info(f"   메모리 절약: {results['clip']['improvement']['memory_save']:.1f}%")
        
        # 모델 간 비교
        logger.info(f"\n🏆 모델 간 비교:")
        kosmos2_fp16_fps = kosmos2_fp16_results['fps']
        clip_fp16_fps = clip_fp16_results['fps']
        logger.info(f"   Kosmos2 FP16: {kosmos2_fp16_fps:.1f} FPS")
        logger.info(f"   CLIP FP16: {clip_fp16_fps:.1f} FPS")
        logger.info(f"   CLIP이 Kosmos2보다 {clip_fp16_fps/kosmos2_fp16_fps:.2f}x 빠름")
        
        # 4. 결과 저장
        with open('quantization_comparison_results.json', 'w') as f:
            # Tensor 객체 제거 후 저장
            clean_results = {}
            for model_name, model_results in results.items():
                clean_results[model_name] = {}
                for test_name, test_results in model_results.items():
                    if test_name == 'improvement':
                        clean_results[model_name][test_name] = test_results
                    else:
                        clean_results[model_name][test_name] = {
                            'inference_time_ms': test_results['inference_time_ms'],
                            'inference_time_std': test_results['inference_time_std'],
                            'memory_usage_mb': test_results['memory_usage_mb'],
                            'peak_memory_mb': test_results['peak_memory_mb'],
                            'fps': test_results['fps']
                        }
            
            json.dump(clean_results, f, indent=2)
        
        logger.info("\n✅ 결과가 quantization_comparison_results.json에 저장되었습니다")
        
        return results

def main():
    tester = QuantizationComparisonTest()
    results = tester.run_comparison()
    
    return results

if __name__ == "__main__":
    main()
