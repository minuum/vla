#!/usr/bin/env python3
"""
최종 양자화 성능 비교
CLIP 모델 실제 측정 + Kosmos2 이론적 비교
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

class FinalQuantizationComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = 50
        
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _create_clip_model(self):
        """MAE 0.212 모델 구조 (CLIP 기반)"""
        class CLIPBasedModel(nn.Module):
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
                    image_features = self.clip.get_image_features(pixel_values=x)  # [batch, 512]
                    
                sequence_features = image_features.unsqueeze(1)  # [batch, 1, 512]
                rnn_out, _ = self.rnn(sequence_features)  # [batch, 1, 4096]
                actions = self.actions(rnn_out.squeeze(1))  # [batch, 2]
                return actions
        
        return CLIPBasedModel()
    
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
        for _ in range(5):
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
    
    def _estimate_kosmos2_performance(self, clip_results):
        """Kosmos2 성능 추정 (구조 분석 기반)"""
        logger.info("🔍 Kosmos2 성능 추정 (구조 분석 기반)")
        
        # Kosmos2 vs CLIP 구조 차이
        kosmos2_input_size = 2048  # Kosmos2 출력 크기
        clip_input_size = 512      # CLIP 출력 크기
        
        # 복잡도 비율 계산 (입력 크기 차이로 인한 RNN 연산량 차이)
        complexity_ratio = (kosmos2_input_size / clip_input_size) ** 2  # RNN 연산량은 입력 크기의 제곱에 비례
        
        # Kosmos2 성능 추정
        kosmos2_original_time = clip_results['original']['inference_time_ms'] * complexity_ratio
        kosmos2_fp16_time = clip_results['fp16']['inference_time_ms'] * complexity_ratio
        
        # FP16 양자화 효과는 동일하다고 가정
        fp16_speedup = clip_results['fp16']['inference_time_ms'] / clip_results['original']['inference_time_ms']
        
        kosmos2_original_fps = 1000 / kosmos2_original_time
        kosmos2_fp16_fps = 1000 / kosmos2_fp16_time
        
        logger.info(f"   복잡도 비율: {complexity_ratio:.2f}x")
        logger.info(f"   추정 원본 시간: {kosmos2_original_time:.2f} ms")
        logger.info(f"   추정 FP16 시간: {kosmos2_fp16_time:.2f} ms")
        logger.info(f"   추정 원본 FPS: {kosmos2_original_fps:.1f}")
        logger.info(f"   추정 FP16 FPS: {kosmos2_fp16_fps:.1f}")
        
        return {
            'original': {
                'inference_time_ms': kosmos2_original_time,
                'fps': kosmos2_original_fps
            },
            'fp16': {
                'inference_time_ms': kosmos2_fp16_time,
                'fps': kosmos2_fp16_fps
            },
            'improvement': {
                'speedup': 1 / fp16_speedup,
                'fps_improvement': kosmos2_fp16_fps / kosmos2_original_fps
            },
            'estimated': True
        }
    
    def run_comparison(self):
        """최종 양자화 성능 비교 실행"""
        logger.info("🚀 최종 양자화 성능 비교 시작")
        
        # 입력 데이터 생성
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        results = {}
        
        # 1. CLIP 모델 (MAE 0.212) 실제 측정
        logger.info("\n" + "="*60)
        logger.info("🎯 MAE 0.212 모델 (CLIP) 실제 양자화 테스트")
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
        
        # 2. Kosmos2 모델 (MAE 0.222) 성능 추정
        logger.info("\n" + "="*60)
        logger.info("🎯 MAE 0.222 모델 (Kosmos2) 성능 추정")
        logger.info("="*60)
        
        kosmos2_results = self._estimate_kosmos2_performance(results['clip'])
        results['kosmos2'] = kosmos2_results
        
        # 3. 결과 출력
        logger.info("\n" + "="*80)
        logger.info("📊 최종 양자화 성능 비교 결과")
        logger.info("="*80)
        
        # CLIP 결과 (실제 측정)
        logger.info(f"\n🥈 MAE 0.212 모델 (CLIP) - 실제 측정:")
        logger.info(f"   원본: {clip_original_results['inference_time_ms']:.2f}ms, {clip_original_results['fps']:.1f} FPS")
        logger.info(f"   FP16: {clip_fp16_results['inference_time_ms']:.2f}ms, {clip_fp16_results['fps']:.1f} FPS")
        logger.info(f"   속도 향상: {results['clip']['improvement']['speedup']:.2f}x")
        logger.info(f"   메모리 절약: {results['clip']['improvement']['memory_save']:.1f}%")
        
        # Kosmos2 결과 (추정)
        logger.info(f"\n🥇 MAE 0.222 모델 (Kosmos2) - 성능 추정:")
        logger.info(f"   원본: {kosmos2_results['original']['inference_time_ms']:.2f}ms, {kosmos2_results['original']['fps']:.1f} FPS")
        logger.info(f"   FP16: {kosmos2_results['fp16']['inference_time_ms']:.2f}ms, {kosmos2_results['fp16']['fps']:.1f} FPS")
        logger.info(f"   속도 향상: {kosmos2_results['improvement']['speedup']:.2f}x")
        logger.info(f"   메모리 절약: {results['clip']['improvement']['memory_save']:.1f}% (CLIP과 동일 추정)")
        
        # 모델 간 비교
        logger.info(f"\n🏆 모델 간 비교:")
        kosmos2_fp16_fps = kosmos2_results['fp16']['fps']
        clip_fp16_fps = clip_fp16_results['fps']
        logger.info(f"   Kosmos2 FP16: {kosmos2_fp16_fps:.1f} FPS (추정)")
        logger.info(f"   CLIP FP16: {clip_fp16_fps:.1f} FPS (실제)")
        logger.info(f"   CLIP이 Kosmos2보다 {clip_fp16_fps/kosmos2_fp16_fps:.2f}x 빠름")
        
        # 4. 결과 저장
        with open('final_quantization_comparison_results.json', 'w') as f:
            # Tensor 객체 제거 후 저장
            clean_results = {}
            for model_name, model_results in results.items():
                clean_results[model_name] = {}
                for test_name, test_results in model_results.items():
                    if test_name == 'improvement':
                        clean_results[model_name][test_name] = test_results
                    elif test_name == 'estimated':
                        clean_results[model_name][test_name] = test_results
                    else:
                        clean_results[model_name][test_name] = {
                            'inference_time_ms': test_results['inference_time_ms'],
                            'fps': test_results['fps']
                        }
                        if 'inference_time_std' in test_results:
                            clean_results[model_name][test_name]['inference_time_std'] = test_results['inference_time_std']
                        if 'memory_usage_mb' in test_results:
                            clean_results[model_name][test_name]['memory_usage_mb'] = test_results['memory_usage_mb']
                        if 'peak_memory_mb' in test_results:
                            clean_results[model_name][test_name]['peak_memory_mb'] = test_results['peak_memory_mb']
            
            json.dump(clean_results, f, indent=2)
        
        logger.info("\n✅ 결과가 final_quantization_comparison_results.json에 저장되었습니다")
        
        return results

def main():
    tester = FinalQuantizationComparison()
    results = tester.run_comparison()
    
    return results

if __name__ == "__main__":
    main()
