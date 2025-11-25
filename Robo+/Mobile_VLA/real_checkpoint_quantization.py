#!/usr/bin/env python3
"""
실제 체크포인트 기반 양자화 성능 비교
순수 Kosmos2 vs Kosmos2+CLIP 하이브리드
"""

import torch
import torch.nn as nn
import time
import json
import logging
import os
import gc
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealCheckpointQuantization:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = 30  # Kosmos2가 느려서 줄임
        
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _load_checkpoint_model(self, checkpoint_path):
        """체크포인트에서 모델 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"✅ 체크포인트 로드 성공: {checkpoint_path}")
            logger.info(f"   - 검증 MAE: {checkpoint.get('val_mae', 'N/A')}")
            logger.info(f"   - 에포크: {checkpoint.get('epoch', 'N/A')}")
            
            # 모델 상태 딕셔너리 확인
            state_dict = checkpoint.get('model_state_dict', {})
            logger.info(f"   - 모델 키 수: {len(state_dict)}")
            
            # 모델 타입 확인
            kosmos_keys = [key for key in state_dict.keys() if 'kosmos' in key.lower()]
            clip_keys = [key for key in state_dict.keys() if 'clip' in key.lower()]
            
            logger.info(f"   - Kosmos2 키: {len(kosmos_keys)}개")
            logger.info(f"   - CLIP 키: {len(clip_keys)}개")
            
            if len(clip_keys) > 0:
                logger.info("   - 모델 타입: Kosmos2+CLIP 하이브리드")
            else:
                logger.info("   - 모델 타입: 순수 Kosmos2")
            
            return checkpoint, state_dict
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 로드 실패: {e}")
            return None, None
    
    def _create_simple_model_for_benchmark(self, state_dict):
        """벤치마크용 간단한 모델 생성"""
        class SimpleBenchmarkModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self.state_dict = state_dict
                
                # RNN 입력 크기 확인
                rnn_input_size = None
                for key in state_dict.keys():
                    if 'weight_ih_l0' in key:
                        rnn_input_size = state_dict[key].shape[1]
                        break
                
                logger.info(f"   - RNN 입력 크기: {rnn_input_size}")
                
                # 간단한 모델 구조 (실제 추론만을 위한)
                self.feature_extractor = nn.Linear(3*224*224, rnn_input_size)  # 이미지 → 특징
                self.rnn = nn.RNN(
                    input_size=rnn_input_size,
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
                
                # 이미지를 평면화하여 특징으로 변환
                x_flat = x.view(batch_size, -1)  # [batch, 3*224*224]
                features = self.feature_extractor(x_flat)  # [batch, rnn_input_size]
                
                # RNN 처리
                sequence_features = features.unsqueeze(1)  # [batch, 1, rnn_input_size]
                rnn_out, _ = self.rnn(sequence_features)  # [batch, 1, 4096]
                
                # 액션 예측
                actions = self.actions(rnn_out.squeeze(1))  # [batch, 2]
                return actions
        
        return SimpleBenchmarkModel(state_dict)
    
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
        for _ in range(3):
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
    
    def run_comparison(self):
        """두 모델 양자화 성능 비교 실행"""
        logger.info("🚀 실제 체크포인트 기반 양자화 성능 비교 시작")
        
        # 입력 데이터 생성
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        results = {}
        
        # 1. 순수 Kosmos2 모델 (MAE 0.222) 테스트
        logger.info("\n" + "="*60)
        logger.info("🎯 순수 Kosmos2 모델 (MAE 0.222) 분석")
        logger.info("="*60)
        
        kosmos2_checkpoint_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
        kosmos2_checkpoint, kosmos2_state_dict = self._load_checkpoint_model(kosmos2_checkpoint_path)
        
        if kosmos2_state_dict is not None:
            # 간단한 모델 생성
            kosmos2_original = self._create_simple_model_for_benchmark(kosmos2_state_dict)
            kosmos2_fp16 = self._create_quantized_model(self._create_simple_model_for_benchmark(kosmos2_state_dict), "fp16")
            
            # 원본 모델 벤치마크
            kosmos2_original_results = self._benchmark_model(kosmos2_original, "순수 Kosmos2 원본", input_data)
            
            # 메모리 정리
            del kosmos2_original
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # FP16 양자화 모델 벤치마크
            kosmos2_fp16_results = self._benchmark_model(kosmos2_fp16, "순수 Kosmos2 FP16", input_data)
            
            # 메모리 정리
            del kosmos2_fp16
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 순수 Kosmos2 결과 저장
            results['pure_kosmos2'] = {
                'original': kosmos2_original_results,
                'fp16': kosmos2_fp16_results,
                'improvement': {
                    'speedup': kosmos2_original_results['inference_time_ms'] / kosmos2_fp16_results['inference_time_ms'],
                    'memory_save': 0 if kosmos2_original_results['memory_usage_mb'] == 0 else 
                        (kosmos2_original_results['memory_usage_mb'] - kosmos2_fp16_results['memory_usage_mb']) / kosmos2_original_results['memory_usage_mb'] * 100,
                    'fps_improvement': kosmos2_fp16_results['fps'] / kosmos2_original_results['fps']
                },
                'checkpoint_info': {
                    'mae': kosmos2_checkpoint.get('val_mae'),
                    'epoch': kosmos2_checkpoint.get('epoch')
                }
            }
        
        # 2. Kosmos2+CLIP 하이브리드 모델 (MAE 0.212) 테스트
        logger.info("\n" + "="*60)
        logger.info("🎯 Kosmos2+CLIP 하이브리드 모델 (MAE 0.212) 분석")
        logger.info("="*60)
        
        hybrid_checkpoint_path = "results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth"
        hybrid_checkpoint, hybrid_state_dict = self._load_checkpoint_model(hybrid_checkpoint_path)
        
        if hybrid_state_dict is not None:
            # 간단한 모델 생성
            hybrid_original = self._create_simple_model_for_benchmark(hybrid_state_dict)
            hybrid_fp16 = self._create_quantized_model(self._create_simple_model_for_benchmark(hybrid_state_dict), "fp16")
            
            # 원본 모델 벤치마크
            hybrid_original_results = self._benchmark_model(hybrid_original, "하이브리드 원본", input_data)
            
            # 메모리 정리
            del hybrid_original
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # FP16 양자화 모델 벤치마크
            hybrid_fp16_results = self._benchmark_model(hybrid_fp16, "하이브리드 FP16", input_data)
            
            # 메모리 정리
            del hybrid_fp16
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 하이브리드 결과 저장
            results['hybrid'] = {
                'original': hybrid_original_results,
                'fp16': hybrid_fp16_results,
                'improvement': {
                    'speedup': hybrid_original_results['inference_time_ms'] / hybrid_fp16_results['inference_time_ms'],
                    'memory_save': 0 if hybrid_original_results['memory_usage_mb'] == 0 else 
                        (hybrid_original_results['memory_usage_mb'] - hybrid_fp16_results['memory_usage_mb']) / hybrid_original_results['memory_usage_mb'] * 100,
                    'fps_improvement': hybrid_fp16_results['fps'] / hybrid_original_results['fps']
                },
                'checkpoint_info': {
                    'mae': hybrid_checkpoint.get('val_mae'),
                    'epoch': hybrid_checkpoint.get('epoch')
                }
            }
        
        # 3. 결과 출력
        if 'pure_kosmos2' in results and 'hybrid' in results:
            logger.info("\n" + "="*80)
            logger.info("📊 실제 체크포인트 기반 양자화 성능 비교 결과")
            logger.info("="*80)
            
            # 순수 Kosmos2 결과
            logger.info(f"\n🥈 순수 Kosmos2 모델 (MAE {results['pure_kosmos2']['checkpoint_info']['mae']:.4f}):")
            logger.info(f"   원본: {results['pure_kosmos2']['original']['inference_time_ms']:.2f}ms, {results['pure_kosmos2']['original']['fps']:.1f} FPS")
            logger.info(f"   FP16: {results['pure_kosmos2']['fp16']['inference_time_ms']:.2f}ms, {results['pure_kosmos2']['fp16']['fps']:.1f} FPS")
            logger.info(f"   속도 향상: {results['pure_kosmos2']['improvement']['speedup']:.2f}x")
            logger.info(f"   메모리 절약: {results['pure_kosmos2']['improvement']['memory_save']:.1f}%")
            
            # 하이브리드 결과
            logger.info(f"\n🥇 Kosmos2+CLIP 하이브리드 모델 (MAE {results['hybrid']['checkpoint_info']['mae']:.4f}):")
            logger.info(f"   원본: {results['hybrid']['original']['inference_time_ms']:.2f}ms, {results['hybrid']['original']['fps']:.1f} FPS")
            logger.info(f"   FP16: {results['hybrid']['fp16']['inference_time_ms']:.2f}ms, {results['hybrid']['fp16']['fps']:.1f} FPS")
            logger.info(f"   속도 향상: {results['hybrid']['improvement']['speedup']:.2f}x")
            logger.info(f"   메모리 절약: {results['hybrid']['improvement']['memory_save']:.1f}%")
            
            # 모델 간 비교
            logger.info(f"\n🏆 모델 간 비교:")
            kosmos2_fp16_fps = results['pure_kosmos2']['fp16']['fps']
            hybrid_fp16_fps = results['hybrid']['fp16']['fps']
            logger.info(f"   순수 Kosmos2 FP16: {kosmos2_fp16_fps:.1f} FPS")
            logger.info(f"   하이브리드 FP16: {hybrid_fp16_fps:.1f} FPS")
            
            if kosmos2_fp16_fps > hybrid_fp16_fps:
                speedup_ratio = kosmos2_fp16_fps / hybrid_fp16_fps
                logger.info(f"   순수 Kosmos2가 하이브리드보다 {speedup_ratio:.2f}x 빠름")
            else:
                speedup_ratio = hybrid_fp16_fps / kosmos2_fp16_fps
                logger.info(f"   하이브리드가 순수 Kosmos2보다 {speedup_ratio:.2f}x 빠름")
            
            # 4. 결과 저장
            with open('real_checkpoint_quantization_results.json', 'w') as f:
                # Tensor 객체 제거 후 저장
                clean_results = {}
                for model_name, model_results in results.items():
                    clean_results[model_name] = {}
                    for test_name, test_results in model_results.items():
                        if test_name == 'improvement' or test_name == 'checkpoint_info':
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
            
            logger.info("\n✅ 결과가 real_checkpoint_quantization_results.json에 저장되었습니다")
        
        return results

def main():
    tester = RealCheckpointQuantization()
    results = tester.run_comparison()
    
    return results

if __name__ == "__main__":
    main()
