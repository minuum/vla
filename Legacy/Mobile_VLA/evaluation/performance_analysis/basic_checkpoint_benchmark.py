#!/usr/bin/env python3
"""
모든 체크포인트 기본 벤치마킹 스크립트
실제 체크포인트들의 성능을 측정하고 비교
"""

import torch
import torch.nn as nn
import time
import json
import logging
import os
import gc
import numpy as np
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicCheckpointBenchmark:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = 50
        
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _load_checkpoint_info(self, checkpoint_path):
        """체크포인트 정보 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'path': checkpoint_path,
                'mae': checkpoint.get('val_mae', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'model_keys': len(checkpoint.get('model_state_dict', {})),
                'checkpoint_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
            }
            
            # 모델 타입 판별
            state_dict = checkpoint.get('model_state_dict', {})
            kosmos_keys = [key for key in state_dict.keys() if 'kosmos' in key.lower()]
            clip_keys = [key for key in state_dict.keys() if 'clip' in key.lower()]
            
            if len(clip_keys) > 0 and len(kosmos_keys) > 0:
                info['model_type'] = 'Kosmos2+CLIP Hybrid'
            elif len(kosmos_keys) > 0:
                info['model_type'] = 'Pure Kosmos2'
            elif len(clip_keys) > 0:
                info['model_type'] = 'CLIP Only'
            else:
                info['model_type'] = 'Unknown'
            
            return info, checkpoint
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 로드 실패 {checkpoint_path}: {e}")
            return None, None
    
    def _create_simple_benchmark_model(self, state_dict):
        """벤치마킹용 간단한 모델 생성"""
        class SimpleBenchmarkModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                
                # RNN 입력 크기 확인
                rnn_input_size = None
                for key in state_dict.keys():
                    if 'weight_ih_l0' in key:
                        rnn_input_size = state_dict[key].shape[1]
                        break
                
                if rnn_input_size is None:
                    rnn_input_size = 2048  # 기본값
                
                # 간단한 모델 구조
                self.feature_extractor = nn.Linear(3*224*224, rnn_input_size)
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
                x_flat = x.view(batch_size, -1)
                features = self.feature_extractor(x_flat)
                sequence_features = features.unsqueeze(1)
                rnn_out, _ = self.rnn(sequence_features)
                actions = self.actions(rnn_out.squeeze(1))
                return actions
        
        return SimpleBenchmarkModel(state_dict)
    
    def _benchmark_model(self, model, name, input_data):
        """모델 벤치마크 수행"""
        logger.info(f"📊 {name} 벤치마크 시작...")
        
        model = model.to(self.device)
        model.eval()
        
        # 워밍업
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_data)
        
        # 메모리 측정
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        # 추론 시간 측정
        times = []
        
        for i in range(self.num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
            
            if (i + 1) % 10 == 0:
                logger.info(f"   진행률: {i+1}/{self.num_runs}")
        
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
            'fps': fps
        }
    
    def run_all_checkpoints_benchmark(self):
        """모든 체크포인트 벤치마크 실행"""
        logger.info("🚀 모든 체크포인트 기본 벤치마킹 시작")
        
        # 체크포인트 경로들
        checkpoint_paths = [
            "results/simple_lstm_results_extended/final_simple_lstm_model.pth",
            "results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
            "results/mobile_vla_epoch_3.pt",
            "models/experimental/simplified_robovlms_best.pth"
        ]
        
        # 입력 데이터 생성
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        results = {}
        
        for checkpoint_path in checkpoint_paths:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"⚠️ 체크포인트 파일이 없습니다: {checkpoint_path}")
                continue
            
            logger.info("\n" + "="*60)
            logger.info(f"🎯 {checkpoint_path} 분석")
            logger.info("="*60)
            
            # 체크포인트 정보 로드
            info, checkpoint = self._load_checkpoint_info(checkpoint_path)
            
            if info is None:
                continue
            
            logger.info(f"   모델 타입: {info['model_type']}")
            logger.info(f"   MAE: {info['mae']}")
            logger.info(f"   에포크: {info['epoch']}")
            logger.info(f"   모델 키: {info['model_keys']}개")
            logger.info(f"   파일 크기: {info['checkpoint_size_mb']:.2f} MB")
            
            # 벤치마크 모델 생성
            state_dict = checkpoint.get('model_state_dict', {})
            benchmark_model = self._create_simple_benchmark_model(state_dict)
            
            # 벤치마크 수행
            benchmark_results = self._benchmark_model(benchmark_model, info['model_type'], input_data)
            
            # 결과 저장
            model_name = Path(checkpoint_path).stem
            results[model_name] = {
                'checkpoint_info': info,
                'benchmark_results': benchmark_results
            }
            
            # 메모리 정리
            del benchmark_model
            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 결과 출력
        if results:
            logger.info("\n" + "="*80)
            logger.info("📊 모든 체크포인트 벤치마킹 결과")
            logger.info("="*80)
            
            # 성능 순위
            performance_ranking = []
            for model_name, data in results.items():
                mae = data['checkpoint_info']['mae']
                fps = data['benchmark_results']['fps']
                inference_time = data['benchmark_results']['inference_time_ms']
                
                performance_ranking.append({
                    'model_name': model_name,
                    'model_type': data['checkpoint_info']['model_type'],
                    'mae': mae,
                    'fps': fps,
                    'inference_time_ms': inference_time,
                    'memory_mb': data['benchmark_results']['memory_usage_mb']
                })
            
            # FPS 기준 정렬
            performance_ranking.sort(key=lambda x: x['fps'], reverse=True)
            
            logger.info("\n🏆 성능 순위 (FPS 기준):")
            for i, model in enumerate(performance_ranking, 1):
                logger.info(f"{i}. {model['model_name']} ({model['model_type']})")
                logger.info(f"   MAE: {model['mae']}")
                logger.info(f"   FPS: {model['fps']:.2f}")
                logger.info(f"   추론 시간: {model['inference_time_ms']:.2f} ms")
                logger.info(f"   메모리: {model['memory_mb']:.2f} MB")
                logger.info()
            
            # MAE 기준 정렬
            mae_ranking = [m for m in performance_ranking if m['mae'] != 'N/A']
            mae_ranking.sort(key=lambda x: x['mae'])
            
            if mae_ranking:
                logger.info("🎯 정확도 순위 (MAE 기준):")
                for i, model in enumerate(mae_ranking, 1):
                    logger.info(f"{i}. {model['model_name']} ({model['model_type']})")
                    logger.info(f"   MAE: {model['mae']}")
                    logger.info(f"   FPS: {model['fps']:.2f}")
                    logger.info()
            
            # 결과 저장
            output_path = "results/all_checkpoints_benchmark_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"\n✅ 결과가 {output_path}에 저장되었습니다")
        
        return results

def main():
    benchmark = BasicCheckpointBenchmark()
    results = benchmark.run_all_checkpoints_benchmark()
    return results

if __name__ == "__main__":
    main()
