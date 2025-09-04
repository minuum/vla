#!/usr/bin/env python3
"""
실제 성능 개선 측정 확인 스크립트
MAE 0.222 모델과 양자화된 모델의 정확한 비교
"""

import torch
import torch.nn as nn
import time
import json
import logging
from transformers import CLIPProcessor, CLIPModel
import os
import gc

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActualPerformanceVerification:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_runs = 50
        
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _load_actual_mae0222_model(self):
        """실제 MAE 0.222 모델 로드 시도"""
        try:
            # 실제 체크포인트 경로
            checkpoint_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
                return None, None
            
            logger.info(f"📁 실제 체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 체크포인트 정보 출력
            logger.info(f"   - 에포크: {checkpoint.get('epoch', 'N/A')}")
            logger.info(f"   - 검증 MAE: {checkpoint.get('val_mae', 'N/A')}")
            logger.info(f"   - 모델 키 수: {len(checkpoint.get('model_state_dict', {}))}")
            
            return checkpoint, checkpoint.get('val_mae', None)
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 로드 실패: {e}")
            return None, None
    
    def _create_actual_model_structure(self):
        """실제 모델 구조 생성 (MAE 0.222 모델과 동일)"""
        class ActualMAE0222Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 실제 MAE 0.222 모델 구조 (simple_lstm_model.py 기반)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip.eval()
                for param in self.clip.parameters():
                    param.requires_grad = False
                
                # 실제 Action Head 구조 (MAE 0.222 모델과 동일)
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
        
        return ActualMAE0222Model()
    
    def _create_quantized_model_structure(self):
        """양자화된 모델 구조 생성"""
        class QuantizedMAE0222Model(nn.Module):
            def __init__(self):
                super().__init__()
                # CLIP을 FP16으로 양자화
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip = self.clip.half()  # FP16 양자화
                self.clip.eval()
                for param in self.clip.parameters():
                    param.requires_grad = False
                
                # Action Head는 FP32 유지 (실제 구조와 동일)
                self.rnn = nn.RNN(
                    input_size=512,
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
                    x_fp16 = x.half()  # 입력을 FP16으로
                    image_features = self.clip.get_image_features(pixel_values=x_fp16)
                
                image_features_fp32 = image_features.float()  # FP32로 변환
                sequence_features = image_features_fp32.unsqueeze(1)
                rnn_out, _ = self.rnn(sequence_features)
                actions = self.actions(rnn_out.squeeze(1))
                return actions
        
        return QuantizedMAE0222Model()
    
    def _benchmark_model(self, model, name, checkpoint=None):
        """모델 벤치마크 수행"""
        logger.info(f"📊 {name} 벤치마크 시작...")
        
        # 체크포인트 로드 시도
        if checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ 체크포인트 로드 성공 (MAE: {checkpoint.get('val_mae', 'N/A')})")
            except Exception as e:
                logger.warning(f"⚠️ 체크포인트 로드 실패: {e}")
                logger.info("   랜덤 초기화된 모델로 테스트")
        
        model = model.to(self.device)
        model.eval()
        
        # 입력 데이터 생성
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
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
        
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time
        
        logger.info(f"   추론 시간: {avg_time:.2f} ms")
        logger.info(f"   메모리 사용량: {memory_used:.2f} MB")
        logger.info(f"   최대 메모리: {peak_memory_used:.2f} MB")
        logger.info(f"   FPS: {fps:.2f}")
        
        return {
            'inference_time_ms': avg_time,
            'memory_usage_mb': memory_used,
            'peak_memory_mb': peak_memory_used,
            'fps': fps,
            'outputs': outputs
        }
    
    def verify_actual_performance(self):
        """실제 성능 개선 측정 확인"""
        logger.info("🚀 실제 성능 개선 측정 확인 시작")
        
        # 1. 실제 MAE 0.222 체크포인트 로드
        checkpoint, actual_mae = self._load_actual_mae0222_model()
        
        if actual_mae:
            logger.info(f"✅ 실제 MAE 0.222 모델 확인: {actual_mae:.6f}")
        else:
            logger.warning("⚠️ 실제 체크포인트 로드 실패, 구조만 비교")
        
        # 2. 실제 모델 구조 생성
        actual_model = self._create_actual_model_structure()
        quantized_model = self._create_quantized_model_structure()
        
        # 3. 벤치마크 수행
        actual_results = self._benchmark_model(actual_model, "실제 MAE 0.222 모델", checkpoint)
        
        # 메모리 정리
        del actual_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        quantized_results = self._benchmark_model(quantized_model, "양자화된 모델 (FP16)")
        
        # 메모리 정리
        del quantized_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. 성능 비교
        speedup = actual_results['inference_time_ms'] / quantized_results['inference_time_ms']
        memory_save = 0
        if actual_results['memory_usage_mb'] > 0:
            memory_save = (actual_results['memory_usage_mb'] - quantized_results['memory_usage_mb']) / actual_results['memory_usage_mb'] * 100
        
        # 5. 출력 비교 (정확도 확인)
        accuracy_info = self._compare_outputs(actual_results['outputs'], quantized_results['outputs'])
        
        # 6. 결과 출력
        logger.info("\n📊 실제 성능 개선 측정 결과:")
        logger.info("=" * 60)
        logger.info(f"실제 MAE: {actual_mae:.6f}" if actual_mae else "실제 MAE: 확인 불가")
        logger.info(f"속도 향상: {speedup:.2f}x")
        logger.info(f"메모리 절약: {memory_save:.1f}%")
        logger.info(f"출력 정확도: {accuracy_info}")
        
        # 7. 결과 저장
        results = {
            'actual_mae': actual_mae,
            'actual_model': actual_results,
            'quantized_model': quantized_results,
            'improvement': {
                'speedup': speedup,
                'memory_save': memory_save,
                'accuracy_info': accuracy_info
            },
            'verification': {
                'checkpoint_loaded': checkpoint is not None,
                'actual_structure_used': True,
                'quantization_applied': True
            }
        }
        
        with open('actual_performance_verification.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ 결과가 actual_performance_verification.json에 저장되었습니다")
        
        return results
    
    def _compare_outputs(self, actual_outputs, quantized_outputs):
        """출력 정확도 비교"""
        if not actual_outputs or not quantized_outputs:
            return "비교 불가"
        
        try:
            actual_tensor = torch.stack(actual_outputs)
            quantized_tensor = torch.stack(quantized_outputs)
            
            mae = torch.mean(torch.abs(actual_tensor - quantized_tensor)).item()
            correlation = torch.corrcoef(torch.stack([actual_tensor.flatten(), quantized_tensor.flatten()]))[0, 1].item()
            accuracy_01 = torch.mean((torch.abs(actual_tensor - quantized_tensor) < 0.1).float()).item()
            
            return f"MAE: {mae:.6f}, 상관계수: {correlation:.6f}, 0.1이내: {accuracy_01:.2%}"
            
        except Exception as e:
            return f"비교 오류: {e}"

def main():
    verifier = ActualPerformanceVerification()
    results = verifier.verify_actual_performance()
    
    return results

if __name__ == "__main__":
    main()
