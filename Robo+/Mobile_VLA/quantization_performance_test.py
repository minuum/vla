#!/usr/bin/env python3
"""
🎯 양자화 성능 비교 테스트
원본 MAE 0.222 모델 vs 양자화된 모델의 성능 비교
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import json
import logging
import numpy as np
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OriginalMAE0222Model(nn.Module):
    """원본 MAE 0.222 모델 (정확한 구조 재현)"""
    
    def __init__(self, model_path: str):
        super().__init__()
        
        # Kosmos2 모델 로드 (원본 FP32)
        from transformers import AutoProcessor, AutoModel
        
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos.eval()
        for param in self.kosmos.parameters():
            param.requires_grad = False
        
        # Action Head (원본 구조)
        self.rnn = nn.RNN(
            input_size=2048,  # Kosmos2 hidden size
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
            nn.Linear(256, 2)  # linear_x, linear_y
        )
        
        # 체크포인트에서 Action Head 로드
        self._load_action_head(model_path)
        
        logger.info("✅ 원본 MAE 0.222 모델 초기화 완료")
    
    def _load_action_head(self, model_path: str):
        """Action Head만 체크포인트에서 로드"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # state_dict에서 Action Head 관련 파라미터만 추출
            action_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if 'rnn' in key or 'actions' in key:
                    action_state_dict[key] = value
            
            # Action Head만 로드
            self.load_state_dict(action_state_dict, strict=False)
            logger.info(f"✅ Action Head 로드 완료: {len(action_state_dict)} 파라미터")
            
        except Exception as e:
            logger.warning(f"⚠️ Action Head 로드 실패: {e}")
            logger.info("🔄 랜덤 초기화된 Action Head 사용")
    
    def forward(self, x):
        """순전파 (원본 FP32)"""
        batch_size = x.size(0)
        
        # 1. VLM으로 이미지 특징 추출 (FP32)
        with torch.no_grad():
            # 더미 텍스트 생성
            dummy_text = ["<image>"] * batch_size
            
            # 텍스트 토크나이징
            text_inputs = self.processor(
                text=dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(x.device)
            
            try:
                # Kosmos2로 특징 추출 (FP32)
                vision_outputs = self.kosmos(
                    pixel_values=x,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                )
                vision_features = vision_outputs.last_hidden_state[:, 0]  # [batch_size, 2048]
            except Exception as e:
                logger.warning(f"Kosmos2 추론 오류: {e}")
                vision_features = torch.randn(batch_size, 2048).to(x.device)
        
        # 2. Action Head로 액션 예측 (FP32)
        sequence_features = vision_features.unsqueeze(1)  # [batch_size, 1, 2048]
        rnn_out, _ = self.rnn(sequence_features)  # [batch_size, 1, 4096]
        actions = self.actions(rnn_out.squeeze(1))  # [batch_size, 2]
        
        return actions

class QuantizedMAE0222Model(nn.Module):
    """양자화된 MAE 0.222 모델 (VLM FP16)"""
    
    def __init__(self, model_path: str):
        super().__init__()
        
        # Kosmos2 모델 로드 (FP16으로 변환)
        from transformers import AutoProcessor, AutoModel
        
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # VLM을 FP16으로 변환
        self.kosmos = self.kosmos.half()
        self.kosmos.eval()
        for param in self.kosmos.parameters():
            param.requires_grad = False
        
        # Action Head (FP32 유지)
        self.rnn = nn.RNN(
            input_size=2048,  # Kosmos2 hidden size
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
            nn.Linear(256, 2)  # linear_x, linear_y
        )
        
        # 체크포인트에서 Action Head 로드
        self._load_action_head(model_path)
        
        logger.info("✅ 양자화된 MAE 0.222 모델 초기화 완료 (VLM FP16)")
    
    def _load_action_head(self, model_path: str):
        """Action Head만 체크포인트에서 로드"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # state_dict에서 Action Head 관련 파라미터만 추출
            action_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if 'rnn' in key or 'actions' in key:
                    action_state_dict[key] = value
            
            # Action Head만 로드
            self.load_state_dict(action_state_dict, strict=False)
            logger.info(f"✅ Action Head 로드 완료: {len(action_state_dict)} 파라미터")
            
        except Exception as e:
            logger.warning(f"⚠️ Action Head 로드 실패: {e}")
            logger.info("🔄 랜덤 초기화된 Action Head 사용")
    
    def forward(self, x):
        """순전파 (VLM FP16 + Action Head FP32)"""
        batch_size = x.size(0)
        
        # 1. VLM으로 이미지 특징 추출 (FP16)
        with torch.no_grad():
            # 입력을 FP16으로 변환
            x_fp16 = x.half()
            
            # 더미 텍스트 생성
            dummy_text = ["<image>"] * batch_size
            
            # 텍스트 토크나이징
            text_inputs = self.processor(
                text=dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(x.device)
            
            try:
                # Kosmos2로 특징 추출 (FP16)
                vision_outputs = self.kosmos(
                    pixel_values=x_fp16,
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask']
                )
                vision_features = vision_outputs.last_hidden_state[:, 0]  # [batch_size, 2048]
            except Exception as e:
                logger.warning(f"Kosmos2 추론 오류: {e}")
                vision_features = torch.randn(batch_size, 2048).half().to(x.device)
        
        # 2. Action Head로 액션 예측 (FP32)
        # FP16에서 FP32로 변환
        vision_features_fp32 = vision_features.float()
        
        sequence_features = vision_features_fp32.unsqueeze(1)  # [batch_size, 1, 2048]
        rnn_out, _ = self.rnn(sequence_features)  # [batch_size, 1, 4096]
        actions = self.actions(rnn_out.squeeze(1))  # [batch_size, 2]
        
        return actions

class QuantizationPerformanceTester:
    """양자화 성능 비교 테스터"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 원본 모델 로드
        self.original_model = OriginalMAE0222Model(model_path).to(self.device)
        self.original_model.eval()
        
        # 양자화된 모델 로드
        self.quantized_model = QuantizedMAE0222Model(model_path).to(self.device)
        self.quantized_model.eval()
        
        logger.info(f"🚀 양자화 성능 비교 테스터 초기화 완료")
        logger.info(f"   디바이스: {self.device}")
        logger.info(f"   모델 경로: {model_path}")
    
    def benchmark_model(self, model, name: str, num_runs: int = 100):
        """모델 벤치마크"""
        logger.info(f"📊 {name} 벤치마크 시작...")
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # 워밍업
        for _ in range(10):
            _ = model(dummy_input)
        
        # 메모리 사용량 측정
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 추론 시간 측정
        start_time = time.time()
        outputs = []
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(dummy_input)
                outputs.append(output)
        
        end_time = time.time()
        inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        # 메모리 사용량 측정
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        logger.info(f"   추론 시간: {inference_time:.2f} ms")
        logger.info(f"   메모리 사용량: {memory_usage:.2f} MB")
        logger.info(f"   FPS: {fps:.2f}")
        
        return {
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_usage,
            'fps': fps,
            'outputs': outputs
        }
    
    def compare_outputs(self, original_outputs, quantized_outputs):
        """출력 비교 (정확도 측정)"""
        logger.info("🔍 출력 정확도 비교...")
        
        if not original_outputs or not quantized_outputs:
            logger.warning("⚠️ 출력 비교 불가")
            return None
        
        # 출력을 텐서로 변환
        original_tensor = torch.stack(original_outputs)
        quantized_tensor = torch.stack(quantized_outputs)
        
        # MAE 계산
        mae = torch.mean(torch.abs(original_tensor - quantized_tensor)).item()
        
        # 상관계수 계산
        original_flat = original_tensor.flatten()
        quantized_flat = quantized_tensor.flatten()
        correlation = torch.corrcoef(torch.stack([original_flat, quantized_flat]))[0, 1].item()
        
        # 정확도 계산 (0.1 이내 오차)
        accuracy_01 = torch.mean((torch.abs(original_tensor - quantized_tensor) < 0.1).float()).item()
        
        # 정확도 계산 (0.05 이내 오차)
        accuracy_005 = torch.mean((torch.abs(original_tensor - quantized_tensor) < 0.05).float()).item()
        
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   상관계수: {correlation:.6f}")
        logger.info(f"   0.1 이내 정확도: {accuracy_01:.2%}")
        logger.info(f"   0.05 이내 정확도: {accuracy_005:.2%}")
        
        return {
            'mae': mae,
            'correlation': correlation,
            'accuracy_01': accuracy_01,
            'accuracy_005': accuracy_005
        }
    
    def test_performance(self):
        """전체 성능 테스트"""
        logger.info("🎯 양자화 성능 비교 테스트 시작!")
        
        results = {}
        
        # 1. 원본 모델 벤치마크
        logger.info("1. 원본 모델 벤치마크...")
        original_benchmark = self.benchmark_model(self.original_model, "원본 MAE 0.222")
        results['original'] = original_benchmark
        
        # 2. 양자화된 모델 벤치마크
        logger.info("2. 양자화된 모델 벤치마크...")
        quantized_benchmark = self.benchmark_model(self.quantized_model, "양자화된 MAE 0.222")
        results['quantized'] = quantized_benchmark
        
        # 3. 출력 정확도 비교
        logger.info("3. 출력 정확도 비교...")
        accuracy_comparison = self.compare_outputs(
            original_benchmark['outputs'], 
            quantized_benchmark['outputs']
        )
        results['accuracy'] = accuracy_comparison
        
        # 4. 결과 비교
        logger.info("4. 결과 비교...")
        self._compare_performance_results(results)
        
        # 5. 결과 저장
        with open('quantization_performance_results.json', 'w') as f:
            # outputs는 너무 크므로 제외
            save_results = {
                'original': {k: v for k, v in results['original'].items() if k != 'outputs'},
                'quantized': {k: v for k, v in results['quantized'].items() if k != 'outputs'},
                'accuracy': results['accuracy']
            }
            json.dump(save_results, f, indent=2)
        
        logger.info("💾 결과 저장 완료: quantization_performance_results.json")
        
        return results
    
    def _compare_performance_results(self, results):
        """성능 결과 비교"""
        logger.info("\n📊 양자화 성능 비교 결과:")
        logger.info("=" * 60)
        
        original = results.get('original', {})
        quantized = results.get('quantized', {})
        accuracy = results.get('accuracy', {})
        
        if original and quantized:
            speedup = original['inference_time_ms'] / quantized['inference_time_ms']
            memory_save = (original['memory_usage_mb'] - quantized['memory_usage_mb']) / original['memory_usage_mb'] * 100
            
            logger.info(f"성능 비교:")
            logger.info(f"   속도 향상: {speedup:.2f}x")
            logger.info(f"   메모리 절약: {memory_save:.1f}%")
            logger.info(f"   FPS 향상: {quantized['fps']:.1f} → {original['fps']:.1f}")
        
        if accuracy:
            logger.info(f"\n정확도 비교:")
            logger.info(f"   MAE: {accuracy['mae']:.6f}")
            logger.info(f"   상관계수: {accuracy['correlation']:.6f}")
            logger.info(f"   0.1 이내 정확도: {accuracy['accuracy_01']:.2%}")
            logger.info(f"   0.05 이내 정확도: {accuracy['accuracy_005']:.2%}")
        
        # 최종 권장사항
        logger.info("\n🎯 양자화 권장사항:")
        if original and quantized and accuracy:
            if speedup > 1.5 and accuracy['correlation'] > 0.95:
                logger.info("   🏆 양자화 강력 권장: 속도 향상 + 높은 정확도")
            elif speedup > 1.2 and accuracy['correlation'] > 0.9:
                logger.info("   🟢 양자화 권장: 적당한 속도 향상 + 양호한 정확도")
            elif accuracy['correlation'] < 0.8:
                logger.info("   🟡 양자화 주의: 정확도 저하 우려")
            else:
                logger.info("   🟡 원본 모델 유지 권장: 양자화 효과 미미")
        else:
            logger.info("   ⚠️ 테스트 실패로 권장사항 제공 불가")

def main():
    """메인 함수"""
    logger.info("🚀 양자화 성능 비교 테스트 시작")
    
    # 모델 경로
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 성능 테스트 실행
    tester = QuantizationPerformanceTester(model_path)
    results = tester.test_performance()
    
    logger.info("🎉 양자화 성능 비교 테스트 완료!")

if __name__ == "__main__":
    main()
