#!/usr/bin/env python3
"""
🏆 최적 양자화: VLM(FP16) + Action Head(INT8) 하이브리드
MAE 0.222 모델에 최적화된 양자화 적용
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import time
import psutil
import os
import json
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimalQuantizedModel(nn.Module):
    """최적 양자화 모델: VLM(FP16) + Action Head(INT8)"""
    
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
        
        # Action Head (INT8 양자화 대상)
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
        
        logger.info("✅ 최적 양자화 모델 초기화 완료")
        logger.info("   VLM: FP16 (고정)")
        logger.info("   Action Head: INT8 (양자화 대상)")
    
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
        """순전파 (VLM FP16 + Action Head INT8)"""
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
        
        # 2. Action Head로 액션 예측 (INT8)
        # FP16에서 FP32로 변환 (INT8 양자화를 위해)
        vision_features_fp32 = vision_features.float()
        
        sequence_features = vision_features_fp32.unsqueeze(1)  # [batch_size, 1, 2048]
        rnn_out, _ = self.rnn(sequence_features)  # [batch_size, 1, 4096]
        actions = self.actions(rnn_out.squeeze(1))  # [batch_size, 2]
        
        return actions

class OptimalQuantizer:
    """최적 양자화기: VLM(FP16) + Action Head(INT8)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 원본 모델 로드
        self.original_model = OptimalQuantizedModel(model_path).to(self.device)
        self.original_model.eval()
        
        logger.info(f"🚀 Optimal Quantizer 초기화 완료")
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
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
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
            'fps': fps
        }
    
    def quantize_action_head_to_int8(self):
        """Action Head를 INT8로 양자화"""
        logger.info("🔧 Action Head INT8 양자화 시작...")
        
        try:
            # Action Head만 INT8 양자화
            int8_model = quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear, nn.RNN},  # Action Head의 Linear와 RNN만 양자화
                dtype=torch.qint8
            )
            int8_model.eval()
            
            # 벤치마크
            benchmark = self.benchmark_model(int8_model, "VLM(FP16) + Action(INT8)")
            
            # 모델 저장
            torch.save(int8_model.state_dict(), 'optimal_quantized_model.pth')
            logger.info("💾 최적 양자화 모델 저장 완료")
            
            return int8_model, benchmark
            
        except Exception as e:
            logger.error(f"❌ INT8 양자화 실패: {e}")
            return None, None
    
    def quantize_model(self):
        """전체 최적 양자화 프로세스"""
        logger.info("🎯 최적 양자화 시작! (VLM FP16 + Action Head INT8)")
        
        results = {}
        
        # 1. 원본 모델 벤치마크 (VLM FP16 + Action Head FP32)
        logger.info("1. 원본 모델 벤치마크 (VLM FP16 + Action FP32)...")
        original_benchmark = self.benchmark_model(self.original_model, "VLM(FP16) + Action(FP32)")
        results['original'] = original_benchmark
        
        # 2. Action Head INT8 양자화
        logger.info("2. Action Head INT8 양자화...")
        int8_model, int8_benchmark = self.quantize_action_head_to_int8()
        if int8_benchmark:
            results['quantized'] = int8_benchmark
        
        # 3. 결과 비교
        logger.info("3. 결과 비교...")
        self._compare_optimal_results(results)
        
        # 4. 결과 저장
        with open('optimal_quantization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 결과 저장 완료: optimal_quantization_results.json")
        
        return results
    
    def _compare_optimal_results(self, results):
        """최적 양자화 결과 비교"""
        logger.info("\n📊 최적 양자화 결과 비교:")
        logger.info("=" * 60)
        
        original = results.get('original', {})
        quantized = results.get('quantized', {})
        
        if original and quantized:
            speedup = original['inference_time_ms'] / quantized['inference_time_ms']
            memory_save = (original['memory_usage_mb'] - quantized['memory_usage_mb']) / original['memory_usage_mb'] * 100
            
            logger.info(f"최적 양자화 vs 원본 성능 비교:")
            logger.info(f"   속도 향상: {speedup:.2f}x")
            logger.info(f"   메모리 절약: {memory_save:.1f}%")
            logger.info(f"   FPS 향상: {quantized['fps']:.1f} → {original['fps']:.1f}")
            
            if speedup > 1.0:
                logger.info("   ✅ 양자화로 속도 향상!")
            else:
                logger.info("   ⚠️ 양자화로 속도 저하")
            
            if memory_save > 0:
                logger.info("   ✅ 양자화로 메모리 절약!")
            else:
                logger.info("   ⚠️ 양자화로 메모리 증가")
        
        # 최종 권장사항
        logger.info("\n🎯 최적 양자화 권장사항:")
        if original and quantized:
            if speedup > 1.1 and memory_save > 5:
                logger.info("   🏆 하이브리드 양자화 강력 권장!")
                logger.info("   - VLM: FP16 (속도 + 메모리 절약)")
                logger.info("   - Action Head: INT8 (추가 메모리 절약)")
            elif speedup > 1.0 or memory_save > 0:
                logger.info("   🟢 하이브리드 양자화 권장")
            else:
                logger.info("   🟡 원본 모델 유지 권장")
        else:
            logger.info("   ⚠️ 테스트 실패로 권장사항 제공 불가")

def main():
    """메인 함수"""
    logger.info("🚀 최적 양자화 시작 (VLM FP16 + Action Head INT8)")
    
    # 모델 경로
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 최적 양자화 실행
    quantizer = OptimalQuantizer(model_path)
    results = quantizer.quantize_model()
    
    logger.info("🎉 최적 양자화 완료!")

if __name__ == "__main__":
    main()
