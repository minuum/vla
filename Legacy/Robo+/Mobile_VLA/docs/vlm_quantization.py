#!/usr/bin/env python3
"""
🎯 VLM + Action Head 통합 양자화
VLM(Kosmos2)과 Action Head 모두에 양자화 적용
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
from transformers import AutoProcessor, AutoModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMQuantizedModel(nn.Module):
    """VLM과 Action Head 모두 양자화된 모델"""
    
    def __init__(self, model_path: str):
        super().__init__()
        
        # Kosmos2 모델 로드
        self.processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # Action Head (LSTM + MLP)
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
        
        # 체크포인트에서 Action Head만 로드
        self._load_action_head(model_path)
        
        # VLM을 평가 모드로 설정 (훈련하지 않음)
        self.kosmos.eval()
        for param in self.kosmos.parameters():
            param.requires_grad = False
    
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
        """순전파 (VLM + Action Head)"""
        batch_size = x.size(0)
        
        # 1. VLM으로 이미지 특징 추출
        with torch.no_grad():
            # 더미 텍스트 생성
            dummy_text = ["<image>"] * batch_size
            
            # 텍스트 토크나이징
            text_inputs = self.processor(
                text=dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(x.device)
            
            # Kosmos2로 특징 추출
            vision_outputs = self.kosmos(
                pixel_values=x,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            
            # 특징 추출 (첫 번째 토큰)
            vision_features = vision_outputs.last_hidden_state[:, 0]  # [batch_size, 2048]
        
        # 2. Action Head로 액션 예측
        sequence_features = vision_features.unsqueeze(1)  # [batch_size, 1, 2048]
        rnn_out, _ = self.rnn(sequence_features)  # [batch_size, 1, 4096]
        actions = self.actions(rnn_out.squeeze(1))  # [batch_size, 2]
        
        return actions

class VLMQuantizer:
    """VLM + Action Head 통합 양자화기"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 원본 모델 로드
        self.original_model = VLMQuantizedModel(model_path).to(self.device)
        self.original_model.eval()
        
        logger.info(f"🚀 VLM Quantizer 초기화 완료")
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
    
    def quantize_to_fp16(self):
        """FP16 양자화 (VLM + Action Head)"""
        logger.info("🔧 FP16 양자화 시작...")
        
        try:
            # 모델을 FP16으로 변환
            fp16_model = self.original_model.half()
            fp16_model.eval()
            
            # 벤치마크
            benchmark = self.benchmark_model(fp16_model, "FP16")
            
            # 모델 저장
            torch.save(fp16_model.state_dict(), 'vlm_fp16_model.pth')
            logger.info("💾 FP16 모델 저장 완료")
            
            return fp16_model, benchmark
            
        except Exception as e:
            logger.error(f"❌ FP16 양자화 실패: {e}")
            return None, None
    
    def quantize_to_int8(self):
        """INT8 양자화 (Action Head만)"""
        logger.info("🔧 INT8 양자화 시작...")
        
        try:
            # Action Head만 INT8 양자화 (VLM은 복잡해서 제외)
            int8_model = quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear, nn.RNN},  # Action Head의 Linear와 RNN만 양자화
                dtype=torch.qint8
            )
            int8_model.eval()
            
            # 벤치마크
            benchmark = self.benchmark_model(int8_model, "INT8")
            
            # 모델 저장
            torch.save(int8_model.state_dict(), 'vlm_int8_model.pth')
            logger.info("💾 INT8 모델 저장 완료")
            
            return int8_model, benchmark
            
        except Exception as e:
            logger.error(f"❌ INT8 양자화 실패: {e}")
            return None, None
    
    def quantize_model(self):
        """전체 양자화 프로세스"""
        logger.info("🎯 VLM + Action Head 통합 양자화 시작!")
        
        results = {}
        
        # 1. FP32 벤치마크 (원본)
        logger.info("1. FP32 벤치마크...")
        fp32_benchmark = self.benchmark_model(self.original_model, "FP32")
        results['fp32'] = fp32_benchmark
        
        # 2. FP16 양자화
        logger.info("2. FP16 양자화...")
        fp16_model, fp16_benchmark = self.quantize_to_fp16()
        if fp16_benchmark:
            results['fp16'] = fp16_benchmark
        
        # 3. INT8 양자화
        logger.info("3. INT8 양자화...")
        int8_model, int8_benchmark = self.quantize_to_int8()
        if int8_benchmark:
            results['int8'] = int8_benchmark
        
        # 4. 결과 비교
        logger.info("4. 결과 비교...")
        self._compare_results(results)
        
        # 5. 결과 저장
        with open('vlm_quantization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 결과 저장 완료: vlm_quantization_results.json")
        
        return results
    
    def _compare_results(self, results):
        """결과 비교 및 분석"""
        logger.info("\n📊 양자화 결과 비교:")
        logger.info("=" * 60)
        
        fp32 = results.get('fp32', {})
        fp16 = results.get('fp16', {})
        int8 = results.get('int8', {})
        
        if fp32 and fp16:
            speedup_fp16 = fp32['inference_time_ms'] / fp16['inference_time_ms']
            memory_save_fp16 = (fp32['memory_usage_mb'] - fp16['memory_usage_mb']) / fp32['memory_usage_mb'] * 100
            
            logger.info(f"FP16 vs FP32:")
            logger.info(f"   속도 향상: {speedup_fp16:.2f}x")
            logger.info(f"   메모리 절약: {memory_save_fp16:.1f}%")
        
        if fp32 and int8:
            speedup_int8 = fp32['inference_time_ms'] / int8['inference_time_ms']
            memory_save_int8 = (fp32['memory_usage_mb'] - int8['memory_usage_mb']) / fp32['memory_usage_mb'] * 100
            
            logger.info(f"INT8 vs FP32:")
            logger.info(f"   속도 향상: {speedup_int8:.2f}x")
            logger.info(f"   메모리 절약: {memory_save_int8:.1f}%")
        
        # 최적 권장사항
        logger.info("\n🎯 권장사항:")
        if fp16 and int8:
            if fp16['fps'] > int8['fps']:
                logger.info("   🏆 FP16 권장: 더 빠른 추론 속도")
            else:
                logger.info("   🏆 INT8 권장: 더 작은 메모리 사용량")
        elif fp16:
            logger.info("   🏆 FP16 권장: 안정적인 성능")
        elif int8:
            logger.info("   🏆 INT8 권장: 메모리 효율적")

def main():
    """메인 함수"""
    logger.info("🚀 VLM + Action Head 통합 양자화 시작")
    
    # 모델 경로
    model_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 양자화 실행
    quantizer = VLMQuantizer(model_path)
    results = quantizer.quantize_model()
    
    logger.info("🎉 VLM + Action Head 통합 양자화 완료!")

if __name__ == "__main__":
    main()
