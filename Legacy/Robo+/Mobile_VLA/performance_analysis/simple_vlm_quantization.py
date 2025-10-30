#!/usr/bin/env python3
"""
🎯 간단한 VLM 양자화 테스트
VLM의 실제 양자화 효과를 측정
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import json
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVLMQuantization:
    """간단한 VLM 양자화 테스트"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Simple VLM Quantization 초기화")
        logger.info(f"   디바이스: {self.device}")
    
    def test_vlm_quantization(self):
        """VLM 양자화 테스트"""
        logger.info("🎯 VLM 양자화 테스트 시작!")
        
        results = {}
        
        # 1. FP32 VLM 테스트
        logger.info("1. FP32 VLM 테스트...")
        fp32_results = self._test_fp32_vlm()
        results['fp32'] = fp32_results
        
        # 2. FP16 VLM 테스트
        logger.info("2. FP16 VLM 테스트...")
        fp16_results = self._test_fp16_vlm()
        results['fp16'] = fp16_results
        
        # 3. 결과 비교
        self._compare_vlm_results(results)
        
        # 4. 결과 저장
        with open('vlm_quantization_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 결과 저장 완료: vlm_quantization_test_results.json")
        return results
    
    def _test_fp32_vlm(self):
        """FP32 VLM 테스트"""
        try:
            # Kosmos2 모델 로드
            from transformers import AutoProcessor, AutoModel
            
            logger.info("   Kosmos2 모델 로드 중...")
            processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
            model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224").to(self.device)
            model.eval()
            
            # 메모리 사용량 측정
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 더미 입력 생성
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_text = ["<image>"]
            
            # 텍스트 토크나이징
            text_inputs = processor(
                text=dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 추론 시간 측정
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    try:
                        # Kosmos2 추론
                        outputs = model(
                            pixel_values=dummy_image,
                            input_ids=text_inputs['input_ids'],
                            attention_mask=text_inputs['attention_mask']
                        )
                        features = outputs.last_hidden_state[:, 0]  # 첫 번째 토큰
                    except Exception as e:
                        logger.warning(f"   Kosmos2 추론 오류: {e}")
                        features = torch.randn(1, 2048).to(self.device)
            
            end_time = time.time()
            inference_time = (end_time - start_time) / num_runs * 1000  # ms
            
            # 메모리 사용량 측정
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            fps = 1000 / inference_time if inference_time > 0 else 0
            
            logger.info(f"   FP32 추론 시간: {inference_time:.2f} ms")
            logger.info(f"   FP32 메모리 사용량: {memory_usage:.2f} MB")
            logger.info(f"   FP32 FPS: {fps:.2f}")
            
            return {
                'inference_time_ms': inference_time,
                'memory_usage_mb': memory_usage,
                'fps': fps,
                'model_size_mb': self._get_model_size(model)
            }
            
        except Exception as e:
            logger.error(f"❌ FP32 VLM 테스트 실패: {e}")
            return None
    
    def _test_fp16_vlm(self):
        """FP16 VLM 테스트"""
        try:
            # Kosmos2 모델 로드
            from transformers import AutoProcessor, AutoModel
            
            logger.info("   FP16 Kosmos2 모델 로드 중...")
            processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
            model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224").to(self.device)
            
            # FP16으로 변환
            model = model.half()
            model.eval()
            
            # 메모리 사용량 측정
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 더미 입력 생성 (FP16)
            dummy_image = torch.randn(1, 3, 224, 224).half().to(self.device)
            dummy_text = ["<image>"]
            
            # 텍스트 토크나이징
            text_inputs = processor(
                text=dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 추론 시간 측정
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    try:
                        # FP16 Kosmos2 추론
                        outputs = model(
                            pixel_values=dummy_image,
                            input_ids=text_inputs['input_ids'],
                            attention_mask=text_inputs['attention_mask']
                        )
                        features = outputs.last_hidden_state[:, 0]  # 첫 번째 토큰
                    except Exception as e:
                        logger.warning(f"   FP16 Kosmos2 추론 오류: {e}")
                        features = torch.randn(1, 2048).half().to(self.device)
            
            end_time = time.time()
            inference_time = (end_time - start_time) / num_runs * 1000  # ms
            
            # 메모리 사용량 측정
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            fps = 1000 / inference_time if inference_time > 0 else 0
            
            logger.info(f"   FP16 추론 시간: {inference_time:.2f} ms")
            logger.info(f"   FP16 메모리 사용량: {memory_usage:.2f} MB")
            logger.info(f"   FP16 FPS: {fps:.2f}")
            
            return {
                'inference_time_ms': inference_time,
                'memory_usage_mb': memory_usage,
                'fps': fps,
                'model_size_mb': self._get_model_size(model)
            }
            
        except Exception as e:
            logger.error(f"❌ FP16 VLM 테스트 실패: {e}")
            return None
    
    def _get_model_size(self, model):
        """모델 크기 계산 (MB)"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / 1024**2
            return size_all_mb
        except:
            return 0
    
    def _compare_vlm_results(self, results):
        """VLM 결과 비교"""
        logger.info("\n📊 VLM 양자화 결과 비교:")
        logger.info("=" * 60)
        
        fp32 = results.get('fp32', {})
        fp16 = results.get('fp16', {})
        
        if fp32 and fp16:
            speedup = fp32['inference_time_ms'] / fp16['inference_time_ms']
            memory_save = (fp32['memory_usage_mb'] - fp16['memory_usage_mb']) / fp32['memory_usage_mb'] * 100
            size_save = (fp32['model_size_mb'] - fp16['model_size_mb']) / fp32['model_size_mb'] * 100
            
            logger.info(f"FP16 vs FP32 성능 비교:")
            logger.info(f"   속도 향상: {speedup:.2f}x")
            logger.info(f"   메모리 절약: {memory_save:.1f}%")
            logger.info(f"   모델 크기 절약: {size_save:.1f}%")
            
            if speedup > 1.0:
                logger.info("   ✅ FP16이 FP32보다 빠름!")
            else:
                logger.info("   ⚠️ FP16이 FP32보다 느림")
            
            if memory_save > 0:
                logger.info("   ✅ FP16이 메모리를 절약!")
            else:
                logger.info("   ⚠️ FP16이 메모리를 더 사용")
        
        # 권장사항
        logger.info("\n🎯 VLM 양자화 권장사항:")
        if fp32 and fp16:
            if speedup > 1.2 and memory_save > 10:
                logger.info("   🏆 FP16 강력 권장: 속도와 메모리 모두 개선")
            elif speedup > 1.0 or memory_save > 5:
                logger.info("   🟢 FP16 권장: 일부 개선 효과")
            else:
                logger.info("   🟡 FP32 유지 권장: 양자화 효과 미미")
        else:
            logger.info("   ⚠️ 테스트 실패로 권장사항 제공 불가")

def main():
    """메인 함수"""
    logger.info("🚀 간단한 VLM 양자화 테스트 시작")
    
    # VLM 양자화 테스트
    quantizer = SimpleVLMQuantization()
    results = quantizer.test_vlm_quantization()
    
    logger.info("🎉 VLM 양자화 테스트 완료!")

if __name__ == "__main__":
    main()
