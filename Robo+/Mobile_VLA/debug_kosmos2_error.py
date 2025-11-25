#!/usr/bin/env python3
"""
Kosmos2 NoneType 에러 디버깅 스크립트
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_kosmos2_error():
    """Kosmos2 NoneType 에러 디버깅"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 디바이스: {device}")
    
    try:
        # 1. Processor 로드
        logger.info("1. Processor 로드 중...")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        logger.info("✅ Processor 로드 완료")
        
        # 2. Model 로드
        logger.info("2. Model 로드 중...")
        model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        model = model.to(device)
        model.eval()
        logger.info("✅ Model 로드 완료")
        
        # 3. 입력 데이터 생성
        logger.info("3. 입력 데이터 생성...")
        batch_size = 1
        dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_text = ["<image>"] * batch_size
        
        logger.info(f"   이미지 shape: {dummy_image.shape}")
        logger.info(f"   텍스트: {dummy_text}")
        
        # 4. Processor 처리
        logger.info("4. Processor 처리...")
        text_inputs = processor(
            text=dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        logger.info(f"   input_ids shape: {text_inputs['input_ids'].shape}")
        logger.info(f"   attention_mask shape: {text_inputs['attention_mask'].shape}")
        
        # 5. 디바이스로 이동
        logger.info("5. 텐서를 디바이스로 이동...")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # 6. 모델 추론
        logger.info("6. 모델 추론 시작...")
        with torch.no_grad():
            logger.info("   pixel_values 전달...")
            logger.info(f"   pixel_values shape: {dummy_image.shape}")
            logger.info(f"   pixel_values dtype: {dummy_image.dtype}")
            
            logger.info("   input_ids 전달...")
            logger.info(f"   input_ids shape: {text_inputs['input_ids'].shape}")
            logger.info(f"   input_ids dtype: {text_inputs['input_ids'].dtype}")
            
            logger.info("   attention_mask 전달...")
            logger.info(f"   attention_mask shape: {text_inputs['attention_mask'].shape}")
            logger.info(f"   attention_mask dtype: {text_inputs['attention_mask'].dtype}")
            
            # 모델 호출
            outputs = model(
                pixel_values=dummy_image,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            
            logger.info("✅ 모델 추론 완료")
            logger.info(f"   outputs type: {type(outputs)}")
            logger.info(f"   outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            
            if hasattr(outputs, 'last_hidden_state'):
                logger.info(f"   last_hidden_state shape: {outputs.last_hidden_state.shape}")
                vision_features = outputs.last_hidden_state[:, 0]
                logger.info(f"   vision_features shape: {vision_features.shape}")
                logger.info("✅ Vision features 추출 성공!")
            else:
                logger.error("❌ last_hidden_state가 없습니다!")
                
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")

def test_kosmos2_simple():
    """간단한 Kosmos2 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 간단한 Kosmos2 테스트 - 디바이스: {device}")
    
    try:
        # 간단한 모델 로드
        model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # CPU에서 테스트
        model = model.cpu()
        model.eval()
        
        # 간단한 입력
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_text = ["<image>"]
        
        # Processor 처리
        text_inputs = processor(
            text=dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 추론
        with torch.no_grad():
            outputs = model(
                pixel_values=dummy_image,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            
            logger.info("✅ CPU에서 추론 성공!")
            logger.info(f"   last_hidden_state shape: {outputs.last_hidden_state.shape}")
            
    except Exception as e:
        logger.error(f"❌ CPU 테스트 실패: {e}")

if __name__ == "__main__":
    logger.info("🚀 Kosmos2 NoneType 에러 디버깅 시작")
    
    # 1. 간단한 CPU 테스트
    test_kosmos2_simple()
    
    # 2. 상세한 GPU 테스트
    debug_kosmos2_error()
