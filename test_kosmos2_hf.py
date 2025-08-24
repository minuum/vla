#!/usr/bin/env python3
"""
Microsoft Kosmos-2 Hugging Face 모델 테스트 스크립트
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import time
import psutil
import os

def get_memory_usage():
    """메모리 사용량 확인"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def test_kosmos2_loading():
    """Kosmos-2 모델 로딩 테스트"""
    print("🚀 Microsoft Kosmos-2 모델 로딩 테스트 시작")
    print("=" * 50)
    
    # 초기 메모리 상태
    initial_memory = get_memory_usage()
    print(f"📊 초기 메모리 사용량: {initial_memory['rss']:.1f} MB (RSS)")
    
    try:
        # 모델명 설정 (2B 파라미터 버전)
        model_name = "microsoft/kosmos-2-patch14-224"
        
        print(f"📦 모델 로딩 중: {model_name}")
        start_time = time.time()
        
        # 프로세서 로딩
        print("🔧 프로세서 로딩 중...")
        processor = AutoProcessor.from_pretrained(model_name)
        processor_time = time.time() - start_time
        print(f"✅ 프로세서 로딩 완료: {processor_time:.2f}초")
        
        # 모델 로딩 (메모리 최적화)
        print("🧠 모델 로딩 중...")
        model_start_time = time.time()
        
        # 메모리 최적화 설정
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # FP16 사용
            device_map="auto",  # 자동 디바이스 매핑
            low_cpu_mem_usage=True,  # 낮은 CPU 메모리 사용
            trust_remote_code=True
        )
        
        model_time = time.time() - model_start_time
        total_time = time.time() - start_time
        
        print(f"✅ 모델 로딩 완료: {model_time:.2f}초")
        print(f"⏱️ 총 로딩 시간: {total_time:.2f}초")
        
        # 로딩 후 메모리 상태
        loaded_memory = get_memory_usage()
        memory_increase = loaded_memory['rss'] - initial_memory['rss']
        print(f"📊 로딩 후 메모리 사용량: {loaded_memory['rss']:.1f} MB (RSS)")
        print(f"📈 메모리 증가량: {memory_increase:.1f} MB")
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 모델 파라미터 수: {total_params:,}")
        print(f"📊 훈련 가능한 파라미터 수: {trainable_params:,}")
        print(f"📊 모델 크기 (예상): {total_params * 2 / (1024**3):.2f} GB (FP16)")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None, None

def test_kosmos2_inference(model, processor):
    """Kosmos-2 추론 테스트"""
    print("\n🎯 Kosmos-2 추론 테스트 시작")
    print("=" * 50)
    
    try:
        # 테스트 이미지 생성 (더미 이미지)
        print("🖼️ 테스트 이미지 생성 중...")
        dummy_image = torch.randn(3, 224, 224)  # RGB 이미지
        
        # 텍스트 프롬프트
        text_prompt = "이 이미지를 자세히 설명해주세요."
        
        print(f"📝 프롬프트: {text_prompt}")
        
        # 추론 시작
        print("🧠 추론 실행 중...")
        inference_start = time.time()
        
        # 입력 준비
        inputs = processor(
            images=dummy_image,
            text=text_prompt,
            return_tensors="pt"
        )
        
        # GPU로 이동 (가능한 경우)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
            print("🚀 GPU 사용 중")
        else:
            print("💻 CPU 사용 중")
        
        # 추론 실행
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                early_stopping=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - inference_start
        
        # 결과 디코딩
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ 추론 완료: {inference_time:.2f}초")
        print(f"📝 생성된 텍스트: {generated_text}")
        
        # 메모리 사용량 확인
        final_memory = get_memory_usage()
        print(f"📊 추론 후 메모리 사용량: {final_memory['rss']:.1f} MB (RSS)")
        
        return True
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return False

def test_memory_optimized_inference():
    """메모리 최적화된 추론 테스트"""
    print("\n💾 메모리 최적화된 추론 테스트")
    print("=" * 50)
    
    try:
        model_name = "microsoft/kosmos-2-patch14-224"
        
        # 8비트 양자화로 로딩
        print("🔧 8비트 양자화 모델 로딩 중...")
        start_time = time.time()
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            load_in_8bit=True,  # 8비트 양자화
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        print(f"✅ 8비트 모델 로딩 완료: {load_time:.2f}초")
        
        # 메모리 사용량
        memory = get_memory_usage()
        print(f"📊 8비트 모델 메모리 사용량: {memory['rss']:.1f} MB (RSS)")
        
        # 간단한 추론 테스트
        dummy_image = torch.randn(3, 224, 224)
        text_prompt = "이미지 설명"
        
        inputs = processor(
            images=dummy_image,
            text=text_prompt,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=2
            )
        
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 8비트 모델 결과: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ 8비트 모델 테스트 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🎯 Microsoft Kosmos-2 Hugging Face 모델 테스트")
    print("=" * 60)
    
    # 시스템 정보 출력
    print(f"💻 CPU 코어 수: {psutil.cpu_count()}")
    print(f"💾 총 메모리: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"🚀 CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"🚀 CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # 1. 기본 모델 로딩 테스트
    model, processor = test_kosmos2_loading()
    
    if model is not None and processor is not None:
        # 2. 추론 테스트
        inference_success = test_kosmos2_inference(model, processor)
        
        if inference_success:
            print("\n✅ 모든 테스트 성공!")
        else:
            print("\n⚠️ 추론 테스트 실패")
    
    # 3. 메모리 최적화 테스트
    print("\n" + "=" * 60)
    test_memory_optimized_inference()
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    main()
