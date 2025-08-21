"""
🔍 Claw Matrix 차원 디버깅 스크립트
차원과 데이터타입 문제를 정확히 진단
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor

# 현재 구현의 Claw Matrix 임포트
from robovlms_style_single_image_model import RoboVLMStyleSingleImageModel, ClawMatrixFusion

def debug_claw_matrix():
    """Claw Matrix 차원 문제 디버깅"""
    
    print("🔍 Claw Matrix 차원 디버깅 시작")
    print("=" * 50)
    
    # 기본 설정
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 초기화 (Claw Matrix만 활성화)
    model = RoboVLMStyleSingleImageModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=3,
        hidden_dim=512,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=False,
        use_advanced_attention=False,
        z_axis_weight=0.05
    ).to(device)
    
    print(f"📊 모델 초기화 완료")
    print(f"   - Vision dim: {model.vision_dim}")
    print(f"   - Language dim: {model.language_dim}")
    print(f"   - Hidden dim: {model.hidden_dim}")
    
    # 더미 입력 생성
    batch_size = 8
    dummy_image = torch.randn(batch_size, 3, 720, 1280).to(device)
    dummy_text = "Navigate to target"
    
    print(f"\n🎯 입력 데이터:")
    print(f"   - Image shape: {dummy_image.shape}")
    print(f"   - Text: {dummy_text}")
    
    # 단계별 디버깅
    try:
        print(f"\n🔍 단계별 차원 추적:")
        
        # 1. Vision 특징 추출
        print(f"   1. Vision 특징 추출...")
        vision_features = model.extract_vision_features(dummy_image)
        print(f"      - Vision features shape: {vision_features.shape}")
        print(f"      - Vision features dtype: {vision_features.dtype}")
        
        # 2. Language 특징 추출
        print(f"   2. Language 특징 추출...")
        language_features = model.extract_language_features(dummy_text, batch_size)
        print(f"      - Language features shape: {language_features.shape}")
        print(f"      - Language features dtype: {language_features.dtype}")
        
        # 3. 기본 융합
        print(f"   3. 기본 융합...")
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        print(f"      - Fused features shape: {fused_features.shape}")
        print(f"      - Fused features dtype: {fused_features.dtype}")
        
        # 4. Claw Matrix 입력 분석
        print(f"   4. Claw Matrix 입력 분석...")
        print(f"      - Claw Matrix 초기화 확인:")
        if hasattr(model, 'claw_matrix'):
            claw = model.claw_matrix
            print(f"         - Vision proj: {claw.vision_proj}")
            print(f"         - Language proj: {claw.language_proj}")
            print(f"         - Action proj: {claw.action_proj}")
            
            # Claw Matrix 내부 차원 분석
            total_dim = fused_features.shape[-1]
            vision_dim_split = total_dim // 2
            language_dim_split = total_dim - vision_dim_split
            
            print(f"         - 입력 총 차원: {total_dim}")
            print(f"         - Vision 분할 차원: {vision_dim_split}")
            print(f"         - Language 분할 차원: {language_dim_split}")
            
            # 분할된 특징 확인
            vision_split = fused_features[:, :vision_dim_split]
            language_split = fused_features[:, vision_dim_split:]
            
            print(f"         - Vision split shape: {vision_split.shape}")
            print(f"         - Language split shape: {language_split.shape}")
            
            # 프로젝션 테스트
            try:
                print(f"   5. 프로젝션 테스트...")
                vision_proj_out = claw.vision_proj(vision_split)
                print(f"      - Vision projection 성공: {vision_proj_out.shape}")
            except Exception as e:
                print(f"      - Vision projection 실패: {e}")
                
            try:
                language_proj_out = claw.language_proj(language_split)
                print(f"      - Language projection 성공: {language_proj_out.shape}")
            except Exception as e:
                print(f"      - Language projection 실패: {e}")
        
        # 5. 전체 순전파 테스트
        print(f"   6. 전체 순전파 테스트...")
        try:
            output = model(dummy_image, dummy_text)
            print(f"      - ✅ 전체 순전파 성공!")
            print(f"      - Output shape: {output.shape}")
            print(f"      - Output dtype: {output.dtype}")
        except Exception as e:
            print(f"      - ❌ 전체 순전파 실패: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 디버깅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def test_claw_matrix_dimensions():
    """Claw Matrix 차원 호환성 테스트"""
    
    print("\n🧪 Claw Matrix 차원 호환성 테스트")
    print("=" * 40)
    
    # 다양한 차원 조합 테스트
    test_configs = [
        {"vision_dim": 1024, "language_dim": 1024, "hidden_dim": 512},
        {"vision_dim": 512, "language_dim": 512, "hidden_dim": 256},
        {"vision_dim": 1024, "language_dim": 2048, "hidden_dim": 512},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n테스트 {i+1}: {config}")
        
        try:
            # Claw Matrix 단독 테스트
            claw = ClawMatrixFusion(
                vision_dim=config['vision_dim'],
                language_dim=config['language_dim'],
                action_dim=3,
                hidden_dim=config['hidden_dim'],
                dropout=0.1
            )
            
            # 테스트 입력 생성
            batch_size = 4
            total_dim = config['vision_dim'] + config['language_dim']
            test_input = torch.randn(batch_size, total_dim)
            
            print(f"   - 입력 shape: {test_input.shape}")
            
            # 순전파 테스트
            output = claw(test_input)
            print(f"   - ✅ 성공! 출력 shape: {output.shape}")
            
        except Exception as e:
            print(f"   - ❌ 실패: {e}")

if __name__ == "__main__":
    debug_claw_matrix()
    test_claw_matrix_dimensions()
