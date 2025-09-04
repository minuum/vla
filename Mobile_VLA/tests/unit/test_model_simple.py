#!/usr/bin/env python3
"""
간단한 Mobile VLA 모델 로더 테스트
"""

import torch
import torch.nn as nn
import os

class SimpleMobileVLAModel(nn.Module):
    """간단한 Mobile VLA 모델 구조"""
    
    def __init__(self):
        super().__init__()
        
        # Vision Encoder
        self.vision_encoder = nn.Linear(2048, 4096)
        
        # Text Encoder  
        self.text_encoder = nn.Linear(2048, 4096)
        
        # LSTM
        self.lstm = nn.LSTM(8192, 4096, batch_first=True)
        
        # Action Head
        self.action_head = nn.Linear(4096, 2)
    
    def forward(self, vision_features, text_features):
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)
        lstm_out, _ = self.lstm(combined.unsqueeze(1))
        actions = self.action_head(lstm_out.squeeze(1))
        return actions

def test_cuda():
    """CUDA 테스트"""
    print("🔧 CUDA 테스트 중...")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        
        # 간단한 CUDA 연산 테스트
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        print(f"CUDA 연산 성공: {z.shape}")
    else:
        print("❌ CUDA를 사용할 수 없습니다.")

def test_model():
    """모델 테스트"""
    print("\n🤖 모델 테스트 중...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 생성
    model = SimpleMobileVLAModel().to(device)
    model.eval()
    
    # 테스트 입력
    batch_size = 2
    vision_features = torch.randn(batch_size, 2048).to(device)
    text_features = torch.randn(batch_size, 2048).to(device)
    
    # 추론
    with torch.no_grad():
        actions = model(vision_features, text_features)
        print(f"추론 성공: {actions.shape}")
        print(f"출력 값: {actions}")

def test_checkpoint():
    """체크포인트 테스트"""
    print("\n📦 체크포인트 테스트 중...")
    
    # 체크포인트 파일 찾기
    checkpoint_paths = [
        "./Robo+/Mobile_VLA/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
        "./mobile-vla-omniwheel/best_simple_lstm_model.pth"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ 체크포인트 발견: {path}")
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   크기: {size_mb:.1f} MB")
            
            try:
                # 체크포인트 로드 시도
                checkpoint = torch.load(path, map_location='cpu')
                print(f"   로드 성공: {type(checkpoint)}")
                
                if isinstance(checkpoint, dict):
                    print(f"   키들: {list(checkpoint.keys())}")
                
            except Exception as e:
                print(f"   로드 실패: {e}")
        else:
            print(f"❌ 체크포인트 없음: {path}")

def main():
    """메인 함수"""
    print("🧪 Mobile VLA 모델 로더 테스트")
    print("=" * 50)
    
    test_cuda()
    test_model()
    test_checkpoint()
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()
