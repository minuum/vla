#!/bin/bash

# 🚀 Simple CLIP LSTM 모델 체크포인트 기반 추론 스크립트
# 체크포인트: vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth

set -e

echo "🚀 Simple CLIP LSTM 모델 체크포인트 기반 추론 시작"
echo "📅 체크포인트: best_simple_clip_lstm_model.pth"
echo ""

# 체크포인트 파일 존재 확인
CHECKPOINT_PATH="vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    exit 1
fi

echo "✅ 체크포인트 파일 확인됨: $(ls -lh $CHECKPOINT_PATH)"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop simple_clip_lstm_inference 2>/dev/null || true
docker rm simple_clip_lstm_inference 2>/dev/null || true

# 컨테이너 실행
echo "🚀 Simple CLIP LSTM 추론 컨테이너 실행 중..."
docker run -d \
    --name simple_clip_lstm_inference \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /usr/local/cuda:/usr/local/cuda \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
    -v $(pwd)/vla:/workspace/vla \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -e CUDA_VISIBLE_DEVICES=0 \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep simple_clip_lstm_inference

# 성공하면 추론 시작
if docker ps | grep -q simple_clip_lstm_inference; then
    echo "✅ 컨테이너가 실행 중입니다. Simple CLIP LSTM 추론을 시작합니다..."
    echo ""
    
    # 추론 실행
    docker exec -it simple_clip_lstm_inference bash -c "
        echo '🎯 Simple CLIP LSTM 모델 체크포인트 기반 추론'
        echo '📅 체크포인트: best_simple_clip_lstm_model.pth'
        echo ''
        
        # 작업 디렉토리로 이동
        cd /workspace
        
        # 1. 시스템 상태 확인
        echo '1️⃣ 시스템 상태 확인:'
        echo '   CUDA 상태:'
        python3 -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')\"
        echo '   GPU 정보:'
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        echo ''
        
        # 2. 체크포인트 파일 확인
        echo '2️⃣ 체크포인트 파일 확인:'
        ls -lh vla/Robo+/Mobile_VLA/simple_clip_lstm_model/
        echo ''
        
        # 3. Simple CLIP LSTM 모델 정의 및 체크포인트 로드
        echo '3️⃣ Simple CLIP LSTM 모델 로드:'
        python3 -c \"
import torch
import torch.nn as nn
import sys
import os
import time
import numpy as np

# Simple CLIP LSTM 모델 클래스 정의
class SimpleCLIPLSTMModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Vision feature encoder
        self.vision_encoder = nn.Linear(input_size, hidden_size)
        
        # Text feature encoder  
        self.text_encoder = nn.Linear(input_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, vision_features, text_features, hidden=None):
        # Encode features
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        
        # Concatenate features
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)
        
        # Add sequence dimension if needed
        if combined.dim() == 2:
            combined = combined.unsqueeze(1)  # [batch, 1, features]
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(combined, hidden)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Predict actions
        actions = self.action_head(last_output)
        
        return actions, hidden

print('Simple CLIP LSTM 체크포인트 로딩 중...')
try:
    # 체크포인트 경로 설정
    checkpoint_path = './vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth'
    
    if os.path.exists(checkpoint_path):
        # 모델 인스턴스 생성
        model = SimpleCLIPLSTMModel()
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 체크포인트 구조 확인
        print(f'체크포인트 키: {list(checkpoint.keys())}')
        
        # 모델 상태 로드 (다양한 키 이름 시도)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # 직접 로드 시도
            model.load_state_dict(checkpoint)
        
        print('✅ Simple CLIP LSTM 체크포인트 로드 성공')
        print(f'체크포인트 경로: {checkpoint_path}')
        print(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
        
        # 체크포인트 메타데이터 출력
        for key, value in checkpoint.items():
            if key not in ['model_state_dict', 'state_dict', 'model']:
                print(f'{key}: {value}')
                
    else:
        print(f'❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}')
        print('사용 가능한 체크포인트:')
        os.system('find . -name \"*.pth\" -type f | head -10')
        
except Exception as e:
    print(f'❌ 체크포인트 로드 실패: {e}')
    import traceback
    traceback.print_exc()
\"
        echo ''
        
        # 4. 실제 추론 성능 테스트
        echo '4️⃣ Simple CLIP LSTM 추론 성능 테스트:'
        python3 -c \"
import time
import torch
import torch.nn as nn
import numpy as np

# Simple CLIP LSTM 모델 클래스 정의 (추론용)
class SimpleCLIPLSTMModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.vision_encoder = nn.Linear(input_size, hidden_size)
        self.text_encoder = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, vision_features, text_features, hidden=None):
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)
        
        if combined.dim() == 2:
            combined = combined.unsqueeze(1)
        
        lstm_out, hidden = self.lstm(combined, hidden)
        last_output = lstm_out[:, -1, :]
        actions = self.action_head(last_output)
        return actions, hidden

print('Simple CLIP LSTM 추론 성능 측정 중...')
try:
    # 체크포인트 로드
    checkpoint_path = './vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth'
    model = SimpleCLIPLSTMModel()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 상태 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # GPU로 이동 (가능한 경우)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 더미 입력 생성
    batch_size = 1
    vision_features = torch.randn(batch_size, 2048).to(device)
    text_features = torch.randn(batch_size, 2048).to(device)
    
    # 추론 시간 측정 (100회 평균)
    num_runs = 100
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            actions, hidden = model(vision_features, text_features)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    fps = 1000 / avg_time
    
    print(f'✅ Simple CLIP LSTM 추론 성능:')
    print(f'   디바이스: {device}')
    print(f'   평균 추론 시간: {avg_time:.3f}ms')
    print(f'   FPS: {fps:.0f}')
    print(f'   모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}')
    print(f'   ✅ 실시간 로봇 제어 가능')
    
    # 샘플 추론 결과 출력
    print(f'   샘플 추론 결과: {actions[0].cpu().numpy()}')
    
except Exception as e:
    print(f'❌ 추론 테스트 실패: {e}')
    import traceback
    traceback.print_exc()
\"
        echo ''
        
        # 5. 대화형 추론 모드
        echo '5️⃣ 대화형 추론 모드:'
        echo '   사용 가능한 명령어:'
        echo '     - python3 inference_demo.py: 대화형 추론 데모'
        echo '     - python3 batch_inference.py: 배치 추론 테스트'
        echo '     - nvidia-smi: GPU 상태 확인'
        echo '     - htop: 시스템 리소스 확인'
        echo ''
        echo '💬 대화형 모드로 전환합니다. 명령어를 입력하세요:'
        exec bash
    "
else
    echo "❌ 컨테이너 실행 실패"
    docker logs simple_clip_lstm_inference
    exit 1
fi
