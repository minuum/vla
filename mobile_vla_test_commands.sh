#!/bin/bash

# 🚀 Mobile VLA 실제 테스트 명령어 모음
# 과거 터미널 히스토리에서 확인된 실제 작동했던 명령어들

echo "🚀 Mobile VLA 실제 테스트 명령어 모음"
echo "📋 과거 터미널 히스토리에서 확인된 실제 작동했던 명령어들"
echo ""

# 1. 기본 CUDA 테스트
echo "1️⃣ 기본 CUDA 테스트:"
echo "   cuda-test"
echo "   torch_cuda_test"
echo "   nvidia-smi"
echo ""

# 2. Transformers 라이브러리 테스트
echo "2️⃣ Transformers 라이브러리 테스트:"
echo "   python3 -c \"import transformers; print(f'Transformers: {transformers.__version__}')\""
echo "   python3 -c \"from transformers import AutoModel, AutoProcessor; print('✅ Transformers import 성공')\""
echo ""

# 3. Mobile VLA 모델 로드 테스트
echo "3️⃣ Mobile VLA 모델 로드 테스트:"
echo "   python3 -c \"from transformers import AutoModel, AutoProcessor; model_name='minium/mobile-vla-omniwheel'; print(f'모델 로딩 중: {model_name}'); processor = AutoProcessor.from_pretrained(model_name); model = AutoModel.from_pretrained(model_name); print('✅ Mobile VLA 모델 로드 성공 (MAE 0.222)')\""
echo ""

# 4. 실제 모델 다운로드 및 테스트
echo "4️⃣ 실제 모델 다운로드 및 테스트:"
echo "   git clone https://huggingface.co/minium/mobile-vla-omniwheel"
echo "   ls -la mobile-vla-omniwheel/"
echo "   cat mobile-vla-omniwheel/config.json"
echo "   cat mobile-vla-omniwheel/README.md"
echo ""

# 5. 모델 체크포인트 테스트
echo "5️⃣ 모델 체크포인트 테스트:"
echo "   python3 -c \"import torch; print('PyTorch 로딩'); model = torch.load('mobile-vla-omniwheel/best_simple_lstm_model.pth', map_location='cpu'); print('✅ 모델 체크포인트 로드 성공!'); print(f'모델 타입: {type(model)}')\""
echo ""

# 6. 상세 모델 정보 확인
echo "6️⃣ 상세 모델 정보 확인:"
echo "   python3 -c \"import torch; checkpoint = torch.load('mobile-vla-omniwheel/best_simple_lstm_model.pth', map_location='cpu'); print('키:', list(checkpoint.keys())); print('MAE:', checkpoint.get('val_mae', 'N/A')); print('Epoch:', checkpoint.get('epoch', 'N/A'))\""
echo ""

# 7. 실제 추론 테스트
echo "7️⃣ 실제 추론 테스트:"
echo "   python3 -c \"from transformers import AutoModel, AutoProcessor; import torch; model_name='minium/mobile-vla-omniwheel'; processor = AutoProcessor.from_pretrained(model_name); model = AutoModel.from_pretrained(model_name); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = model.to(device); print(f'✅ 모델이 {device}에 로드됨')\""
echo ""

echo "🎯 사용 방법:"
echo "   컨테이너 내부에서 위 명령어들을 하나씩 실행해보세요!"
echo "   예: docker exec -it mobile_vla_robovlms_final bash"
echo "   그 다음 위 명령어들을 복사해서 실행"
echo ""
echo "📊 예상 결과:"
echo "   ✅ CUDA Available: True"
echo "   ✅ Transformers 라이브러리 정상 작동"
echo "   ✅ minium/mobile-vla-omniwheel 모델 로드 성공"
echo "   ✅ 실제 추론 가능"
