#!/bin/bash

# Mobile VLA Poetry 환경 활성화 스크립트

export PATH="$HOME/.local/bin:$PATH"

echo "🔧 Poetry 환경 정보:"
poetry env info

echo "🐍 Python 인터프리터 확인:"
PYTHON_PATH="/home/billy/.cache/pypoetry/virtualenvs/mobile-vla-FNblWQUj-py3.10/bin/python"
echo "경로: $PYTHON_PATH"
echo "존재 여부: $(test -f $PYTHON_PATH && echo '✅ 존재' || echo '❌ 없음')"
echo "실행 가능 여부: $(test -x $PYTHON_PATH && echo '✅ 실행 가능' || echo '❌ 실행 불가')"

echo "🧪 환경 테스트:"
$PYTHON_PATH -c "
import sys
print(f'Python 버전: {sys.version}')
print(f'실행 경로: {sys.executable}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except:
    print('PyTorch: 설치되지 않음')
try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except:
    print('Transformers: 설치되지 않음')
"

echo "✅ 환경 활성화 완료!"
echo "이제 다음 명령어로 Python을 실행하세요:"
echo "source /home/billy/.cache/pypoetry/virtualenvs/mobile-vla-FNblWQUj-py3.10/bin/activate"

