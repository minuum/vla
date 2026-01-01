# Phase 2 추론 엔진 테스트 오류 히스토리

**일시**: 2026-01-02 01:00 ~ 02:24 KST  
**목표**: 로컬 추론 엔진 (`test_local_inference_engine.py`) 테스트

---

## 📊 오류 발생 히스토리

| 시도 | 시간 | 오류 유형 | 오류 메시지 | 원인 분석 | 해결 시도 |
|------|------|-----------|-------------|-----------|-----------|
| **#1** | ~01:03 | **Import Error** | `ModuleNotFoundError: No module named 'robovlms.train.mobile_vla_trainer'` | PYTHONPATH 미설정 | `run_phase2_test.sh`에 PYTHONPATH 추가 |
| **#2** | ~01:04 | **Import Error** | `ModuleNotFoundError: No module named 'robovlms.train.mobile_vla_trainer'` | `poetry run`이 PYTHONPATH를 전달하지 못함 | Import 경로를 `Robo+` 구조에 맞게 수정 시도 |
| **#3** | ~01:10 | **Import Error** | `ModuleNotFoundError: No module named 'robovlms.train.mobile_vla_trainer'` | 파일 내부에서 여전히 잘못된 import 경로 사용 | `robovlms_mobile_vla_inference.py` 179번 라인 수정: `from Mobile_VLA.core.train_core.mobile_vla_trainer` |
| **#4** | ~01:15 | **File Error** | 코드 중복 (264-303라인) | `replace_file_content` 툴 사용 시 중복 코드 삽입 | 중복 코드 블록 제거 |
| **#5** | ~01:20 | **Logic Error** | `forward_continuous` 메소드 호출 실패 예상 | MobileVLATrainer는 일반 `forward` 사용, Lightning 아님 | 체크포인트 로딩 방식 변경: `torch.load()` + `load_state_dict()` |
| **#6** | ~01:25 | **Logic Error** | 모델 forward 호출 방식 불일치 | `forward_continuous`를 호출했으나 MobileVLATrainer는 일반 `forward` | `forward()` 호출로 변경, 출력 형식 수정 |
| **#7** | ~02:00 | **Performance** | 모델 로딩 중 무한 대기 (2분+) | 6.4GB 체크포인트를 Jetson Orin에서 CPU→FP16→GPU로 로딩하는데 시간 소요 | 대기 중 (정상적인 로딩 과정) |

---

## 🔍 주요 오류 상세 분석

### 1️⃣ Import Error (시도 #1~3)

**문제**:
```python
ModuleNotFoundError: No module named 'robovlms.train.mobile_vla_trainer'
```

**원인**:
- `src/robovlms_mobile_vla_inference.py` 179번 라인에서 잘못된 import 경로 사용
- 실제 파일 위치: `Robo+/Mobile_VLA/core/train_core/mobile_vla_trainer.py`
- Import 시도 경로: `robovlms.train.mobile_vla_trainer` (존재하지 않음)

**해결 과정**:
```python
# Before (잘못된 경로)
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.data_utils import get_text_function

# After (올바른 경로)
# 1. Robo+ 경로를 sys.path에 추가
robo_plus_path = Path(__file__).parent.parent / "Robo+"
sys.path.insert(0, str(robo_plus_path))

# 2. 올바른 import
from Mobile_VLA.core.train_core.mobile_vla_trainer import MobileVLATrainer
```

---

### 2️⃣ 체크포인트 로딩 방식 오류 (시도 #5)

**문제**:
```python
# 잘못된 방식 (PyTorch Lightning 방식)
self.model = MobileVLATrainer.load_from_checkpoint(
    self.config.checkpoint_path,
    map_location='cpu'
)
```

**원인**:
- `MobileVLATrainer`는 PyTorch Lightning Trainer가 아님
- 일반 PyTorch 모델이므로 `load_from_checkpoint` 메소드 없음

**해결**:
```python
# 올바른 방식 (일반 PyTorch 방식)
# 1. 체크포인트 로드
checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')

# 2. Trainer 생성 (설정 정보로)
trainer = MobileVLATrainer(
    model_name=checkpoint['config']['model_name'],
    action_dim=checkpoint['config']['action_dim'],
    ...
)

# 3. State dict 로드
trainer.model.load_state_dict(checkpoint['model_state_dict'])
```

---

### 3️⃣ Forward 호출 방식 불일치 (시도 #6)

**문제**:
```python
# 잘못된 방식
result = self.model.model.forward_continuous(
    images, text_tokens,
    attention_mask=text_mask,
    ...
)
```

**원인**:
- `MobileVLATrainer.model`은 Kosmos 기반 커스텀 모델
- `forward_continuous` 메소드 없음
- 일반 `forward(pixel_values, input_ids, attention_mask)` 사용

**해결**:
```python
# 올바른 방식
result = self.model(
    pixel_values=images,  # (1, window_size, 3, H, W)
    input_ids=text_tokens,
    attention_mask=text_mask
)

# 출력 형식
# result = {'predicted_actions': (B, chunk_size, action_dim), ...}
actions = result['predicted_actions'][0]  # (chunk_size, action_dim)
```

---

## 📝 현재 상태

### ✅ 해결된 문제들
1. ✅ Import 경로 수정 완료
2. ✅ 체크포인트 로딩 방식 수정 완료  
3. ✅ Forward 호출 방식 수정 완료
4. ✅ 중복 코드 제거 완료
5. ✅ Python 캐시 클리어 완료

### ⏳ 진행 중
- **모델 로딩 대기**: Jetson Orin에서 6.4GB 체크포인트 로딩 중
- **예상 소요 시간**: 
  - Kosmos-2 Processor 로딩: ~30초 ✅ (완료)
  - Checkpoint 로딩 (CPU): ~1-2분
  - FP16 변환: ~30초
  - GPU 전송: ~30초
  - **총 예상**: 3-5분

### 🔄 다음 단계
1. 모델 로딩 완료 대기
2. 추론 테스트 실행
3. 성능 측정 (FPS, 지연시간)
4. ROS2 노드 통합 테스트

---

## 💡 학습한 교훈

| 교훈 | 설명 |
|------|------|
| **1. 디렉토리 구조 확인** | Import 경로 수정 전에 반드시 실제 파일 위치 확인 (`find` 명령어 활용) |
| **2. 모델 프레임워크 확인** | PyTorch Lightning vs 일반 PyTorch 구분 필수 |
| **3. API 문서 확인** | 모델 클래스의 forward 메소드 시그니처 확인 |
| **4. 캐시 관리** | Python `__pycache__` 삭제로 변경사항 즉시 반영 |
| **5. Jetson 성능 고려** | 대용량 모델 로딩 시 충분한 시간 확보 (3-5분) |

---

## 🔧 최종 수정 파일

### `/home/soda/vla/src/robovlms_mobile_vla_inference.py`

**주요 변경사항**:
1. **Import 경로** (177-188번 라인):
   ```python
   robo_plus_path = Path(__file__).parent.parent / "Robo+"
   sys.path.insert(0, str(robo_plus_path))
   from Mobile_VLA.core.train_core.mobile_vla_trainer import MobileVLATrainer
   ```

2. **체크포인트 로딩** (195-248번 라인):
   ```python
   checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')
   trainer = MobileVLATrainer(...)
   trainer.model.load_state_dict(checkpoint['model_state_dict'])
   ```

3. **Forward 호출** (307-325번 라인):
   ```python
   result = self.model(pixel_values=images, input_ids=text_tokens, ...)
   actions = result['predicted_actions'][0]
   ```

### `/home/soda/vla/scripts/run_phase2_test.sh`

**주요 변경사항**:
- PYTHONPATH 환경 변수 추가 (Robo+ 경로 포함)

---

## 📌 참고 자료

- 체크포인트 경로: `runs/mobile_vla_no_chunk_20251209/.../epoch_epoch=06-val_loss=val_loss=0.067.ckpt`
- 체크포인트 크기: **6.4GB**
- 모델 구조: Kosmos-2 + Action Head (LSTM + MLP)
- 훈련 설정: window_size=2, chunk_size=10, action_dim=2
