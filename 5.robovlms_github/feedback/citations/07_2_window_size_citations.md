# 07_2 Window Size Analysis - RoboVLMs GitHub Citations

## 📊 **Window Size Technical Analysis**

### **7.2.1 Window Size Definition**
- **Source**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:561`  # GitHub 코드에서 확인된 정의 
- **Definition**: `window_size: the history length of the image / action`  # 이미지/액션의 히스토리 길이
- **Purpose**: Historical context for action prediction  # 액션 예측을 위한 히스토리 컨텍스트

### **7.2.2 Window Size Implementation**
- **Source**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:535`  # GitHub 코드에서 확인된 구현
- **Default Value**: `window_size: int = 16`  # 기본값 16
- **Usage**: Controls the number of historical observations used for prediction  # 예측에 사용되는 히스토리 관찰 수 제어

### **7.2.3 Window Size in Configuration**
- **Source**: `RoboVLMs/README.md:217`  # GitHub README에서 확인된 설정
- **Configuration Example**:  # 설정 예시
  ```json
  "window_size": 8,        # 슬라이딩 윈도우 크기 (히스토리 길이)
  "fwd_pred_next_n": 10,  # 예측할 대상 액션 청크 수
  ```

### **7.2.4 Window Size in Data Processing**
- **Source**: `RoboVLMs/robovlms/data/data_utils.py:249`  # GitHub 코드에서 확인된 데이터 처리
- **Function**: `generate_chunck_data(data, window_size, chunk_size)`  # 청크 데이터 생성 함수
- **Assertion**: `seq_len == window_size + chunk_size`  # 시퀀스 길이 = 윈도우 크기 + 청크 크기

### **7.2.5 Window Size in Model Architecture**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:42`  # GitHub 코드에서 확인된 모델 아키텍처
- **Parameter**: `window_size=None`  # 윈도우 크기 파라미터
- **Usage**: `self.window_size = window_size`  # 윈도우 크기 설정

### **7.2.6 Window Size in Training**
- **Source**: `RoboVLMs/robovlms/train/base_trainer.py:386`  # GitHub 코드에서 확인된 훈련 과정
- **Training Logic**:  # 훈련 로직
  ```python
  seq_len = self.configs["window_size"]    # 시퀀스 길이 = 윈도우 크기
  language = batch["text"].cuda()          # 언어 데이터 로딩
  text_mask = batch["text_mask"].cuda()    # 텍스트 마스크 로딩
  ```

### **7.2.7 Window Size in CALVIN Dataset**
- **Source**: `RoboVLMs/robovlms/data/calvin_dataset.py:707`  # GitHub 코드에서 확인된 CALVIN 데이터셋
- **Episode Processing**:  # 에피소드 처리
  ```python
  right_pad = end_idx - start_idx - self.act_step - self.window_size + 1  # 오른쪽 패딩 계산
  for idx in range(start_idx, end_idx + 1 - self.window_size):            # 윈도우 크기만큼 반복
  ```

### **7.2.8 Window Size in Mobile VLA**
- **Source**: `RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:28`  # GitHub 코드에서 확인된 Mobile VLA
- **Mobile VLA Configuration**:  # Mobile VLA 설정
  ```python
  window_size: int = 16,      # 윈도우 크기 16
  fwd_pred_next_n: int = 2,   # 순방향 예측 스텝 2
  ```

### **7.2.9 Window Size in Data Chunking**
- **Source**: `RoboVLMs/robovlms/data/data_utils.py:702`  # GitHub 코드에서 확인된 데이터 청킹
- **Chunk Generation**:  # 청크 생성
  ```python
  def get_chunked_episode(
      window_sample: Literal["sliding", "range"],  # 윈도우 샘플링 방법
      left_pad: bool,                              # 왼쪽 패딩 여부
      window_size: int,                            # 윈도우 크기
      fwd_pred_next_n: int,                        # 순방향 예측 스텝
      episode_idx_range: np.ndarray,               # 에피소드 인덱스 범위
  ):
  ```

### **7.2.10 Window Size in Model Forward Pass**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:910`  # GitHub 코드에서 확인된 모델 순전파
- **Forward Pass**:  # 순전파
  ```python
  bs, window_size = vision_x.shape[:2]    # 배치 크기, 윈도우 크기 = 비전 입력의 첫 두 차원
  ```

## 🎯 **Key Findings**

### **7.2.11 Window Size Technical Summary**
1. **History Length**: Window size determines how many historical observations are used  # 윈도우 크기는 사용되는 히스토리 관찰 수를 결정
2. **Context Window**: Provides temporal context for action prediction  # 액션 예측을 위한 시간적 컨텍스트 제공
3. **Sliding Window**: Implements sliding window approach for sequential data  # 순차 데이터를 위한 슬라이딩 윈도우 접근법 구현
4. **Data Chunking**: Enables efficient processing of long sequences  # 긴 시퀀스의 효율적 처리 가능

### **7.2.12 Window Size Implementation Details**
- **Default Value**: 16 (commonly used across configurations)  # 기본값: 16 (설정에서 일반적으로 사용)
- **Range**: Typically 8-16 for different model configurations  # 범위: 다양한 모델 설정에서 일반적으로 8-16
- **Relationship**: `seq_len = window_size + fwd_pred_next_n`  # 관계: 시퀀스 길이 = 윈도우 크기 + 순방향 예측 스텝
- **Memory**: Controls memory usage for historical data  # 히스토리 데이터의 메모리 사용량 제어

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/data/base_action_prediction_dataset.py` (L535-561)  # 액션 예측 데이터셋 기본 클래스
- `RoboVLMs/robovlms/data/data_utils.py` (L249-270)  # 데이터 유틸리티 함수
- `RoboVLMs/robovlms/model/backbone/base_backbone.py` (L42-60)  # 기본 백본 모델
- `RoboVLMs/robovlms/train/base_trainer.py` (L386-388)  # 기본 트레이너
- `RoboVLMs/robovlms/data/calvin_dataset.py` (L707-708)  # CALVIN 데이터셋
- `RoboVLMs/robovlms/data/mobile_vla_action_dataset.py` (L28-29)  # Mobile VLA 액션 데이터셋
- `RoboVLMs/README.md` (L217)  # README 설정 예시
- `RoboVLMs/configs/oxe_training/finetune_kosmos_mobile_vla.json` (L11-12)  # Mobile VLA 설정 파일
