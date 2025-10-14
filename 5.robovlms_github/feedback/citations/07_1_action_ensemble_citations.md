# 07_1 Action Ensemble Analysis - RoboVLMs GitHub Citations

## 📊 **Action Ensemble Technical Analysis**

### **7.1.1 Action Ensemble Definition**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:38`  # GitHub 코드에서 확인된 정의
- **Parameter**: `action_ensemble=False`  # 액션 앙상블 사용 여부
- **Purpose**: Action history management and weighted averaging  # 액션 히스토리 관리 및 가중 평균

### **7.1.2 Action Ensemble Implementation**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:154-185`  # GitHub 코드에서 확인된 구현
- **Function**: `ensemble_action(self, action)`  # 액션 앙상블 함수
- **Core Logic**:  # 핵심 로직
  ```python
  def ensemble_action(self, action):
      """액션 앙상블 함수 (히스토리 기반 가중 평균)"""
      # 차원 처리
      if action.ndim >= 3:
          action = action.squeeze()          # 3차원 이상 → 압축
      
      if action.ndim == 1:
          action = action.unsqueeze(0)       # 1차원 → 차원 확장
      
      self.action_hist_list.append(action)  # 액션 히스토리에 추가
      
      act_cache = []
      max_len = self.fwd_pred_next_n        # 순방향 예측 스텝 수
      max_len = 1                           # 실제로는 1로 고정
      
      # 히스토리 길이 제한 (오래된 액션 제거)
      while len(self.action_hist_list) > max_len:
          self.action_hist_list.pop(0)      # 가장 오래된 액션 제거
      
      idx = 0
      for act in self.action_hist_list[::-1]:  # 역순으로 처리 (최신 → 과거)
          act_cache.append(act[idx])           # 액션 캐시에 추가
          idx += 1
      
      act_cache = torch.stack(act_cache, dim=0)  # 텐서로 스택
      
      # 가중치 계산 (fwd_decay_ratio = 1)
      weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
      weights = weights / weights.sum()         # 가중치 정규화
      
      # 가중 평균 계산
      weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)
      
      return weighted_act
  ```

### **7.1.3 Action Ensemble Usage**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:370`  # GitHub 코드에서 확인된 사용
- **Usage**: `action = self.ensemble_action(action)`  # 액션 앙상블 적용
- **Context**: Applied after action prediction and scaling  # 액션 예측 및 스케일링 후 적용

### **7.1.4 Action Ensemble Parameters**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:25`  # GitHub 코드에서 확인된 파라미터
- **Decay Ratio**: `fwd_decay_ratio = 1`  # 순방향 감쇠 비율
- **Max Length**: `max_len = 1`  # 최대 길이 (실제로는 1로 고정)
- **History Management**: `self.action_hist_list`  # 히스토리 관리

### **7.1.5 Action Ensemble Technical Details**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:154-185`  # GitHub 코드에서 확인된 기술적 세부사항
- **Dimension Handling**:  # 차원 처리
  - **Input**: `action.ndim >= 3` → `squeeze()`  # 입력: 3차원 이상 → 압축
  - **Input**: `action.ndim == 1` → `unsqueeze(0)`  # 입력: 1차원 → 차원 확장
- **History Management**:  # 히스토리 관리
  - **Append**: `self.action_hist_list.append(action)`  # 추가
  - **Pop**: `self.action_hist_list.pop(0)`  # 제거
- **Weight Calculation**:  # 가중치 계산
  - **Formula**: `weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])`  # 공식
  - **Normalization**: `weights = weights / weights.sum()`  # 정규화

### **7.1.6 Action Ensemble Weighted Average**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:180-185`  # GitHub 코드에서 확인된 가중 평균
- **Weight Calculation**:  # 가중치 계산
  ```python
  # 가중치 계산 (감쇠 비율 기반)
  weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
  weights = weights / weights.sum()         # 가중치 정규화
  ```
- **Weighted Average**:  # 가중 평균
  ```python
  # 가중 평균 계산 (가중치 적용)
  weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)
  ```

### **7.1.7 Action Ensemble Current Limitations**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:165-166`  # GitHub 코드에서 확인된 현재 한계
- **Max Length**: `max_len = 1`  # 최대 길이 1로 고정
- **Decay Ratio**: `fwd_decay_ratio = 1`  # 감쇠 비율 1로 고정
- **Effect**: Limited ensemble effect due to single action history  # 단일 액션 히스토리로 인한 제한적 앙상블 효과

### **7.1.8 Action Ensemble Benefits**
- **Action Smoothing**: Reduces abrupt action changes  # 급격한 액션 변화 완화
- **Noise Reduction**: Averages out individual action noise  # 개별 액션 노이즈 평균화
- **Stability**: Provides more stable robot movements  # 더 안정적인 로봇 움직임 제공
- **History Management**: Maintains action history for context  # 컨텍스트를 위한 액션 히스토리 유지

### **7.1.9 Action Ensemble Technical Architecture**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:28-39`  # GitHub 코드에서 확인된 기술적 아키텍처
- **Class**: `CustomModel`  # 커스텀 모델 클래스
- **Initialization**:  # 초기화
  ```python
  def __init__(
      self,
      ckpt_path,
      configs,
      device,
      save_dir=None,
      raw_calvin=True,
      debug=False,
      action_ensemble=False,  # 액션 앙상블 사용 여부
  ):
  ```

### **7.1.10 Action Ensemble Future Potential**
- **Expandable**: Framework allows for future expansion  # 향후 확장 가능한 프레임워크
- **Configurable**: Can be modified for different ensemble strategies  # 다양한 앙상블 전략을 위해 수정 가능
- **Scalable**: Can handle multiple action history lengths  # 여러 액션 히스토리 길이 처리 가능

## 🎯 **Key Findings**

### **7.1.11 Action Ensemble Technical Summary**
1. **History Management**: Maintains action history for context  # 컨텍스트를 위한 액션 히스토리 유지
2. **Weighted Averaging**: Calculates weighted average of historical actions  # 히스토리 액션의 가중 평균 계산
3. **Action Smoothing**: Reduces abrupt action changes  # 급격한 액션 변화 완화
4. **Noise Reduction**: Averages out individual action noise  # 개별 액션 노이즈 평균화

### **7.1.12 Action Ensemble Implementation Status**
- **Current State**: Limited to single action history (max_len = 1)  # 현재 상태: 단일 액션 히스토리로 제한 (최대 길이 = 1)
- **Decay Ratio**: Fixed at 1 (no decay effect)  # 감쇠 비율: 1로 고정 (감쇠 효과 없음)
- **Future Potential**: Framework ready for expansion  # 향후 잠재력: 확장을 위한 프레임워크 준비

## 📁 **Supporting Files**
- `RoboVLMs/eval/calvin/model_wrapper.py` (L25-185)  # CALVIN 모델 래퍼
- `RoboVLMs/eval/calvin/model_wrapper.py` (L38)  # 액션 앙상블 파라미터
- `RoboVLMs/eval/calvin/model_wrapper.py` (L154-185)  # 액션 앙상블 구현
- `RoboVLMs/eval/calvin/model_wrapper.py` (L370)  # 액션 앙상블 사용
