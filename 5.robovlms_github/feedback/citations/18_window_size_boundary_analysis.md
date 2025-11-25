# 18. Window Size 경계 처리 분석

## Critical Issue: Window Size를 벗어난 시점에서의 액션 예측 처리

### 문제점
Window size를 벗어난 시점 (t > window_size)에서 액션 예측을 어떻게 처리하는지, 특히 `fwd_pred_next_n` 값이 어떻게 조정되는지 불명확.

---

## 1. 핵심 발견사항

### **1.1 unfold 함수의 정확한 동작**

```python
# RoboVLMs/robovlms/data/calvin_dataset.py:884-887
action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
```

**unfold 함수 분석**:
- `unfold(dim=1, size=fwd_pred_next_n, step=1)`
- **결과 shape**: `[batch, window_size - fwd_pred_next_n + 1, fwd_pred_next_n, action_dim]`
- **핵심**: `window_size - fwd_pred_next_n + 1`개의 chunk만 생성됨

### **1.2 구체적 예시 (window_size=8, fwd_pred_next_n=10)**

```
원본 데이터: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]

unfold(1, 10, 1) 결과:
- Chunk 0 (t0): [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]     ← 10개
- Chunk 1 (t1): [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]    ← 10개  
- Chunk 2 (t2): [a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]  ← 10개
- Chunk 3 (t3): [a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] ← 10개
- Chunk 4 (t4): [a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] ← 10개
- Chunk 5 (t5): [a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] ← 10개
- Chunk 6 (t6): [a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] ← 10개
- Chunk 7 (t7): [a7, a8, a9, a10, a11, a12, a13, a14, a15, a16] ← 10개
- Chunk 8 (t8): [a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] ← 10개

총 9개 chunk 생성 (8 - 10 + 1 = -1 → 불가능!)
```

**문제**: `window_size < fwd_pred_next_n`인 경우 **음수 개수**의 chunk가 생성되어야 함!

---

## 2. 실제 코드에서의 처리 방식

### **2.1 Mobile VLA Dataset의 처리**

```python
# RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:222-238
# 길이 정합성 보장: images=18(window+fwd), actions=17(window+fwd-1) 필요
target_img_len = self.window_size + self.fwd_pred_next_n
if images.shape[0] > target_img_len:
    images = images[:target_img_len]
elif images.shape[0] < target_img_len:
    pad = target_img_len - images.shape[0]
    last = images[-1:]
    images = np.concatenate([images, np.repeat(last, pad, axis=0)], axis=0)

# actions가 18이면 마지막 1개를 제거해 윈도우 규칙(window=16, fwd=2)에 맞춤
if actions.shape[0] > 17:
    actions = actions[:17]
elif actions.shape[0] < 17 and images.shape[0] == 18:
    # 부족 시 마지막 액션 반복으로 채움
    pad = 17 - actions.shape[0]
    last = actions[-1:]
    actions = np.concatenate([actions, np.repeat(last, pad, axis=0)], axis=0)
```

**핵심**: `window_size + fwd_pred_next_n` 길이의 데이터를 준비

### **2.2 Base Action Prediction Dataset의 Assertion**

```python
# RoboVLMs/robovlms/data/base_action_prediction_dataset.py:261
if self.mode == "train":
    assert action.shape[0] == self.fwd_pred_next_n + self.window_size - 1
    window_size = self.window_size
```

**핵심**: Training 시 `action.shape[0] = fwd_pred_next_n + window_size - 1`이어야 함

### **2.3 Data Utils의 Chunk 생성**

```python
# RoboVLMs/robovlms/data/data_utils.py:255-257
assert (
    seq_len == window_size + chunk_size
), f"The sequence length should be {window_size + chunk_size}"
```

**핵심**: `seq_len = window_size + fwd_pred_next_n`이어야 함

---

## 3. 정확한 처리 방식 분석

### **3.1 데이터 준비 단계**

```python
# 실제 데이터 길이: window_size + fwd_pred_next_n
total_length = window_size + fwd_pred_next_n

# 예: window_size=8, fwd_pred_next_n=10
# total_length = 8 + 10 = 18
# 데이터: [a0, a1, a2, ..., a17] (18개)
```

### **3.2 unfold 결과**

```python
# unfold(1, fwd_pred_next_n, 1) = unfold(1, 10, 1)
# 결과: [batch, window_size - fwd_pred_next_n + 1, fwd_pred_next_n, action_dim]
#      [batch, 8 - 10 + 1, 10, action_dim] = [batch, -1, 10, action_dim] ← 불가능!
```

**문제**: `window_size < fwd_pred_next_n`이면 unfold가 불가능!

### **3.3 실제 해결책: 데이터 구조 변경**

```python
# RoboVLMs/robovlms/data/calvin_dataset.py:870-887
# Image chunk 생성
image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 5, 2, 3, 4)[:, 1:]
#                                                                                    ^^^^^^^^
#                                                                                    첫 번째 제거!

image_tensors = image_tensors[:, : self.window_size]
#                            ^^^^^^^^^^^^^^^^^^^^
#                            window_size까지만 사용
```

**핵심**: 
1. **전체 데이터**: `window_size + fwd_pred_next_n` 길이
2. **unfold 적용**: `fwd_pred_next_n` 크기로 sliding
3. **첫 번째 제거**: `[:, 1:]`로 첫 번째 chunk 제거
4. **최종 사용**: `window_size` 길이만 사용

---

## 4. 구체적 예시 (window_size=8, fwd_pred_next_n=10)

### **4.1 데이터 준비**

```
원본 데이터: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17]
            |<-- window_size=8 -->|<-- fwd_pred_next_n=10 -->|
            |<------------- total_length=18 ---------------->|
```

### **4.2 unfold 적용**

```
unfold(1, 10, 1) 결과:
- Chunk 0: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]     ← t0~t9
- Chunk 1: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]    ← t1~t10  
- Chunk 2: [a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]  ← t2~t11
- Chunk 3: [a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] ← t3~t12
- Chunk 4: [a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] ← t4~t13
- Chunk 5: [a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] ← t5~t14
- Chunk 6: [a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] ← t6~t15
- Chunk 7: [a7, a8, a9, a10, a11, a12, a13, a14, a15, a16] ← t7~t16
- Chunk 8: [a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] ← t8~t17

총 9개 chunk (18 - 10 + 1 = 9)
```

### **4.3 첫 번째 제거 후**

```
[:, 1:] 적용 후:
- Chunk 0: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]    ← t1~t10
- Chunk 1: [a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]  ← t2~t11
- Chunk 2: [a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] ← t3~t12
- Chunk 3: [a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] ← t4~t13
- Chunk 4: [a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] ← t5~t14
- Chunk 5: [a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] ← t6~t15
- Chunk 6: [a7, a8, a9, a10, a11, a12, a13, a14, a15, a16] ← t7~t16
- Chunk 7: [a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] ← t8~t17

총 8개 chunk (window_size와 동일)
```

### **4.4 최종 사용 데이터**

```
image_tensors = image_tensors[:, : self.window_size]
# [a0, a1, a2, a3, a4, a5, a6, a7] (window_size=8)

action_chunks:
- t0: [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]     ← t0 시점에서 미래 10개
- t1: [a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]   ← t1 시점에서 미래 10개
- t2: [a3, a4, a5, a6, a7, a8, a9, a10, a11, a12]  ← t2 시점에서 미래 10개
- t3: [a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] ← t3 시점에서 미래 10개
- t4: [a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] ← t4 시점에서 미래 10개
- t5: [a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] ← t5 시점에서 미래 10개
- t6: [a7, a8, a9, a10, a11, a12, a13, a14, a15, a16] ← t6 시점에서 미래 10개
- t7: [a8, a9, a10, a11, a12, a13, a14, a15, a16, a17] ← t7 시점에서 미래 10개
```

---

## 5. 핵심 결론

### **5.1 fwd_pred_next_n 값은 변경되지 않음**

- **각 시간 단계마다 동일한 `fwd_pred_next_n` 개수**의 액션을 예측
- **경계를 벗어나도 `fwd_pred_next_n`을 1 빼지 않음**
- **대신 데이터 길이를 `window_size + fwd_pred_next_n`으로 확장**

### **5.2 실제 처리 방식**

1. **데이터 확장**: `window_size + fwd_pred_next_n` 길이로 데이터 준비
2. **unfold 적용**: `fwd_pred_next_n` 크기로 sliding window 생성
3. **첫 번째 제거**: `[:, 1:]`로 첫 번째 chunk 제거 (boundary effect 방지)
4. **최종 사용**: `window_size` 길이의 입력과 `window_size` 개의 chunk 사용

### **5.3 왜 이렇게 처리하는가?**

1. **Temporal Consistency**: 모든 시간 단계에서 동일한 예측 개수 유지
2. **Boundary Effect 방지**: 첫 번째 chunk 제거로 경계 효과 최소화
3. **학습 안정성**: 일관된 예측 구조로 학습 안정성 확보

---

## 6. 코드 근거 요약

**출처**:
- `RoboVLMs/robovlms/data/calvin_dataset.py:870-887`
- `RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:222-238`
- `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:261`
- `RoboVLMs/robovlms/data/data_utils.py:255-257`

**핵심**: Window size 경계를 벗어나도 `fwd_pred_next_n` 값은 변경되지 않고, 대신 데이터 길이를 확장하여 처리합니다.
