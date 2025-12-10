# 수요일 미팅 준비 계획 (2025-12-11 16:00)

**현재 시각**: 2025-12-10 13:05  
**미팅까지**: 약 27시간  
**교수님 의견**: "Option 2가 의미있을 것 같다"

---

## 🎯 교수님 요구사항

### Option 1: Unfrozen + LoRA + Action Head
**설정**:
- Backbone: **Trainable** (not frozen)
- LoRA: Enabled
- Action Head: Trainable
- Data: 1000-3000 episodes 필요

**상태**: ❌ **준비 안됨**
- 데이터 부족 (현재 500 episodes)
- 시간 부족 (학습에 최소 12시간+)
- Config 미생성

**교수님 판단**: 데이터양이 많아야 의미있음

---

### Option 2: Frozen + Action Head (현재 방식) ⭐
**설정**:
- Backbone: **Frozen**
- LoRA: Enabled
- Action Head: Trainable

**상태**: ✅ **준비됨!**
- Case 1-9 모두 이 방식
- 의미 벡터(context vector) 추출 가능
- **교수님이 더 의미있다고 판단**

---

## 📊 핵심 과제: 의미 벡터(Latent Space) 비교

### 질문
**"Frozen VLM의 latent space에서 나오는 의미 벡터가 어떤 특성을 가지는가?"**

### 비교 대상
1. **Left direction** context vectors
2. **Right direction** context vectors

### 측정 방법
- **Cosine Similarity** (코사인 유사도)
- **L2 Distance** 
- **Feature Correlation**
- **Distribution Analysis** (KS test, Wasserstein)

---

## ✅ 기존 준비 상황

### 완료된 작업
1. **Context Vector 분석 도구**:
   - `docs/RoboVLMs_validation/compare_vectors_metrics.py` ✅
   - Cosine similarity, L2 distance 구현됨
   - 시각화 기능 포함

2. **Context Vector 구조 문서**:
   - `docs/CONTEXT_VECTOR_SHAPE_EXPLAINED.md` ✅
   - Shape: [batch, 8_frames, 64_tokens, 2048_features]

3. **기존 분석 결과**:
   - `docs/reports/Q1_Context_Vector_Report.md` ✅

### 부족한 부분
1. ❌ **Left vs Right 의미 벡터 직접 비교**
2. ❌ **Latent space 시각화** (t-SNE, UMAP)
3. ❌ **논문 예시 찾기**

---

## 📝 수요일까지 할 일

### Phase 1: Context Vector 추출 (오늘, 3시간)
**목표**: Left/Right direction의 context vectors 추출

```python
# 스크립트: extract_left_right_contexts.py
# Input: Case 5 checkpoint (best model)
# Output: 
#   - left_contexts.npy: [N, 8, 64, 2048]
#   - right_contexts.npy: [N, 8, 64, 2048]
```

**방법**:
1. Case 5 checkpoint 로드
2. Left episodes (250개) → context 추출
3. Right episodes (250개) → context 추출
4. .npy로 저장

---

### Phase 2: 유사도 비교 (오늘 저녁, 2시간)
**목표**: Left vs Right context의 차이/유사도 측정

**측정 항목**:
1. **Cosine Similarity** (가장 중요!)
   - Left-Left: 얼마나 similar?
   - Right-Right: 얼마나 similar?
   - Left-Right: 얼마나 different?

2. **L2 Distance**
   - 의미 벡터 간 거리

3. **Feature Analysis**
   - 어떤 feature들이 direction을 구분?
   - Feature importance

4. **Distribution**
   - Left/Right의 분포 차이

**출력 형식**:
```
Left-Left Similarity: 0.95 ± 0.03
Right-Right Similarity: 0.94 ± 0.04
Left-Right Similarity: 0.75 ± 0.12

→ 같은 방향끼리는 매우 유사 (0.95)
→ 다른 방향끼리는 구분됨 (0.75)
```

---

### Phase 3: 시각화 (내일 오전, 2시간)
**목표**: 직관적인 시각화

**시각화 1: t-SNE Projection**
```python
# 2048D → 2D로 차원 축소
# Left: 파란색 점
# Right: 빨간색 점
# → 클러스터링 확인
```

**시각화 2: Similarity Matrix**
```python
# Heatmap
# Row: Left samples
# Col: Right samples
# Value: Cosine similarity
```

**시각화 3: Feature Importance**
```python
# 어떤 feature들이 Left/Right를 구분?
# Top 20 discriminative features
```

---

### Phase 4: 논문 예시 찾기 (내일 오후, 3시간)
**목표**: 비슷한 분석을 한 논문 찾기

**검색 키워드**:
- "latent space analysis VLA"
- "context vector similarity robot"
- "frozen VLM representation"
- "vision-language latent space"

**참고 논문 후보**:
1. **RoboFlamingo**: Frozen VLM analysis
2. **RT-2**: Representation learning
3. **OpenVLA**: Latent space analysis
4. **CLIP**: Contrastive learning

**찾을 내용**:
- Latent space 비교 방법
- Similarity 측정 기준
- 시각화 예시
- 해석 방법

---

### Phase 5: 결과 정리 (수요일 오전, 2시간)
**목표**: 미팅 자료 완성

**문서**:
1. **실험 결과 리포트**
   - Context vector 추출 방법
   - 유사도 측정 결과
   - 시각화
   - 해석

2. **미팅 발표 자료**
   - 핵심 발견 3가지
   - 시각화 3개
   - 논문 근거

---

## 🔧 구현 계획 상세

### Script 1: extract_left_right_contexts.py
```python
# Pseudo code
import torch
from pathlib import Path

# 1. Load best checkpoint (Case 5, Epoch 4)
checkpoint = torch.load("runs/mobile_vla_no_chunk_20251209/.../epoch=04.ckpt")

# 2. Load dataset
left_episodes = load_h5_files("*left*.h5")   # 250 episodes
right_episodes = load_h5_files("*right*.h5") # 250 episodes

# 3. Extract contexts
left_contexts = []
for ep in left_episodes:
    images = ep['images'][:8]  # window_size=8
    with torch.no_grad():
        context = model.encode_images(images)
        # context shape: [8, 64, 2048]
    left_contexts.append(context)

# 4. Save
np.save("left_contexts.npy", np.array(left_contexts))  # [250, 8, 64, 2048]
np.save("right_contexts.npy", np.array(right_contexts))
```

---

### Script 2: compare_left_right.py
```python
# Pseudo code
import numpy as np
from scipy.spatial.distance import cosine

# 1. Load contexts
left = np.load("left_contexts.npy")  # [250, 8, 64, 2048]
right = np.load("right_contexts.npy")

# 2. Flatten to [N, D]
left_flat = left.reshape(250, -1)   # [250, 8*64*2048]
right_flat = right.reshape(250, -1)

# 3. Compute similarities
# 3a. Left-Left similarity
left_left_sim = []
for i in range(250):
    for j in range(i+1, 250):
        sim = 1 - cosine(left_flat[i], left_flat[j])
        left_left_sim.append(sim)

# 3b. Right-Right similarity
right_right_sim = [...]

# 3c. Left-Right similarity
left_right_sim = []
for i in range(250):
    for j in range(250):
        sim = 1 - cosine(left_flat[i], right_flat[j])
        left_right_sim.append(sim)

# 4. Statistics
print(f"Left-Left: {np.mean(left_left_sim):.4f} ± {np.std(left_left_sim):.4f}")
print(f"Right-Right: {np.mean(right_right_sim):.4f} ± {np.std(right_right_sim):.4f}")
print(f"Left-Right: {np.mean(left_right_sim):.4f} ± {np.std(left_right_sim):.4f}")
```

---

### Script 3: visualize_latent_space.py
```python
# Pseudo code
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Load and prepare data
left = np.load("left_contexts.npy").reshape(250, -1)
right = np.load("right_contexts.npy").reshape(250, -1)

# 2. t-SNE
combined = np.vstack([left, right])
labels = ['Left']*250 + ['Right']*250

tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(combined)

# 3. Plot
plt.figure(figsize=(10, 8))
plt.scatter(embedded[:250, 0], embedded[:250, 1], c='blue', label='Left', alpha=0.6)
plt.scatter(embedded[250:, 0], embedded[250:, 1], c='red', label='Right', alpha=0.6)
plt.legend()
plt.title("Latent Space: Left vs Right Directions")
plt.savefig("tsne_left_right.png", dpi=300)
```

---

## 📅 타임라인

### 오늘 (12/10)
- 14:00-17:00: Context vector 추출 스크립트 작성 및 실행
- 18:00-20:00: 유사도 비교 및 분석
- 21:00-22:00: 초기 시각화

### 내일 (12/11)
- 09:00-11:00: t-SNE, UMAP 시각화
- 11:00-14:00: 논문 예시 찾기 및 정리
- 14:00-15:30: 미팅 자료 작성
- 15:30-16:00: 최종 리허설

### 미팅 (12/11 16:00)
- Option 2 분석 결과 발표
- Latent space 시각화 제시
- 논문 근거 설명

---

## 🎯 Expected Outcomes

### 핵심 발견 (예상)
1. **방향 구분 가능**: Left와 Right의 context vectors는 명확히 구분됨
   - Left-Left similarity: ~0.95
   - Right-Right similarity: ~0.95
   - Left-Right similarity: ~0.70-0.80
   
2. **Latent space clustering**: t-SNE에서 두 클러스터 형성

3. **Feature specialization**: 일부 features가 방향 구분에 중요

### 교수님 질문 대응
**Q**: "왜 Option 2가 의미있는가?"

**A**: 
- "Frozen VLM의 latent space가 이미 방향(direction)을 구분하는 의미 있는 표현을 학습했습니다"
- "Left vs Right similarity가 0.75로, 같은 방향(0.95)보다 낮습니다"
- "t-SNE 시각화에서 명확한 클러스터링이 확인되었습니다"
- "RoboFlamingo [논문]에서도 비슷한 분석을 통해 Frozen VLM의 효과를 검증했습니다"

---

**작성**: 2025-12-10 13:05  
**미팅**: 2025-12-11 16:00 (27시간 후)  
**상태**: 계획 수립 완료, 구현 시작 필요
