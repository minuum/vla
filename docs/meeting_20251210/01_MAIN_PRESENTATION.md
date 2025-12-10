# 메인 발표 - VLA Training 결과 (25분)

**일시**: 2025-12-10 16:00  
**발표자**: Billy  
**시간**: 25분

---

## 1. Task & Environment (3분)

### Holonomic Obstacle Avoidance Navigation

**Robot**: Omnidirectional (Holonomic)
- 전방향 이동 가능
- linear_x + linear_y 독립 제어

**환경**:
```
[Holonomic Robot]
       ↓
    [Box] ← 장애물 (중앙 고정)
       ↓
   [Bottle] ← 목표 (중앙 고정)
```

**Instruction**:
- **"...on the left"** → [1.15, +1.15] (45도 왼쪽 대각선)
- **"...on the right"** → [1.15, -1.15] (45도 오른쪽 대각선)

**Goal**: 박스를 옆으로 피하며 bottle 도달

---

## 2. 실험 설계 (2분)

### 변수
1. **Action Chunking**: fwd_pred_next_n
   - Chunk=1: 1 step만 예측 (reactive)
   - Chunk=10: 10 steps 예측 (RoboVLMs 기본)

2. **Data**: 500 (L+R) vs 250 (R only)

3. **Strategy**: Baseline, Aug, Abs

### 고정
- Model: Kosmos-2 (Frozen) + LoRA (r=32)
- Window: 8 frames
- Action: 2 DOF (linear_x, linear_y)

---

## 3. 핵심 발견 (10분) ⭐

### Finding 1: No Chunk 압도적 (98% 개선)

| Chunk | Val Loss | Improvement |
|:---:|---:|---:|
| **1** | **0.000532** | - |
| 10 | 0.027 | 98% worse |

**왜 Chunk=1이 완벽한가?**

**Holonomic navigation 특성**:
1. **Coupled control**: linear_x + linear_y 동시 조정
2. **Smooth trajectory**: 부드러운 대각선 경로
3. **Real-time obstacle**: 장애물 거리 즉시 반응
4. **Omnidirectional**: 즉각 횡이동

**Chunk=10의 한계**:
- 10 steps 미리 경로 예측
- Holonomic 장점 못 살림
- Obstacle 실시간 대응 어려움

**Note**: Manipulation에는 Chunk=10 유용할 수 있음!

---

### Finding 2: Simple Baseline이 최고

| Strategy | Val Loss | vs Baseline |
|:---|---:|---:|
| **Baseline** | **0.000532** | - |
| Abs | 0.00243 | 4.6x worse |
| Aug+Abs | 0.004 | 7.5x worse |

**결론**: Language instruction 충분히 informative

---

### Finding 3: Data Diversity 중요

| Data | Episodes | Val Loss |
|:---|:---:|---:|
| **L+R** | 500 | **0.000532** |
| R only | 250 | 0.016 |

**개선**: 30x

---

## 4. Best Model (5분)

### Case 5: Champion 🏆

**설정**:
- Chunk: 1 (No Chunk)
- Data: L+R 500 episodes
- Strategy: Baseline

**성능**:
- Val Loss: **0.000532**
- Train Loss: ~0.0001
- Epochs: 4

**Action 검증**:
- Left: [1.15, +1.15] ✅
- Right: [1.15, -1.15] ✅

---

## 5. 향후 계획 (5분)

### 교수님 의견: Approach 2가 의미 있을 듯 ⭐

#### Approach 2: Frozen VLM + Action Head
**목적**: Latent space에서 의미 벡터 비교

**방법**:
1. VLM frozen
2. Action head만 학습
3. Latent space 추출 (hidden states)
4. Left vs Right 의미 벡터 비교
5. **코사인 유사도** 등으로 측정

**참고**: OpenVLA, RT-2 논문 예시

#### Approach 1: UnFrozen VLM (비교용)
- LoRA + Action head
- 데이터 1000-3000 필요
- 의미 벡터 비교

**비교 분석**:
- Frozen vs UnFrozen의 latent space 차이
- 어떤 approach가 더 의미있는 representation?

---

## 핵심 요약 (3줄)

1. **No Chunk (Chunk=1)로 98% 성능 개선**
2. **Holonomic navigation = Reactive control 필요**
3. **향후: Frozen VLM latent space 분석** (교수님 추천)

---

**준비**: 완료 ✅  
**시간**: 25분  
**자신감**: Very High
