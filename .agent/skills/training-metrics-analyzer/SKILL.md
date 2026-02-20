---
name: training-metrics-analyzer
description: Enforce a structured visualization and analysis of training logs (TensorBoard) to detect overfitting and identify optimal checkpoints.
---

# Training Metrics Analyzer Skill

이 스킬은 VLA 모델 학습 과정에서 생성된 TensorBoard 로그를 분석하여 학습 상태를 시각화하고, 특히 과적합(Overfitting) 발생 시점을 데이터 기반으로 판별하는 데 특화되어 있습니다.

## Core Instructions

1. **로그 경로 인식**: `runs/` 디렉토리 하위의 `version_X` 구조를 탐색하여 최신 이벤트 파일을 자동으로 식별합니다.
2. **지표 추출**: `train_loss`, `val_loss`, `train_acc_velocity_act` 등 핵심 지표를 추출합니다.
3. **시각화 규칙**:
    - Train Loss는 흐린 선으로 전체 추세를 표시합니다.
    - Val Loss는 점과 실선으로 명확하게 표시하여 변화 지점을 강조합니다.
    - Y축은 Loss의 세밀한 변화를 보기 위해 로그 스케일(`plt.yscale('log')`) 사용을 권장합니다.
4. **과적합 분석**:
    - Val Loss가 최저점을 찍고 다시 상승하는 지점을 'Best Checkpoint' 및 'Overfitting Start' 지점으로 자동 마킹합니다.

## Usage Guide

### Metrics Visualization
학습 로그 데이터로부터 그래프를 생성하려면 다음 스크립트를 실행하십시오:

```bash
python3 .agent/skills/training-metrics-analyzer/scripts/visualize_metrics.py
```

### Checkpoint Recommendation
과적합 분석 결과를 바탕으로 주행 테스트에 사용할 최적의 에포크를 추천합니다. (예: Epoch 2가 Val Loss 최저인 경우 해당 모델 사용 권장)

## Outputs
- `docs/plots/`: 생성된 PNG 그래프 파일들이 저장됩니다.
- Analysis Reports: 분석 결과는 Markdown 형식으로 `docs/reports/`에 정리됩니다.
