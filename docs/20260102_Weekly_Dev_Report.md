# 2026-01-02 주간 개발 리포트 및 트러블슈팅 로그

## 1. Summary
**Mobile VLA** 모델의 **Jetson Orin** 온디바이스 배포를 위한 환경 구축, ROS2 추론 노드 개발, 그리고 실제 주행 테스트까지의 파이프라인을 성공적으로 구축하였습니다. 초기 발생했던 SSH 연결 끊김(OOM) 문제와 의존성 충돌을 모두 해결하고, 현재 안정적인 **18프레임 연속 추론 및 데이터 시각화**가 가능합니다.

---

## 2. Weekly Achievements (이번 주 완료 사항)

| 카테고리 | 상세 내용 | 상태 |
| :--- | :--- | :---: |
| **Environment** | **Jetson Orin Nano/AGX 환경 구축 완료**<br>- PyTorch 2.1.0 + TorchVision 0.16.0 (JP 6.0 호환)<br>- `transformers` 버전 충돌 해결 (4.35.0)<br>- `bitsandbytes` 빌드 및 설치 완료 | ✅ |
| **ROS2 Dev** | **Mobile VLA Inference Node 개발**<br>- `mobile_vla_data_collector.py` 로직 이식 (0.4s Move -> Stop)<br>- Action Gain (60.0) 및 Deadzone (0.15) 튜닝 적용 | ✅ |
| **Optimization** | **메모리 안정화 및 자동화**<br>- 추론 중 `matplotlib` 제거로 OOM(SSH 끊김) 방지<br>- JSON 데이터 저장 → `subprocess` 자동 시각화 파이프라인 구축 | ✅ |
| **Visualization** | **분석 도구 개발**<br>- 실시간 메모리(RAM/VRAM) 사용량 측정<br>- Robot Trajectory (X-Y Plane) 궤적 시각화 도구 구현 | ✅ |

---

## 3. Daily Troubleshooting Log (2026-01-02)

오늘 진행된 테스트 과정에서 발생한 주요 이슈와 해결 과정을 정리한 표입니다.

| 이슈 (Issue) | 원인 분석 (Root Cause) | 해결 방법 (Solution) | 결과 |
| :--- | :--- | :--- | :---: |
| **SSH 연결 끊김** | 추론 노드 내에서 `matplotlib.pyplot` 등 무거운 GUI 라이브러리를 반복 호출하여 **시스템 메모리 부족(OOM)** 발생, OS가 프로세스 및 네트워크 세션 강제 종료 | **1. 시각화 분리:** 노드 내 그래프 생성 코드 제거<br>**2. 경량화:** 데이터만 JSON으로 저장 후, 노드 종료 시 `subprocess`로 별도 시각화 스크립트 실행 | **해결**<br>(안정적 완주) |
| **Import Error** | 코드 수정 과정에서 `import os`, `import torch` 구문이 누락됨<br>(`NameError: name 'torch' is not defined`) | 누락된 라이브러리(`os`, `json`, `torch`, `sys`) 명시적 임포트 추가 | **해결** |
| **모델 로딩 에러** | `chunk_size`(10)가 `window_size`(2)보다 크게 설정되어 모델 내부 텐서 연산에서 `IndexError` 발생 | `load_model` 메서드 내에 **Chunk Size 자동 조정 로직** 추가 (`min(chunk, window)`) | **해결** |
| **로봇 이동 방향** | 모델이 전진해야 할 상황에서 `x` 값이 음수(예: -4.72)로 출력되어 후진/대각선 이동 발생 | **궤적 시각화 도구**를 통해 실제 모델 출력값 패턴 확인.<br>(현재 데이터셋 좌표계와 모델 출력 좌표계 간의 정합성 확인 필요) | **분석 중** |

---

## 4. Inference Test Analysis (주행 테스트 분석)

### 4.1 테스트 환경
- **Target Frames**: 18 Steps
- **Action Gain**: 60.0 (Data Scale Matching)
- **Deadzone**: 0.15
- **Input Instruction**: "Navigate to the target"

### 4.2 결과 데이터 (Log 기반)
최근 성공한 18프레임 주행 데이터(`inference_data_*.json`) 분석 결과입니다.

- **안정성**: 중간 끊김 없이 18프레임 전체 수행 완료.
- **리소스 사용량**:
    - CPU RAM: 안정적 유지
    - GPU VRAM: 3.12GB ~ 3.22GB 유지 (OOM 없음)
- **이동 궤적 패턴**:
    - **X축 (전진/후진)**: 지속적인 음수 값 (`-4.6` ~ `-4.8`) 관측 → **후진**
    - **Y축 (좌/우)**: 양수 값 (`3.5` ~ `3.6`) 관측 → **좌측**
    - **결론**: 로봇이 **"좌측 후방(142도 방향)"**으로 일관되게 이동 중.

### 4.3 원인 가설 및 향후 계획
모델이 일관된 방향성을 보이나, 방향이 반대(전진 대신 후진)인 현상이 관측됩니다.
1.  **가설 1 (좌표계 불일치)**: 학습 데이터는 `X=후진`인데 로봇 제어기는 `X=전진`일 가능성 (혹은 그 반대).
2.  **가설 2 (데이터 정규화)**: Action Normalization/Denormalization 과정에서 부호가 반전되었을 가능성.

**Next Action:**
- 데이터셋의 `linear_x`, `linear_y` 원시 데이터 분포 재확인.
- 추론 엔진의 `denormalize_action` 로직 검토.
- 필요 시 `mobile_vla_inference_node.py`에서 Action 부호 반전(`x = -x`) 임시 적용 테스트.

---

## 5. Script & Artifacts

현재 시스템에서 사용 가능한 주요 스크립트입니다.

1.  **실행 노드**: `vla-inference` (ROS2 Node)
    - 위치: `Feature: Memory Profiling & Auto-JSON Save`
2.  **시각화 도구**: `scripts/visualize_inference_log.py`
    - 기능: JSON 로그를 읽어 궤적 및 리소스 그래프 생성
3.  **기록 경로**: `docs/memory_analysis/`
    - 산출물: `.json` (Raw Data), `_report.png` (Visualization)
