# Jetson 리소스 최적화 및 코드 정리 가이드

## 1. 메모리 리소스 분석 (Jetson AGX Orin)

Jetson은 **Unified Memory Architecture**를 사용하므로, 시스템 RAM(CPU)과 VRAM(GPU)이 물리적 메모리를 공유합니다.

### 현재 메모리 상태
- **총 메모리**: 15GB (System + GPU 공유)
- **현재 사용 중**: 3.8GB (OS + 필수 서비스)
- **여유 메모리**: 약 **11GB**

### Antigravity IDE 점유율
- **메모리 점유**: 약 **1.0 ~ 1.2 GB**
    - `vscode-server`: ~800MB
    - `node` 프로세스들: ~300MB
- **CPU 점유**: 간헐적 (파일 인덱싱 등 작업 시 상승)

### 모델 요구사항 (Fine-tuned Mobile VLA)
- **메모리**: **3.12 GB** (FP16 로드 시)
- **추론 시**: 약 3.5 GB (피크)

---

## 2. 실행 전략 비교 및 권장사항

| 방식 | 장점 | 단점 | 메모리 영향 | 권장 상황 |
|------|------|------|------------|-----------|
| **A. IDE 실행** | 디버깅 편리, 코드 수정 즉시 반영 | 메모리 ~1GB 추가 점유 | **Low** (여유 충분) | 개발/테스트/디버깅 단계 |
| **B. SSH 실행** | 리소스 최소화, 최고 성능 | 로그 확인 불편 (터미널) | **None** (<50MB) | **실제 운영/로봇 구동 단계** |

### 🚀 최종 권장 설계
**현재 여유 메모리(11GB)가 모델 요구량(3.5GB)보다 훨씬 크므로, IDE를 켜두어도 성능 저하는 거의 없습니다.**

1.  **Phase 4 (테스트/통합)**: **IDE (A안)** 유지
    - 실시간 로그 확인 및 긴급 수정이 중요합니다.
    - 메모리 1GB를 더 써도 7GB 이상 남습니다.
2.  **Deployment (실배포)**: **SSH (B안)** 전환
    - 로봇에 모니터 없이 탑재되거나, 장시간 가동 시 안정성을 위해 SSH 또는 서비스 등록을 권장합니다.

---

## 3. 코드베이스 정리 (Cleanup)

`inference` 관련 파일이 57개나 되어 혼란스럽습니다. **3가지 핵심 파일**만 남기고 나머지는 정리(Archive)하는 것을 추천합니다.

### ✅ 남겨야 할 Core 파일 (현재 사용 중)
| 경로 | 역할 | 비고 |
|------|------|------|
| `src/robovlms_mobile_vla_inference.py` | **추론 엔진 (Engine)** | 핵심 로직, autocast 적용됨 |
| `ROS_action/.../mobile_vla_inference_node.py` | **ROS2 노드 (Node)** | data_collector 스타일 제어 적용됨 |
| `scripts/test_finetuned_mobile_vla.py` | **테스트 스크립트** | 검증용, 주기적 실행 가능 |

### 🗑️ 정리 대상 (헷갈리는 파일들)
다음 파일들은 **삭제**하거나 별도 `archive/` 폴더로 이동해야 합니다.

**1. 이름이 비슷해서 가장 위험한 파일들**
- `ROS_action/.../mobile_vla_inference.py` (Node 파일 아님, 구버전 추론 로직)
- `src/final_mobile_vla_inference.py` (이름만 final, 구버전)
- `jetson_local_complete_inference.py` (Phase 2 시도 파일)

**2. 더 이상 안 쓰는 테스트 스크립트**
- `config/scripts/*.py` (대부분 구버전)
- `scripts/test_int8_inference_phase2.py` (Phase 3 파일로 대체됨)
- `scripts/test_phase3_mobile_vla.py` (Fine-tuned 테스트로 대체됨)

---

## 4. 실행 액션 (복사/붙여넣기용)

### [Step 1] 정리 폴더 생성 및 이동
```bash
# 아카이브 폴더 생성
mkdir -p scripts/archive_inference
mkdir -p src/archive_inference

# 헷갈리는 파일 이동
mv src/final_mobile_vla_inference.py src/archive_inference/
mv jetson_local_complete_inference.py scripts/archive_inference/
mv ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_inference.py scripts/archive_inference/mobile_vla_inference_legacy.py

# 구버전 스크립트 이동
mv config/scripts/*_inference*.py scripts/archive_inference/
mv scripts/test_int8_inference_phase2.py scripts/archive_inference/
mv scripts/test_phase3_mobile_vla.py scripts/archive_inference/ 
```

### [Step 2] 메모리 모니터링 (실행 중 확인)
```bash
# 1초마다 메모리 사용량 갱신 (watch 명령)
watch -n 1 "free -h && echo '' && ps -eo cmd,%mem --sort=-%mem | head -5"
```
