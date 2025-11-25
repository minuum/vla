# Mobile-VLA 연구 문서 인덱스

> **작성일**: 2025-11-26  
> **프로젝트**: Mobile-VLA - 2DOF Mobile Robot을 위한 Vision-Language-Action 모델

---

## 📚 문서 구조

### 🎯 연구 계획
- **[RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md)**: 전체 연구 로드맵 (5 Phases)
- **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)**: 구현 상세 계획 및 실험 설계

### 📊 데이터셋
- **[../data_augmentation/DATASET_ANALYSIS_REPORT.md](../data_augmentation/DATASET_ANALYSIS_REPORT.md)**: 468개 H5 파일 분석 결과
- **[../data_augmentation/NON_SIMULATION_AUGMENTATION.md](../data_augmentation/NON_SIMULATION_AUGMENTATION.md)**: 비시뮬레이션 증강 전략
- **[../data_augmentation/SIMULATION_SETUP.md](../data_augmentation/SIMULATION_SETUP.md)**: Habitat-AI 시뮬레이션 환경

### 📋 데이터 명세
- **[../../DATA_FORMAT_SPEC.md](../../DATA_FORMAT_SPEC.md)**: VLA 데이터 형식 명세서

### 📝 보고서
- **[../../reports/MEETING_BRIEF_20251120.md](../../reports/MEETING_BRIEF_20251120.md)**: 연구 미팅 브리프
- **[../../reports/LFS_ISSUE_REPORT_20251125.md](../../reports/LFS_ISSUE_REPORT_20251125.md)**: Git LFS 이슈 보고

---

## 🚀 Quick Start

### 현재 상태 (2025-11-26)
- ✅ 468개 H5 데이터 분석 완료
- ✅ Language instruction 추가 완료
- ✅ 데이터 증강 전략 수립
- 🔄 Context vector 추출 준비 중

### 다음 단계
1. **RoboVLMs Context Vector 추출** (최우선)
2. **ControlNet 증강** (468 → 4,680)
3. **전체 데이터 학습** (468개)

---

## 📖 핵심 질문

### 1. Context Vector 유의미성
**질문**: RoboVLMs가 mobile robot 이미지에서 의미있는 context를 추출하는가?  
**문서**: IMPLEMENTATION_PLAN.md - "Context Vector 의미성 검증"

### 2. 7DOF → 2DOF 적응
**질문**: Manipulator용 VLM을 Mobile robot에 전이 가능한가?  
**문서**: IMPLEMENTATION_PLAN.md - "7DOF → 2DOF 적응 가능성"

### 3. 데이터 증강 효과
**질문**: 500개 → 5,000개 증강이 성능 개선에 효과적인가?  
**문서**: NON_SIMULATION_AUGMENTATION.md

---

## 📅 타임라인

| Week | Phase | 주요 작업 |
|------|-------|----------|
| 1-2 | Phase 1 | Context Vector 추출 및 분석 |
| 3-4 | Phase 2 | ControlNet 증강 (×10) |
| 5 | Phase 2 | CAST 후진 동작 생성 |
| 7 | Phase 3 | 전체 468개 학습 |
| 8-9 | Phase 3 | 증강 데이터 학습 |
| 11-12 | Phase 4 | 실시간 추론 시스템 |
| 15-16 | Phase 5 | 논문 작성 |

---

**업데이트**: 2025-11-26  
**다음 리뷰**: Context Vector 추출 완료 시
