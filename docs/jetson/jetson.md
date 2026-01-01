# Jetson On-Device Execution

**최종 업데이트**: 2026-01-01

## Phase 1 완료: 메모리 측정
- Vision Encoder: 1.89GB
- LLM: 3.59GB
- 전체 파이프라인: 2.79GB
- 여유: 7.8GB

## 결론
FP16으로 충분! 경량화 불필요.

## 다음
Phase 2: 로컬 추론 노드 테스트

---

## 관련 문서
- `memory_profiling.md` - 상세 측정 결과
- `ondevice_plan.md` - 최적화 계획
