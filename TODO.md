# TODO: 다음 작업 항목

## 우선순위 1: 오른쪽 경로 가이드 정의
- [ ] `1box_right` 시나리오의 기본 가이드 정의 필요
  - 현재 테스트 가이드: `W W D E E E E E E E W W Q Q Q W W`
  - 이 가이드를 검증하고 `mobile_vla_data_collector.py`의 `default_guides`에 추가
  - 위치: `get_core_pattern_guide_keys` 함수 내부

## 현재 데이터셋 상태 (2025-12-03 18:20)
- 총 파일: 250개
- 패턴: `W W A Q Q Q Q Q Q Q W W E E E W W` (100% 일치)
- 시나리오: `1box_left` (250/250 완료)
- 다음 시나리오: `1box_right` (0/250)

## 다음 수집 목표
- `1box_right` 시나리오 250개 수집
- 목표 패턴: 오른쪽 경로에 맞는 가이드 정의 후 수집
