# Mobile-VLA 연구 로드맵

## Phase 1: RoboVLMs 검증 (Context Vector 분석)

### [/] 1.1 Pre-trained RoboVLMs Context Vector 추출
- [x] RoboVLMs 모델 로드 환경 구축
- [ ] Mobile-VLA 이미지 샘플링 (대표 50개)
- [ ] Context vector 추출 스크립트 작성
  - [ ] Model hook 구현
  - [ ] Forward pass 중간 출력 캡처
  - [ ] Vector 저장 (NPY/H5)
- [ ] Context vector 시각화
  - [ ] t-SNE/UMAP 차원 축소
  - [ ] 클러스터링 분석
  - [ ] Manipulator vs Mobile 비교

### [ ] 1.2 7DOF → 2DOF 적응 가능성 검증
- [ ] RoboVLMs action head 구조 분석
- [ ] 7DOF → 2DOF 매핑 전략 수립
  - [ ] Linear layer 차원 변경
  - [ ] Adapter 추가
  - [ ] Fine-tuning 필요 레이어 식별
- [ ] 소규모 실험 (50개 데이터)
  - [ ] Adapter only 학습
  - [ ] Full fine-tuning 비교
  - [ ] 수렴 가능성 확인

## Phase 2: 데이터셋 증강 (500 → 5,000)

### [x] 2.1 데이터셋 현황 분석
- [x] 468개 H5 파일 분석 완료
- [x] 액션 분포 파악 (후진 0%)
- [x] Language instruction 추가

### [ ] 2.2 ControlNet 기반 이미지 증강
- [ ] ControlNet 환경 구축
  - [ ] Diffusion 모델 설치
  - [ ] Depth estimation 모델 준비
- [ ] 증강 파이프라인 구현
  - [ ] Depth map 추출
  - [ ] 10가지 프롬프트 적용
  - [ ] 배치 프로세싱 (468 → 4,680)
- [ ] 품질 검증
  - [ ] Depth 일관성 확인
  - [ ] 액션 레이블 유효성 검증

### [ ] 2.3 CAST 기반 액션 증강 (후진 동작)
- [ ] VLM (GPT-4V/LLaVA) 선택
- [ ] Counterfactual 생성
  - [ ] "후진이 필요한 상황" 질의
  - [ ] BACKWARD 액션 생성
  - [ ] 궤적 validity 검증
- [ ] 후진 데이터 추가 (목표 +500)

### [ ] 2.4 시뮬레이션 증강 (선택)
- [ ] Habitat-AI 설치 (GPU 머신)
- [ ] 사무실 환경 구축
- [ ] Domain randomization 구현

## Phase 3: Mobile-VLA 학습 및 검증

### [ ] 3.1 전체 데이터셋 활용
- [ ] Git LFS 이슈 해결
- [ ] 468개 전체 학습 데이터 활용
  - [ ] Train: 375 (80%)
  - [ ] Val: 93 (20%)
- [ ] Baseline 재학습
  - [ ] LoRA 설정 동일 유지
  - [ ] Val Loss 개선 확인 (0.213 → 0.17 목표)

### [ ] 3.2 증강 데이터 학습
- [ ] ControlNet 증강 데이터 학습
  - [ ] 4,680개 데이터셋 준비
  - [ ] 학습 및 검증
  - [ ] Val Loss 측정
- [ ] CAST 증강 데이터 추가 학습
  - [ ] 후진 동작 포함
  - [ ] 다양한 액션 패턴 학습

### [ ] 3.3 Language-conditioned 학습
- [ ] Language instruction 활용
  - [ ] Text encoder 통합
  - [ ] Cross-attention 구현
- [ ] Left/Right conditioning 검증

## Phase 4: 추론 시스템 구축

### [ ] 4.1 실시간 추론 파이프라인
- [ ] 추론 주기 설계 (0.4초 vs 0.2초)
- [ ] Action chunk 처리
  - [ ] 10개 액션 한 번에 예측
  - [ ] 0.02초 간격으로 실행
- [ ] 카메라-모델-로봇 루프 구현
  - [ ] 이미지 캡처
  - [ ] VLM 추론
  - [ ] 속도 명령 발행

### [ ] 4.2 Inference Test
- [ ] 실제 로봇 테스트
  - [ ] 거리 측정 (초기)
  - [ ] 목표까지 내비게이션
  - [ ] 성공률 측정
- [ ] 다양한 조건 테스트
  - [ ] 조명 변화
  - [ ] 장애물 배치 변경
  - [ ] 목표 거리 변경

### [ ] 4.3 성능 벤치마크
- [ ] 추론 속도 측정
  - [ ] VLM forward pass 시간
  - [ ] 전체 루프 지연시간
- [ ] 제어 정확도 측정
  - [ ] 목표 도달 오차
  - [ ] 경로 효율성

## Phase 5: 논문 작성

### [ ] 5.1 Ablation Study
- [ ] 증강 효과 비교
  - [ ] Baseline (468)
  - [ ] +ControlNet (4,680)
  - [ ] +CAST (5,180)
- [ ] 7DOF → 2DOF 적응 분석
- [ ] Language conditioning 효과

### [ ] 5.2 Mobile-VLA 관련 연구 조사
- [ ] Mobile robot VLA 선행 연구
- [ ] Domain adaptation 연구
- [ ] Few-shot learning 연구

### [ ] 5.3 논문 작성
- [ ] Introduction
- [ ] Method
- [ ] Experiments
- [ ] Results & Discussion

---

## 우선순위 (다음 2주)

### Week 1
1. **RoboVLMs Context Vector 추출** (Phase 1.1)
2. **ControlNet 증강 시작** (Phase 2.2)
3. **전체 데이터셋 학습** (Phase 3.1)

### Week 2
4. **7DOF → 2DOF 검증** (Phase 1.2)
5. **CAST 후진 생성** (Phase 2.3)
6. **증강 데이터 학습** (Phase 3.2)

---

## 주요 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| Context vector가 의미없음 | 높음 | Pre-training 필요, 다른 backbone 시도 |
| 500개로 7DOF→2DOF 불가 | 높음 | 시뮬레이션 대량 증강, Transfer learning |
| 추론 속도 느림 (0.4초 초과) | 중간 | 모델 경량화, TensorRT, Quantization |
| Mobile VLA 선행 연구 부족 | 중간 | Novelty 강조, Manipulator 연구 참고 |

---

**업데이트**: 2025-11-26  
**다음 리뷰**: Context vector 추출 완료 후
