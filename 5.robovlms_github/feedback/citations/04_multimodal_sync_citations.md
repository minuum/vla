# 4. Multimodal Synchronization - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed from @RoboVLMs)**

### **4.1 Multimodal Fusion Function**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:323-392` (Updated from @RoboVLMs)
- **Implementation**: `merge_multi_modal_input()` function for vision-language-action fusion
- **Core Code**:
```python
def merge_multi_modal_input(
    self,
    input_embeds: torch.Tensor,
    multimodal_feats: torch.Tensor = None,
    labels: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    is_image=True,
    insert_idx=1,
    fill_zero=False,
):
    """멀티모달 입력 융합 (Vision-Language-Action)"""
    bs = input_embeds.shape[0]  # 배치 크기
    
    if is_image:
        # 이미지 인코딩
        rgb_feats = self.encode_images(multimodal_feats)
        
        # 이미지 토큰 시작/끝 마커 추가
        if self.start_image_token_id is not None:
            # 이미지 시작 토큰 임베딩
            image_start_embed = (
                self.word_embedding(self.start_image_token_id.to(self.model.device))
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(*rgb_feats.shape[:2], 1, 1)
            )
            
            # 이미지 끝 토큰 ID 설정
            if self.end_image_token_id is None:
                end_image_token_id = self.start_image_token_id + 1
            else:
                end_image_token_id = self.end_image_token_id
            # 이미지 끝 토큰 임베딩
            image_end_embed = (
                self.word_embedding(end_image_token_id.to(self.model.device))
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(*rgb_feats.shape[:2], 1, 1)
            )
            
            # 시작-이미지-끝 토큰 결합
            rgb_feats = torch.cat([image_start_embed, rgb_feats, image_end_embed], dim=2)
        
        # 시퀀스 차원 평탄화
        rgb_feats = rearrange(rgb_feats, "b l n d -> b (l n) d")
    else:
        rgb_feats = multimodal_feats
    
    added_seq_len = rgb_feats.shape[1]  # 추가된 시퀀스 길이
    
    # 텍스트와 이미지 임베딩 결합
    multimodal_embeds = torch.cat(
        [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
        dim=1,
    )
```

### **4.2 BaseRoboVLM Class Structure**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:34-57` (Updated from @RoboVLMs)
- **Implementation**: `BaseRoboVLM` class for multimodal VLA architecture
- **Code**:
```python
class BaseRoboVLM(nn.Module):
    """멀티모달 VLA 아키텍처 기본 클래스"""
    def __init__(
        self,
        configs,                    # 모델 설정
        train_setup_configs,        # 학습 설정
        act_encoder_configs=None,   # 액션 인코더 설정
        act_head_configs=None,      # 액션 헤드 설정
        fwd_head_configs=None,      # 순방향 헤드 설정
        window_size=None,          # 윈도우 크기
        use_obs_queries=True,       # 관찰 쿼리 사용 여부
        use_act_queries=True,      # 액션 쿼리 사용 여부
        use_hand_rgb=False,        # 손 RGB 사용 여부
        use_pixel_loss=True,       # 픽셀 손실 사용 여부
        use_mim_obs_loss=False,    # MIM 관찰 손실 사용 여부
        use_time_causal_attn=True, # 시간 인과적 어텐션 사용 여부
        vision_masked_ratio=0.9,   # 비전 마스킹 비율
        use_tube_mask=False,       # 튜브 마스크 사용 여부
        fwd_pred_next_n=1,         # 순방향 예측 스텝 수
        use_vision_resampler=False, # 비전 리샘플러 사용 여부
        vision_resampler_configs=None, # 비전 리샘플러 설정
        use_clip_norm=False,       # CLIP 정규화 사용 여부
        use_state=False,           # 상태 사용 여부
        **kwargs,
    ):
```

### **4.3 RoboFlamingo Multimodal Integration**
- **File**: `RoboVLMs/robovlms/model/backbone/roboflamingo.py:200-254` (Updated from @RoboVLMs)
- **Implementation**: `cat_multi_input_ids()` function for multimodal input concatenation
- **Code**:
```python
def cat_multi_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_ids: torch.Tensor = None,
    insert_idx: int = 0,
    attention_masks: torch.Tensor = None,
):
    """멀티모달 입력 ID 결합"""
    bs, seq_len = input_ids.shape[:2]  # 배치 크기, 시퀀스 길이
    device = input_ids.device
    
    if insert_idx >= 0:
        # 텍스트와 멀티모달 ID 결합
        return_ids = torch.cat(
            (input_ids[:, :insert_idx], multimodal_ids, input_ids[:, insert_idx:]),
            dim=1,
        )
        # 삽입 마스크 생성 (멀티모달 부분만 1)
        insert_masks = torch.cat(
            (
                torch.zeros(bs, insert_idx),           # 텍스트 앞부분 (0)
                torch.ones(multimodal_ids.shape),      # 멀티모달 부분 (1)
                torch.zeros(bs, seq_len - insert_idx), # 텍스트 뒷부분 (0)
            ),
            dim=1,
        )
    return return_ids, insert_masks
```

### **4.3 Multimodal Feature Processing**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1384-1442`
- **Implementation**: `pred_action_discrete()` function
- **Code**:
```python
def pred_action_discrete(self, instr_and_action_ids, vision_x, ...):
    """이산 액션 예측 (멀티모달 융합)"""
    action_dim = self.act_head_configs["action_dim"]  # 7DOF 액션 차원 (x,y,z,rx,ry,rz,gripper)
    generated_ids = []                                # 생성된 액션 토큰 ID 리스트
    kv_cache = None                                   # Key-Value 캐시 (어텐션 최적화)
    self.fwd_pred_next_n = 1                         # 순방향 예측 스텝 수
    
    for i in range(action_dim * self.fwd_pred_next_n):  # 7개 액션 차원 × 1스텝
        output_hs = self.model(                        # VLM 모델 순전파
            inputs_embeds=multimodal_embeds,           # 멀티모달 임베딩 (이미지+텍스트+액션)
            past_key_values=kv_cache,                  # 이전 어텐션 상태 재사용
            use_cache=True,                            # 캐시 사용으로 효율성 향상
        )
        kv_cache = output_hs.past_key_values           # 어텐션 상태 업데이트
        cur_id = output_hs.logits[:, -1].argmax(dim=-1) # 현재 액션 토큰 예측
        generated_ids.append(cur_id)                   # 예측된 토큰 저장
```

## 📊 **Synchronization Method Evidence**

### **4.4 Vision-Language Fusion**
- **Image Encoding**: 다양한 비전 인코더 지원 (이미지 → 시각적 특징)
  - **RoboFlamingo**: CLIP-based vision encoder (`clip_vision_encoder`)
  - **RoboKosMos**: Kosmos-2 vision model (`vision_model`)
  - **RoboUform**: Uform image encoder (`image_encoder`)
  - **RoboPaligemma**: Paligemma vision tower (`vision_tower`)
- **Text Encoding**: Language model tokenizer (텍스트 → 언어적 특징)
- **Fusion**: Attention-based multimodal fusion (어텐션 기반 멀티모달 융합)
- **Output**: Unified multimodal representation (통합된 멀티모달 표현)

### **4.5 Action Integration**
- **Action Head**: Dedicated action prediction head (전용 액션 예측 헤드)
- **History Modeling**: Temporal action sequence processing (시간적 액션 시퀀스 처리)
- **End-to-End**: Joint vision-language-action learning (통합된 시각-언어-액션 학습)

### **4.6 Multimodal Synchronization Features**
- **Token-based Fusion**: 이미지/텍스트/액션을 토큰으로 통일
- **Causal Attention**: 시간적 인과적 어텐션 (과거 → 미래)
- **Cache Optimization**: Key-Value 캐시로 효율성 향상
- **Sequential Generation**: 순차적 액션 토큰 생성

## 🎯 **Key Findings**

1. **Unified Architecture**: Single model for vision, language, and action (통합 아키텍처)
2. **Attention-based Fusion**: Advanced multimodal attention mechanisms (어텐션 기반 융합)
3. **Temporal Modeling**: History-aware action prediction (시간적 모델링)
4. **End-to-End Learning**: Joint optimization of all modalities (통합 최적화)
5. **Token-based Synchronization**: 모든 모달리티를 토큰으로 통일 (토큰 기반 동기화)
6. **Causal Generation**: 순차적 액션 토큰 생성으로 시간적 일관성 보장

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/model/backbone/base_backbone.py` (기본 멀티모달 융합)
- `RoboVLMs/robovlms/model/backbone/robokosmos.py` (Kosmos-2 비전 모델)
- `RoboVLMs/robovlms/model/backbone/robouform.py` (Uform 이미지 인코더)
- `RoboVLMs/robovlms/model/backbone/robopaligemma.py` (Paligemma 비전 타워)
- `RoboVLMs/robovlms/model/backbone/roboflamingo.py` (CLIP 비전 인코더)
- `RoboVLMs/robovlms/model/vision_encoder/vision_transformer.py` (CLIP 비전 인코더 구현)
