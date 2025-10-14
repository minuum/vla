# 07_3 Training & Inference Process Analysis - RoboVLMs GitHub Citations

## 📊 **Training & Inference Process Technical Analysis**

### **7.3.1 Training Process - Batch Processing**
- **Source**: `RoboVLMs/robovlms/train/base_trainer.py:345-395`  # GitHub 코드에서 확인된 훈련 과정
- **Batch Structure**:  # 배치 구조
  ```python
  def _process_batch(self, batch):
      """
      액션 예측 배치 처리
      args: rgb, language, attention_mask, hand_rgb, action
      reformat: action to input and target (seq_len = window size + chunck size)
      """
      # RGB 데이터 처리
      if len(rgb.shape) == 4:
          rgb = rgb.unsqueeze(1)              # 4차원 → 5차원으로 확장
      assert len(rgb.shape) == 5              # (batch, seq_len, channels, height, width)
      
      # 시퀀스 길이 설정
      seq_len = self.configs["window_size"]   # 윈도우 크기로 시퀀스 길이 설정
      language = batch["text"].cuda()         # 언어 데이터 GPU로 이동
      text_mask = batch["text_mask"].cuda()   # 텍스트 마스크 GPU로 이동
  ```

### **7.3.2 Training Process - Sequence Length**
- **Source**: `RoboVLMs/robovlms/train/base_trainer.py:349`  # GitHub 코드에서 확인된 시퀀스 길이
- **Sequence Length Formula**: `seq_len = window_size + chunk_size`  # 시퀀스 길이 = 윈도우 크기 + 청크 크기
- **Example**: `window_size=16, fwd_pred_next_n=2` → `seq_len=18`  # 예시: 윈도우 16, 순방향 예측 2 → 시퀀스 길이 18

### **7.3.3 Training Process - Data Chunking**
- **Source**: `RoboVLMs/robovlms/data/data_utils.py:249-270`  # GitHub 코드에서 확인된 데이터 청킹
- **Chunk Generation**:  # 청크 생성
  ```python
  def generate_chunck_data(data, window_size, chunk_size):
      """데이터 청킹 생성 함수"""
      bs, seq_len = data.shape[:2]                    # 배치 크기, 시퀀스 길이
      assert seq_len == window_size + chunk_size      # 시퀀스 길이 = 윈도우 크기 + 청크 크기
      data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)  # 윈도우 크기만큼 반복
      mask = claw_matrix(seq_len, chunk_size - 1, data_flatten.device)        # 클로 매트릭스 마스크
      mask = mask[:window_size].bool()                 # 윈도우 크기만큼 마스크 자르기
      data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)  # 최종 데이터 형태
      return data_flatten
  ```

### **7.3.4 Training Process - Mobile VLA Example**
- **Source**: `RoboVLMs/robovlms/data/mobile_vla_action_dataset.py:223-238`  # GitHub 코드에서 확인된 Mobile VLA 예시
- **Length Consistency**:  # 길이 일관성
  ```python
  # 길이 정합성 보장: images=18(window+fwd), actions=17(window+fwd-1) 필요
  target_img_len = self.window_size + self.fwd_pred_next_n  # 16 + 2 = 18
  if images.shape[0] > target_img_len:
      images = images[:target_img_len]                      # 이미지 길이 자르기
  elif images.shape[0] < target_img_len:
      pad = target_img_len - images.shape[0]               # 패딩 계산
      last = images[-1:]                                    # 마지막 이미지
      images = np.concatenate([images, np.repeat(last, pad, axis=0)], axis=0)  # 패딩 추가
  
  # actions가 18이면 마지막 1개를 제거해 윈도우 규칙(window=16, fwd=2)에 맞춤
  if actions.shape[0] > 17:
      actions = actions[:17]                               # 액션 길이 자르기
  ```

### **7.3.5 Training Process - Forward Pass**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:910-928`  # GitHub 코드에서 확인된 순전파
- **Forward Pass**:  # 순전파
  ```python
  def forward_discrete(self, vision_x, lang_x, ...):
      """이산 액션 순전파"""
      assert vision_x is not None
      bs, window_size = vision_x.shape[:2]  # 배치 크기, 윈도우 크기
      
      # 2차원인 경우 윈도우 크기만큼 반복
      if instr_and_action_ids.ndim == 2:
          instr_and_action_ids = instr_and_action_ids.unsqueeze(1).repeat(1, window_size, 1)
          instr_and_action_labels = instr_and_action_labels.unsqueeze(1).repeat(1, window_size, 1)
          instr_and_action_mask = instr_and_action_mask.unsqueeze(1).repeat(1, window_size, 1)
      
      # 차원 평탄화
      instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
      vision_x = vision_x.flatten(0, 1)  # (bs * window_size, ...)
  ```

### **7.3.6 Inference Process - Single Image Sequential Processing**
- **Source**: `RoboVLMs/vla_test/standalone_vla_test.py:87-124`  # GitHub 코드에서 확인된 단일 이미지 순차 처리
- **Sequential Single Image Inference**:  # 순차적 단일 이미지 추론
  ```python
  def infer_from_image_and_text(self, image: np.ndarray, text_prompt: str) -> str:
      """이미지와 텍스트로부터 VLA 추론 수행 (단일 이미지 순차 처리)"""
      # 단일 이미지 처리 (한 번에 하나씩)
      if len(image.shape) == 3 and image.shape[2] == 3:
          rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
      else:
          rgb_image = image
      
      pil_image = PilImage.fromarray(rgb_image)  # 단일 이미지를 PIL 이미지로 변환
      
      # 모델 입력 준비 (단일 이미지)
      inputs = self.processor(
          images=pil_image,      # 단일 이미지
          text=text_prompt,      # 텍스트 프롬프트
          return_tensors="pt"    # PyTorch 텐서로 반환
      ).to(self.device)
      
      # 추론 실행 (단일 이미지에 대해)
      with torch.no_grad():
          outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
          result = self.processor.decode(outputs[0], skip_special_tokens=True)
  ```

### **7.3.6.1 Inference Process - Step Function Sequential Processing**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:318-378`  # GitHub 코드에서 확인된 Step 함수 순차 처리
- **Sequential Step Processing**:  # 순차적 Step 처리
  ```python
  def step(self, obs, goal):
      """Step function - 한 번에 하나의 관찰 처리"""
      input_dict = dict()
      image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)
      
      input_dict["rgb"] = image_x  # 단일 이미지
      input_dict["hand_rgb"] = gripper_x  # 단일 그리퍼 이미지
      input_dict["text"] = text_x  # 단일 텍스트
      input_dict["text_mask"] = mask  # 단일 마스크
      
      # 단일 관찰에 대한 추론
      with torch.no_grad():
          action = self.policy.inference_step(input_dict)["action"]
      
      # 액션 후처리 (단일 액션)
      if self.action_space != "discrete":
          if action[0].ndim == action[1].ndim + 1:
              action = (action[0], action[1].unsqueeze(2))
          action = torch.cat([action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()], dim=-1)
      
      # 액션 앙상블 적용 (단일 액션에 대해)
      action = self.ensemble_action(action)
      
      if isinstance(action, torch.Tensor):
          action = action.squeeze()
          if action.ndim == 2:
              action = action[0]  # 단일 액션 반환
  ```

### **7.3.6.2 Inference Process - VLA Node Sequential Processing**
- **Source**: `RoboVLMs/vla_node.py:258-281`  # GitHub 코드에서 확인된 VLA 노드 순차 처리
- **Sequential VLA Processing**:  # 순차적 VLA 처리
  ```python
  def infer_and_parse(current_raw_image, current_prompt):
      """단일 이미지와 프롬프트에 대한 순차 처리"""
      img_width, img_height = current_raw_image.size
      print(f"Input image size: ({img_width}, {img_height}) for prompt: '{current_prompt}'")
      
      # 단일 이미지 처리
      inputs_data = processor(text=current_prompt, images=current_raw_image, return_tensors="pt").to(device)
      
      print("Performing inference...")
      with torch.inference_mode():  # 단일 이미지에 대한 추론
          try:
              output_ids = model.generate(**inputs_data, max_new_tokens=max_new_tokens, do_sample=False)
              generated_text_output = processor.decode(output_ids[0], skip_special_tokens=True)
              
              # 단일 결과 파싱
              parsed_detections = parse_segmentation_output(generated_text_output, img_width, img_height, current_prompt)
  ```

### **7.3.6.3 Inference Process - Test Script Sequential Processing**
- **Source**: `RoboVLMs/test.py:144-177`  # GitHub 코드에서 확인된 테스트 스크립트 순차 처리
- **Sequential Test Processing**:  # 순차적 테스트 처리
  ```python
  def inference(model, image, instruction, device="cpu"):
      """단일 이미지와 지시문에 대한 순차 추론"""
      logger.info("인퍼런스 시작...")
      start_time = time.time()
      
      try:
          # 단일 이미지 전처리
          logger.info("이미지 전처리 중...")
          if isinstance(image, str):
              image = load_image(image)  # 단일 이미지 로드
          
          preprocessed_image = preprocess_image(image, model.configs["image_size"])
          preprocessed_image = preprocessed_image.to(device)
          
          # 단일 텍스트 인코딩
          logger.info("텍스트 인코딩 중...")
          encoded_text = model.encode_text(instruction)
          
          # 단일 이미지에 대한 모델 추론
          logger.info("모델 인퍼런스 실행 중...")
          with torch.no_grad():
              output = model.generate(
                  preprocessed_image,  # 단일 이미지
                  encoded_text,  # 단일 텍스트
                  max_new_tokens=128, 
                  temperature=0.7
              )
  ```

### **7.3.7 Inference Process - Model Wrapper**
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:28-39`  # GitHub 코드에서 확인된 모델 래퍼
- **Model Wrapper**:  # 모델 래퍼
  ```python
  class CustomModel:
      def __init__(
          self,
          ckpt_path,
          configs,
          device,
          save_dir=None,
          raw_calvin=True,
          debug=False,
          action_ensemble=False,
      ):
          self.model = BaseTrainer(configs=configs)
          self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
  ```

### **7.3.8 Training vs Inference Process Comparison**
- **Source**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:177-181`  # GitHub 코드에서 확인된 훈련 vs 추론 비교
- **Training Mode**:  # 훈련 모드
  ```python
  if self.mode == "train":
      assert action.shape[0] == self.window_size + self.fwd_pred_next_n - 1
      window_size = self.window_size
  else:
      window_size = action.shape[0] + 1
  ```

### **7.3.9 Image Processing in Model**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:188-221`  # GitHub 코드에서 확인된 모델의 이미지 처리
- **Image Encoding**:  # 이미지 인코딩
  ```python
  def encode_images(self, images, image_sizes=None):
      # input: images: list of b,c,h,w or b,t,c,h,w
      # output: image_features: b, t, n, d
      
      if images.ndim == 4:
          images = images.unsqueeze(1)  # (b, c, h, w) -> (b, 1, c, h, w)
      
      bs, seq_len = images.shape[:2]  # 배치 크기, 시퀀스 길이
      
      if type(images) is list or images.ndim == 5:
          if type(images) is list:
              images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
          concat_images = torch.cat([image for image in images], dim=0)
          image_features = self.model_encode_images(concat_images)
      else:
          image_features = self.model_encode_images(images)
      
      image_features = torch.stack(image_features, dim=0).view(
          bs, seq_len, -1, image_features[0].shape[-1]
      )
  ```

### **7.3.10 Data Collation Process**
- **Source**: `RoboVLMs/robovlms/data/concat_dataset.py:110-122`  # GitHub 코드에서 확인된 데이터 수집 과정
- **Data Collation**:  # 데이터 수집
  ```python
  fwd_rgb_chunck = generate_chunck_data(
      image_tensors, self.window_size, self.fwd_pred_next_n
  )
  fwd_hand_rgb_chunck = generate_chunck_data(
      gripper_tensors, self.window_size, self.fwd_pred_next_n
  )
  chunck_mask = generate_chunck_data(
      image_mask, self.window_size, self.fwd_pred_next_n
  )
  action_chunck = generate_chunck_data(
      action_tensors, self.window_size, self.fwd_pred_next_n
  )
  ```

## 🎯 **Key Findings**

### **7.3.11 Training vs Inference Process Summary**
1. **Training Process**:  # 훈련 과정
   - **Batch Processing**: Processes multiple sequences simultaneously  # 여러 시퀀스를 동시에 처리
   - **Sequence Length**: `window_size + fwd_pred_next_n` (e.g., 16 + 2 = 18)  # 시퀀스 길이: 윈도우 크기 + 순방향 예측 스텝
   - **Data Chunking**: Uses sliding window approach for temporal context  # 시간적 컨텍스트를 위한 슬라이딩 윈도우 접근법 사용

2. **Inference Process**:  # 추론 과정
   - **Single Image Sequential Processing**: Processes one image at a time sequentially  # 한 번에 하나의 이미지를 순차적으로 처리
   - **Step-by-Step**: Each step processes single observation  # 각 단계마다 단일 관찰 처리
   - **Real-time**: Suitable for real-time robot control  # 실시간 로봇 제어에 적합
   - **Sequential**: Processes images one by one in sequence  # 이미지를 하나씩 순차적으로 처리

### **7.3.12 Sequential Processing Evidence Summary**
- **Source**: `RoboVLMs/vla_test/standalone_vla_test.py:87-124`  # Standalone VLA 순차 처리
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:318-378`  # CALVIN Step 함수 순차 처리
- **Source**: `RoboVLMs/vla_node.py:258-281`  # VLA 노드 순차 처리
- **Source**: `RoboVLMs/test.py:144-177`  # 테스트 스크립트 순차 처리

### **7.3.13 Sequential Processing Technical Details**
1. **Single Image Input**: All inference functions take single image as input  # 모든 추론 함수는 단일 이미지를 입력으로 받음
2. **One-by-One Processing**: Each function processes one image at a time  # 각 함수는 한 번에 하나의 이미지 처리
3. **Sequential Execution**: Images are processed in sequence, not in batches  # 이미지는 배치가 아닌 시퀀스로 처리
4. **Real-time Capability**: Designed for real-time robot control  # 실시간 로봇 제어를 위해 설계됨

### **7.3.12 Technical Implementation Details**
- **Window Size**: Controls historical context length  # 히스토리 컨텍스트 길이 제어
- **Forward Prediction**: Predicts multiple future actions  # 여러 미래 액션 예측
- **Data Structure**: `(batch_size, window_size, channels, height, width)`  # 데이터 구조
- **Memory Management**: Efficient processing of long sequences  # 긴 시퀀스의 효율적 처리

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/train/base_trainer.py` (L345-395)  # 기본 트레이너
- `RoboVLMs/robovlms/data/data_utils.py` (L249-270)  # 데이터 유틸리티
- `RoboVLMs/robovlms/data/mobile_vla_action_dataset.py` (L223-238)  # Mobile VLA 액션 데이터셋
- `RoboVLMs/robovlms/model/backbone/base_backbone.py` (L188-221, 910-928)  # 기본 백본 모델
- `RoboVLMs/vla_test/standalone_vla_test.py` (L87-124)  # 독립 VLA 테스트 - 순차 처리
- `RoboVLMs/eval/calvin/model_wrapper.py` (L28-39, 318-378)  # CALVIN 모델 래퍼 - Step 함수 순차 처리
- `RoboVLMs/vla_node.py` (L258-281)  # VLA 노드 - 순차 처리
- `RoboVLMs/test.py` (L144-177)  # 테스트 스크립트 - 순차 처리
- `RoboVLMs/robovlms/data/concat_dataset.py` (L110-122)  # 연결 데이터셋
