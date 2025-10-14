# 7. Calvin Dataset Analysis - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed from @RoboVLMs)**

### **7.1 CALVIN Dataset Class Implementation**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:521-602` (Updated from @RoboVLMs)
- **Implementation**: `DiskCalvinDataset` class for CALVIN dataset loading
- **Code**:
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """
    디스크에서 개별 파일로 에피소드를 로드하는 데이터셋
    Args:
        skip_frames: 언어 데이터셋을 위한 윈도우 스킵 수
        save_format: datasets_dir의 파일 형식 (pkl 또는 npz)
        pretrain: 사전 훈련 시 True로 설정
    """
    def __init__(
        self,
        image_fn: Callable,           # 이미지 처리 함수
        tokenizer: Callable,          # 토크나이저 함수
        *args: Any,                   # 추가 인수들
        skip_frames: int = 1,         # 프레임 스킵 수
        save_format: str = "npz",     # 저장 형식
        pretrain: bool = False,       # 사전 훈련 여부
        partial_data=False,          # 부분 데이터 사용 여부
        decoder_type="lstm",          # 디코더 타입
        discrete_action=False,        # 이산 액션 사용 여부
        action_tokenizer=None,        # 액션 토크나이저
        model_name="vicuna",          # 모델 이름
        predict_stop_token=True,      # 정지 토큰 예측 여부
        use_mu_law=False,            # μ-law 사용 여부
        mu_val=255,                   # μ-law 값
        n_bin=256,                    # 이산화 빈 수
        min_action=-1,                # 액션 최소값
        max_action=1,                 # 액션 최대값
        task_type="calvin_action",    # 태스크 타입
        tcp_rel=False,                # TCP 상대 좌표 사용 여부
        few_shot=False,               # Few-shot 학습 여부
        exclude_tasks=[],             # 제외할 태스크 목록
        **kwargs: Any,                # 추가 키워드 인수들
    ):
```

### **7.2 CALVIN Evaluation Framework**
- **File**: `RoboVLMs/eval/calvin/eval_utils.py:64-120` (Updated from @RoboVLMs)
- **Implementation**: CALVIN evaluation metrics and success counting
- **Code**:
```python
def count_success(results):
    """CALVIN 성공률 계산 함수"""
    step_success = []
    # 1-5개 연속 태스크 성공률 계산
    for i in range(1, 6):
        success_count = sum(1 for result in results if result >= i)  # i개 이상 성공한 경우
        success_rate = success_count / len(results)  # 성공률 계산
        step_success.append(success_rate)
    return step_success

def print_and_save(results, sequences, eval_result_path, epoch=None):
    """CALVIN 평가 결과 출력 및 저장"""
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)  # 평균 성공 시퀀스 길이
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}  # 체인 성공률
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")  # i개 연속 성공률 출력
    
    cnt_success = Counter()  # 성공한 태스크 카운터
    cnt_fail = Counter()      # 실패한 태스크 카운터
    
    # 각 결과와 시퀀스에 대해 성공/실패 카운팅
    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:  # 성공한 태스크들
            cnt_success[successful_tasks] += 1
        if result < len(sequence):  # 실패한 경우
            failed_task = sequence[result]  # 실패한 태스크
            cnt_fail[failed_task] += 1
    
    total = cnt_success + cnt_fail  # 전체 태스크 수
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        # 각 태스크별 성공률 출력
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")
```

### **7.3 CALVIN Model Wrapper**
- **File**: `RoboVLMs/eval/calvin/model_wrapper.py:28-147` (Updated from @RoboVLMs)
- **Implementation**: `CustomModel` class for CALVIN evaluation
- **Code**:
```python
class CustomModel:
    """CALVIN 평가를 위한 커스텀 모델 래퍼"""
    def __init__(
        self,
        ckpt_path,                    # 체크포인트 파일 경로
        configs,                     # 모델 설정
        device,                      # 실행 디바이스
        save_dir=None,               # 저장 디렉토리
        raw_calvin=True,             # 원본 CALVIN 데이터 사용 여부
        debug=False,                 # 디버그 모드 여부
        action_ensemble=False,       # 액션 앙상블 사용 여부
    ):
        self.ckpt_path = ckpt_path           # 체크포인트 경로 저장
        self.configs = configs               # 설정 정보 저장
        self.device = device                 # 디바이스 정보 저장
        self.save_dir = save_dir             # 저장 디렉토리 저장
        self.raw_calvin = raw_calvin         # 원본 CALVIN 사용 여부 저장
        self.debug = debug                   # 디버그 모드 저장
        self.action_ensemble = action_ensemble  # 액션 앙상블 설정 저장
        # 모델 초기화 및 설정
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)
```

### **7.4 CALVIN Benchmark Results**
- **File**: `RoboVLMs/README.md:113-136` (Updated from @RoboVLMs)
- **Implementation**: CALVIN benchmark performance results
- **Code**:
```python
# CALVIN Benchmark Results from README
# ABCD -> D Split
# KosMos P.H. (RoboVLMs): 96.7% success rate, 4.49 average length
# ABC -> D Split  
# KosMos P.H. (RoboVLMs): 98.0% success rate, 4.25 average length
```

### **7.3 CALVIN Evaluation Framework**
- **File**: `RoboVLMs/eval/calvin/eval_utils.py:64-120`
- **Implementation**: CALVIN evaluation metrics and success counting
- **Code**:
```python
def count_success(results):
    """CALVIN 성공률 계산 함수"""
    # results: 각 시퀀스에서 성공한 태스크 수의 리스트
    step_success = []  # 각 단계별 성공률 저장
    for i in range(1, 6):  # 1-5개 연속 태스크 성공률 계산
        success_count = sum(1 for result in results if result >= i)  # i개 이상 성공한 시퀀스 수
        success_rate = success_count / len(results)  # 전체 대비 성공률
        step_success.append(success_rate)  # 단계별 성공률 추가
    return step_success  # [1개 성공률, 2개 성공률, ..., 5개 성공률] 반환

def print_and_save(results, sequences, eval_result_path, epoch=None):
    """CALVIN 평가 결과 출력 및 저장"""
    print(f"Results for Epoch {epoch}:")  # 현재 에포크 결과 출력
    avg_seq_len = np.mean(results)  # 평균 성공 시퀀스 길이 계산
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}  # 체인 성공률 딕셔너리 생성
    print(f"Average successful sequence length: {avg_seq_len}")  # 평균 성공 길이 출력
    print("Success rates for i instructions in a row:")  # 연속 성공률 헤더
    for i, sr in chain_sr.items():  # 각 연속 성공률 출력
        print(f"{i}: {sr * 100:.1f}%")  # 백분율로 성공률 출력
    
    cnt_success = Counter()  # 성공한 태스크 카운터
    cnt_fail = Counter()  # 실패한 태스크 카운터
    
    for result, (_, sequence) in zip(results, sequences):  # 결과와 시퀀스 매칭
        for successful_tasks in sequence[:result]:  # 성공한 태스크들 카운트
            cnt_success[successful_tasks] += 1
        if result < len(sequence):  # 실패한 경우
            failed_task = sequence[result]  # 실패한 태스크 식별
            cnt_fail[failed_task] += 1  # 실패 태스크 카운트
    
    total = cnt_success + cnt_fail  # 전체 태스크 수 계산
    task_info = {}  # 태스크별 정보 딕셔너리
    for task in total:  # 각 태스크별 성공률 계산
        task_info[task] = {"success": cnt_success[task], "total": total[task]}  # 성공/전체 수 저장
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")  # 태스크별 성공률 출력
    
    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}  # 결과 데이터 구성
    current_data[epoch] = data  # 현재 에포크 데이터 저장
    
    print()  # 빈 줄 출력
    previous_data = {}  # 이전 데이터 초기화
    json_data = {**previous_data, **current_data}  # JSON 데이터 병합
    with open(eval_result_path, "w") as file:  # 결과 파일 저장
        json.dump(json_data, file)  # JSON 형태로 저장
    print(f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "  # 최고 성능 에포크 출력
          f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}")  # 최고 평균 길이 출력
```

### **7.4 CALVIN Model Wrapper Implementation**
- **File**: `RoboVLMs/eval/calvin/model_wrapper.py:28-147`
- **Implementation**: CALVIN 모델 래퍼 클래스
- **Code**:
```python
class CustomModel:
    """CALVIN 평가를 위한 커스텀 모델 래퍼"""
    def __init__(self, ckpt_path, configs, device, save_dir=None, raw_calvin=True, debug=False, action_ensemble=False):
        self.ckpt_path = ckpt_path  # 체크포인트 경로 저장
        self.configs = configs  # 설정 정보 저장
        self.device = device  # 디바이스 정보 저장
        self.save_dir = save_dir  # 저장 디렉토리 설정
        self.raw_calvin = raw_calvin  # 원시 CALVIN 데이터 사용 여부
        self.debug = debug  # 디버그 모드 설정
        self.action_ensemble = action_ensemble  # 액션 앙상블 사용 여부
        self.init_config(ckpt_path, configs, device, save_dir, raw_calvin, debug)  # 설정 초기화

    def init_config(self, ckpt_path, configs, device, save_dir=None, raw_calvin=False, debug=False):
        """모델 설정 초기화"""
        self.model = self.load_model(ckpt_path)  # 모델 로드
        self.configs = configs  # 설정 저장
        self.device = device  # 디바이스 설정
        self.save_dir = save_dir  # 저장 경로 설정
        self.raw_calvin = raw_calvin  # 원시 CALVIN 플래그 설정
        self.debug = debug  # 디버그 모드 설정
        
        # 데이터 타입 설정 (FP16 또는 FP32)
        if self.configs["trainer"]["precision"] == "fp16":
            dtype = torch.float16  # FP16 정밀도 설정
        else:
            dtype = torch.float32  # FP32 정밀도 설정
        self.dtype = dtype  # 데이터 타입 저장
        self.act_head_configs = self.configs["act_head"]  # 액션 헤드 설정 저장
        self.raw_calvin = raw_calvin  # 원시 CALVIN 플래그 재설정
        self.tcp_rel = self.configs.get("tcp_rel", False)  # TCP 상대 좌표 사용 여부
        
        print(f"raw action: {self.raw_calvin}")  # 원시 액션 사용 여부 출력
        
        self.device = device  # 디바이스 재설정
        self.policy = self.model  # 정책 모델 설정
        self.policy = self.policy.to(self.dtype)  # 정책 모델을 지정된 데이터 타입으로 변환
        self.policy.to(self.device)  # 정책 모델을 지정된 디바이스로 이동
        self.policy.eval()  # 정책 모델을 평가 모드로 설정
        
        # 언어 모델 헤드가 없는 경우 액션 헤드를 언어 모델 헤드로 사용
        if not hasattr(self.policy.model, "lm_head"):
            self.policy.model.lm_head = self.policy.model.act_head  # 액션 헤드를 언어 모델 헤드로 설정
        
        self.tokenizer = build_tokenizer(self.configs["tokenizer"])  # 토크나이저 빌드
        
        self.window_size = configs["window_size"]  # 윈도우 크기 설정
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]  # 순방향 예측 스텝 수 설정
        self.act_step = self.fwd_pred_next_n + 1  # 액션 스텝 계산
        self.seq_len = self.configs["seq_len"]  # 시퀀스 길이 설정
        self.use_hand_rgb = self.configs["use_hand_rgb"]  # 손목 카메라 RGB 사용 여부
        
        # 정책 설정에 따른 데이터 믹스 설정
        if hasattr(self, "policy_setup"):
            data_mix = "bridge" if self.policy_setup == "widowx_bridge" else "rt_1"  # 정책에 따른 데이터 믹스 선택
            configs["train_dataset"]["data_mix"] = data_mix  # 훈련 데이터 믹스 설정
            configs["val_dataset"]["data_mix"] = data_mix  # 검증 데이터 믹스 설정
        
        # 이미지 전처리 함수 설정
        image_preprocess = self.model.model.image_processor  # 모델의 이미지 프로세서 가져오기
        self.image_preprocess = functools.partial(  # 이미지 전처리 함수 부분 적용
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        )
        
        # 텍스트 전처리 함수 설정
        self.text_preprocess = get_text_function(  # 텍스트 전처리 함수 가져오기
            self.model.model.tokenizer, configs["model"]
        )
        
        # 액션 공간 설정 (연속 또는 이산)
        self.action_space = self.configs["act_head"].get("action_space", "continuous")  # 액션 공간 타입 설정
        if self.action_space == "discrete":  # 이산 액션 공간인 경우
            self.action_tokenizer = ActionTokenizer(  # 액션 토크나이저 생성
                self.tokenizer,
                bins=self.act_head_configs["n_bin"],  # 빈 수 설정
                min_action=self.act_head_configs["min_action"],  # 최소 액션 값 설정
                max_action=self.act_head_configs["max_action"],  # 최대 액션 값 설정
            )
        
        print(f"Evaluating checkpoint {ckpt_path}")  # 평가할 체크포인트 출력
        
        # 평가를 위한 리스트 초기화
        self.rgb_list = []  # RGB 이미지 리스트 초기화
        self.hand_rgb_list = []  # 손목 RGB 이미지 리스트 초기화
        self.action_hist_list = []  # 액션 히스토리 리스트 초기화
```

## 📊 **Dataset Analysis Evidence**

### **7.5 CALVIN Dataset Structure**
- **Dataset Class**: `DiskCalvinDataset` for loading episodes from disk  # 디스크에서 에피소드 로딩을 위한 DiskCalvinDataset 클래스
- **File Format**: NPZ format for efficient data loading  # 효율적인 데이터 로딩을 위한 NPZ 형식
- **Splits**: A, B, C, D for training and evaluation  # 훈련 및 평가를 위한 A, B, C, D 분할
- **Action Space**: 7-DOF continuous action space  # 7자유도 연속 액션 공간

### **7.5.1 CALVIN Dataset Official Specifications**
- **Source**: CALVIN Official Documentation  # CALVIN 공식 문서
- **Total Data**: 6 hours of teleoperated play data in each of 4 environments  # 4개 환경 각각에서 6시간의 텔레오퍼레이션 플레이 데이터
- **Download Sizes**:  # 다운로드 크기
  - **Split D→D**: 166 GB  # D→D 분할: 166GB
  - **Split ABC→D**: 517 GB  # ABC→D 분할: 517GB
  - **Split ABCD→D**: 656 GB  # ABCD→D 분할: 656GB
  - **Debug Dataset**: 1.3 GB  # 디버그 데이터셋: 1.3GB

### **7.6 Language Instructions**
- **Natural Language**: Human-readable task descriptions  # 인간이 읽을 수 있는 태스크 설명
- **Task Categories**: Pick, place, open, close, push, pull, etc.  # 집기, 놓기, 열기, 닫기, 밀기, 당기기 등
- **Instruction Format**: "Pick up the red block and place it in the box"  # "빨간 블록을 집어서 상자에 넣어라"

### **7.6.1 CALVIN Language Annotations Structure**
- **Source**: CALVIN Official Documentation  # CALVIN 공식 문서
- **Language Embeddings**: Precomputed language embeddings available  # 사전 계산된 언어 임베딩 사용 가능
- **Available Embeddings**:  # 사용 가능한 임베딩들
  - **lang_all-distilroberta-v1**: DistilRoBERTa 기반 임베딩
  - **lang_all-MiniLM-L6-v2**: MiniLM 기반 임베딩
  - **lang_all-mpnet-base-v2**: MPNet 기반 임베딩
  - **lang_BERT**: BERT 기반 임베딩
  - **lang_clip_resnet50**: CLIP ResNet50 기반 임베딩
  - **lang_clip_ViTB32**: CLIP ViT-B/32 기반 임베딩
- **Data Structure**:  # 데이터 구조
  - **`['language']['ann']`**: Raw language annotations  # 원시 언어 주석
  - **`['language']['task']`**: Task ID list  # 태스크 ID 리스트
  - **`['language']['emb']`**: Precomputed MiniLM embeddings  # 사전 계산된 MiniLM 임베딩
  - **`['info']['indx']`**: Start and end indices for language embeddings  # 언어 임베딩의 시작 및 끝 인덱스

### **7.7 Action Space**
- **7-DOF Actions**: TCP position (3) + orientation (3) + gripper (1)  # TCP 위치(3) + 방향(3) + 그리퍼(1)
- **Coordinate Systems**: Both absolute and relative coordinates  # 절대 및 상대 좌표계 모두 지원
- **Action Normalization**: Scaled to (-1, 1) range  # (-1, 1) 범위로 스케일링

### **7.7.1 CALVIN Action Space Detailed Specifications**
- **Source**: CALVIN Official Documentation  # CALVIN 공식 문서
- **Absolute Actions (`['actions']`)**:  # 절대 액션
  - **TCP Position (3)**: x, y, z in absolute world coordinates  # 절대 월드 좌표계의 x, y, z
  - **TCP Orientation (3)**: Euler angles x, y, z in absolute world coordinates  # 절대 월드 좌표계의 Euler 각 x, y, z
  - **Gripper Action (1)**: Binary (close = -1, open = 1)  # 이진값 (닫기 = -1, 열기 = 1)
- **Relative Actions (`['rel_actions']`)**:  # 상대 액션
  - **TCP Position (3)**: x, y, z in relative world coordinates, normalized and clipped to (-1, 1) with scaling factor 50  # 상대 월드 좌표계의 x, y, z, 정규화 및 (-1, 1) 클리핑, 스케일링 팩터 50
  - **TCP Orientation (3)**: Euler angles x, y, z in relative world coordinates, normalized and clipped to (-1, 1) with scaling factor 20  # 상대 월드 좌표계의 Euler 각 x, y, z, 정규화 및 (-1, 1) 클리핑, 스케일링 팩터 20
  - **Gripper Action (1)**: Binary (close = -1, open = 1)  # 이진값 (닫기 = -1, 열기 = 1)

### **7.8 CALVIN Evaluation Metrics**
- **Success Rate**: Percentage of successful task completions  # 성공적인 태스크 완료 비율
- **Chain Success Rate**: Success rate for consecutive tasks (1-5 tasks)  # 연속 태스크 성공률 (1-5개)
- **Average Sequence Length**: Mean length of successful sequences  # 성공한 시퀀스의 평균 길이
- **Task-specific Success Rate**: Individual task performance metrics  # 개별 태스크 성능 메트릭

### **7.8.1 CALVIN Camera Observations**
- **Source**: CALVIN Official Documentation  # CALVIN 공식 문서
- **RGB Observations**:  # RGB 관찰
  - **`['rgb_static']`**: Static camera RGB (200×200×3, uint8)  # 정적 카메라 RGB (200×200×3, uint8)
  - **`['rgb_gripper']`**: Gripper camera RGB (84×84×3, uint8)  # 그리퍼 카메라 RGB (84×84×3, uint8)
  - **`['rgb_tactile']`**: Tactile camera RGB (160×120×6, uint8)  # 촉각 카메라 RGB (160×120×6, uint8)
- **Depth Observations**:  # 깊이 관찰
  - **`['depth_static']`**: Static camera depth (200×200, float32)  # 정적 카메라 깊이 (200×200, float32)
  - **`['depth_gripper']`**: Gripper camera depth (84×84, float32)  # 그리퍼 카메라 깊이 (84×84, float32)
  - **`['depth_tactile']`**: Tactile camera depth (160×120×2, float32)  # 촉각 카메라 깊이 (160×120×2, float32)

### **7.8.2 CALVIN State Observations**
- **Source**: CALVIN Official Documentation  # CALVIN 공식 문서
- **Scene State (`['scene_obs']`)**:  # 장면 상태
  - **Sliding Door (1)**: Joint state  # 슬라이딩 도어 (1): 관절 상태
  - **Drawer (1)**: Joint state  # 서랍 (1): 관절 상태
  - **Button (1)**: Joint state  # 버튼 (1): 관절 상태
  - **Switch (1)**: Joint state  # 스위치 (1): 관절 상태
  - **Lightbulb (1)**: on=1, off=0  # 전구 (1): 켜짐=1, 꺼짐=0
  - **Green Light (1)**: on=1, off=0  # 녹색 불빛 (1): 켜짐=1, 꺼짐=0
  - **Red Block (6)**: (x, y, z, euler_x, euler_y, euler_z)  # 빨간 블록 (6): (x, y, z, euler_x, euler_y, euler_z)
  - **Blue Block (6)**: (x, y, z, euler_x, euler_y, euler_z)  # 파란 블록 (6): (x, y, z, euler_x, euler_y, euler_z)
  - **Pink Block (6)**: (x, y, z, euler_x, euler_y, euler_z)  # 분홍 블록 (6): (x, y, z, euler_x, euler_y, euler_z)
- **Robot State (`['robot_obs']`)**:  # 로봇 상태
  - **TCP Position (3)**: x, y, z in world coordinates  # 월드 좌표계의 x, y, z
  - **TCP Orientation (3)**: Euler angles x, y, z in world coordinates  # 월드 좌표계의 Euler 각 x, y, z
  - **Gripper Opening Width (1)**: in meters  # 그리퍼 열림 폭 (1): 미터 단위
  - **Arm Joint States (7)**: in radians  # 팔 관절 상태 (7): 라디안 단위
  - **Gripper Action (1)**: Binary (close = -1, open = 1)  # 그리퍼 액션 (1): 이진값 (닫기 = -1, 열기 = 1)

### **7.9 CALVIN Performance Benchmarks**
- **Source**: `RoboVLMs/README.md:125` (CALVIN Benchmark Table)  # CALVIN 벤치마크 테이블에서 확인된 성능 데이터
- **ABCD→D Split**: 96.7% single task success rate, 4.49 average length  # ABCD→D 분할: 96.7% 단일 태스크 성공률, 4.49 평균 길이
- **ABC→D Split**: 98.0% single task success rate, 4.25 average length  # ABC→D 분할: 98.0% 단일 태스크 성공률, 4.25 평균 길이
- **RoboVLMs Performance**: Best performing VLA model on CALVIN  # RoboVLMs 성능: CALVIN에서 최고 성능 VLA 모델

## 🎯 **Key Findings**

### **7.10 Technical Achievements**
1. **Dataset Implementation**: `DiskCalvinDataset` class for efficient data loading  # 효율적인 데이터 로딩을 위한 DiskCalvinDataset 클래스
2. **Language Grounding**: Natural language task descriptions  # 자연어 태스크 설명
3. **Evaluation Framework**: Standardized evaluation pipeline with `CustomModel`  # CustomModel을 사용한 표준화된 평가 파이프라인
4. **Performance Metrics**: Success rate calculation with `count_success()` function  # count_success() 함수를 사용한 성공률 계산

### **7.11 Implementation Details**
- **Dataset Loading**: `DiskCalvinDataset` class for efficient data loading  # 효율적인 데이터 로딩을 위한 DiskCalvinDataset 클래스
- **Evaluation Pipeline**: `CustomModel` wrapper for model evaluation  # 모델 평가를 위한 CustomModel 래퍼
- **Success Counting**: `count_success()` function for performance metrics  # 성능 메트릭을 위한 count_success() 함수
- **Result Storage**: JSON format for evaluation results  # 평가 결과를 위한 JSON 형식

### **7.12 CALVIN Benchmark Results**
- **Source**: `RoboVLMs/README.md:125` (Verified from GitHub)  # GitHub에서 확인된 성능 데이터
- **RoboVLMs (KosMos P.H.)**: 96.7% success rate on ABCD→D  # RoboVLMs (KosMos P.H.): ABCD→D에서 96.7% 성공률
- **Average Length**: 4.49 tasks in successful sequences  # 평균 길이: 성공한 시퀀스에서 4.49개 태스크
- **Chain Performance**: 93.0% for 2 tasks, 89.9% for 3 tasks (from README table)  # 체인 성능: 2개 태스크 93.0%, 3개 태스크 89.9% (README 테이블에서)
- **State-of-the-art**: Best performing VLA model on CALVIN benchmark  # 최신 기술: CALVIN 벤치마크에서 최고 성능 VLA 모델

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/data/calvin_dataset.py` (L521-873)  # CALVIN 데이터셋 클래스 구현
- `RoboVLMs/eval/calvin/eval_utils.py` (L64-120)  # CALVIN 평가 유틸리티 함수
- `RoboVLMs/eval/calvin/model_wrapper.py` (L28-147)  # CALVIN 모델 래퍼 구현

## 📚 **Additional CALVIN Documentation**
- **CALVIN Official Documentation**: Complete dataset specifications and download instructions  # CALVIN 공식 문서: 완전한 데이터셋 사양 및 다운로드 지침
- **Language Embeddings**: 10 different precomputed language embeddings available  # 언어 임베딩: 10가지 사전 계산된 언어 임베딩 사용 가능
- **Visualization Tools**: Scripts for dataset visualization and language annotation visualization  # 시각화 도구: 데이터셋 시각화 및 언어 주석 시각화를 위한 스크립트
- **Data Integrity**: SHA256 checksums for verifying downloaded dataset integrity  # 데이터 무결성: 다운로드된 데이터셋 무결성 검증을 위한 SHA256 체크섬
- `RoboVLMs/README.md` (L113-136)  # CALVIN 벤치마크 결과
- `5.robovlms_github/feedback/calvin_dataset_analysis.md`  # CALVIN 데이터셋 분석 문서
