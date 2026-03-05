import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset


class MobileVLAActionDataset(ActionPredictionDataset):
    """
    Mobile VLA HDF5 → RoboVLMs ActionPredictionDataset 어댑터

    - images: [T, H, W, 3] (uint8)
    - actions: [T, 3] (float32) -> [T, 7]로 패딩하여 (ax, ay, az, 0, 0, 0, gripper) 형태로 매핑
    - action_event_types: ['episode_start'|'start_action'|'stop_action']
      → gripper(open/close 유사 이진): start_action=1, 그 외=0
    - 길이 설정: window_size=16, fwd_pred_next_n=2 → images 길이=18, actions 길이=17을 만족
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "kosmos",
        mode: str = "train",
        organize_type: str = "interleave",
        window_size: int = 16,
        fwd_pred_next_n: int = 2,
        discrete: bool = False,
        norm_action: bool = True,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        use_mu_law: bool = False,
        regular_action: bool = False,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        image_history: bool = True,
        action_history: bool = True,
        predict_stop_token: bool = False,
        special_history_id: int = -100,
        tokenizer: Optional[Dict] = None,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        **kwargs,
    ):
        # 시나리오 명령어는 Line 151-160에서 정의됨 (중복 제거)
        
        # ActionPredictionDataset 초기화 (토크나이저/이미지 전처리/변환기 구성)
        # GRDataModule가 is_training을 kwargs로 전달하므로 이를 반영해 mode 자동 결정
        resolved_mode = "train" if kwargs.get("is_training", mode == "train") else "inference"

        super().__init__(
            model_name=model_name,
            mode=resolved_mode,
            organize_type=organize_type,
            discrete=discrete,
            action_history=action_history,
            image_history=image_history,
            predict_stop_token=predict_stop_token,
            special_history_id=special_history_id,
            window_size=window_size,
            fwd_pred_next_n=fwd_pred_next_n,
            tokenizer=tokenizer or {
                "type": "AutoProcessor",
                "pretrained_model_name_or_path": "microsoft/kosmos-2-patch14-224",
                "tokenizer_type": "kosmos",
                "max_text_len": 256,
                "use_local_files": False,
            },
            rgb_pad=rgb_pad,
            gripper_pad=gripper_pad,
            **kwargs,
        )

        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"data_dir not found: {data_dir}"
        # HDF5 유효 파일만 채택 (LFS 포인터/손상 파일 제외)
        all_h5 = sorted([p for p in self.data_dir.glob("*.h5") if p.is_file()])
        valid_files: List[Path] = []

        def _is_valid_h5(path: Path) -> bool:
            """mobile_vla_data_collector.py 형태인지 검증"""
            try:
                if not h5py.is_hdf5(path.as_posix()):
                    return False
                with h5py.File(path, "r") as f:
                    # 필수 키 검사
                    required_keys = ["images", "actions", "action_event_types"]
                    if not all(k in f for k in required_keys):
                        return False
                    
                    # 필수 속성 검사
                    required_attrs = ["episode_name", "num_frames"]
                    if not all(attr in f.attrs for attr in required_attrs):
                        return False
                    
                    # 데이터 형태 검사
                    images = f["images"]
                    actions = f["actions"]
                    events = f["action_event_types"]
                    
                    # 차원 검사: images [T, H, W, 3], actions [T, 3], events [T]
                    if len(images.shape) != 4 or images.shape[3] != 3:
                        return False
                    if len(actions.shape) != 2 or actions.shape[1] != 3:
                        return False
                    if len(events.shape) != 1:
                        return False
                    
                    # 길이 일치 검사
                    T = images.shape[0]
                    if actions.shape[0] != T or events.shape[0] != T:
                        return False
                    
                    # 이벤트 타입이 올바른 문자열인지 검사
                    sample_event = events[0]
                    if isinstance(sample_event, bytes):
                        sample_event = sample_event.decode('utf-8')
                    valid_events = ['episode_start', 'start_action', 'stop_action']
                    if sample_event not in valid_events:
                        return False
                    
                    return True
            except Exception:
                return False

        for p in all_h5:
            if _is_valid_h5(p):
                # 시나리오 태그 확인 (unknown 제외)
                scenario = self._extract_scenario_from_filename(p.name)
                if scenario == "unknown":
                    print(f"파일 제외 (시나리오 태그 없음): {p.name}")
                    continue
                valid_files.append(p)

        self.h5_files = valid_files
        assert len(self.h5_files) > 0, f"No valid .h5 files in {data_dir}"

        # Scenario → English instruction
        # 위치 명시 표현("visible on the left/right side of the frame")을 사용하여
        # VLM의 시각 위치 인식과 Action Head의 instruction grounding을 동일한 텍스트로 통합.
        # base_action_prediction_dataset.py의 prompt builder가 이를
        # "What action should the robot take to {task_description}?" 형식으로 래핑.
        self.scenario_instructions = {
            "1box_vert_left":  "Navigate to the cup visible on the left side of the frame",
            "1box_vert_right": "Navigate to the cup visible on the right side of the frame",
            "1box_hori_left":  "Navigate to the cup visible on the left side of the frame",
            "1box_hori_right": "Navigate to the cup visible on the right side of the frame",
            "2box_vert_left":  "Navigate to the cup visible on the left side of the frame",
            "2box_vert_right": "Navigate to the cup visible on the right side of the frame",
            "2box_hori_left":  "Navigate to the cup visible on the left side of the frame",
            "2box_hori_right": "Navigate to the cup visible on the right side of the frame",
        }

    def _extract_scenario_from_filename(self, filename: str) -> str:
        """파일명에서 시나리오 추출 (mobile_vla_data_collector.py 방식)"""
        for scenario in self.scenario_instructions.keys():
            if scenario in filename:
                return scenario
        return "unknown"

    def __len__(self) -> int:
        return len(self.h5_files)

    @staticmethod
    def _to_text(e: Any) -> str:
        if isinstance(e, bytes):
            return e.decode("utf-8", errors="ignore")
        try:
            import numpy as _np

            if isinstance(e, _np.bytes_):
                return e.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return str(e)

    def _extract_scenario(self, episode_name: str) -> str:
        for k in self.scenario_instructions.keys():
            if k in episode_name:
                return k
        return "unknown"

    def _convert_to_pil_images(self, images_array):
        """numpy 배열을 PIL Image 리스트로 변환 (Kosmos processor용)"""
        from PIL import Image
        pil_images = []
        for img in images_array:
            # numpy uint8 [H, W, 3] → PIL Image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        return pil_images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 안전하게 유효 파일로 접근 (드물게 런타임 손상 발생 대비)
        start_idx = idx % len(self.h5_files)
        attempts = 0
        while attempts < len(self.h5_files):
            h5_path = self.h5_files[(start_idx + attempts) % len(self.h5_files)]
            try:
                with h5py.File(h5_path, "r") as f:
                    images = f["images"][:]  # [T, H, W, 3]
                    actions = f["actions"][:]  # [18, 3]
                    events = f["action_event_types"][:]
                    episode_name = f.attrs.get("episode_name", os.path.basename(h5_path))
                break
            except Exception:
                attempts += 1
                continue
        else:
            raise OSError("No readable HDF5 files available for this batch.")

        # 길이 정합성 보장: images=18(window+fwd), actions=17(window+fwd-1) 필요
        target_img_len = self.window_size + self.fwd_pred_next_n
        if images.shape[0] > target_img_len:
            images = images[:target_img_len]
        elif images.shape[0] < target_img_len:
            pad = target_img_len - images.shape[0]
            last = images[-1:]
            images = np.concatenate([images, np.repeat(last, pad, axis=0)], axis=0)

        # actions가 18이면 마지막 1개를 제거해 윈도우 규칙(window=16, fwd=2)에 맞춤
        if actions.shape[0] > 17:
            actions = actions[:17]
        elif actions.shape[0] < 17 and images.shape[0] == 18:
            # 부족 시 마지막 액션 반복으로 채움
            pad = 17 - actions.shape[0]
            last = actions[-1:]
            actions = np.concatenate([actions, np.repeat(last, pad, axis=0)], axis=0)

        # Mobile 태스크는 gripper 사용 안함 → action_event_types를 action_mask로 변환
        event_text = [self._to_text(e) for e in events]
        # episode_start=0, start_action=1, stop_action=0 (실제 동작 구간만 1)
        action_validity = np.array([1 if e == "start_action" else 0 for e in event_text], dtype=np.float32)
        action_validity = action_validity[: actions.shape[0]]  # actions 길이에 맞춤(17)

        # 3D → 7D 패딩: [linear_x, linear_y, angular_z, 0, 0, 0, 0] (gripper=0 고정)
        t = actions.shape[0]
        padded_actions = np.zeros((t, 7), dtype=np.float32)
        # Mobile VLA 액션 매핑: [linear_x, linear_y, angular_z] → [x, y, 0, 0, 0, rz, 0]
        padded_actions[:, 0] = actions[:, 0]  # linear_x → x
        padded_actions[:, 1] = actions[:, 1]  # linear_y → y
        padded_actions[:, 5] = actions[:, 2]  # angular_z → rz
        # 나머지는 0으로 유지 (z, rx, ry, gripper)

        # 에피소드 마스크: images(18) 기준으로 모두 유효
        episode_mask = np.ones((images.shape[0],), dtype=bool)

        # 태스크 설명(한국어 지시문)
        scenario = self._extract_scenario(str(episode_name))
        task_description = self.scenario_instructions.get(scenario, "컵까지 가세요")

        # base_action_prediction_dataset.py의 convert_image가 numpy 배열을 기대하므로
        # numpy 배열 그대로 전달 (convert_image에서 PIL로 변환됨)
        
        # 변환 함수 호출 → 모델/콜레이터가 요구하는 키들 생성
        return self.batch_transform(
            task_description=task_description,
            action=padded_actions,  # [17, 7] - Mobile VLA 액션 매핑 적용
            episode_mask=episode_mask,  # [18]
            images=images,  # [18, H, W, 3] numpy 배열 (convert_image에서 PIL로 변환됨)
            gripper_images=None,
        )


