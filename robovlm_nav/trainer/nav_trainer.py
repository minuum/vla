import os
import torch
from robovlms.train.base_trainer import BaseTrainer

class NavTrainer(BaseTrainer):
    """
    V4 학습을 위한 커스텀 트레이너.
    기본 트레이너의 그래디언트 관리 로직을 보완하고 로깅을 강화함.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 초기화 직후 그래디언트 설정 강제 수행
        self._enable_gradients()

    @classmethod
    def from_checkpoint(cls, checkpoint_path, load_source="nav", variant=None):
        """
        체크포인트로부터 트레이너 인스턴스를 생성하는 팩토리 메서드.
        """
        print(f"🚀 [NavTrainer] Creating NavTrainer from checkpoint: {checkpoint_path}", flush=True)
        instance = super().from_checkpoint(checkpoint_path, load_source, variant)
        # 로드 후 다시 한 번 파라미터 체크 (로드 과정에서 바뀔 수 있으므로)
        if hasattr(instance, "_enable_gradients"):
            instance._enable_gradients()
        return instance

    def _enable_gradients(self):
        """학습이 필요한 파라미터들의 requires_grad를 True로 설정"""
        if self.model is None:
             print("⚠️ [NavTrainer] Model is None, skipping gradient enablement.", flush=True)
             return
             
        print(f"📊 [NavTrainer] === Parameter Gradient Check (Model ID: {id(self.model)}) ===", flush=True)
        
        # 1. BaseRoboVLM 수준에서 unfreeze 시도
        if hasattr(self.model, 'unfreeze_params'):
            print("  [Model] Calling unfreeze_params()...", flush=True)
            self.model.unfreeze_params()

        # 2. 명시적으로 필요한 컴포넌트들 활성화 (문자열 매칭)
        # 'lora' (PEFT), 'act_head' (Policy), 'projector' (Vision-to-Lang), 'resampler'
        target_keywords = ['lora', 'act_head', 'projector', 'resampler', 'action_token', 'mm_projector']
        matched_count = 0
        
        for name, param in self.model.named_parameters():
            if any(key in name.lower() for key in target_keywords):
                param.requires_grad = True
                matched_count += 1
                if matched_count <= 5: # 처음 5개만 예시로 로깅
                    print(f"  [Enabled] {name}", flush=True)

        print(f"✅ [NavTrainer] Total keywords-matched parameters enabled: {matched_count}", flush=True)
        
        # 3. 최종 요약
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        grad_true_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"🔍 [NavTrainer] Total trainable parameters: {trainable_params_count:,}", flush=True)
        print(f"🔍 [NavTrainer] Count of params with requires_grad=True: {grad_true_count}", flush=True)

    def _process_batch(self, batch):
        """
        BaseTrainer의 _process_batch에서 발생하는 Action slicing 에러 방지 및 리턴 규격(19개) 준수
        """
        if self.configs.get("discrete_action", False):
            # batch에서 데이터 추출
            rgb = batch["rgb"].to(self.device).to(self.dtype)
            language = batch["text"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            arm_action = batch["action"].to(self.device) 
            raw_text = batch.get("raw_text", None)
            data_source = "mobile_vla_action"

            # 19개 리턴 규격 (BaseTrainer.py:468-488 참고)
            # BaseTrainer.training_step은 arm_action_chunck(10번째)를 action_labels로 사용함
            return (
                rgb,                    # 01: rgb
                None,                   # 02: hand_rgb
                None,                   # 03: attention_mask
                language,               # 04: language
                text_mask,              # 05: text_mask
                None,                   # 06: fwd_rgb_chunck
                None,                   # 07: fwd_hand_rgb_chunck
                None,                   # 08: arm_action
                None,                   # 09: gripper_action
                arm_action,             # 10: arm_action_chunck (여기 전달해야 training_step에서 사용됨)
                None,                   # 11: gripper_action_chunck
                None,                   # 12: chunck_mask
                None,                   # 13: fwd_mask
                None,                   # 14: instr_and_action_ids
                None,                   # 15: instr_and_action_labels
                None,                   # 16: instr_and_action_mask
                raw_text,               # 17: raw_text
                None,                   # 18: rel_state
                data_source             # 19: data_source
            )
        
        return super()._process_batch(batch)

    def training_step(self, batch, batch_idx):
        """
        BaseTrainer.training_step을 오버라이딩하여 Loss 전달 과정을 투명하게 관리
        """
        # 데이터 전처리
        processed_batch = self._process_batch(batch)
        
        # 모델 Forward (BaseTrainer.training_step과 동일한 인자 구성)
        prediction = self.model.forward(
            processed_batch[0],      # rgb
            processed_batch[3],      # language
            attention_mask=processed_batch[4], # text_mask
            action_labels=(processed_batch[9], processed_batch[10]), # arm_action_chunck, gripper_action_chunck
            action_mask=processed_batch[11], # chunck_mask
            raw_text=processed_batch[16],
            data_source=processed_batch[18]
        )

        # Loss 추출
        loss_dict = self._get_loss(prediction)
        train_loss = loss_dict["loss"]

        # 로그 기록
        for k, v in loss_dict.items():
            if v is not None and isinstance(v, torch.Tensor):
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k=="loss"))
        
        # Gradient 체크 (디버깅)
        if hasattr(train_loss, "requires_grad") and not train_loss.requires_grad:
            print(f"❌ [NavTrainer] CRITICAL: Training loss has NO gradients! prediction keys={list(prediction.keys())}", flush=True)
            
        return train_loss

    def on_validation_epoch_start(self):
        """검증 시작 전 메모리 정리"""
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation 시에도 동일한 로직 적용 + 메모리 절약
        """
        # 확실하게 grad 계산 차단
        with torch.no_grad():
            processed_batch = self._process_batch(batch)
            prediction = self.model.forward(
                processed_batch[0],      # rgb
                processed_batch[3],      # language
                attention_mask=processed_batch[4],
                action_labels=(processed_batch[9], processed_batch[10]),
                action_mask=processed_batch[11],
                raw_text=processed_batch[16],
                data_source=processed_batch[18]
            )
            loss_dict = self._get_loss(prediction)
            
            for k, v in loss_dict.items():
                if v is not None and isinstance(v, torch.Tensor):
                    self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """검증 종료 후 메모리 정리"""
        torch.cuda.empty_cache()

    def _get_loss(self, prediction):
        """
        NavPolicy가 리턴한 Loss를 안전하게 추출.

        배경: base_backbone._update_loss(loss, action_loss, "act") 호출 시
        NavPolicy.loss()의 키들에 suffix '_act'가 붙는다.
          예) "loss_arm_act" -> "loss_arm_act_act"
              "loss_velocity" -> "loss_velocity_act"
              "acc_arm_act"   -> "acc_arm_act_act"

        또한 _format_loss()가 모든 "loss_*" 값들을 합산해 "loss" 키로 저장한다.
        따라서 "loss" 키를 우선 사용하고, 없으면 여러 후보 키를 순서대로 탐색한다.
        """
        # 1) _format_loss()가 만들어주는 통합 "loss" 키를 우선 사용
        loss = prediction.get("loss", None)

        # 2) loss가 없거나 requires_grad=False이면 후보 키에서 재탐색
        if loss is None or (isinstance(loss, torch.Tensor) and not loss.requires_grad):
            candidates = [
                "loss_arm_act_act",   # _update_loss(..., "act") suffix 버전
                "loss_velocity_act",
                "loss_arm_act",
                "loss_velocity",
            ]
            for key in candidates:
                v = prediction.get(key, None)
                if v is not None and isinstance(v, torch.Tensor) and v.requires_grad:
                    loss = v
                    print(f"[NavTrainer] Using '{key}' as loss (requires_grad=True)", flush=True)
                    break

        # 3) 여전히 None이면 경고
        if loss is None:
            print(f"❌ [NavTrainer._get_loss] No valid loss! prediction keys={list(prediction.keys())}", flush=True)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # accuracy: suffix 버전 우선, 없으면 원본 키
        acc = prediction.get("acc_arm_act_act",
              prediction.get("acc_arm_act",
              prediction.get("acc_velocity_act",
              prediction.get("acc_velocity", 0.0))))

        return {
            "loss":         loss,
            "loss_arm_act": loss,
            "acc_arm_act":  acc,
        }
