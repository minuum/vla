# 22. LSTM Layer 학습 분석

## 1. LSTM Layer 구조 분석

### **1.1 LSTMDecoder 클래스 구조**

**핵심 LSTM 구현**:
```python
class LSTMDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,                    # 기본 4층 LSTM
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        super(LSTMDecoder, self).__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.history_memory = []
        self.hidden_size = hidden_size
        
        # LSTM 네트워크 정의
        self.rnn = lstm_decoder(
            in_features * latent, 
            hidden_size * latent, 
            num_layers, 
            policy_rnn_dropout_p
        )
        
        # Action Head 정의
        self.actions = MLPTanhHead(
            self.hidden_size * latent, 
            fwd_pred_next_n * (self.action_dim - 1)  # 6-DOF arm
        )
        self.gripper = MLPSigmoidHead(
            self.hidden_size * latent, 
            fwd_pred_next_n  # 1-DOF gripper
        )
        self.hidden_state = None
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:387-426`

### **1.2 LSTM Forward Pass**

**LSTM Forward Pass 구현**:
```python
def forward(self, tok_seq, h_0=None, **kwargs):
    # Down sampling 처리
    if self.down_sample == "pooling":
        bs, seq_len = tok_seq.shape[:2]
        tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
        tok_seq = self.global_1d_pool(
            tok_seq.permute(0, 2, 1)
        )
        tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
    elif self.down_sample == "none":
        tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
    
    # 히스토리 메모리 관리
    if tok_seq.shape[1] == 1:
        self.history_memory.append(tok_seq)
        if len(self.history_memory) <= self.history_len:
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n
            x = x[:, -1].unsqueeze(1)
            self.rnn_out = x.squeeze(1)
        else:
            # 히스토리 윈도우 초과 시 새로고침
            cur_len = len(self.history_memory)
            for _ in range(cur_len - self.history_len):
                self.history_memory.pop(0)
            assert len(self.history_memory) == self.history_len
            hist_feature = torch.cat(self.history_memory, dim=1)
            self.hidden_state = None
            x, h_n = self.rnn(hist_feature, self.hidden_state)
            x = x[:, -1].unsqueeze(1)
    else:
        # 배치 학습 시
        self.hidden_state = h_0
        x, h_n = self.rnn(tok_seq, self.hidden_state)
        self.hidden_state = h_n
    
    # Action 예측
    actions = self.actions(x)
    gripper = self.gripper(x)
    
    return actions, gripper
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:432-485`

---

## 2. LSTM 학습 과정

### **2.1 Loss 계산**

**LSTM Loss 함수**:
```python
def loss(self, pred_action, labels, attention_mask=None):
    """
    pred_action_logits: [bs, seq_len, chunck_size, 7]
    labels: (pose gt [bs, seq_len, chunck_size, 6], gripper gt [bs, seq_len, chunck_size])
    attention_mask: [bs, seq_len, chunck_size]
    """
    if labels is None or labels[0] is None:
        return {"loss": None}
    
    # 예측값과 라벨 결합
    if isinstance(pred_action, tuple) or isinstance(pred_action, list):
        if pred_action[0].ndim == pred_action[1].ndim:
            pred_action = torch.cat(pred_action, dim=-1)
        elif pred_action[0].ndim == pred_action[1].ndim + 1:
            pred_action = torch.cat(
                [pred_action[0], pred_action[1].unsqueeze(-1)], dim=-1
            )
    
    # Loss 계산
    if attention_mask is None:
        # Pose Loss (6-DOF arm)
        pose_loss = torch.nn.functional.huber_loss(pred_action[..., :6], labels[0])
        # Gripper Loss (1-DOF)
        gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_action[..., -1], labels[1]
        )
    else:
        # Attention mask 적용
        pose_loss = torch.nn.functional.huber_loss(
            pred_action[..., :6], labels[0], reduction="none"
        )
        attention_mask = attention_mask.bool()
        pose_loss = pose_loss[attention_mask].mean()
        
        gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_action[..., -1], labels[1], reduction="none"
        )
        gripper_loss = gripper_loss[attention_mask].mean()
    
    # Gripper 정확도 계산
    gripper_action_preds = (F.sigmoid(pred_action[..., -1]) > 0.5).float()
    acc_gripper = (gripper_action_preds == labels[1]).float().mean()
    
    return {
        "loss_arm": pose_loss,
        "loss_gripper": gripper_loss,
        "acc_gripper": acc_gripper
    }
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:118-160`

### **2.2 학습 루프 예시**

**LSTM 학습 루프**:
```python
# LSTM 학습 예시 (10,000회 반복)
net = LSTMDecoder(
    in_features=1024,
    action_dim=7,
    down_sample="pooling",
    latent=1,
    fwd_pred_next_n=2,
    window_size=12,
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
bs = 5
window_size = 12
text_len = 8
tokens = torch.randn(bs, window_size, text_len, 1024)
labels = (torch.randn(bs, window_size, 2, 6), torch.ones(bs, window_size, 2))
att_mask = torch.ones(bs, window_size, 2)

for i in range(10000):
    actions, gripper = net(tokens)
    pred_action_logitss = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
    optimizer.zero_grad()
    loss = net.loss(pred_action_logitss, labels, att_mask)
    
    loss_arm = loss["loss_arm"]
    loss_gripper = loss["loss_gripper"]
    acc_gripper = loss["acc_gripper"]
    loss_act = loss_arm + 0.01 * loss_gripper  # 가중치 조합
    
    loss_act.backward()
    optimizer.step()
    print(
        "iter: {}, loss: {} gripper: {} acc: {}".format(
            i, loss_act.item(), loss_gripper.item(), acc_gripper
        )
    )
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:625-642`

---

## 3. 전체 학습 파이프라인에서의 LSTM

### **3.1 BaseTrainer에서의 LSTM 학습**

**Training Step**:
```python
def training_step(self, batch, batch_idx):
    # 배치 데이터 처리
    (
        rgb, hand_rgb, attention_mask, language, text_mask,
        fwd_rgb_chunck, fwd_hand_rgb_chunck,
        arm_action, gripper_action,
        arm_action_chunck, gripper_action_chunck,
        chunck_mask, fwd_mask,
        instr_and_action_ids, instr_and_action_labels,
        instr_and_action_mask, raw_text, rel_state, data_source,
    ) = self._process_batch(batch)
    
    # 모델 Forward Pass
    prediction = self.model.forward(
        rgb, language,
        attention_mask=text_mask,
        action_labels=(arm_action_chunck, gripper_action_chunck),
        action_mask=chunck_mask,
        vision_gripper=hand_rgb,
        fwd_rgb_labels=fwd_rgb_chunck,
        fwd_hand_rgb_labels=fwd_hand_rgb_chunck,
        fwd_mask=fwd_mask,
        instr_and_action_ids=instr_and_action_ids,
        instr_and_action_labels=instr_and_action_labels,
        instr_and_action_mask=instr_and_action_mask,
        raw_text=raw_text,
        data_source=data_source,
        rel_state=rel_state,
    )
    
    # Loss 계산
    output = self._get_loss(prediction)
    return output
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:565-621`

### **3.2 Loss 조합**

**전체 Loss 계산**:
```python
def _get_loss(self, prediction):
    loss_arm_act = prediction.get("loss_arm_act", None)
    loss_gripper_act = prediction.get("loss_gripper_act", None)
    loss_obs = prediction.get("loss_obs_fwd", None)
    loss_hand_obs = prediction.get("loss_hand_obs_fwd", None)
    acc_arm_act = prediction.get("acc_arm_act", None)
    acc_gripper_act = prediction.get("acc_gripper_act", None)
    loss_cap = prediction.get("loss_cap", None)
    loss_kl = prediction.get("loss_kl", None)
    loss_vl_cotrain = prediction.get("loss_vl_cotrain", None)
    
    loss = torch.tensor(0.0).to(self.device)
    
    if self.act_pred:
        # LSTM Action Loss
        loss_act = (loss_arm_act if loss_arm_act is not None else 0) + (
            loss_gripper_act * self.arm_gripper_loss_ratio
            if loss_gripper_act is not None
            else 0
        )
        loss += loss_act
        
        if loss_kl is not None:
            loss += self.kl_div_ratio * loss_kl
    
    if self.fwd_pred:
        # Forward Prediction Loss
        loss += self.fwd_loss_ratio * (loss_obs if loss_obs is not None else 0)
        if self.fwd_pred_hand:
            loss += self.fwd_loss_ratio * (
                loss_hand_obs if loss_hand_obs is not None else 0
            )
    
    if loss_cap is not None:
        # Caption Loss
        loss += self.cap_loss_ratio * loss_cap
    
    if loss_vl_cotrain is not None:
        # Vision-Language Co-training Loss
        loss += self.vl_cotrain_ratio * loss_vl_cotrain
    
    return {
        "loss": loss,
        "loss_act": loss_act,
        "loss_arm_act": loss_arm_act,
        "loss_gripper_act": loss_gripper_act,
        "acc_arm_act": acc_arm_act,
        "acc_gripper_act": acc_gripper_act
    }
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:269-315`

---

## 4. LSTM 학습 설정

### **4.1 LSTM 하이퍼파라미터**

**기본 설정**:
```python
LSTM_CONFIG = {
    "hidden_size": 1024,           # LSTM hidden size
    "num_layers": 4,                # LSTM layer 수
    "policy_rnn_dropout_p": 0.0,    # LSTM dropout
    "window_size": 8,               # 히스토리 윈도우 크기
    "fwd_pred_next_n": 10,          # 예측할 액션 수
    "action_dim": 7,                # 액션 차원 (6-DOF + gripper)
    "down_sample": "none"           # 다운샘플링 방식
}
```

### **4.2 Loss 가중치**

**Loss 가중치 설정**:
```python
LOSS_WEIGHTS = {
    "arm_gripper_loss_ratio": 0.01,  # Gripper loss 가중치
    "fwd_loss_ratio": 0,             # Forward prediction loss
    "cap_loss_ratio": 0.05,           # Caption loss 가중치
    "vl_cotrain_ratio": 0.1          # Vision-Language co-training
}
```

### **4.3 Optimizer 설정**

**Optimizer 설정**:
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.max_epochs,
        eta_min=self.learning_rate * self.min_lr_scale
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch"
        }
    }
```

---

## 5. LSTM 학습 최적화

### **5.1 메모리 효율성**

**Gradient Checkpointing**:
```python
# 메모리 절약을 위한 설정
training_config = {
    "gradient_checkpointing": False,  # LSTM에서는 비활성화
    "precision": "bf16",              # Mixed precision
    "batch_size": 2,                  # 작은 배치 크기
    "accumulate_grad_batches": 4     # 그래디언트 누적
}
```

### **5.2 학습 안정성**

**Gradient Clipping**:
```python
# 그래디언트 클리핑
trainer_config = {
    "gradient_clip_val": 1.0,        # 그래디언트 클리핑
    "gradient_clip_algorithm": "norm"  # L2 norm 클리핑
}
```

### **5.3 LSTM 특화 설정**

**LSTM 최적화**:
```python
# LSTM 특화 설정
lstm_config = {
    "num_layers": 4,                 # 4층 LSTM (기본값)
    "hidden_size": 1024,             # Hidden size
    "dropout": 0.0,                  # LSTM dropout
    "bidirectional": False,          # 단방향 LSTM
    "batch_first": True              # Batch first
}
```

---

## 6. 핵심 결론

### **6.1 LSTM Layer 구조**

- **4층 LSTM**: 기본적으로 4개 레이어 사용
- **Hidden Size**: 1024 (고정)
- **Action Head**: MLPTanhHead (arm) + MLPSigmoidHead (gripper)
- **히스토리 관리**: window_size 기반 메모리 관리

### **6.2 학습 과정**

- **Loss 함수**: Huber Loss (arm) + BCE Loss (gripper)
- **가중치**: arm_gripper_loss_ratio = 0.01
- **Optimizer**: AdamW + CosineAnnealingLR
- **메모리 최적화**: Mixed precision, 작은 배치 크기

### **6.3 전체 파이프라인 통합**

- **VLM + LSTM**: End-to-End 학습
- **Loss 조합**: Action Loss + Forward Loss + Caption Loss
- **실시간 학습**: 히스토리 메모리 관리

**출처 요약**:
- `RoboVLMs/robovlms/model/policy_head/base_policy.py:387-485`: LSTMDecoder 구현
- `RoboVLMs/robovlms/train/base_trainer.py:565-621`: 학습 스텝
- `RoboVLMs/robovlms/train/base_trainer.py:269-315`: Loss 계산
- `RoboVLMs/robovlms/model/policy_head/base_policy.py:625-642`: 학습 루프 예시

**핵심**: LSTM Layer는 **4층 구조**로 **히스토리 기반 액션 예측**을 수행하며, **Huber Loss + BCE Loss**로 학습됩니다.
