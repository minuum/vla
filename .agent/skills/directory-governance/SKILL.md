---
name: directory-governance
description: Enforce the canonical directory structure of RoboVLM-Nav. Use this skill before creating any new files, directories, or moving code. Prevents structural drift and ensures compatibility across branches.
---

# Directory Governance Skill

## Value Proposition
Prevents structural drift that breaks training scripts and causes import errors.
Every file creation/move must be validated against the canonical structure in
`docs/directory_governance.md`.

## When to Use
- Before creating ANY new .py, .sh, .json, .md file in the project
- Before moving or renaming directories
- When merging code from legacy directories (Robo+, RoboVLMs, src, core, etc.)
- When a user asks to "clean up" or "reorganize" the project

## Instructions

### Step 1: Read governance doc first
```
view_file /home/billy/25-1kp/vla/docs/directory_governance.md
```

### Step 2: Validate placement (use this decision tree)

```
Is it a core training/inference Python module?
  → robovlm_nav/  (datasets/, models/, serve/)

Is it a JSON config file?
  → configs/

Is it a shell training script?
  → scripts/

Is it a one-off analysis or quantization tool?
  → tools/

Is it a Markdown document?
  → docs/  (never root /)

Is it a Docker file?
  → docker/

Is it a test?
  → tests/

Is it a temporary scratch file?
  → /tmp/  (never commit)
```

### Step 3: Verify naming convention before creating

| Type         | Pattern                       | Example                         |
| ------------ | ----------------------------- | ------------------------------- |
| Config       | `mobile_vla_[ver]_[exp].json` | `mobile_vla_v3_exp08_lora.json` |
| Train script | `train_[ver]_[exp].sh`        | `train_v3_exp08_lora.sh`        |
| Python class | `NavXxx` (no Omni prefix)     | `NavDataset`, `NavPolicy`       |

### Step 4: Verify PYTHONPATH in any new .sh script

Must include:
```bash
export PYTHONPATH="/home/billy/25-1kp/vla:/home/billy/25-1kp/vla/third_party/RoboVLMs"
```

### Step 5: Never modify third_party/RoboVLMs directly

- `third_party/RoboVLMs` = **원본 upstream** (https://github.com/Robot-VLAs/RoboVLMs)
- 우리 커스텀 코드는 **모두 `robovlm_nav/`에만** 존재해야 함
- `robovlm_nav/train.py`의 setattr 주입 패턴으로 third_party main.py와 연결:
```python
setattr(robovlms.data, "NavDataset", NavDataset)
setattr(robovlms.model.policy_head, "NavPolicy", NavPolicy)
setattr(robovlms.train, "MobileVLATrainer", NavTrainer)
```

### robovlm_nav/ 내부 구조

```
robovlm_nav/
├── datasets/
│   ├── nav_dataset.py          ← 진입점 (NavDataset 클래스)
│   └── nav_h5_dataset_impl.py  ← 실제 구현 (MobileVLAH5Dataset)
├── models/
│   ├── nav_policy.py           ← 진입점 (NavPolicy, NavPolicyRegression)
│   └── policy_head/
│       ├── nav_policy_impl.py  ← 실제 구현 (LSTM+Classification Decoder)
│       └── hybrid_action_head.py ← 하이브리드 액션 헤드
├── trainer/
│   ├── nav_trainer.py          ← MobileVLATrainer (2D velocity 특화)
│   └── nav_qat_trainer.py      ← QAT Trainer
├── serve/
│   └── inference_server*.py   ← 추론 서버
└── train.py                    ← 모든 컴포넌트를 robovlms에 inject 후 실행
```

## Red Flags (Block immediately if seen)

- ❌ Creating .py files directly in project root `/`
- ❌ Using `OmniDataset`, `OmniPolicy` in new configs
- ❌ Referencing `Mobile_VLA/configs/` (old path)
- ❌ Referencing `omni_robovlm` (renamed to `robovlm_nav`)
- ❌ Creating `RoboVLMs*/` top-level directories (모두 삭제됨)
- ❌ **직접 수정 `third_party/RoboVLMs/robovlms/`** → robovlm_nav/에 구현 후 setattr 주입

## Cleanup Status (2026-03-03 업데이트)

| Directory                   | Action                                 | Status |
| --------------------------- | -------------------------------------- | ------ |
| `RoboVLMs/`                 | 삭제                                   | ✅ 완료 |
| `RoboVLMs_upstream_backup/` | 삭제                                   | ✅ 완료 |
| `v3-exp04-lora/`            | 삭제 (심볼릭 링크만 있었음)            | ✅ 완료 |
| `config/`                   | README → docs/archives, 폴더 삭제      | ✅ 완료 |
| `src/`                      | → `tools/legacy_inference/`            | ✅ 완료 |
| `core/`                     | → `tools/mcp_integration/`             | ✅ 완료 |
| `models/`                   | → `tools/model_analysis/`              | ✅ 완료 |
| `whisper2/`                 | → `ros2_client/whisper/`               | ✅ 완료 |
| `simpler_env_repo/`         | → `third_party/simpler_env/`           | ✅ 완료 |
| `debug_*/`                  | 삭제 (이미지 파일)                     | ✅ 완료 |
| `memora/`                   | **삭제** (menemory로 통합)             | ✅ 완료 |
| `menemory/`                 | 독립 git 저장소 유지 (minuum/menemory) | ✅ 확정 |
| `test_images/`              | → `assets/images/`                     | ✅ 완료 |
| `__pycache__/` (루트)       | 삭제                                   | ✅ 완료 |
| `Robo+/`                    | `.md` 보존, 나머지 → 별도 논의         | 🟡 보류 |
| `git_recovery_backup/`      | 내용 확인 후 결정                      | 🟡 보류 |
| `result/`                   | 내용 확인 후 결정                      | 🟡 보류 |
| `Model_ws/`                 | 사용 여부 확인 필요                    | 🟡 보류 |
| `ROS_action/`               | **데이터셋 원본 - 절대 이동/삭제 금지** | 🔒 보존 |

