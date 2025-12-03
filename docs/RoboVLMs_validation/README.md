# RoboVLMs Validation

## Context Vector Analysis
- **Goal**: Analyze the context vector output of the pretrained RoboVLMs model when fed with collected data.
- **Status**: Planning
- **Key Questions**:
    - What does the context vector look like? (Shape, values)
    - How to hook into the model to extract this vector?
    - Is 500 samples enough? Need sampling strategy.

### 2. Technical Analysis (Context Vector)
- **Location**: `RoboVLMs/robovlms/model/backbone/base_backbone.py` -> `forward_continuous` -> `action_hs`
- **Extraction Method**: PyTorch Forward Hook on `model.act_head`
- **Findings (2025-12-03)**:
    - **Shape**: `(1, 1, 2048)` (Batch, Sequence, Feature Dim)
    - **Feature Dimension**: 2048 (Likely 1024 Vision + 1024 Text concatenation)
    - **Statistics (Single Sample)**:
        - Mean: ~ -0.02
        - Std: ~ 1.01
        - Range: [-6.40, 32.05]
    - **Note**: The input feature dimension to the Action Head (`MobileVLALSTMDecoder`) is 2048, not 1024 as initially assumed. This suggests a concatenation of features (e.g., Kosmos-2 Vision + Text) before the action head.

### 3. Sampling Strategy
- **Goal**: Verify if the context vector distribution is consistent across a larger dataset.
- **Results (Local Model, 2025-12-03)**:
    - **Samples**: 499 frames from 100 episodes
    - **Shape**: `(499, 2048)`
    - **Global Statistics**:
        - Mean: -0.0196
        - Std: 1.0056
        - Min: -7.43
        - Max: 34.31
    - **Stability**: No dead or constant neurons found. The distribution is well-behaved (normalized).

### 4. Original RoboVLMs Model Analysis
- **Goal**: Analyze the context vector of the original `robovlms/RoboVLMs` model from Hugging Face to compare with the local finetuned model.
- **Results (2025-12-03)**:
    - **Model**: `robovlms/RoboVLMs` (Kosmos-2 backbone)
    - **Shape**: `(1, 1, 1, 2048)`
    - **Statistics**:
        - Mean: -0.0196 (Identical to local model)
        - Std: 1.0032 (Very similar to local model's 1.0056)
        - Min: -7.95
        - Max: 9.27 (Significantly lower than local model's 34.31)
    - **Technical Specifications**:
        - **Type**: `torch.Tensor`
        - **Data Type (Dtype)**: `torch.float32`
        - **Device**: `cuda:0` (GPU)
        - **Layout**: `torch.strided`
        - **Memory Stride**: `(2048, 2048, 2048, 1)`
        - **Gradient**: `requires_grad=False` (Inference Mode)
    - **Conclusion**: The context vector structure and central distribution are consistent between the original and finetuned models. The difference in maximum values suggests that finetuning might have pushed some features to more extreme values, or it's an artifact of the specific sample used. The 2048 dimension is confirmed.
- **Context Vector Location**:
    - In `RoboVLMs/robovlms/model/backbone/base_backbone.py`, the method `forward_continuous` computes `output_hs` (hidden states from the LLM).
    - It then extracts `action_hs` (Action Hidden States) which serves as the input to the `act_head` (Action Head).
    - `action_hs` is likely the "context vector" of interest.
    - Code reference:
        ```python
        # base_backbone.py
        output_hs = output.hidden_states[-1].clone()
        # ...
        action_hs = output_hs[action_token_mask].reshape(...)
        # ...
        action_logits, action_loss = self._forward_action_head(action_hs, ...)
        ```

## Implementation Plan
1.  **Hook Strategy**: Use a PyTorch forward hook on `model.act_head` to capture the input (`action_hs`).
    ```python
    def hook_fn(module, input, output):
        context_vector = input[0]
        # save or analyze context_vector
    
    model.act_head.register_forward_hook(hook_fn)
    ```
2.  **Sampling**:
    -   If the dataset is small (~500), we can potentially run all of them or sample 50-100 representative ones.
    -   Need to check the diversity of the collected data.

## Next Steps
-   Write a script to load the model and data.
-   Register the hook.
-   Run inference on samples.
-   Save and visualize `action_hs`.
