# 09_1 Downsampling Methods in RoboVLMs

## 📋 **Overview**

This document analyzes the downsampling methods used in RoboVLMs for VLA (Vision-Language-Action) models, focusing on the differences between pooling, resampler, and none approaches, and their impact on robot control performance.

## 🎯 **Key Findings**

### **9.1.1 Downsampling Methods in RoboVLMs**

#### **Available Methods**
- **Source**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:319-425` (Updated from @RoboVLMs)
- **Implementation**: Three downsampling options in Policy Head classes
- **Methods**:
  ```python
  if self.down_sample == "pooling":
      # 1D 글로벌 풀링 사용
      self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
      
  elif self.down_sample == "resampler":
      # 리샘플러 사용 (미구현)
      raise NotImplementedError
      
  elif self.down_sample == "none":
      # 다운샘플링 없음
      pass
  ```

#### **9.1.2 Method 1: Pooling Downsampling**
- **Source**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:346-348` (Updated from @RoboVLMs)
- **Implementation**: AdaptiveMaxPool1d for 1D global pooling
- **Process**:
  ```python
  if self.down_sample == "pooling":
      bs, seq_len = tok_seq.shape[:2]
      tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
      tok_seq = self.global_1d_pool(
          tok_seq.permute(0, 2, 1)
      )  # bs*seq_len, n_tok, tok_dim -> bs*seq_len, tok_dim
      tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
  ```
- **Characteristics**:
  - **Pooling Type**: Max pooling (most commonly used)
  - **Dimension Reduction**: n_tok → latent (e.g., 196 → 64 tokens)
  - **Information Loss**: Some spatial information may be lost
  - **Computational Cost**: Low (simple max operation)

#### **9.1.3 Method 2: Resampler Downsampling**
- **Source**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:421, 445, 550` (Updated from @RoboVLMs)
- **Implementation**: Currently not implemented
- **Status**: `raise NotImplementedError`
- **Planned Features**:
  - **Learned Downsampling**: Trainable parameters for adaptive downsampling
  - **Better Information Preservation**: More sophisticated than pooling
  - **Higher Computational Cost**: Requires training

#### **9.1.4 Method 3: None Downsampling**
- **Source**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:352, 447, 552` (Updated from @RoboVLMs)
- **Implementation**: Simple dimension rearrangement without downsampling
- **Process**:
  ```python
  elif self.down_sample == "none":
      tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
  ```
- **Characteristics**:
  - **No Information Loss**: All tokens preserved
  - **Higher Memory Usage**: Full token sequence maintained
  - **Computational Cost**: Medium (no pooling, but full sequence processing)

### **9.1.5 Usage Statistics in RoboVLMs**

#### **Configuration File Analysis**
- **Source**: Configuration files in `RoboVLMs/configs/` (Updated from @RoboVLMs)
- **Usage Count**:
  - **"none"**: 12 configuration files (80% of all configs)
  - **"pooling"**: 3 configuration files (20% of all configs)
  - **"resampler"**: 0 configuration files (0% - not implemented)

#### **Specific Usage Cases**
```json
// Most common configuration (12 files)
"act_head": {
    "type": "LSTMDecoder",
    "down_sample": "none",  // 80% of configs use this
    "latent": 1,
    "action_dim": 7
}

// Less common configuration (3 files)
"act_head": {
    "type": "LSTMDecoder", 
    "down_sample": "pooling",  // 20% of configs use this
    "latent": 1,
    "action_dim": 7
}
```

### **9.1.6 Performance Comparison**

#### **Memory Usage**
| Method | Memory Usage | Information Preservation | Computational Cost |
|--------|-------------|-------------------------|-------------------|
| **"none"** | High | Complete | Medium |
| **"pooling"** | Low | Partial | Low |
| **"resampler"** | Medium | High | High (when implemented) |

#### **Information Flow**
```python
# None downsampling: Full information flow
tok_seq: [batch, seq_len, n_tok, feature_dim]
         ↓ (rearrange only)
tok_seq: [batch, seq_len, n_tok * feature_dim]

# Pooling downsampling: Information reduction
tok_seq: [batch, seq_len, n_tok, feature_dim]
         ↓ (AdaptiveMaxPool1d)
tok_seq: [batch, seq_len, latent, feature_dim]
         ↓ (rearrange)
tok_seq: [batch, seq_len, latent * feature_dim]
```

### **9.1.7 Why "none" is Preferred in RoboVLMs**

#### **Robot Control Requirements**
- **Precision**: Robot control requires high precision in action prediction
- **Temporal Consistency**: Sequential processing benefits from full token information
- **Memory vs Performance**: The trade-off favors performance over memory efficiency

#### **LSTM Decoder Benefits**
- **Sequential Processing**: LSTM can handle variable-length sequences effectively
- **Hidden State Management**: Full token information helps maintain context
- **Action Prediction**: More information leads to better action predictions

### **9.1.8 Downsampling in Computer Vision Context**

#### **Reference**: [Wikidocs Downsampling Guide](https://wikidocs.net/147019)
- **Pooling Methods**: Max pooling, Average pooling
- **Convolution-based**: Dilated convolution, Depthwise convolution
- **Advanced Methods**: Depthwise separable convolution

#### **RoboVLMs vs Traditional CV**
| Aspect | Traditional CV | RoboVLMs |
|--------|----------------|----------|
| **Purpose** | Feature extraction | Action prediction |
| **Information Loss** | Acceptable | Critical |
| **Method** | Spatial pooling | Token pooling |
| **Goal** | Classification accuracy | Control precision |

### **9.1.9 Implementation Details**

#### **AdaptiveMaxPool1d Usage**
```python
# Pooling implementation
self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)

# Forward pass
tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
# Input: [batch, n_tok, feature_dim]
# Output: [batch, latent, feature_dim]
```

#### **Dimension Calculations**
```python
# Input dimensions
tok_seq.shape = [batch, seq_len, n_tok, feature_dim]

# After pooling (if pooling used)
tok_seq.shape = [batch, seq_len, latent, feature_dim]

# After rearrange
tok_seq.shape = [batch, seq_len, latent * feature_dim]
```

### **9.1.10 Future Improvements**

#### **Resampler Implementation**
- **Planned Features**: Learnable downsampling with attention mechanisms
- **Benefits**: Better information preservation than pooling
- **Challenges**: Higher computational cost and training complexity

#### **Hybrid Approaches**
- **Adaptive Downsampling**: Dynamic pooling based on task complexity
- **Attention-based**: Use attention weights for selective downsampling
- **Multi-scale**: Different downsampling rates for different modalities

## 🎯 **Key Findings**

1. **"none" is Dominant**: 80% of RoboVLMs configurations use no downsampling
2. **Performance Priority**: Robot control prioritizes precision over memory efficiency
3. **LSTM Compatibility**: Sequential processing works well with full token information
4. **Resampler Potential**: Future implementation could improve information preservation
5. **Memory Trade-off**: Current approach favors performance over memory optimization

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/model/policy_head/base_policy.py`
- `RoboVLMs/configs/calvin_finetune/` (12 files with "none")
- `RoboVLMs/configs/calvin_finetune/finetune_flamingo_mpt_3b_ws-8_act-10_lstm_calvin.json` (pooling example)
- `RoboVLMs/configs/k_project/ros2_automotive.json` (pooling example)
- [Wikidocs Downsampling Guide](https://wikidocs.net/147019)
