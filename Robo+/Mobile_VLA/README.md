# Distance-Aware Mobile VLA Model

## Overview
This is a distance-aware Vision-Language-Action (VLA) model for mobile robot navigation, built on top of Kosmos2 vision backbone.

## Model Architecture
- **Backbone**: Kosmos2 Vision Model (microsoft/kosmos-2-patch14-224)
- **Action Head**: LSTM + MLP
- **Distance Awareness**: Distance embedding and fusion layers
- **Input**: 8-frame image sequence
- **Output**: 2-frame action prediction [linear_x, linear_y, angular_z]

## Performance
- **Overall MAE**: 0.2602
- **Success Rate**: 88.7%
- **Distance-wise Performance**:
  - Close: MAE 0.2617 (76.6% success)
  - Medium: MAE 0.2017 (81.9% success) ⭐ Best
  - Far: MAE 0.3373 (69.8% success)

## Usage
```python
from transformers import AutoProcessor, AutoModel
import torch

# Load model
processor = AutoProcessor.from_pretrained("your-username/distance-aware-mobile-vla")
model = AutoModel.from_pretrained("your-username/distance-aware-mobile-vla")

# Prepare input
images = torch.randn(1, 8, 3, 224, 224)  # 8-frame sequence
distance_labels = torch.tensor([1])  # 0: close, 1: medium, 2: far

# Predict actions
with torch.no_grad():
    predicted_actions = model(images, distance_labels)
```

## Training Details
- **Dataset**: 480 episodes (160 per distance)
- **Augmentation**: Distance-aware specialized augmentation
- **Distance Factors**: Close 8x, Medium 5x, Far 8x
- **Training Epochs**: 15

## Key Features
- ✅ Distance-aware training and inference
- ✅ Kosmos2 vision backbone
- ✅ Temporal modeling with LSTM
- ✅ Specialized data augmentation
- ✅ Balanced performance across distances

## Limitations
- Currently predicts 2 frames from 8 input frames
- SPACE (stop) action accuracy needs improvement
- Far distance performance can be enhanced

## Citation
If you use this model, please cite:
```
@misc{distance_aware_mobile_vla_2024,
  title={Distance-Aware Mobile VLA Model},
  author={Your Name},
  year={2024}
}
```
