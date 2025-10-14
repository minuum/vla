# RoboVLMs GitHub README Analysis

## Project Overview

**RoboVLMs** is a flexible framework for building Vision-Language-Action (VLA) models by integrating various Vision-Language Models (VLMs) into robot manipulation policies. The project provides a unified approach to transfer pre-trained VLMs into VLAs with minimal manual design.

## Key Features

### 1. Flexible VLM Integration
- Support for multiple VLM backbones (LLaVA, Flamingo, KosMos, Qwen-VL, MoonDream, PaliGemma)
- Easy integration of new VLMs within 30 lines of code
- Modular architecture for seamless VLM-to-VLA conversion

### 2. Multiple VLA Architectures
- **One-step models**: Single observation to action prediction
- **History modeling**: Multi-step observation processing
- **Continuous/Discrete action spaces**: Flexible action representation
- **Policy head integration**: Effective history fusion methods

### 3. Comprehensive Benchmarks
- **CALVIN**: Multi-task table-top manipulation simulation
- **SimplerEnv**: Real-to-sim environment evaluation
- **Real-world experiments**: 20 tasks with 74K trajectories

## Installation

### Environment Setup
```bash
# For CALVIN simulation
conda create -n robovlms python=3.8.10 -y

# For SIMPLER simulation  
conda create -n robovlms python=3.10 -y

conda activate robovlms
conda install cudatoolkit cudatoolkit-dev -y
pip install -e .
```

### Benchmark Environment Setup
```bash
# CALVIN Installation
bash scripts/setup_calvin.sh

# SimplerEnv Installation
bash scripts/setup_simplerenv.sh
```

### Verification
```python
# CALVIN simulation verification
python eval/calvin/env_test.py

# SimplerEnv simulation verification
python eval/simpler/env_test.py
```

## VLA Benchmarks Performance

### CALVIN Benchmark Results

**ABCD -> D Split:**
- KosMos P.H. (RoboVLMs): 96.7% success rate, 4.49 avg length
- GR-1: 94.9% success rate, 4.21 avg length
- HULC: 88.9% success rate, 3.06 avg length

**ABC -> D Split:**
- KosMos P.H. (RoboVLMs): 98.0% success rate, 4.25 avg length
- GR-1: 85.4% success rate, 3.06 avg length
- HULC: 41.8% success rate, 0.67 avg length

### SimplerEnv Performance
- Achieves state-of-the-art performance on both WidowX+Bridge and Google Robot environments
- Demonstrates strong generalization across different robot platforms

### Real-World Performance
- 20 manipulation tasks with 5 rollouts each
- Strong performance across Simple, Unseen Distractor, Unseen Background, and Novel Skill Description settings
- Self-correction capabilities in complex manipulation scenarios

## VLM Integration Tutorial

### 1. VLM Attribute Configuration
Required attributes for VLM integration:
- `image_processor`: Image preprocessing
- `hidden_size`: VLM hidden dimensions
- `word_embedding`: Text embedding layer
- `text_tower`: Text processing component
- `vision_tower`: Vision processing component
- `model`: Core VLM backbone

### 2. Example Integration (PaliGemma)
```python
class RoboPaligemma(BaseRoboVLM):
    @property
    def image_processor(self):
        return self.model.processor
    
    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size
    
    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens
    
    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower
    
    @property
    def model(self):
        return self.backbone
    
    def model_encode_images(self, images):
        image_outputs = self.model.vision_tower(images)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.model.config.hidden_size**0.5)
        return image_features
```

### 3. VLA Registration
```python
from .robopaligemma import RoboPaligemma
__all__.append('RoboPaligemma')
```

## Training Pipeline

### Data Preprocessing
- **Action Normalization**: Clamp to 1st/99th quantile, normalize to [-1, 1]
- **Action Discretization**: Map continuous actions to 256 discrete bins
- **History Processing**: Sliding window of historical observations

### Model Architecture
- **BaseRoboVLM**: Core framework for VLM integration
- **Action Heads**: LSTM, FC, or GPT-based action prediction
- **Multimodal Fusion**: Vision-language-action integration

### Training Configuration
```json
{
    "model": {
        "backbone": "kosmos",
        "action_head": "lstm",
        "history_length": 16,
        "action_chunk_size": 10
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 5,
        "warmup_ratio": 0.25
    }
}
```

## Evaluation Process

### CALVIN Evaluation
- 5 consecutive tasks evaluation
- Success rates for 1-5 task sequences
- Average task length (Avg. Len.)
- Generalization from ABC to D split

### SimplerEnv Evaluation
- Google Robot and WidowX+Bridge environments
- Multiple task types: pick, move, open/close, place
- Cross-embodiment generalization testing

### Real-World Evaluation
- 20 manipulation tasks
- 5 evaluation settings per task
- Unseen object, background, and skill description testing

## Supported Backbones

### Encoder-Decoder Architectures
- **Flamingo**: Cross-attention based fusion
- **OFA**: Unified encoder-decoder framework

### Decoder-Only Architectures
- **LLaVA**: Self-attention based fusion
- **KosMos**: Multimodal transformer
- **Qwen-VL**: Large-scale vision-language model
- **MoonDream**: Efficient VLM architecture
- **PaliGemma**: Google's multimodal model

## Core Learning Methods

### 1. Vision-Language Pre-training
- Large-scale web data training
- Robust multimodal representations
- Foundation for robot manipulation

### 2. Action Prediction
- **Continuous Actions**: MSE + BCE loss for pose and gripper
- **Discrete Actions**: Cross-entropy loss for action tokens
- **History Integration**: Policy head for temporal modeling

### 3. Training Strategies
- **Pre-train**: Cross-embodiment data pre-training
- **Post-train**: VLM pre-training + in-domain fine-tuning
- **Finetune**: Direct in-domain training

## Scalability Features

### 1. Modular Design
- Easy VLM backbone swapping
- Flexible action head selection
- Configurable training pipelines

### 2. Performance Optimization
- Memory-efficient training
- Distributed training support
- Checkpoint management

### 3. Evaluation Framework
- Multiple benchmark support
- Automated evaluation pipelines
- Performance monitoring

## Troubleshooting

### Common Issues
1. **CUDA compatibility**: Ensure proper CUDA toolkit installation
2. **Environment conflicts**: Use separate conda environments
3. **Memory issues**: Adjust batch size and model parameters
4. **Benchmark setup**: Follow specific installation guides

### Optimization Tips
1. **Model selection**: Choose appropriate backbone for task
2. **Hyperparameter tuning**: Optimize learning rate and batch size
3. **Data preprocessing**: Ensure proper action normalization
4. **Training strategy**: Use appropriate pre-training approach

## Conclusion

RoboVLMs provides a comprehensive framework for building high-performance VLAs with minimal manual design. The framework's flexibility, comprehensive evaluation, and strong performance make it a valuable tool for robotics research and development.

### Key Advantages
- **Easy Integration**: 30-line VLM integration
- **Strong Performance**: State-of-the-art results on multiple benchmarks
- **Flexible Architecture**: Support for various VLM backbones and VLA structures
- **Comprehensive Evaluation**: Multiple simulation and real-world benchmarks
- **Open Source**: Complete codebase and model weights available

### Applications
- Generalist robot manipulation policies
- Cross-embodiment robot learning
- Vision-language-action model research
- Robotics benchmark evaluation