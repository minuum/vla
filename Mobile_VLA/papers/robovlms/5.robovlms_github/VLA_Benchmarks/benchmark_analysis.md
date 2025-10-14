# RoboVLMs VLA Benchmarks Analysis

## Benchmark Overview

RoboVLMs provides comprehensive evaluation across multiple benchmarks, including simulation environments and real-world robot manipulation tasks. The framework supports three main evaluation scenarios: CALVIN, SimplerEnv, and real-world robot experiments.

## CALVIN Benchmark

### Dataset Characteristics
- **Total Demonstrations**: 24K human-teleoperated demonstrations
- **Basic Skills**: 34 pre-defined manipulation tasks
- **Trajectory Length**: Less than 64 time steps per trajectory
- **Language Instructions**: Annotated with natural language descriptions

### Task Categories
1. **Block Manipulation**: Rotate, move, lift, place blocks
2. **Slider Operations**: Move slider left/right
3. **Light Control**: Turn on/off light bulb and LED
4. **Drawer Operations**: Open/close drawer, push in drawer
5. **Stacking Tasks**: Lift stacked blocks

### Evaluation Splits
- **Scene A, B, C, D**: Different scene configurations
- **Training Splits**: ABCD (full training), ABC (generalization)
- **Test Split**: D (unseen scene evaluation)

### Performance Results

#### ABCD → D Split (Full Training)
| Method | VLA? | 1 Task | 2 Tasks | 3 Tasks | 4 Tasks | 5 Tasks | Avg. Len. |
|--------|------|--------|---------|---------|---------|---------|-----------|
| MCIL | ✖ | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| R3M (Frozen) | ✖ | 0.085 | 0.005 | 0.001 | 0.000 | 0.000 | 0.10 |
| Voltron (Frozen) | ✖ | 0.101 | 0.003 | 0.001 | 0.000 | 0.000 | 0.11 |
| Voltron (Fine-tuned) | ✖ | 0.837 | 0.566 | 0.352 | 0.208 | 0.115 | 2.08 |
| RT-1 | ✖ | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| HULC | ✖ | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06 |
| GR-1 | ✔ | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21 |
| **KosMos P.H. (RoboVLMs)** | ✔ | **0.967** | **0.930** | **0.899** | **0.865** | **0.826** | **4.49** |

#### ABC → D Split (Generalization)
| Method | VLA? | 1 Task | 2 Tasks | 3 Tasks | 4 Tasks | 5 Tasks | Avg. Len. |
|--------|------|--------|---------|---------|---------|---------|-----------|
| MCIL | ✖ | 0.304 | 0.013 | 0.002 | 0.000 | 0.000 | 0.31 |
| Voltron (Frozen) | ✖ | 0.026 | 0.001 | 0.000 | 0.000 | 0.000 | 0.03 |
| Voltron (Fine-tuned) | ✖ | 0.569 | 0.272 | 0.105 | 0.038 | 0.014 | 1.00 |
| RT-1 | ✖ | 0.533 | 0.222 | 0.094 | 0.038 | 0.013 | 0.90 |
| HULC | ✖ | 0.418 | 0.165 | 0.057 | 0.019 | 0.011 | 0.67 |
| GR-1 | ✔ | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| **KosMos P.H. (RoboVLMs)** | ✔ | **0.980** | **0.936** | **0.854** | **0.778** | **0.704** | **4.25** |

### Key Achievements
- **State-of-the-art Performance**: Highest success rates across all consecutive task lengths
- **Strong Generalization**: 12.6% improvement in single task execution (ABC → D)
- **Long-horizon Capability**: 4.25 average task length in zero-shot settings
- **Robust Performance**: Consistent high performance across different training splits

## SimplerEnv Benchmark

### Environment Setup
- **Purpose**: Real-to-sim environment evaluation
- **Robot Platforms**: Google Robot, WidowX+Bridge
- **Task Types**: Pick, move, open/close, place operations
- **Evaluation**: Cross-embodiment generalization testing

### Google Robot Tasks
1. **Pick Coke Can**: Lift empty Coke can from table
   - **Positions**: Horizontal, vertical, standing upright
   - **Grid Points**: 25 positions per orientation
   - **Total Trials**: 75 experiments

2. **Move {obj1} near {obj2}**: Object manipulation
   - **Objects**: 8 different objects (bottles, cans, fruits)
   - **Configurations**: Triangular formations
   - **Total Trials**: 60 experiments

3. **Open/Close Drawer**: Articulated object manipulation
   - **Drawers**: Top, middle, bottom
   - **Actions**: Open/close operations
   - **Total Trials**: 54 experiments

4. **Open Top Drawer & Place Apple**: Sequential manipulation
   - **Steps**: Open drawer, place apple
   - **Positions**: 3 robot locations, 9 apple positions
   - **Total Trials**: 27 experiments

### WidowX+Bridge Tasks
1. **Put Spoon on Towel**: Object placement
   - **Setup**: 15cm square configuration
   - **Orientations**: Horizontal and vertical
   - **Total Trials**: 24 experiments

2. **Put Carrot on Plate**: Similar to spoon task
   - **Objects**: Carrot and plate
   - **Configuration**: Same as spoon task
   - **Total Trials**: 24 experiments

3. **Stack Green Block on Yellow Block**: Block stacking
   - **Block Sizes**: 3cm blocks
   - **Square Sizes**: 10cm and 20cm
   - **Total Trials**: 24 experiments

4. **Put Eggplant in Yellow Basket**: Container manipulation
   - **Setup**: Sink basin configuration
   - **Objects**: Eggplant and yellow basket
   - **Total Trials**: 24 experiments

### Performance Results

#### WidowX+Bridge Performance
- **Put Spoon on Towel**: 70.8% success rate
- **Put Carrot on Plate**: 45.8% success rate
- **Stack Green Block on Yellow Block**: 33.3% success rate
- **Put Eggplant in Yellow Basket**: 20.8% success rate

#### Google Robot Performance
- **Pick Coke Can**: 94.0% success rate
- **Move Near**: 90.3% success rate
- **Open/Close Drawer**: 77.3% success rate
- **Open Drawer & Place Apple**: 43.5% success rate

### Key Achievements
- **Cross-embodiment Generalization**: Strong performance across different robot platforms
- **Multi-task Capability**: Effective handling of diverse manipulation tasks
- **Real-to-sim Transfer**: Successful simulation-to-reality transfer

## Real-World Robot Experiments

### Experimental Setup
- **Robot Platform**: Kinova Gen-3 7-DoF arm with Robotiq 2F-85 gripper
- **Cameras**: Static workspace camera + wrist-mounted camera
- **Workspace**: 55cm × 24cm table with 40+ objects
- **Tasks**: 20 manipulation tasks with 5 rollouts each

### Evaluation Settings
1. **Simple Setting**: Training distribution matching
2. **Unseen Distractor**: Previously unseen distractor objects
3. **Unseen Background**: New tablecloths and backgrounds
4. **Unseen Objects**: Objects not in training data
5. **Novel Skill Description**: GPT-4 generated instruction synonyms

### Task Categories
- **Opening/Closing**: Drawer, oven, toaster operations
- **Pick & Place**: Object manipulation and placement
- **Button Pressing**: Switch and button operations
- **Sequential Tasks**: Multi-step manipulation sequences

### Performance Results

#### Overall Performance
- **Simple Setting**: 75% success rate
- **Unseen Distractor**: 60% success rate
- **Unseen Background**: 50% success rate
- **Novel Skill Description**: 55% success rate
- **Unseen Objects**: 33% success rate

#### Task-Specific Performance
- **Open Drawer**: 75% success rate
- **Pickup Eggplant**: 60% success rate
- **Press Toaster**: 50% success rate
- **Pickup Knife**: 45% success rate
- **Pickup Cucumber**: 40% success rate

### Key Achievements
- **Real-world Deployment**: Successful real-robot manipulation
- **Generalization**: Strong performance across unseen settings
- **Self-correction**: Emergent trajectory correction capabilities
- **Robustness**: Effective handling of diverse scenarios

## Benchmark Characteristics

### 1. CALVIN Characteristics
- **Simulation Environment**: Table-top manipulation
- **Task Complexity**: 34 basic skills
- **Language Instructions**: Natural language descriptions
- **Evaluation Metrics**: Consecutive task success rates, average length

### 2. SimplerEnv Characteristics
- **Real-to-sim Transfer**: Simulation of real-world scenarios
- **Cross-embodiment**: Multiple robot platforms
- **Task Diversity**: Various manipulation operations
- **Evaluation Focus**: Generalization and robustness

### 3. Real-World Characteristics
- **Physical Robot**: 7-DoF arm with gripper
- **Multi-camera Setup**: Workspace and wrist cameras
- **Diverse Objects**: 40+ objects in workspace
- **Evaluation Settings**: Multiple generalization scenarios

## Evaluation Metrics

### 1. Success Rate Metrics
- **Single Task**: Individual task completion rate
- **Consecutive Tasks**: 1-5 task sequence completion
- **Average Length**: Average number of completed tasks
- **Generalization Score**: Cross-domain performance

### 2. Performance Metrics
- **Task Completion**: Binary success/failure
- **Trajectory Quality**: Smoothness and efficiency
- **Generalization**: Unseen scenario performance
- **Robustness**: Distractor and background handling

### 3. Comparative Metrics
- **Baseline Comparison**: Against existing methods
- **VLA vs Non-VLA**: VLA framework effectiveness
- **Cross-embodiment**: Multi-robot platform performance
- **Sim-to-real**: Simulation to reality transfer

## Key Insights

### 1. VLA Effectiveness
- **Superior Performance**: VLA models outperform non-VLA baselines
- **Generalization**: Strong zero-shot performance on unseen scenarios
- **Long-horizon**: Effective handling of consecutive tasks

### 2. Architecture Impact
- **Policy Head**: More effective than interleaved modeling
- **History Integration**: Multi-step observations crucial for performance
- **Action Space**: Continuous actions outperform discrete for long-horizon tasks

### 3. Training Strategy
- **Pre-training**: Cross-embodiment data improves generalization
- **Fine-tuning**: In-domain data essential for task-specific performance
- **Data Efficiency**: VLA models show strong few-shot learning capabilities

### 4. Real-world Applicability
- **Deployment Success**: Successful real-robot manipulation
- **Generalization**: Effective handling of unseen objects and scenarios
- **Self-correction**: Emergent capabilities for trajectory correction
- **Robustness**: Strong performance across diverse evaluation settings