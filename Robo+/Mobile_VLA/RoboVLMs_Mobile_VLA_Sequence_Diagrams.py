#!/usr/bin/env python3
"""
🔄 RoboVLMs vs Mobile VLA 시퀀스 다이어그램 생성

데이터 플로우, 모델 추론, 학습 과정을 시퀀스 다이어그램으로 비교 분석합니다.
"""

from datetime import datetime

def generate_robovlms_sequence_diagram():
    """RoboVLMs 원본 시퀀스 다이어그램"""
    
    return """
RoboVLMs 데이터 플로우 및 추론 과정
```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Dataset as 📊 RLDS Dataset
    participant DataLoader as 🔄 DataLoader
    participant Backbone as 🧠 Backbone Model<br/>(RT-1/OpenVLA)
    participant ActionHead as 🎯 Action Head<br/>(Discrete)
    participant Trainer as 🏋️ BaseTrainer
    
    Note over User, Trainer: 📚 RoboVLMs 원본 아키텍처
    
    User->>Dataset: Load RLDS/TFRecord data
    Dataset->>DataLoader: RLDS format data
    
    loop Training Loop
        DataLoader->>DataLoader: Generate window/chunk<br/>(8 frames -> 2 actions)
        DataLoader->>Backbone: Batched images + text
        
        Backbone->>Backbone: Vision encoding
        Backbone->>Backbone: Text encoding  
        Backbone->>Backbone: Multi-modal fusion
        
        Backbone->>ActionHead: Hidden features
        ActionHead->>ActionHead: Discrete classification<br/>(7-DOF robot arm)
        ActionHead->>Trainer: Predicted actions
        
        Trainer->>Trainer: CrossEntropy loss
        Trainer->>Backbone: Backpropagation
    end
    
    Note over User, Trainer: ✅ 장점: 범용성, 대용량 데이터 처리
    Note over User, Trainer: ❌ 단점: 복잡성, 많은 데이터 필요
```
"""

def generate_mobile_vla_sequence_diagram():
    """Mobile VLA 시퀀스 다이어그램"""
    
    return """
Mobile VLA 데이터 플로우 및 추론 과정
```mermaid
sequenceDiagram
    participant User as 👤 User
    participant HDF5 as 📁 HDF5 Files<br/>(Real Robot Data)
    participant Dataset as 📊 MobileVLADataset
    participant Collate as 🔄 Custom Collate<br/>(PIL->Tensor)
    participant Kosmos as 🌌 Kosmos-2B<br/>(Vision Only)
    participant ActionHead as 🎯 Action Head<br/>(3D Continuous)
    participant Trainer as 🏋️ MobileVLATrainer
    
    Note over User, Trainer: 🚀 Mobile VLA 특화 아키텍처
    
    User->>HDF5: Load .h5 files<br/>(72 episodes)
    HDF5->>Dataset: Real robot data<br/>RGB + 3D actions
    Dataset->>Dataset: Extract scenarios<br/>(1box/2box + left/right)
    
    loop Training Loop
        Dataset->>Collate: PIL images + actions
        Collate->>Collate: PIL -> normalized tensors<br/>(224x224, ImageNet norm)
        
        Collate->>Trainer: Batched data
        Trainer->>Trainer: Window/chunk processing<br/>(8 frames -> 2 actions)
        
        Trainer->>Kosmos: 5D -> 4D conversion<br/>Last frame extraction
        Kosmos->>Kosmos: Vision model only<br/>(Skip text processing)
        Kosmos->>ActionHead: Pooled vision features
        
        ActionHead->>ActionHead: 3D regression<br/>[linear_x, linear_y, angular_z]
        ActionHead->>Trainer: Continuous predictions
        
        Trainer->>Trainer: Huber loss (regression)
        Trainer->>Kosmos: Backpropagation
    end
    
    Note over User, Trainer: ✅ 장점: 데이터 효율적, 실시간, 정밀 제어
    Note over User, Trainer: 🎯 특화: Mobile Robot, 연속 제어, 시나리오 분석
```
"""

def generate_comparison_sequence_diagram():
    """RoboVLMs vs Mobile VLA 비교 다이어그램"""
    
    return """
RoboVLMs vs Mobile VLA 아키텍처 비교
```mermaid
graph TB
    subgraph "🔵 RoboVLMs (Original)"
        A1[📊 RLDS/TFRecord<br/>수백만 데모] --> B1[🔄 Standard DataLoader]
        B1 --> C1[🧠 Multi-modal Backbone<br/>RT-1/OpenVLA]
        C1 --> D1[🎯 Discrete Action Head<br/>7-DOF Classification]
        D1 --> E1[📈 CrossEntropy Loss]
        
        F1[📝 Text Instructions] --> C1
        G1[🖼️ RGB Images] --> C1
    end
    
    subgraph "🟢 Mobile VLA (Specialized)"
        A2[📁 HDF5 Files<br/>72 episodes] --> B2[🔄 Custom Collate<br/>PIL->Tensor]
        B2 --> C2[🌌 Kosmos-2B Vision<br/>Image-only Processing]
        C2 --> D2[🎯 Continuous Action Head<br/>3D Regression]
        D2 --> E2[📈 Huber Loss]
        
        F2[📝 Scenario Descriptions<br/>Obstacle Avoidance] --> C2
        G2[🖼️ Real Robot RGB] --> C2
        
        H2[📊 Scenario Analysis<br/>1box/2box, left/right] --> I2[📈 Detailed Metrics<br/>MAE, R², Per-dimension]
    end
    
    subgraph "🔄 공통 요소 (Inherited)"
        J[📦 Window/Chunk Mechanism<br/>8 frames -> 2 actions]
        K[🏋️ BaseTrainer Pattern<br/>Training Loop Structure]
        L[📊 Dataset Interface<br/>__getitem__ format]
    end
    
    C1 -.-> J
    C2 -.-> J
    E1 -.-> K
    E2 -.-> K
    B1 -.-> L
    B2 -.-> L
    
    style A1 fill:#e1f5fe
    style A2 fill:#e8f5e8
    style C1 fill:#e1f5fe
    style C2 fill:#e8f5e8
    style J fill:#fff3e0
    style K fill:#fff3e0
    style L fill:#fff3e0
```
"""

def generate_data_processing_comparison():
    """데이터 처리 과정 비교"""
    
    return """
데이터 처리 파이프라인 비교
```mermaid
sequenceDiagram
    participant Raw as 📁 Raw Data
    participant Process as 🔄 Processing
    participant Model as 🧠 Model Input
    participant Output as 📊 Output
    
    Note over Raw, Output: 🔵 RoboVLMs Data Pipeline
    
    Raw->>Process: RLDS/TFRecord<br/>Millions of demos
    Process->>Process: Standard preprocessing<br/>Resize, normalize
    Process->>Model: Text + Images<br/>Multi-modal input
    Model->>Output: Discrete actions<br/>Classification logits
    
    Note over Raw, Output: 🟢 Mobile VLA Data Pipeline
    
    Raw->>Process: HDF5 files<br/>72 real robot episodes
    Process->>Process: Custom PIL handling<br/>Scenario extraction
    Process->>Model: Images only<br/>Last frame selection
    Model->>Output: Continuous actions<br/>3D regression values
    
    Note over Raw, Output: 📊 Key Differences
    Note right of Raw: Data Volume:<br/>Millions vs 72 episodes
    Note right of Process: Processing:<br/>Standard vs Custom
    Note right of Model: Input:<br/>Multi-modal vs Vision-only
    Note right of Output: Output:<br/>Discrete vs Continuous
```
"""

def generate_evaluation_comparison():
    """평가 시스템 비교"""
    
    return """
평가 시스템 비교
```mermaid
flowchart TD
    subgraph "🔵 RoboVLMs Evaluation"
        A1[🎯 Task Success Rate] --> B1[📊 Binary Success/Fail]
        B1 --> C1[📈 Overall Accuracy]
        C1 --> D1[📋 Simple Report]
    end
    
    subgraph "🟢 Mobile VLA Evaluation"
        A2[🎯 Action Prediction] --> B2[📊 Continuous Metrics]
        B2 --> C2[📈 Multi-dimensional Analysis]
        
        B2 --> E2[📐 MAE/RMSE/R²]
        B2 --> F2[🎲 Threshold Accuracy]
        B2 --> G2[📋 Per-action Analysis]
        B2 --> H2[🗺️ Scenario-wise Performance]
        
        C2 --> I2[📊 Comprehensive Report]
        E2 --> I2
        F2 --> I2  
        G2 --> I2
        H2 --> I2
        
        I2 --> J2[🎓 Professor Evaluation]
        I2 --> K2[📈 Performance Grading]
        I2 --> L2[🔮 Improvement Roadmap]
    end
    
    subgraph "📊 Evaluation Depth"
        M[🔵 Surface Level<br/>Success/Fail only]
        N[🟢 Deep Analysis<br/>Multi-metric, Multi-scenario]
    end
    
    D1 -.-> M
    L2 -.-> N
    
    style A1 fill:#e1f5fe
    style A2 fill:#e8f5e8
    style M fill:#ffebee
    style N fill:#e8f5e8
```
"""

def generate_all_sequence_diagrams():
    """모든 시퀀스 다이어그램 생성"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    content = f"""# 🔄 RoboVLMs vs Mobile VLA 시퀀스 다이어그램 비교

**생성 일시:** {timestamp}

{generate_robovlms_sequence_diagram()}

{generate_mobile_vla_sequence_diagram()}

{generate_comparison_sequence_diagram()}

{generate_data_processing_comparison()}

{generate_evaluation_comparison()}

## 📊 핵심 차이점 요약

### 🔵 RoboVLMs 특징
- **데이터**: RLDS/TFRecord 형식, 수백만 데모
- **모델**: Multi-modal backbone (vision + text)
- **액션**: Discrete classification (7-DOF)
- **평가**: 단순 성공률
- **장점**: 범용성, 확장성
- **단점**: 복잡성, 많은 데이터 필요

### 🟢 Mobile VLA 특징  
- **데이터**: HDF5 형식, 72개 실제 로봇 에피소드
- **모델**: Kosmos-2B vision-only processing
- **액션**: Continuous regression (3D mobile)
- **평가**: 다차원 상세 분석
- **장점**: 데이터 효율적, 실시간, 정밀 제어
- **특화**: Mobile robot, 장애물 회피 시나리오

### 🔄 공통 유지 요소
- **Window/Chunk**: 8프레임 관찰 → 2프레임 예측
- **BaseTrainer**: 학습 루프 구조
- **Dataset Interface**: 데이터 로딩 인터페이스

### 🚀 Mobile VLA 혁신점
- **커스텀 Collate**: PIL 이미지 처리
- **5D→4D 변환**: Kosmos 특화 처리  
- **Huber Loss**: 연속 액션 회귀
- **시나리오 분석**: 장애물 회피 성능 분석
- **종합 평가**: MAE, R², 임계값 정확도

---
*Sequence Diagrams Analysis - {timestamp}*
"""

    filename = f'RoboVLMs_Mobile_VLA_Sequence_Diagrams_{timestamp}.md'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

def create_mermaid_diagrams():
    """Mermaid 다이어그램 생성 및 표시"""
    
    print("🔄 RoboVLMs vs Mobile VLA 시퀀스 다이어그램 생성")
    print("=" * 60)
    
    diagrams = {
        "RoboVLMs 원본": generate_robovlms_sequence_diagram(),
        "Mobile VLA 특화": generate_mobile_vla_sequence_diagram(), 
        "아키텍처 비교": generate_comparison_sequence_diagram(),
        "데이터 처리 비교": generate_data_processing_comparison(),
        "평가 시스템 비교": generate_evaluation_comparison()
    }
    
    for title, diagram in diagrams.items():
        print(f"\n📊 {title}")
        print("-" * 50)
        print(diagram)
    
    # 통합 파일 생성
    filename = generate_all_sequence_diagrams()
    print(f"\n📄 통합 다이어그램 파일 생성: {filename}")
    
    return filename

def main():
    """메인 시퀀스 다이어그램 생성 실행"""
    
    print("🔄 시퀀스 다이어그램 기반 비교 분석 시작!")
    
    # Mermaid 다이어그램 생성
    diagram_file = create_mermaid_diagrams()
    
    print(f"\n🎉 시퀀스 다이어그램 분석 완료!")
    print(f"📋 핵심 인사이트:")
    print(f"   🔵 RoboVLMs: 범용적, 복잡한 multi-modal 처리")
    print(f"   🟢 Mobile VLA: 특화된, 효율적인 vision-only 처리")
    print(f"   🔄 공통점: Window/Chunk 메커니즘 완전 유지")
    print(f"   🚀 혁신점: 데이터 효율성 + 실시간 성능")
    
    return diagram_file

if __name__ == "__main__":
    main()
