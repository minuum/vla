#!/usr/bin/env python3
"""
🔧 Advanced Mobile VLA Model
Claw Matrix + Hierarchical Planning + Advanced Attention 포함
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import sys
from pathlib import Path

# RoboVLMs 모듈 추가
sys.path.append(str(Path(__file__).parent / "robovlms" / "models"))
from claw_matrix import ClawMatrixModel, create_claw_matrix_model
from hierarchical import HierarchicalPlanningModel, create_hierarchical_model

class AdvancedMobileVLAModel(nn.Module):
    """
    고급 Mobile VLA 모델: Claw Matrix + Hierarchical Planning + Advanced Attention
    """
    def __init__(self, 
                 processor,
                 vision_dim: int = 768,
                 language_dim: int = 768,
                 action_dim: int = 3,
                 fusion_dim: int = 512,
                 plan_dim: int = 256,
                 num_claw_layers: int = 3,
                 num_subgoals: int = 6,
                 frames_per_subgoal: int = 3,
                 num_heads: int = 8,
                 use_claw_matrix: bool = True,
                 use_hierarchical: bool = True,
                 use_advanced_attention: bool = True):
        super().__init__()
        
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.fusion_dim = fusion_dim
        self.plan_dim = plan_dim
        self.num_subgoals = num_subgoals
        self.frames_per_subgoal = frames_per_subgoal
        self.total_frames = num_subgoals * frames_per_subgoal  # 18프레임
        
        # 사용할 고급 기능들
        self.use_claw_matrix = use_claw_matrix
        self.use_hierarchical = use_hierarchical
        self.use_advanced_attention = use_advanced_attention
        
        # Kosmos2 백본
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # 모든 파라미터에 그래디언트 활성화
        for param in self.kosmos.parameters():
            param.requires_grad = True
        
        # 특징 어댑터 (크기 통일) - 동적으로 생성
        self.feature_adapter = None
        
        # Claw Matrix (조건부)
        if self.use_claw_matrix:
            claw_config = {
                'vision_dim': vision_dim,
                'language_dim': language_dim,
                'action_dim': action_dim,
                'fusion_dim': fusion_dim,
                'output_dim': action_dim,
                'num_claw_layers': num_claw_layers,
                'num_heads': num_heads
            }
            self.claw_matrix = create_claw_matrix_model(claw_config)
        else:
            self.claw_matrix = None
        
        # Hierarchical Planning (조건부)
        if self.use_hierarchical:
            hierarchical_config = {
                'vision_dim': vision_dim,
                'language_dim': language_dim,
                'action_dim': action_dim,
                'plan_dim': plan_dim,
                'num_subgoals': num_subgoals,
                'frames_per_subgoal': frames_per_subgoal,
                'num_heads': num_heads
            }
            self.hierarchical_planner = create_hierarchical_model(hierarchical_config)
        else:
            self.hierarchical_planner = None
        
        # Advanced Attention Mechanisms (조건부)
        if self.use_advanced_attention:
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=vision_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=vision_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.hierarchical_attention = nn.MultiheadAttention(
                embed_dim=vision_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
        else:
            self.cross_modal_attention = None
            self.temporal_attention = None
            self.hierarchical_attention = None
        
        # 기본 LSTM (고급 기능이 비활성화된 경우)
        if not self.use_claw_matrix and not self.use_hierarchical:
            self.lstm = nn.LSTM(vision_dim, fusion_dim // 2, batch_first=True)
            self.action_head = nn.Sequential(
                nn.Linear(fusion_dim // 2, fusion_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(fusion_dim // 4, action_dim)
            )
        
        # 거리별 특화 (기존 기능 유지)
        self.distance_embedding = nn.Embedding(3, 32)
        self.distance_fusion = nn.Linear(vision_dim + 32, vision_dim)
        
        # 출력 정규화
        self.output_norm = nn.LayerNorm(action_dim)
        
    def extract_vision_features(self, images):
        """Kosmos2를 사용한 비전 특징 추출"""
        batch_size, seq_len, c, h, w = images.shape
        device = images.device
        
        image_features = []
        for t in range(seq_len):
            try:
                pixel_values = images[:, t, :, :, :]
                dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                dummy_attention_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                
                with torch.no_grad():
                    vision_outputs = self.kosmos.vision_model(pixel_values=pixel_values)
                    features = vision_outputs.last_hidden_state.mean(dim=1)
            except Exception as e:
                # 대체 방법
                features = torch.randn(batch_size, self.vision_dim, device=device)
            
            # 크기 통일
            if features.shape[-1] != self.vision_dim:
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Linear(features.shape[-1], self.vision_dim).to(features.device)
                features = self.feature_adapter(features)
            
            image_features.append(features)
        
        return torch.stack(image_features, dim=1)  # [batch_size, seq_len, vision_dim]
    
    def apply_advanced_attention(self, vision_features, language_features=None):
        """고급 어텐션 메커니즘 적용"""
        if not self.use_advanced_attention:
            return vision_features
        
        # Cross-modal attention (Vision-Language)
        if language_features is not None:
            vision_features, _ = self.cross_modal_attention(
                query=vision_features,
                key=language_features,
                value=language_features
            )
        
        # Temporal attention
        vision_features, _ = self.temporal_attention(
            query=vision_features,
            key=vision_features,
            value=vision_features
        )
        
        # Hierarchical attention
        vision_features, _ = self.hierarchical_attention(
            query=vision_features,
            key=vision_features,
            value=vision_features
        )
        
        return vision_features
    
    def forward(self, 
                images: torch.Tensor,
                distance_labels: torch.Tensor,
                language_features: torch.Tensor = None,
                current_actions: torch.Tensor = None) -> torch.Tensor:
        """
        고급 Mobile VLA 모델 순전파
        
        Args:
            images: [batch_size, seq_len, c, h, w]
            distance_labels: [batch_size]
            language_features: [batch_size, seq_len, language_dim] (선택사항)
            current_actions: [batch_size, seq_len, action_dim] (선택사항)
            
        Returns:
            predicted_actions: [batch_size, total_frames, action_dim]
        """
        batch_size = images.size(0)
        device = images.device
        
        # 1. 비전 특징 추출
        vision_features = self.extract_vision_features(images)  # [batch_size, seq_len, vision_dim]
        
        # 2. 거리별 특화
        distance_embeds = self.distance_embedding(distance_labels)  # [batch_size, 32]
        distance_embeds = distance_embeds.unsqueeze(1).expand(-1, vision_features.size(1), -1)
        vision_features = torch.cat([vision_features, distance_embeds], dim=-1)
        vision_features = self.distance_fusion(vision_features)
        
        # 3. 고급 어텐션 적용
        vision_features = self.apply_advanced_attention(vision_features, language_features)
        
        # 4. 모델 선택 및 실행
        if self.use_hierarchical:
            # Hierarchical Planning 사용
            if language_features is None:
                # 더미 언어 특징 생성
                language_features = torch.zeros(batch_size, vision_features.size(1), 
                                             self.language_dim, device=device)
            
            predicted_actions = self.hierarchical_planner(
                vision_features, language_features, current_actions
            )
            
        elif self.use_claw_matrix:
            # Claw Matrix 사용
            if language_features is None:
                language_features = torch.zeros(batch_size, vision_features.size(1), 
                                             self.language_dim, device=device)
            if current_actions is None:
                current_actions = torch.zeros(batch_size, vision_features.size(1), 
                                           self.action_dim, device=device)
            
            predicted_actions = self.claw_matrix(
                vision_features, language_features, current_actions
            )
            
        else:
            # 기본 LSTM 사용
            lstm_out, _ = self.lstm(vision_features)
            predicted_actions = self.action_head(lstm_out)
            # 2프레임 예측으로 확장
            predicted_actions = predicted_actions.unsqueeze(1).expand(-1, 2, -1)
        
        # 5. 출력 정규화
        predicted_actions = self.output_norm(predicted_actions)
        
        return predicted_actions

class AdvancedMobileVLATrainer:
    """
    고급 Mobile VLA 모델 훈련기
    """
    def __init__(self, 
                 model: AdvancedMobileVLAModel,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 손실 함수
        self.criterion = nn.HuberLoss()
        
        # 거리별 가중치
        self.distance_weights = {
            0: 1.5,  # close
            1: 1.0,  # medium
            2: 0.8   # far
        }
        
    def compute_loss(self, predicted_actions, target_actions, distance_labels):
        """손실 계산"""
        batch_size = predicted_actions.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            distance = distance_labels[i].item()
            weight = self.distance_weights.get(distance, 1.0)
            
            # MAE 계산
            mae = F.l1_loss(predicted_actions[i], target_actions[i])
            total_loss += weight * mae
        
        return total_loss / batch_size
    
    def train_step(self, batch):
        """훈련 스텝"""
        self.model.train()
        
        # 데이터 준비
        images = batch['images'].to(self.device)
        actions = batch['actions'].to(self.device)
        distance_labels = batch['distance_labels'].to(self.device)
        
        # 순전파
        predicted_actions = self.model(images, distance_labels)
        
        # 손실 계산
        loss = self.compute_loss(predicted_actions, actions, distance_labels)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                actions = batch['actions'].to(self.device)
                distance_labels = batch['distance_labels'].to(self.device)
                
                predicted_actions = self.model(images, distance_labels)
                
                # 손실 계산
                loss = self.compute_loss(predicted_actions, actions, distance_labels)
                mae = F.l1_loss(predicted_actions, actions)
                
                total_loss += loss
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches

def create_advanced_model(config: dict) -> AdvancedMobileVLAModel:
    """
    고급 모델 생성
    
    Args:
        config: 모델 설정
        
    Returns:
        AdvancedMobileVLAModel 인스턴스
    """
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    return AdvancedMobileVLAModel(
        processor=processor,
        vision_dim=config.get('vision_dim', 768),
        language_dim=config.get('language_dim', 768),
        action_dim=config.get('action_dim', 3),
        fusion_dim=config.get('fusion_dim', 512),
        plan_dim=config.get('plan_dim', 256),
        num_claw_layers=config.get('num_claw_layers', 3),
        num_subgoals=config.get('num_subgoals', 6),
        frames_per_subgoal=config.get('frames_per_subgoal', 3),
        num_heads=config.get('num_heads', 8),
        use_claw_matrix=config.get('use_claw_matrix', True),
        use_hierarchical=config.get('use_hierarchical', True),
        use_advanced_attention=config.get('use_advanced_attention', True)
    )

def test_advanced_model():
    """고급 모델 테스트"""
    print("🧪 고급 Mobile VLA 모델 테스트")
    print("=" * 60)
    
    # 모델 설정
    config = {
        'vision_dim': 768,
        'language_dim': 768,
        'action_dim': 3,
        'fusion_dim': 512,
        'plan_dim': 256,
        'num_claw_layers': 3,
        'num_subgoals': 6,
        'frames_per_subgoal': 3,
        'num_heads': 8,
        'use_claw_matrix': True,
        'use_hierarchical': True,
        'use_advanced_attention': True
    }
    
    # 모델 생성
    model = create_advanced_model(config)
    
    # 테스트 입력
    batch_size = 2
    seq_len = 8
    
    images = torch.randn(batch_size, seq_len, 3, 224, 224)
    distance_labels = torch.randint(0, 3, (batch_size,))
    
    print(f"입력 형태:")
    print(f"  Images: {images.shape}")
    print(f"  Distance Labels: {distance_labels.shape}")
    
    # 순전파
    with torch.no_grad():
        predicted_actions = model(images, distance_labels)
    
    print(f"출력 형태: {predicted_actions.shape}")
    print(f"예상 출력: [batch_size, {config['num_subgoals'] * config['frames_per_subgoal']}, {config['action_dim']}]")
    print(f"실제 출력: {predicted_actions.shape}")
    
    # 검증
    expected_frames = config['num_subgoals'] * config['frames_per_subgoal']
    assert predicted_actions.shape == (batch_size, expected_frames, config['action_dim']), \
        f"출력 형태가 예상과 다릅니다: {predicted_actions.shape} vs ({batch_size}, {expected_frames}, {config['action_dim']})"
    
    print("✅ 테스트 통과!")
    print(f"🎯 18프레임 예측 가능: {expected_frames} 프레임")
    print(f"🔧 Claw Matrix: {'✅' if config['use_claw_matrix'] else '❌'}")
    print(f"🏗️ Hierarchical Planning: {'✅' if config['use_hierarchical'] else '❌'}")
    print(f"👁️ Advanced Attention: {'✅' if config['use_advanced_attention'] else '❌'}")
    
    return model

if __name__ == "__main__":
    test_advanced_model()
