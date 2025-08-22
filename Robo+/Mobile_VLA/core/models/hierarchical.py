#!/usr/bin/env python3
"""
🏗️ Hierarchical Planning Implementation
18프레임 예측을 위한 계층적 계획 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

class GoalDecomposition(nn.Module):
    """
    목표 분해: 장기 목표를 단기 목표로 분해
    """
    def __init__(self, 
                 input_dim: int = 768,
                 goal_dim: int = 256,
                 num_subgoals: int = 6,  # 18프레임을 6개 구간으로 분할
                 num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.num_subgoals = num_subgoals
        self.num_heads = num_heads
        
        # 목표 인코더
        self.goal_encoder = nn.Sequential(
            nn.Linear(input_dim, goal_dim * 2),
            nn.LayerNorm(goal_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(goal_dim * 2, goal_dim)
        )
        
        # 하위 목표 생성기
        self.subgoal_generator = nn.MultiheadAttention(
            embed_dim=goal_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 하위 목표 프로젝션
        self.subgoal_proj = nn.Linear(goal_dim, goal_dim)
        
        # 시간적 위치 인코딩
        self.temporal_embedding = nn.Embedding(num_subgoals, goal_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        목표 분해
        
        Args:
            features: [batch_size, seq_len, input_dim]
            
        Returns:
            subgoals: [batch_size, num_subgoals, goal_dim]
        """
        batch_size = features.size(0)
        
        # 목표 인코딩
        goal_features = self.goal_encoder(features.mean(dim=1))  # [batch_size, goal_dim]
        goal_features = goal_features.unsqueeze(1).expand(-1, self.num_subgoals, -1)
        
        # 시간적 위치 인코딩
        temporal_pos = torch.arange(self.num_subgoals, device=features.device)
        temporal_emb = self.temporal_embedding(temporal_pos).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 하위 목표 생성
        subgoals, _ = self.subgoal_generator(
            query=goal_features + temporal_emb,
            key=goal_features + temporal_emb,
            value=goal_features + temporal_emb
        )
        
        subgoals = self.subgoal_proj(subgoals)
        
        return subgoals

class HierarchicalPlanner(nn.Module):
    """
    계층적 계획자: 장기 계획과 단기 실행을 분리
    """
    def __init__(self, 
                 input_dim: int = 768,
                 plan_dim: int = 256,
                 action_dim: int = 3,
                 num_subgoals: int = 6,
                 frames_per_subgoal: int = 3,  # 각 하위 목표당 3프레임
                 num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.plan_dim = plan_dim
        self.action_dim = action_dim
        self.num_subgoals = num_subgoals
        self.frames_per_subgoal = frames_per_subgoal
        self.total_frames = num_subgoals * frames_per_subgoal  # 18프레임
        
        # 목표 분해
        self.goal_decomposition = GoalDecomposition(
            input_dim=input_dim,
            goal_dim=plan_dim,
            num_subgoals=num_subgoals,
            num_heads=num_heads
        )
        
        # 장기 계획자 (High-level planner)
        self.high_level_planner = nn.TransformerEncoderLayer(
            d_model=plan_dim,
            nhead=num_heads,
            dim_feedforward=plan_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # 단기 실행자 (Low-level executor)
        self.low_level_executor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(plan_dim + input_dim, plan_dim),
                nn.LayerNorm(plan_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(plan_dim, plan_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(plan_dim // 2, action_dim)
            ) for _ in range(frames_per_subgoal)
        ])
        
        # 계획-실행 연결
        self.plan_execution_fusion = nn.MultiheadAttention(
            embed_dim=plan_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 출력 정규화
        self.output_norm = nn.LayerNorm(action_dim)
        
    def forward(self, 
                features: torch.Tensor,
                current_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        계층적 계획 및 실행
        
        Args:
            features: [batch_size, seq_len, input_dim]
            current_actions: [batch_size, seq_len, action_dim] (선택사항)
            
        Returns:
            predicted_actions: [batch_size, total_frames, action_dim]
        """
        batch_size = features.size(0)
        
        # 1. 목표 분해
        subgoals = self.goal_decomposition(features)  # [batch_size, num_subgoals, plan_dim]
        
        # 2. 장기 계획 수립
        high_level_plan = self.high_level_planner(subgoals)  # [batch_size, num_subgoals, plan_dim]
        
        # 3. 단기 실행 계획
        all_actions = []
        
        for subgoal_idx in range(self.num_subgoals):
            subgoal = high_level_plan[:, subgoal_idx:subgoal_idx+1, :]  # [batch_size, 1, plan_dim]
            
            # 현재 상태와 하위 목표 융합
            current_state = features.mean(dim=1, keepdim=True)  # [batch_size, 1, input_dim]
            plan_state = torch.cat([subgoal, current_state], dim=-1)  # [batch_size, 1, plan_dim + input_dim]
            
            # 각 프레임별 액션 생성
            subgoal_actions = []
            for frame_idx in range(self.frames_per_subgoal):
                executor = self.low_level_executor[frame_idx]
                action = executor(plan_state)  # [batch_size, 1, action_dim]
                subgoal_actions.append(action)
            
            subgoal_actions = torch.cat(subgoal_actions, dim=1)  # [batch_size, frames_per_subgoal, action_dim]
            all_actions.append(subgoal_actions)
        
        # 모든 액션 결합
        predicted_actions = torch.cat(all_actions, dim=1)  # [batch_size, total_frames, action_dim]
        
        # 출력 정규화
        predicted_actions = self.output_norm(predicted_actions)
        
        return predicted_actions

class HierarchicalPlanningModel(nn.Module):
    """
    완전한 계층적 계획 모델
    """
    def __init__(self, 
                 vision_dim: int = 768,
                 language_dim: int = 768,
                 action_dim: int = 3,
                 plan_dim: int = 256,
                 num_subgoals: int = 6,
                 frames_per_subgoal: int = 3,
                 num_heads: int = 8):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.plan_dim = plan_dim
        self.num_subgoals = num_subgoals
        self.frames_per_subgoal = frames_per_subgoal
        self.total_frames = num_subgoals * frames_per_subgoal
        
        # 특징 융합
        self.feature_fusion = nn.Sequential(
            nn.Linear(vision_dim + language_dim, plan_dim * 2),
            nn.LayerNorm(plan_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(plan_dim * 2, plan_dim)
        )
        
        # 계층적 계획자
        self.hierarchical_planner = HierarchicalPlanner(
            input_dim=plan_dim,
            plan_dim=plan_dim,
            action_dim=action_dim,
            num_subgoals=num_subgoals,
            frames_per_subgoal=frames_per_subgoal,
            num_heads=num_heads
        )
        
        # 시간적 일관성 보장
        self.temporal_consistency = nn.LSTM(
            input_size=action_dim,
            hidden_size=action_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # 최종 액션 헤드
        self.final_action_head = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()  # 액션 범위 제한
        )
        
    def forward(self, 
                vision_features: torch.Tensor,
                language_features: torch.Tensor,
                current_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        계층적 계획 모델 순전파
        
        Args:
            vision_features: [batch_size, seq_len, vision_dim]
            language_features: [batch_size, seq_len, language_dim]
            current_actions: [batch_size, seq_len, action_dim] (선택사항)
            
        Returns:
            predicted_actions: [batch_size, total_frames, action_dim]
        """
        # 특징 융합
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        
        # 계층적 계획
        planned_actions = self.hierarchical_planner(fused_features, current_actions)
        
        # 시간적 일관성 보장
        lstm_out, _ = self.temporal_consistency(planned_actions)
        
        # 최종 액션 생성
        final_actions = self.final_action_head(lstm_out)
        
        return final_actions

def create_hierarchical_model(config: dict) -> HierarchicalPlanningModel:
    """
    계층적 계획 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
        
    Returns:
        HierarchicalPlanningModel 인스턴스
    """
    return HierarchicalPlanningModel(
        vision_dim=config.get('vision_dim', 768),
        language_dim=config.get('language_dim', 768),
        action_dim=config.get('action_dim', 3),
        plan_dim=config.get('plan_dim', 256),
        num_subgoals=config.get('num_subgoals', 6),
        frames_per_subgoal=config.get('frames_per_subgoal', 3),
        num_heads=config.get('num_heads', 8)
    )

def test_hierarchical_model():
    """계층적 모델 테스트"""
    print("🧪 계층적 계획 모델 테스트")
    print("=" * 50)
    
    # 모델 설정
    config = {
        'vision_dim': 768,
        'language_dim': 768,
        'action_dim': 3,
        'plan_dim': 256,
        'num_subgoals': 6,
        'frames_per_subgoal': 3,
        'num_heads': 8
    }
    
    # 모델 생성
    model = create_hierarchical_model(config)
    
    # 테스트 입력
    batch_size = 2
    seq_len = 8
    
    vision_features = torch.randn(batch_size, seq_len, config['vision_dim'])
    language_features = torch.randn(batch_size, seq_len, config['language_dim'])
    
    print(f"입력 형태:")
    print(f"  Vision: {vision_features.shape}")
    print(f"  Language: {language_features.shape}")
    
    # 순전파
    with torch.no_grad():
        predicted_actions = model(vision_features, language_features)
    
    print(f"출력 형태: {predicted_actions.shape}")
    print(f"예상 출력: [batch_size, {config['num_subgoals'] * config['frames_per_subgoal']}, {config['action_dim']}]")
    print(f"실제 출력: {predicted_actions.shape}")
    
    # 검증
    expected_frames = config['num_subgoals'] * config['frames_per_subgoal']
    assert predicted_actions.shape == (batch_size, expected_frames, config['action_dim']), \
        f"출력 형태가 예상과 다릅니다: {predicted_actions.shape} vs ({batch_size}, {expected_frames}, {config['action_dim']})"
    
    print("✅ 테스트 통과!")
    print(f"🎯 18프레임 예측 가능: {expected_frames} 프레임")
    
    return model

if __name__ == "__main__":
    test_hierarchical_model()
