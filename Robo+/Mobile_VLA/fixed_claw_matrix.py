"""
🔧 완전히 수정된 Claw Matrix 구현
차원과 데이터타입 문제를 모두 해결
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedClawMatrixFusion(nn.Module):
    """수정된 Claw Matrix Fusion (차원 문제 해결)"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim  # 입력 차원 (vision_dim + language_dim)
        self.hidden_dim = hidden_dim  # 숨겨진 차원
        self.dropout = dropout
        
        # 입력을 hidden_dim으로 변환
        self.input_adapter = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head Self Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Vision-Language 융합
        self.vl_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Language-Action 융합
        self.la_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Action-Vision 융합
        self.av_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 정규화 레이어들
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)
        
        # 최종 융합
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, features):
        """
        입력: features [batch_size, input_dim]
        출력: fused_features [batch_size, hidden_dim]
        """
        batch_size = features.shape[0]
        
        # 1. 입력을 hidden_dim으로 변환
        adapted_features = self.input_adapter(features)  # [batch, hidden_dim]
        
        # 2. Self Attention
        features_expanded = adapted_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        self_attended, _ = self.self_attention(
            query=features_expanded,
            key=features_expanded,
            value=features_expanded
        )
        self_attended = self_attended.squeeze(1)  # [batch, hidden_dim]
        
        # Residual connection + norm
        features_self = self.norm1(adapted_features + self_attended)
        
        # 3. Vision-Language 융합
        vl_fused = self.vl_fusion(features_self)
        vl_output = self.norm1(features_self + vl_fused)
        
        # 4. Language-Action 융합  
        la_fused = self.la_fusion(vl_output)
        la_output = self.norm2(vl_output + la_fused)
        
        # 5. Action-Vision 융합
        av_fused = self.av_fusion(la_output)
        av_output = self.norm3(la_output + av_fused)
        
        # 6. 최종 융합 (3개 경로 결합)
        combined = torch.cat([vl_output, la_output, av_output], dim=-1)
        final_output = self.final_fusion(combined)
        
        # 최종 정규화
        final_output = self.norm_final(final_output)
        
        return final_output

class FixedHierarchicalPlanner(nn.Module):
    """수정된 Hierarchical Planning"""
    
    def __init__(self, hidden_dim, action_dim, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dropout = dropout
        
        # 고수준 목표 계획
        self.high_level_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 중간 수준 서브골 계획
        self.mid_level_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 저수준 액션 계획
        self.low_level_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # 계층별 융합
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 정규화
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, features):
        """계층적 계획 수행"""
        
        # 각 수준에서 계획 수행
        high_level = self.high_level_planner(features)
        mid_level = self.mid_level_planner(features)
        low_level = self.low_level_planner(features)
        
        # 계층별 결과 융합
        hierarchical_combined = torch.cat([high_level, mid_level, low_level], dim=-1)
        hierarchical_output = self.hierarchical_fusion(hierarchical_combined)
        
        # Residual connection + norm
        output = self.norm(features + hierarchical_output)
        
        return output

class FixedAdvancedAttention(nn.Module):
    """수정된 Advanced Attention"""
    
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Multi-head Self Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal Attention (시간적 의존성)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Spatial Attention (공간적 의존성)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 정규화
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 최종 융합
        self.attention_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, features):
        """고급 어텐션 수행"""
        
        # 입력 확장 (어텐션을 위해)
        features_expanded = features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self Attention
        self_attended, _ = self.self_attention(
            query=features_expanded,
            key=features_expanded, 
            value=features_expanded
        )
        self_attended = self_attended.squeeze(1)
        self_output = self.norm1(features + self_attended)
        
        # Temporal Attention
        temporal_attended, _ = self.temporal_attention(
            query=features_expanded,
            key=features_expanded,
            value=features_expanded
        )
        temporal_attended = temporal_attended.squeeze(1)
        temporal_output = self.norm2(features + temporal_attended)
        
        # Spatial Attention
        spatial_attended, _ = self.spatial_attention(
            query=features_expanded,
            key=features_expanded,
            value=features_expanded
        )
        spatial_attended = spatial_attended.squeeze(1)
        spatial_output = self.norm3(features + spatial_attended)
        
        # 모든 어텐션 결과 융합
        attention_combined = torch.cat([self_output, temporal_output, spatial_output], dim=-1)
        attention_fused = self.attention_fusion(attention_combined)
        
        # FFN 적용
        ffn_output = self.ffn(attention_fused)
        
        # 최종 residual connection
        final_output = features + attention_fused + ffn_output
        
        return final_output

def test_fixed_components():
    """수정된 컴포넌트들 테스트"""
    
    print("🧪 수정된 컴포넌트 테스트")
    print("=" * 40)
    
    batch_size = 8
    input_dim = 2048  # vision_dim + language_dim
    hidden_dim = 512
    
    # 테스트 입력
    test_input = torch.randn(batch_size, input_dim)
    print(f"📊 테스트 입력: {test_input.shape}")
    
    # 1. Fixed Claw Matrix 테스트
    print(f"\n1. Fixed Claw Matrix 테스트:")
    try:
        claw = FixedClawMatrixFusion(input_dim, hidden_dim)
        claw_output = claw(test_input)
        print(f"   ✅ 성공! 출력: {claw_output.shape}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    # 2. Fixed Hierarchical Planner 테스트
    print(f"\n2. Fixed Hierarchical Planner 테스트:")
    try:
        hierarchical = FixedHierarchicalPlanner(hidden_dim, 3)
        # hidden_dim 입력으로 변환
        hidden_input = torch.randn(batch_size, hidden_dim)
        hierarchical_output = hierarchical(hidden_input)
        print(f"   ✅ 성공! 출력: {hierarchical_output.shape}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    # 3. Fixed Advanced Attention 테스트
    print(f"\n3. Fixed Advanced Attention 테스트:")
    try:
        attention = FixedAdvancedAttention(hidden_dim)
        # hidden_dim 입력으로 변환
        hidden_input = torch.randn(batch_size, hidden_dim)
        attention_output = attention(hidden_input)
        print(f"   ✅ 성공! 출력: {attention_output.shape}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    # 4. 전체 파이프라인 테스트
    print(f"\n4. 전체 파이프라인 테스트:")
    try:
        claw = FixedClawMatrixFusion(input_dim, hidden_dim)
        hierarchical = FixedHierarchicalPlanner(hidden_dim, 3)
        attention = FixedAdvancedAttention(hidden_dim)
        
        # 순차 처리
        x = test_input
        x = claw(x)  # input_dim -> hidden_dim
        x = attention(x)  # hidden_dim -> hidden_dim  
        x = hierarchical(x)  # hidden_dim -> hidden_dim
        
        print(f"   ✅ 전체 파이프라인 성공!")
        print(f"   📊 최종 출력: {x.shape}")
        
    except Exception as e:
        print(f"   ❌ 전체 파이프라인 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_components()
