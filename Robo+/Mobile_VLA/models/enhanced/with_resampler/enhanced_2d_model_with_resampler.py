"""
🚀 Enhanced 2D Action Model with RoboVLMs Advanced Features
RoboVLMs의 최신 기능들을 모두 포함한 향상된 2D 액션 모델

추가된 기능:
- Vision Resampler (PerceiverResampler)
- CLIP Normalization
- State Embedding (선택적)
- Enhanced Attention Mechanisms
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json
from einops import rearrange, repeat
from einops_exts import rearrange_many

from transformers import AutoModel
from PIL import Image

def exists(val):
    return val is not None

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttention(nn.Module):
    """Perceiver Attention for Vision Resampling"""
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    """Vision Resampler for reducing image token count"""
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)

class CLIPNormalizationHead(nn.Module):
    """CLIP Normalization for better feature alignment"""
    def __init__(self, hidden_size, clip_dim=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.clip_dim = clip_dim
        
        # Projection to CLIP space
        self.clip_projection = nn.Linear(hidden_size, clip_dim)
        
        # CLIP text encoder (frozen)
        try:
            import open_clip
            self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
            # Freeze CLIP
            for param in self.clip_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: open_clip not available. CLIP normalization disabled.")
            self.clip_model = None
            self.clip_tokenizer = None

    def forward(self, features, raw_text):
        """Compute CLIP normalization loss"""
        if self.clip_model is None:
            return torch.tensor(0.0, device=features.device)
        
        # Project features to CLIP space
        projected_features = self.clip_projection(features)
        projected_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
        
        # Encode text with CLIP
        text_tokens = self.clip_tokenizer(raw_text).to(features.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity loss
        similarity = torch.cosine_similarity(projected_features, text_features, dim=-1)
        loss = 1.0 - similarity.mean()
        
        return loss

class StateEmbedding(nn.Module):
    """State embedding for robot state information"""
    def __init__(self, state_dim=7, hidden_size=512):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        # Arm state embedding (6D)
        self.embed_arm_state = nn.Linear(state_dim - 1, hidden_size)
        
        # Gripper state embedding (1D binary)
        self.embed_gripper_state = nn.Embedding(2, hidden_size)
        
        # Combined state embedding
        self.embed_state = nn.Linear(2 * hidden_size, hidden_size)
        
    def forward(self, state):
        """
        Args:
            state: [batch_size, seq_len, state_dim]
        Returns:
            state_embeddings: [batch_size, seq_len, hidden_size]
        """
        arm_state = state[..., :6]  # First 6 dimensions
        gripper_state = state[..., -1].long()  # Last dimension as binary
        
        arm_embeddings = self.embed_arm_state(arm_state)
        gripper_embeddings = self.embed_gripper_state(gripper_state)
        
        # Combine embeddings
        combined = torch.cat([arm_embeddings, gripper_embeddings], dim=-1)
        state_embeddings = self.embed_state(combined)
        
        return state_embeddings

class Enhanced2DActionDataset(Dataset):
    """Enhanced 2D Action Dataset with state information"""
    
    def __init__(self, data_path, processor, split='train', frame_selection='random', 
                 use_state=False, use_vision_resampler=False):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.frame_selection = frame_selection
        self.use_state = use_state
        self.use_vision_resampler = use_vision_resampler
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} Enhanced 2D 액션 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        print(f"   - 프레임 선택: {frame_selection}")
        print(f"   - Z축 제외: True")
        print(f"   - 상태 정보 사용: {use_state}")
        print(f"   - 비전 리샘플러: {use_vision_resampler}")
    
    def _load_episodes(self):
        """에피소드 로드 (Z축 제외, 2D 액션만)"""
        if os.path.isdir(self.data_path):
            h5_files = list(Path(self.data_path).glob("*.h5"))
        else:
            h5_files = [self.data_path]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]  # [18, H, W, 3]
                        actions = f['actions'][:]  # [18, 3]
                        
                        # 상태 정보가 있는 경우 로드
                        robot_state = None
                        if self.use_state and 'robot_state' in f:
                            robot_state = f['robot_state'][:]  # [18, 15]
                        
                        # 첫 프레임 제외 (프레임 1-17만 사용)
                        valid_frames = list(range(1, 18))  # 1, 2, 3, ..., 17
                        
                        if self.frame_selection == 'random':
                            frame_idx = np.random.choice(valid_frames)
                        elif self.frame_selection == 'middle':
                            frame_idx = valid_frames[len(valid_frames)//2]  # 9
                        elif self.frame_selection == 'all':
                            for frame_idx in valid_frames:
                                single_image = images[frame_idx]  # [H, W, 3]
                                single_action = actions[frame_idx]  # [3]
                                
                                # 2D 액션으로 변환 (Z축 제외)
                                action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                                
                                episode_data = {
                                    'image': single_image,
                                    'action': action_2d,  # 2D 액션
                                    'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                                    'frame_idx': frame_idx,
                                    'original_file': h5_file.name
                                }
                                
                                if robot_state is not None:
                                    episode_data['robot_state'] = robot_state[frame_idx]
                                
                                self.episodes.append(episode_data)
                            continue
                        
                        # 단일 프레임 선택
                        single_image = images[frame_idx]  # [H, W, 3]
                        single_action = actions[frame_idx]  # [3]
                        
                        # 2D 액션으로 변환 (Z축 제외)
                        action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                        
                        episode_data = {
                            'image': single_image,
                            'action': action_2d,  # 2D 액션
                            'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                            'frame_idx': frame_idx,
                            'original_file': h5_file.name
                        }
                        
                        if robot_state is not None:
                            episode_data['robot_state'] = robot_state[frame_idx]
                        
                        self.episodes.append(episode_data)
                        
            except Exception as e:
                print(f"❌ {h5_file} 로드 실패: {e}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지: [H, W, 3] → [3, H, W] (PyTorch 형식)
        image = episode['image']  # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        # 이미지 전처리
        image = Image.fromarray(image.transpose(1, 2, 0))
        inputs = self.processor(images=image, return_tensors="pt")
        image_tensor = inputs['pixel_values'].squeeze(0)  # [3, H, W]
        
        # 액션
        action = torch.FloatTensor(episode['action'])  # [2]
        
        # 텍스트 (더미)
        text = "로봇을 제어하세요"
        
        result = {
            'image': image_tensor,
            'action': action,
            'text': text,
            'episode_id': episode['episode_id']
        }
        
        # 상태 정보 추가
        if 'robot_state' in episode:
            result['robot_state'] = torch.FloatTensor(episode['robot_state'])
        
        return result

class Enhanced2DActionModel(nn.Module):
    """Enhanced 2D Action Model with RoboVLMs features"""
    
    def __init__(self, processor, vision_dim=1024, language_dim=1024, action_dim=2, 
                 hidden_dim=512, dropout=0.2, use_claw_matrix=True, use_hierarchical=True, 
                 use_advanced_attention=True, use_vision_resampler=False, use_clip_norm=False, 
                 use_state=False):
        super().__init__()
        
        # 기본 설정
        self.processor = processor
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.action_dim = action_dim  # 2D 액션
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 기능 플래그
        self.use_claw_matrix = use_claw_matrix
        self.use_hierarchical = use_hierarchical
        self.use_advanced_attention = use_advanced_attention
        self.use_vision_resampler = use_vision_resampler
        self.use_clip_norm = use_clip_norm
        self.use_state = use_state
        
        # Kosmos2 모델 로드
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos_processor = processor
        
        # 디바이스 설정
        self.device = next(self.kosmos.parameters()).device
        self.kosmos = self.kosmos.to(self.device)
        self.kosmos.eval()
        
        # 동적 어댑터들
        self.language_adapter = None
        self.fusion_adapter = None
        
        # 특징 추출기
        self.feature_adapter = nn.Linear(vision_dim, hidden_dim)
        
        # 정규화 및 드롭아웃
        self.layer_norm_vision = nn.LayerNorm(hidden_dim)
        self.layer_norm_language = nn.LayerNorm(language_dim)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        self.dropout_vision = nn.Dropout(dropout)
        self.dropout_language = nn.Dropout(dropout)
        self.dropout_fusion = nn.Dropout(dropout)
        
        # Vision Resampler
        if self.use_vision_resampler:
            self.vision_resampler = PerceiverResampler(
                dim=hidden_dim,
                depth=6,
                dim_head=64,
                heads=8,
                num_latents=64
            )
        else:
            self.vision_resampler = None
        
        # State Embedding
        if self.use_state:
            self.state_embedding = StateEmbedding(state_dim=15, hidden_size=hidden_dim)
        else:
            self.state_embedding = None
        
        # CLIP Normalization
        if self.use_clip_norm:
            self.clip_norm_head = CLIPNormalizationHead(hidden_dim)
        else:
            self.clip_norm_head = None
        
        # 고급 기능들
        if self.use_claw_matrix:
            self.claw_matrix = EnhancedClawMatrixFusion(
                vision_dim, language_dim, action_dim, hidden_dim, dropout
            )
        
        if self.use_hierarchical:
            self.hierarchical_planner = EnhancedHierarchicalPlanner(
                hidden_dim, action_dim, dropout
            )
        
        if self.use_advanced_attention:
            self.advanced_attention = EnhancedAdvancedAttention(
                hidden_dim, dropout
            )
        
        # 2D 액션 예측 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)  # 2D 액션만
        )
        
        print(f"✅ Enhanced 2D Action Model 초기화 완료")
        print(f"   - 액션 차원: {action_dim}D")
        print(f"   - 비전 리샘플러: {use_vision_resampler}")
        print(f"   - CLIP 정규화: {use_clip_norm}")
        print(f"   - 상태 임베딩: {use_state}")
        print(f"   - Claw Matrix: {use_claw_matrix}")
        print(f"   - Hierarchical Planning: {use_hierarchical}")
        print(f"   - Advanced Attention: {use_advanced_attention}")
    
    def extract_vision_features(self, images):
        """비전 특징 추출 (리샘플러 포함)"""
        batch_size = images.shape[0]
        
        # Kosmos2 프로세서로 입력 준비
        inputs = self.kosmos_processor(
            images=images, 
            return_tensors="pt",
            padding=True
        )
        
        # 모든 입력을 모델 디바이스로 이동
        inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Kosmos2 vision 모델 사용
        with torch.no_grad():
            if 'pixel_values' in inputs:
                vision_outputs = self.kosmos.vision_model(inputs['pixel_values'])
                vision_features = vision_outputs.pooler_output  # [batch_size, 1024]
            else:
                vision_features = torch.zeros(batch_size, 1024).to(self.kosmos.device)
        
        # 차원 조정
        vision_features = self.feature_adapter(vision_features)
        
        # Vision Resampler 적용
        if self.use_vision_resampler and self.vision_resampler is not None:
            # Resampler 입력 형태로 변환: [batch_size, 1, 1, num_tokens, hidden_dim]
            vision_features = vision_features.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            vision_features = self.vision_resampler(vision_features)
            vision_features = vision_features.squeeze(1).squeeze(1)  # [batch_size, num_latents, hidden_dim]
            vision_features = vision_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 정규화 및 드롭아웃
        vision_features = self.layer_norm_vision(vision_features)
        vision_features = self.dropout_vision(vision_features)
        
        return vision_features
    
    def extract_language_features(self, text: str, batch_size: int = 1):
        """언어 특징 추출"""
        with torch.no_grad():
            inputs = self.kosmos_processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.kosmos.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Kosmos2 텍스트 모델 사용
            language_outputs = self.kosmos.text_model(**inputs)
            language_features = language_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
        
        # 배치 차원 확장
        language_features = language_features.expand(batch_size, -1)
        
        # 차원 조정 (동적 어댑터 생성)
        if language_features.shape[-1] != self.language_dim:
            if self.language_adapter is None:
                self.language_adapter = nn.Linear(
                    language_features.shape[-1], 
                    self.language_dim
                ).to(language_features.device)
            language_features = self.language_adapter(language_features)
        
        # 정규화 및 드롭아웃
        language_features = self.layer_norm_language(language_features)
        language_features = self.dropout_language(language_features)
        
        return language_features
    
    def forward(self, single_image: torch.Tensor, text: str, robot_state: torch.Tensor = None):
        """향상된 2D 액션 예측"""
        batch_size = single_image.shape[0]
        
        # 특징 추출
        vision_features = self.extract_vision_features(single_image)
        language_features = self.extract_language_features(text, batch_size)
        
        # 상태 임베딩 추가
        if self.use_state and robot_state is not None and self.state_embedding is not None:
            state_features = self.state_embedding(robot_state)
            # 상태 특징을 비전 특징에 추가
            vision_features = vision_features + state_features.mean(dim=1)
        
        # 기본 융합
        fused_features = torch.cat([vision_features, language_features], dim=-1)
        
        # 고급 기능 적용
        if self.use_claw_matrix and hasattr(self, 'claw_matrix'):
            # 2D 액션용 더미 액션 생성
            dummy_actions = torch.zeros(batch_size, self.hidden_dim).to(vision_features.device)
            fused_features = self.claw_matrix(vision_features, language_features, dummy_actions)
        else:
            # Claw Matrix를 사용하지 않는 경우 기본 융합
            if fused_features.shape[-1] != self.hidden_dim:
                if not hasattr(self, 'fusion_adapter'):
                    self.fusion_adapter = nn.Linear(fused_features.shape[-1], self.hidden_dim).to(fused_features.device)
                fused_features = self.fusion_adapter(fused_features)
        
        # 정규화 및 드롭아웃
        fused_features = self.layer_norm_fusion(fused_features)
        fused_features = self.dropout_fusion(fused_features)
        
        # Advanced Attention 적용
        if self.use_advanced_attention and hasattr(self, 'advanced_attention'):
            fused_features = self.advanced_attention(fused_features)
        
        # Hierarchical Planning 적용
        if self.use_hierarchical and hasattr(self, 'hierarchical_planner'):
            fused_features = self.hierarchical_planner(fused_features)
        
        # CLIP 정규화 손실 계산
        clip_loss = 0.0
        if self.use_clip_norm and self.clip_norm_head is not None:
            clip_loss = self.clip_norm_head(fused_features, text)
        
        # 2D 액션 예측 (Z축 제외)
        actions_2d = self.action_head(fused_features)  # [batch_size, 2]
        
        return actions_2d, clip_loss

class EnhancedClawMatrixFusion(nn.Module):
    """향상된 Claw Matrix 융합"""
    
    def __init__(self, vision_dim, language_dim, action_dim, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 프로젝션 레이어들
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)  # 더미 액션용
        
        # Cross-attention 메커니즘들
        self.vl_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.la_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.av_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # 정규화 레이어들
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm4 = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_features, language_features, dummy_actions):
        """향상된 Claw Matrix 융합"""
        # 프로젝션
        vision_proj = self.vision_proj(vision_features)
        language_proj = self.language_proj(language_features)
        action_proj = self.action_proj(dummy_actions)
        
        # Vision-Language Cross-attention
        vl_out, _ = self.vl_cross_attention(
            query=vision_proj.unsqueeze(1),
            key=language_proj.unsqueeze(1),
            value=language_proj.unsqueeze(1)
        )
        vl_out = self.norm1(vl_out.squeeze(1) + vision_proj)
        
        # Language-Action Cross-attention
        la_out, _ = self.la_cross_attention(
            query=language_proj.unsqueeze(1),
            key=action_proj.unsqueeze(1),
            value=action_proj.unsqueeze(1)
        )
        la_out = self.norm2(la_out.squeeze(1) + language_proj)
        
        # Action-Vision Cross-attention
        av_out, _ = self.av_cross_attention(
            query=action_proj.unsqueeze(1),
            key=vl_out.unsqueeze(1),
            value=vl_out.unsqueeze(1)
        )
        av_out = self.norm3(av_out.squeeze(1) + action_proj)
        
        # 최종 융합
        fused = vl_out + la_out + av_out
        
        # Feed-forward
        ffn_out = self.ffn(fused)
        fused = self.norm4(fused + ffn_out)
        
        return fused

class EnhancedHierarchicalPlanner(nn.Module):
    """향상된 Hierarchical Planning"""
    
    def __init__(self, hidden_dim, action_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 목표 분해기
        self.goal_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 액션 계획기
        self.action_planner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 정규화
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, features):
        """향상된 Hierarchical Planning"""
        # 목표 분해
        decomposed = self.goal_decomposer(features)
        
        # 액션 계획
        planned_actions = self.action_planner(decomposed)
        
        # 특징 업데이트
        updated_features = self.norm(features + decomposed)
        
        return updated_features

class EnhancedAdvancedAttention(nn.Module):
    """향상된 Advanced Attention"""
    
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # 정규화 레이어들
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm4 = nn.LayerNorm(hidden_dim)
    
    def forward(self, features):
        """향상된 Advanced Attention"""
        # Self Attention
        attn_out, _ = self.self_attention(features, features, features)
        features = self.norm1(features + attn_out)
        
        # Temporal Attention (시퀀스가 있는 경우)
        if features.dim() == 3:
            temp_out, _ = self.temporal_attention(features, features, features)
            features = self.norm2(features + temp_out)
        
        # Spatial Attention (공간 정보가 있는 경우)
        if features.dim() == 3:
            spatial_out, _ = self.spatial_attention(features, features, features)
            features = self.norm3(features + spatial_out)
        
        # Feedforward
        ffn_out = self.ffn(features)
        features = self.norm4(features + ffn_out)
        
        return features

def create_enhanced_data_loaders(data_path, processor, batch_size=4, train_split=0.8, 
                                frame_selection='random', use_state=False, use_vision_resampler=False):
    """향상된 데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = Enhanced2DActionDataset(
        data_path, processor, 'full', frame_selection, use_state, use_vision_resampler
    )
    
    # 훈련/검증 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"📊 향상된 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 액션 차원: 2D (Z축 제외)")
    print(f"   - 상태 정보: {use_state}")
    print(f"   - 비전 리샘플러: {use_vision_resampler}")
    
    return train_loader, val_loader

def train_enhanced_2d_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda',
    clip_loss_weight=0.1
):
    """향상된 2D 액션 모델 훈련"""
    
    model = model.to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # 손실 함수
    criterion = nn.MSELoss()
    
    # 조기 종료
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"🚀 향상된 2D 액션 모델 훈련 시작")
    print(f"   - 에포크: {num_epochs}")
    print(f"   - 학습률: {learning_rate}")
    print(f"   - CLIP 손실 가중치: {clip_loss_weight}")
    
    for epoch in range(num_epochs):
        # 훈련
        model.train()
        train_loss = 0
        train_clip_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            
            # 상태 정보
            robot_states = None
            if 'robot_state' in batch:
                robot_states = batch['robot_state'].to(device)
            
            # 순전파
            predictions, clip_loss = model(images, texts, robot_states)
            
            # 손실 계산
            action_loss = criterion(predictions, actions)
            total_loss = action_loss + clip_loss_weight * clip_loss
            
            # 역전파
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += action_loss.item()
            train_clip_loss += clip_loss.item()
            num_batches += 1
        
        # 검증
        model.eval()
        val_loss = 0
        val_clip_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = batch['image'].to(device)
                actions = batch['action'].to(device)
                texts = batch['text']
                
                # 상태 정보
                robot_states = None
                if 'robot_state' in batch:
                    robot_states = batch['robot_state'].to(device)
                
                # 순전파
                predictions, clip_loss = model(images, texts, robot_states)
                
                # 손실 계산
                action_loss = criterion(predictions, actions)
                
                val_loss += action_loss.item()
                val_clip_loss += clip_loss.item()
                val_batches += 1
        
        # 평균 손실 계산
        avg_train_loss = train_loss / num_batches
        avg_train_clip_loss = train_clip_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_val_clip_loss = val_clip_loss / val_batches
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Action Loss: {avg_train_loss:.6f}, CLIP Loss: {avg_train_clip_loss:.6f}")
        print(f"  Val   - Action Loss: {avg_val_loss:.6f}, CLIP Loss: {avg_val_clip_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 조기 종료 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'use_vision_resampler': model.use_vision_resampler,
                    'use_clip_norm': model.use_clip_norm,
                    'use_state': model.use_state,
                    'action_dim': model.action_dim
                }
            }, f'enhanced_2d_model_epoch_{epoch+1}.pth')
            print(f"  ✅ 모델 저장됨 (Val Loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  🛑 조기 종료 (Patience: {early_stopping_patience})")
                break
    
    print(f"🎉 향상된 2D 액션 모델 훈련 완료!")
    print(f"   최고 검증 손실: {best_val_loss:.6f}")
    
    return model

if __name__ == "__main__":
    # 설정
    data_path = "path/to/your/h5/data"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 향상된 모델 생성
    model = Enhanced2DActionModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=2,
        hidden_dim=512,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        use_vision_resampler=True,  # 비전 리샘플러 활성화
        use_clip_norm=True,         # CLIP 정규화 활성화
        use_state=False             # 상태 정보는 선택적
    )
    
    # 데이터 로더 생성
    train_loader, val_loader = create_enhanced_data_loaders(
        data_path=data_path,
        processor=processor,
        batch_size=4,
        train_split=0.8,
        frame_selection='random',
        use_state=False,
        use_vision_resampler=True
    )
    
    # 모델 훈련
    trained_model = train_enhanced_2d_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        learning_rate=1e-4,
        weight_decay=1e-4,
        early_stopping_patience=5,
        device=device,
        clip_loss_weight=0.1
    )
    
    print("✅ 향상된 2D 액션 모델 훈련 완료!")
