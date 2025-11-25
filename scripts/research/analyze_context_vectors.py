#!/usr/bin/env python3
"""
Context Vector Analysis Script for Mobile-VLA
---------------------------------------------
This script validates whether the pre-trained RoboVLMs model (designed for 7DOF manipulators)
extracts meaningful context vectors from 2DOF mobile robot data.

It performs the following steps:
1. Loads the RoboPaligemma model using a standard config.
2. Loads Mobile-VLA H5 episodes.
3. Extracts visual features (context vectors) from the model's vision tower.
4. Reduces dimensionality using PCA and t-SNE.
5. Visualizes the vectors, colored by action categories (WASD), to check for clustering.

Usage:
    python3 scripts/research/analyze_context_vectors.py --data_dir mobile_vla_dataset --output_dir results/context_analysis
"""

import os
import sys
import json
import argparse
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add RoboVLMs to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
ROBOVLMS_ROOT = os.path.join(PROJECT_ROOT, "RoboVLMs")
sys.path.append(ROBOVLMS_ROOT)

try:
    from robovlms.model.backbone.robopaligemma import RoboPaligemma
    from robovlms.model.backbone.base_backbone import load_config
except ImportError as e:
    print(f"Error importing RoboVLMs: {e}")
    print(f"Please ensure {ROBOVLMS_ROOT} is in your PYTHONPATH or the directory structure is correct.")
    sys.exit(1)

def categorize_action(action):
    """
    Categorize 2D action (linear_x, linear_y) into WASD classes.
    """
    linear_x, linear_y = action[0], action[1]
    thresh = 0.1
    
    if abs(linear_x) < thresh and abs(linear_y) < thresh:
        return 'Stop', 0
    elif linear_x > thresh:
        if abs(linear_y) < thresh: return 'Forward (W)', 1
        elif linear_y > thresh: return 'Forward-Left (Q)', 2
        else: return 'Forward-Right (E)', 3
    elif linear_x < -thresh:
        if abs(linear_y) < thresh: return 'Backward (S)', 4
        elif linear_y > thresh: return 'Backward-Left (Z)', 5
        else: return 'Backward-Right (C)', 6
    else: # linear_x is small, but linear_y is large
        if linear_y > thresh: return 'Left (A)', 7
        else: return 'Right (D)', 8

def load_model(config_path, device):
    """Load RoboPaligemma model."""
    print(f"Loading config from {config_path}...")
    configs = load_config(config_path)
    
    # Ensure train_setup exists
    if "train_setup" not in configs:
        configs["train_setup"] = {
            "train_vision": False,
            "freeze_backbone": True,
            "bits": 16 if "cuda" in device else 32
        }

    print("Initializing RoboPaligemma model...")
    model = RoboPaligemma(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs.get("window_size", 8),
        use_hand_rgb=False,
        act_head_configs=configs.get("act_head", None),
        fwd_pred_next_n=configs.get("fwd_pred_next_n", 1),
    )
    
    model.to(device)
    model.eval()
    return model

def process_episode(h5_path, model, device, max_frames=50):
    """Process a single episode and extract features."""
    features = []
    labels = []
    label_names = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check for action chunks
            if 'action_chunks' in f:
                # Iterate through chunks
                chunks = list(f['action_chunks'].keys())
                # Subsample chunks if too many
                if len(chunks) > max_frames:
                    indices = np.linspace(0, len(chunks)-1, max_frames, dtype=int)
                    chunks = [chunks[i] for i in indices]
                
                for chunk_name in chunks:
                    chunk = f['action_chunks'][chunk_name]
                    
                    # Get images (take the last one in the window as current)
                    if 'images' in chunk:
                        imgs = chunk['images'][:] # [frames, H, W, C]
                        if len(imgs) > 0:
                            img = imgs[-1] # Current frame
                            
                            # Get future action (first one)
                            if 'future_actions' in chunk:
                                fut_acts = chunk['future_actions'][:]
                                if len(fut_acts) > 0:
                                    action = fut_acts[0] # [linear_x, linear_y, angular_z]
                                    cat_name, cat_id = categorize_action(action)
                                    
                                    # Preprocess image
                                    # Convert BGR to RGB if needed (assuming cv2 saved as BGR)
                                    # But h5 usually stores as RGB if from PIL, check collector.
                                    # Collector used cv2 (BGR) -> h5. So likely BGR.
                                    img_rgb = img[..., ::-1].copy() # BGR to RGB
                                    
                                    # Convert to PIL
                                    pil_img = Image.fromarray(img_rgb)
                                    
                                    # Process with model processor
                                    inputs = model.processor(images=pil_img, return_tensors="pt")
                                    pixel_values = inputs.pixel_values.to(device)
                                    if inputs.pixel_values.dtype != torch.float16 and "cuda" in device:
                                         pixel_values = pixel_values.to(torch.float16)

                                    # Extract features
                                    with torch.no_grad():
                                        # Use model_encode_images
                                        # It expects [batch, seq, C, H, W] or similar?
                                        # robopaligemma.py: image_outputs = self.model.vision_tower(images)
                                        # It likely expects pixel_values directly.
                                        
                                        # Check input shape expectation
                                        # If pixel_values is [1, 3, 224, 224], we might need to unsqueeze if model expects seq dim
                                        if pixel_values.ndim == 4:
                                            pixel_values = pixel_values.unsqueeze(1) # [B, 1, C, H, W]
                                            
                                        feat = model.model_encode_images(pixel_values)
                                        # feat shape: [B, Seq, Hidden]
                                        
                                        # Average over sequence if needed, or take last
                                        feat_vec = feat.mean(dim=1).squeeze().cpu().numpy()
                                        
                                        features.append(feat_vec)
                                        labels.append(cat_id)
                                        label_names.append(cat_name)

    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        
    return features, labels, label_names

def main():
    parser = argparse.ArgumentParser(description="Analyze Context Vectors")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to H5 dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/context_analysis", help="Output directory")
    parser.add_argument("--config_path", type=str, 
                        default=os.path.join(ROBOVLMS_ROOT, "configs/calvin_finetune/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json"),
                        help="Path to model config")
    parser.add_argument("--max_episodes", type=int, default=20, help="Max episodes to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    model = load_model(args.config_path, args.device)
    
    # Find H5 files
    h5_files = glob.glob(os.path.join(args.data_dir, "*.h5"))
    if not h5_files:
        print(f"No H5 files found in {args.data_dir}")
        return
        
    print(f"Found {len(h5_files)} episodes. Processing {min(len(h5_files), args.max_episodes)}...")
    
    all_features = []
    all_labels = []
    all_label_names = []
    
    for i, h5_path in enumerate(tqdm(h5_files[:args.max_episodes])):
        feats, labs, names = process_episode(h5_path, model, args.device)
        all_features.extend(feats)
        all_labels.extend(labs)
        all_label_names.extend(names)
        
    if not all_features:
        print("No features extracted.")
        return
        
    X = np.array(all_features)
    y = np.array(all_labels)
    y_names = np.array(all_label_names)
    
    print(f"Extracted features shape: {X.shape}")
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plotting
    unique_labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # PCA Plot
    for i, label in enumerate(unique_labels):
        mask = y == label
        name = y_names[mask][0]
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6)
    axes[0].set_title("PCA of Context Vectors")
    axes[0].legend()
    
    # t-SNE Plot
    for i, label in enumerate(unique_labels):
        mask = y == label
        name = y_names[mask][0]
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=name, alpha=0.6)
    axes[1].set_title("t-SNE of Context Vectors")
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "context_vector_analysis.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Save raw data
    np.savez(os.path.join(args.output_dir, "features.npz"), X=X, y=y, names=y_names)

if __name__ == "__main__":
    main()
