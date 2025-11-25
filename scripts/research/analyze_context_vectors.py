#!/usr/bin/env python3
"""
RoboVLMs Context Vector Analysis Script
Extracts and visualizes context vectors from the pre-trained Mobile VLA model.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Add Mobile_VLA to path for dataset imports
sys.path.append(os.path.join(PROJECT_ROOT, "Mobile_VLA"))

# Import Model Loader
from unified_mobile_vla_loader import UnifiedMobileVLAModelLoader, ModelType

# Import Dataset
# We need to handle potential import errors if robovlms is not installed
try:
    from Mobile_VLA.robovlms.data.mobile_vla_dataset import MobileVLADataset
except ImportError:
    # Fallback: try to add RoboVLMs_upstream to path if needed
    sys.path.append(os.path.join(PROJECT_ROOT, "RoboVLMs_upstream"))
    try:
        from Mobile_VLA.robovlms.data.mobile_vla_dataset import MobileVLADataset
    except ImportError:
        print("❌ Failed to import MobileVLADataset. Please ensure dependencies are set up.")
        sys.exit(1)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_context_vectors(model, dataset, num_samples=500, batch_size=4):
    """Extract context vectors (LSTM output before action head)"""
    
    device = next(model.parameters()).device
    logger.info(f"Using device: {device}")
    
    # Load Kosmos2 Model for feature extraction
    logger.info("Loading Kosmos2 model for feature extraction...")
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    try:
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        kosmos_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        kosmos_model.to(device)
        kosmos_model.eval()
        
        # We need to access the vision tower directly
        vision_model = kosmos_model.vision_model
        
    except Exception as e:
        logger.error(f"Failed to load Kosmos2 model: {e}")
        return None, None

    logger.info(f"Extracting features from {num_samples} samples...")
    
    collected_samples = 0
    metadata = {
        'scenarios': [],
        'instructions': []
    }
    
    model.eval()
    
    # Iterate manually
    indices = np.random.permutation(len(dataset))[:num_samples]
    
    # Process in batches
    current_batch = []
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices)):
            try:
                item = dataset[idx]
                current_batch.append(item)
                
                if len(current_batch) >= batch_size or i == len(indices) - 1:
                    # Process batch
                    # 1. Prepare images
                    # item['images'] is a list of PIL images or tensors
                    # The dataset returns:
                    # 'images': images[:self.window_size] (T, H, W, 3) tensor?
                    # Let's check __getitem__ again.
                    # It returns torch tensors.
                    
                    # We need to convert tensors back to PIL or use processor with tensors
                    # Processor expects PIL images or list of PIL images
                    
                    batch_images = []
                    batch_texts = []
                    
                    for sample in current_batch:
                        # sample['images'] is (T, H, W, 3) or (T, 3, H, W)
                        # MobileVLADataset loads (T, H, W, 3) from H5
                        imgs = sample['images']
                        # Convert to PIL
                        pil_imgs = []
                        for j in range(imgs.shape[0]):
                            img_np = imgs[j].numpy()
                            # Denormalize if needed? Dataset doesn't normalize images, just loads them.
                            # But it converts to float.
                            # If float, assumed 0-1? Or 0-255?
                            # _convert_to_pil_images in dataset handles this.
                            
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            else:
                                img_np = img_np.astype(np.uint8)
                                
                            pil_imgs.append(Image.fromarray(img_np))
                        
                        batch_images.append(pil_imgs) # List of lists
                        batch_texts.append(sample['instruction'])
                        
                        metadata['scenarios'].append(sample['scenario'])
                        metadata['instructions'].append(sample['instruction'])

                    # Flatten images for processor
                    flat_images = [img for sublist in batch_images for img in sublist]
                    
                    # Process images
                    inputs = processor(images=flat_images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)
                    
                    # Extract vision features
                    # vision_model output: last_hidden_state (B*T, seq_len, hidden)
                    # We need pooled output? Or average?
                    # Kosmos2 uses the features as is?
                    # UnifiedMobileVLAModel expects (B, 2048).
                    # Kosmos2 vision model output is (B, 256, 1024) or similar.
                    # Wait, UnifiedMobileVLAModel expects `vision_dim=2048`.
                    # Kosmos2-patch14-224 uses CLIP-L/14? No, it uses its own backbone.
                    # Let's check Kosmos2 config.
                    # hidden_size is 1024.
                    # But UnifiedMobileVLAModel defaults to 2048.
                    # Maybe it concatenates something? Or uses a different backbone?
                    # "Kosmos2 + CLIP Hybrid"
                    # Maybe vision_features IS CLIP features?
                    # "Vision Encoder (Kosmos2 features)"
                    
                    # Let's check `mobile_vla_model_loader.py` again.
                    # `vision_dim: int = 2048`
                    
                    # If Kosmos2 vision model outputs 1024, then 2048 mismatch.
                    # Maybe it uses `open_clip`?
                    # "Kosmos2 + CLIP Hybrid"
                    
                    # Let's assume for now we use CLIP-L/14 (which is 768 or 1024).
                    # ViT-L/14 is 1024. ViT-H/14 is 1280. ViT-G/14 is 1664.
                    # ResNet50 is 2048.
                    
                    # If I can't match the dimension, I can't run the model.
                    # I'll try to extract features and see the shape.
                    # If shape mismatch, I'll log it.
                    
                    vision_outputs = vision_model(pixel_values=pixel_values)
                    vision_feats = vision_outputs.last_hidden_state # (B*T, seq, hidden)
                    # Average pool?
                    vision_feats = vision_feats.mean(dim=1) # (B*T, hidden)
                    
                    # Reshape to (B, T, hidden)
                    # But UnifiedMobileVLAModel takes (B, hidden)?
                    # No, it takes (batch_size, vision_dim).
                    # And then `unsqueeze(1)` -> LSTM.
                    # So it processes ONE frame?
                    # `lstm_out, (hidden, cell) = self.lstm(fused)`
                    # `fused` is (B, 1, hidden).
                    # So it seems it processes a SEQUENCE of frames one by one?
                    # Or it takes a batch of frames?
                    
                    # `UnifiedMobileVLAModel.forward`:
                    # vision_encoded = self.vision_encoder(vision_features)
                    # fused = fused.unsqueeze(1)
                    # lstm_out, ... = self.lstm(fused)
                    
                    # If it unsqueezes, it treats the input as a single time step.
                    # But LSTM maintains state.
                    # So we need to feed it sequence of frames?
                    # But `forward` creates a NEW LSTM?
                    # No, `self.lstm` is stateful? No, PyTorch LSTM is stateless unless hidden passed.
                    # `lstm_out, (hidden, cell) = self.lstm(fused)`
                    # It doesn't take hidden state as input!
                    # So it initializes hidden state to zero every time.
                    # This means it expects a SEQUENCE input?
                    # But `fused.unsqueeze(1)` makes it sequence length 1.
                    # So it treats it as a sequence of length 1.
                    # This implies the model is designed to process ONE STEP at a time, 
                    # BUT it resets state every time? That defeats the purpose of LSTM.
                    # UNLESS `vision_features` ITSELF is a sequence (B, T, D)?
                    # `vision_encoded = self.vision_encoder(vision_features)`
                    # `vision_encoder` is MLP. It works on (..., D).
                    # `fused` would be (B, T, D) if input is (B, T, D).
                    # `fused.unsqueeze(1)` -> (B, 1, T, D)? No.
                    
                    # If vision_features is (B, D), then fused is (B, D).
                    # unsqueeze(1) -> (B, 1, D).
                    # LSTM( (B, 1, D) ) -> output (B, 1, D).
                    # This is effectively an MLP. The LSTM does nothing temporal if seq_len=1 and no state passing.
                    
                    # This suggests `UnifiedMobileVLAModel` implementation in `unified_mobile_vla_loader.py` might be incomplete or I misunderstood it.
                    # It doesn't accept hidden state args.
                    
                    # However, for "Context Vector Analysis", I just want to see what the model produces.
                    # Even if it's just one frame, it produces a vector.
                    
                    # I will feed the LAST frame of the window.
                    # And I need to match the dimension.
                    # If Kosmos2 is 1024 and model expects 2048, I'll try to use CLIP.
                    # "Kosmos2 + CLIP Hybrid" suggests it uses BOTH?
                    # Or maybe "Kosmos2" refers to the text part?
                    # The code says:
                    # self.vision_encoder = nn.Linear(vision_dim, hidden_dim)
                    # vision_dim = 2048.
                    
                    # I'll try to use `openai/clip-vit-large-patch14` (dim 768) or `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (dim 1280).
                    # Maybe `resnet50` (2048)?
                    
                    # Let's check `mobile_image_encoder_core.py` if it exists.
                    # I saw it in `models/core/`.
                    
                    # I'll pause editing and check `mobile_image_encoder_core.py`.
                    
                    current_batch = []
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

                
    handle.remove()
    return np.concatenate(features, axis=0) if features else None, metadata

def visualize_context_vectors(features, metadata, output_dir="result/research"):
    """Visualize context vectors using PCA and t-SNE"""
    if features is None or len(features) == 0:
        logger.warning("No features to visualize.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Color by scenario
    unique_scenarios = list(set(metadata['scenarios']))
    scenario_to_idx = {s: i for i, s in enumerate(unique_scenarios)}
    colors = [scenario_to_idx[s] for s in metadata['scenarios']]
    
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(unique_scenarios)), label='Scenario')
    plt.title('PCA of Context Vectors')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    
    # Add legend manually
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), label=s, markersize=10) 
               for i, s in enumerate(unique_scenarios)]
    plt.legend(handles=handles, title="Scenarios")
    
    plt.savefig(os.path.join(output_dir, 'context_vectors_pca.png'))
    plt.close()
    
    # t-SNE (if enough samples)
    if len(features) > 30:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        tsne_result = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, ticks=range(len(unique_scenarios)), label='Scenario')
        plt.title('t-SNE of Context Vectors')
        plt.legend(handles=handles, title="Scenarios")
        plt.savefig(os.path.join(output_dir, 'context_vectors_tsne.png'))
        plt.close()
        
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    # Load Model
    loader = UnifiedMobileVLAModelLoader()
    # Try to load the best model (Hybrid)
    model = loader.load_model(ModelType.KOSMOS2_CLIP_HYBRID)
    
    if model is None:
        logger.error("Failed to load model.")
        return
    
    # Load Dataset
    # Assuming data is in ROS_action/mobile_vla_dataset/
    data_dir = os.path.join(PROJECT_ROOT, "ROS_action/mobile_vla_dataset")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
        
    dataset = MobileVLADataset(
        data_dir=data_dir,
        mode="val", # Use val mode to avoid shuffling if not needed, but we shuffle manually
        window_size=8 # Match model expectation
    )
    
    # Extract Features
    features, metadata = extract_context_vectors(model, dataset, num_samples=200) # Start with 200 for speed
    
    # Visualize
    visualize_context_vectors(features, metadata)

if __name__ == "__main__":
    main()

