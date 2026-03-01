
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset
import os

def test_normalization_discrepancy():
    print("=== Testing Normalization Discrepancy ===")
    
    # 1. Setup paths
    data_dir = "/home/billy/25-1kp/vla/ROS_action/basket_dataset"
    model_path = ".vlms/kosmos-2-patch14-224"
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return

    # 2. Instantiate Dataset (simulating Training)
    print("\n[Training Pipeline Verification]")
    try:
        dataset = MobileVLAH5Dataset(
            data_dir=data_dir,
            episode_pattern="*left*.h5",
            window_size=1,
            action_chunk_size=1,
            model_name="kosmos",
            train_split=0.9
        )
        
        sample = dataset[0]
        rgb = sample['rgb'] # (seq_len, C, H, W)
        print(f"Dataset RGB Shape: {rgb.shape}")
        print(f"Dataset RGB Range: Min={rgb.min().item():.4f}, Max={rgb.max().item():.4f}")
        print(f"Dataset RGB Mean: {rgb.mean().item():.4f}, Std: {rgb.std().item():.4f}")
        
        if rgb.min() >= 0.0 and rgb.max() <= 1.0:
            print("=> Dataset Output is [0, 1] (Un-normalized)")
        else:
            print("=> Dataset Output is likely Normalized")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

    # 3. Instantiate Processor (simulating Inference)
    print("\n[Inference Pipeline Verification]")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Create dummy image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        
        # Process
        inputs = processor(text="test", images=dummy_img, return_tensors="pt")
        proc_rgb = inputs['pixel_values']
        
        print(f"Processor RGB Shape: {proc_rgb.shape}")
        print(f"Processor RGB Range: Min={proc_rgb.min().item():.4f}, Max={proc_rgb.max().item():.4f}")
        print(f"Processor RGB Mean: {proc_rgb.mean().item():.4f}, Std: {proc_rgb.std().item():.4f}")
        
        if proc_rgb.min() < 0.0:
            print("=> Processor Output is Normalized (contains negatives)")
        else:
            print("=> Processor Output is [0, 1]")
            
    except Exception as e:
        print(f"Error loading processor: {e}")

if __name__ == "__main__":
    test_normalization_discrepancy()
