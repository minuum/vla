import torch
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as T
import sys
import os

# Add RoboVLMs to path
sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def get_direction_from_text(instruction):
    """
    Extract direction from text instruction.
    Coordinate System:
    - Left: Positive linear_y
    - Right: Negative linear_y
    """
    instr_lower = instruction.lower()
    if 'left' in instr_lower:
        return 1.0
    elif 'right' in instr_lower:
        return -1.0
    else:
        # Default or warning
        # For safety, maybe 0.0 or maintain previous?
        # Here we default to 0.0 (straight) if direction is unclear
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Inference for Mobile VLA with Abs Action Strategy")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (abs_action)')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--text', type=str, default="Navigate to the left bottle", help='Instruction')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint} on {device}...")
    
    # Load Model
    model = MobileVLATrainer.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()
    model.to(device)
    
    print("Model loaded successfully.")

    # Image Transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # Simulation of inference loop (single image)
    if args.image:
        img = Image.open(args.image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).unsqueeze(0).to(device) # (B, Seq, C, H, W) -> (1, 1, 3, 224, 224)
        # Note: Model expects sequence, typically we pass just 1 frame for react policy or window
        # MobileVLAPolicy expects (B, Seq, C, H, W)
        
        # Text Logic
        direction = get_direction_from_text(args.text)
        print(f"Instruction: '{args.text}' -> Direction Multiplier: {direction}")

        with torch.no_grad():
            vision_x = img_tensor
            # Forward pass
            # Note: robovlms logic is complex, usually involves encode_images -> act_head
            
            # Encode
            vision_x = vision_x.squeeze(1).unsqueeze(0) # Logic adjustment might be needed depending on dims
            # Let's rely on standard forward or act
            
            # Simplified forward for inference:
            # 1. Encode images
            image_features = model.model.encode_images(img_tensor.squeeze(0)) # Expects (Seq, C, H, W) or (B, Seq...)
            # MobileVLAImageEncoder handles (B, T, C, H, W)
            
            # ... Actually, let's use the standard way we tested earlier:
            bs, seq_len, c, h, w = img_tensor.shape
            
            encoded = model.model.vision_encoder(img_tensor.view(-1, c, h, w))
            encoded = encoded.view(bs, seq_len, -1, encoded.shape[-1]) # (B, Seq, num_tokens, dim)
            
            # In simple inference, just take the last frame's token
            # But wait, MobileVLAPolicy.act_head expects action_history?
            # If we don't have history, we pass None or zeros.
            
            # Extract just the action embedding token (last one usually or mean?)
            # In Kosmos-2 backbone, we take the special tokens.
            
            # Let's simply call model.predict_step if available, or reproduce basic forward
            action_hs = encoded[:, :, -1:, :].squeeze(2) # (B, Seq, Hidden)
            
            action_mask = torch.ones(bs, seq_len, dtype=torch.bool).to(device)
            
            # Predict
            pred_actions = model.model.act_head(action_hs, actions=None, action_masks=action_mask)
            if isinstance(pred_actions, tuple):
                pred_actions = pred_actions[0]
                
            # Result: (B, Seq, Pred_Len, Action_Dim)
            # Take last step
            raw_action = pred_actions[0, -1, 0].cpu().numpy() # [linear_x, abs_linear_y]
            
            # Apply Direction
            final_linear_x = raw_action[0]
            final_linear_y = abs(raw_action[1]) * direction
            
            print("="*30)
            print(f"Raw Prediction (Abs): {raw_action}")
            print(f"Final Action: linear_x={final_linear_x:.3f}, linear_y={final_linear_y:.3f}")
            print("="*30)
            
    else:
        print("Provide --image to test inference.")

if __name__ == "__main__":
    main()
