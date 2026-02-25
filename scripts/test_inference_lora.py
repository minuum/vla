import argparse
import json
import torch
import os
import sys
import time
from PIL import Image
import numpy as np

sys.path.append(os.path.join('/home/billy/25-1kp/vla/RoboVLMs_upstream'))
def patch_tokenizer_utils():
    import robovlms.utils.model_utils
    orig_dtc = robovlms.utils.model_utils.default_tokenizer_config
    def new_dtc(tokenizer):
        if tokenizer == 'kosmos':
            return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': '/home/billy/25-1kp/vla/.vlms/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
        try:
            return orig_dtc(tokenizer)
        except Exception:
            return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': '/home/billy/25-1kp/vla/.vlms/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
    robovlms.utils.model_utils.default_tokenizer_config = new_dtc

def load_vla_model(ckpt_path, config_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config_dict['use_hand_rgb'] = config_dict.get('use_hand_rgb', False)
    config_dict['fwd_pred_next_n'] = config_dict.get('act_head', {}).get('fwd_pred_next_n', 1)
    config_dict['fwd_head'] = config_dict.get('fwd_head', {})
    config_dict['window_size'] = config_dict.get('act_head', {}).get('window_size', 8)
    config_dict['cap_loss_ratio'] = config_dict.get('cap_loss_ratio', 1.0)
    config_dict['arm_gripper_loss_ratio'] = config_dict.get('arm_gripper_loss_ratio', 1.0)
    config_dict['fwd_loss_ratio'] = config_dict.get('fwd_loss_ratio', 1.0)
    
    vlm = config_dict.get('vlm')
    if vlm and not os.path.exists(vlm.get('pretrained_model_name_or_path', '')):
        vlm['pretrained_model_name_or_path'] = '/home/billy/25-1kp/vla/.vlms/kosmos-2-patch14-224'
    
    config_dict['tokenizer'] = {'type': 'AutoProcessor', 'pretrained_model_name_or_path': '/home/billy/25-1kp/vla/.vlms/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
    
    patch_tokenizer_utils()
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    print("Initializing MobileVLATrainer...")
    trainer = MobileVLATrainer(config_dict)
    model = trainer.model
    
    print(f"Loading checkpoint weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    unwrapped_sd = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            unwrapped_sd[k.replace('model.', '', 1)] = v
        else:
            unwrapped_sd[k] = v
            
    missing, unexpected = model.load_state_dict(unwrapped_sd, strict=False)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Required for inference
    model.half()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("/home/billy/25-1kp/vla/.vlms/kosmos-2-patch14-224")
    
    return model, processor

def preprocess_image(img_arr, processor):
    img = Image.fromarray(img_arr, mode='RGB')
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(1)
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    return pixel_values.half()

def tokenize_instruction(instruction, tokenizer):
    text = f"<image>{instruction}<|endofchunk|>{tokenizer.eos_token}"
    inputs = tokenizer(
        text,
        max_length=256,
        padding="longest",
        truncation="longest_first",
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
    return input_ids, attention_mask

def decode_action(logits):
    class_map = {
        0: [0.0, 0.0],
        1: [1.15, 0.0],
        2: [-1.15, 0.0],
        3: [0.0, 1.15],
        4: [0.0, -1.15],
        5: [1.15, 1.15],
        6: [1.15, -1.15],
        7: [-1.15, 1.15],
        8: [-1.15, -1.15]
    }
    action_idx = torch.argmax(logits, dim=-1).item()
    return np.array(class_map.get(action_idx, [0.0, 0.0]))

def run_test():
    ckpt_path = '/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_exp04_lora/2026-02-22/v3-exp04-lora/merged_v3_exp04_best.ckpt'
    config_path = '/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp04_inference.json'
    
    model, processor = load_vla_model(ckpt_path, config_path)
    
    # Load real images
    img1 = np.array(Image.open('/home/billy/25-1kp/vla/test_images/robot_view_test.png').convert('RGB'))
    img2 = np.array(Image.open('/home/billy/25-1kp/vla/test_images/left_sample.jpg').convert('RGB'))
    img3 = np.array(Image.open('/home/billy/25-1kp/vla/test_images/right_sample.jpg').convert('RGB'))
    
    images = [img1, img2, img3, img1]
    
    instructions = [
        "Navigate to the brown pot on the left",
        "Navigate to the black cabinet on the right",
        "Move forward",
        "Stop"
    ]
    
    for i, inst in enumerate(instructions):
        input_ids, attention_mask = tokenize_instruction(inst, processor.tokenizer)
        image_tensor = preprocess_image(images[i], processor)

        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.inference(vision_x=image_tensor, lang_x=input_ids, attention_mask=attention_mask)
            logits = outputs['action'][0] if isinstance(outputs.get('action'), tuple) else outputs.get('action')
            action = decode_action(logits)
        
        latency = (time.time() - start_time) * 1000
        print(f"Instruction: {inst}")
        print(f"Action: {action} (Latency: {latency:.1f}ms)")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
