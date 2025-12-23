import os
import sys
import torch
import psutil
import logging
import argparse
import json
import types
from PIL import Image

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# MOCK LLAVA module
llava = types.ModuleType("llava")
sys.modules["llava"] = llava
llava_train = types.ModuleType("llava.train")
sys.modules["llava.train"] = llava_train
llava_train_train = types.ModuleType("llava.train.train")
sys.modules["llava.train.train"] = llava_train_train

def find_all_linear_names(model):
    import torch
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            # Append the leaf name
            lora_module_names.add(names[-1])
    
    # Common exclude
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

llava_train_train.find_all_linear_names = find_all_linear_names

# Import necessary modules from RoboVLMs
try:
    from robovlms.model.policy_head import LSTMDecoder
    import robovlms.model.policy_head as policy_head
    
    # Inject MobileVLALSTMDecoder alias
    class MobileVLALSTMDecoder(LSTMDecoder):
        pass
    policy_head.MobileVLALSTMDecoder = MobileVLALSTMDecoder
    
    from robovlms.model.backbone.robokosmos import RoboKosMos
except ImportError as e:
    print(f"Failed to import RoboVLMs modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def get_gpu_memory_info(device=0):
    if not torch.cuda.is_available():
        return 0, 0, 0
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    return allocated, reserved, max_allocated

def print_memory_stats(stage):
    if torch.cuda.is_available():
        allocated, reserved, max_alloc = get_gpu_memory_info()
        logger.info(f"[{stage}] GPU Memory: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB, Max Allocated={max_alloc:.2f} GB")
    else:
        process = psutil.Process(os.getpid())
        logger.info(f"[{stage}] System Memory: {process.memory_info().rss / (1024**3):.2f} GB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/home/soda/vla/ROS_action/last.ckpt")
    parser.add_argument("--base_model_path", type=str, default="/home/soda/vla/.vlms/kosmos-2-patch14-224")
    args = parser.parse_args()

    print_memory_stats("Start")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Reconstruct config
    config = {
        "robovlm_name": "RoboKosMos",
        "model": "kosmos",
        "model_url": args.base_model_path,
        "image_size": 224,
        "window_size": 8,
        "fwd_pred_next_n": 10,
        "train_setup": {
            "train_vision": False,
            "freeze_backbone": True,
            "bits": 16 if device == "cuda" else 32,
            "lora_enable": True,
            "lora_r": 32,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_bias": 'none',
            "train_text_embedding": False
        },
        "act_head": {
            "type": "MobileVLALSTMDecoder",
            "hidden_size": 512,
            "action_dim": 2,
            "down_sample": "none",
            "latent": 1,
            "fwd_pred_next_n": 10,
            "window_size": 8, 
            "action_space": "continuous",
            "with_history": True,
            "history_type": "post"
        },
        "vlm": {
            "type": "AutoModelForVision2Seq",
            "pretrained_model_name_or_path": args.base_model_path,
            "name": "kosmos"
        },
        "tokenizer": {
            "type": "AutoProcessor",
            "pretrained_model_name_or_path": args.base_model_path,
            "tokenizer_type": "kosmos",
            "max_text_len": 256
        }
    }

    try:
        logger.info(f"Initializing RoboKosMos with LoRA enabled...")
        
        # Initialize model structure
        model = RoboKosMos(
            configs=config,
            train_setup_configs=config["train_setup"],
            fwd_head_configs=None,
            window_size=config["window_size"],
            use_hand_rgb=False,
            act_head_configs=config["act_head"],
            fwd_pred_next_n=config["fwd_pred_next_n"],
            use_state=True
        )
        model.to(device)
        
        print_memory_stats("Model Structure Initialized")

        if os.path.exists(args.ckpt_path):
            logger.info(f"Loading checkpoint from {args.ckpt_path}...")
            # Load only state_dict to save memory if possible, but map_location is key
            checkpoint = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            
            # Adjust keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if(k.startswith("model.")):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            logger.info(f"Checkpoint keys sample (after adjustment): {list(new_state_dict.keys())[:5]}")
            logger.info(f"Model keys sample: {list(model.state_dict().keys())[:5]}")

            # Load state dict
            logger.info("Loading state dict into model...")
            msg = model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Missing keys (sample): {msg.missing_keys[:5]}")
            logger.info(f"Unexpected keys (sample): {msg.unexpected_keys[:5]}")

            
            # Cleanup
            del checkpoint
            del state_dict
            del new_state_dict
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print_memory_stats("Checkpoint Loaded")
        else:
            logger.warning(f"Checkpoint not found at {args.ckpt_path}, skipping load.")

        # Dummy inference
        logger.info("Running dummy inference...")
        
        bs = 1
        seq_len = 1
        # Determine dtype from model
        param = next(model.parameters())
        dtype = param.dtype
        logger.info(f"Model param dtype: {dtype}")

        vision_x = torch.zeros((bs, seq_len, 3, 224, 224), dtype=dtype).to(device)
        lang_x = torch.ones((bs, 10), dtype=torch.long).to(device)
        attention_mask = torch.ones((bs, 10)).bool().to(device)
        # Assuming rel_state is required for this model config
        rel_state = torch.randn((bs, seq_len, 7), dtype=dtype).to(device)
        
        with torch.no_grad():
             output = model(
                 vision_x=vision_x,
                 lang_x=lang_x,
                 attention_mask=attention_mask,
                 rel_state=rel_state
             )
        
        logger.info(f"Inference output keys: {output.keys() if hasattr(output, 'keys') else 'tuple'}")
        
        print_memory_stats("After Inference")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
