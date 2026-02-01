import torch
from transformers import PaliGemmaForConditionalGeneration, AutoConfig

model_id = "google/paligemma-3b-pt-224"
print(f"Testing load for {model_id}...")

try:
    print("Loading config...")
    config = AutoConfig.from_pretrained(model_id)
    print("Config loaded successfully")
except Exception as e:
    print(f"Config load failed: {e}")

try:
    print("Loading model (PaliGemmaForConditionalGeneration)...")
    # Load with low_cpu_mem_usage=True to be fast if it's just verification
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}")
