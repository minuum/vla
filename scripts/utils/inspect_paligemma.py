from transformers import PaliGemmaForConditionalGeneration
import torch

model_id = "google/paligemma-3b-pt-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

print("\n=== Model Structure ===")
print(model)

print("\n=== Checking Attributes ===")
try:
    print(f"model.language_model: {model.language_model}")
except AttributeError:
    print("model.language_model not found")

try:
    print(f"model.language_model.model: {model.language_model.model}")
except AttributeError:
    print("model.language_model.model not found")
    
try:
    print(f"model.language_model.model.embed_tokens: {model.language_model.model.embed_tokens}")
except AttributeError:
    print("model.language_model.model.embed_tokens not found")
