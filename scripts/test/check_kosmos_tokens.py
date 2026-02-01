from transformers import AutoProcessor
import torch

processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
text = "An image of a robot Navigate to the basket"
processed = processor(text=text, return_tensors="pt")
print(f"Tokens without <grounding>: {processor.tokenizer.convert_ids_to_tokens(processed['input_ids'][0])}")

text_with_g = "<grounding>An image of a robot Navigate to the basket"
processed_with_g = processor(text=text_with_g, return_tensors="pt")
print(f"Tokens with <grounding>: {processor.tokenizer.convert_ids_to_tokens(processed_with_g['input_ids'][0])}")
