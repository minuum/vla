import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

print("Starting Kosmos-2 model test...")

# Load model
print("Downloading model...")
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
print("Model loaded successfully!")

# Set prompt
prompt = "<grounding>An image of"

# Download test image
print("Downloading test image...")
url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
image = Image.open(requests.get(url, stream=True).raw)

# Save and reload image (same as original demo)
image.save("new_image.jpg")
image = Image.open("new_image.jpg")
print("Image prepared!")

# Prepare model inputs
print("Preparing model inputs...")
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate text
print("Generating text...")
generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=128,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Process results
print("Processing results...")
# Raw generated text
processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
print("\nRaw generated text:")
print(processed_text)

# Cleaned text and entities
processed_text, entities = processor.post_process_generation(generated_text)
print("\nCleaned text:")
print(processed_text)
print("\nDetected entities:")
print(entities)

print("\nKosmos-2 test completed!")
