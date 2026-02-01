from huggingface_hub import snapshot_download
import os

model_id = "google/paligemma-3b-pt-224"
print(f"=== Downloading {model_id} ===")

try:
    path = snapshot_download(
        repo_id=model_id, 
        repo_type="model",
        resume_download=True
    )
    print(f"\n✅ Success! Model is located at:")
    print(path)
    os.system(f"du -sh {path}")
except Exception as e:
    print(f"\n❌ Error: {e}")
