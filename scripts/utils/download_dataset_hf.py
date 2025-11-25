import os
import argparse
from huggingface_hub import snapshot_download, login

def download_dataset(repo_id, local_dir, token=None):
    """
    Downloads a dataset from Hugging Face Hub to a local directory.
    
    Args:
        repo_id (str): The ID of the Hugging Face repository (e.g., 'username/dataset_name').
        local_dir (str): The local directory to save the dataset to.
        token (str, optional): Hugging Face API token. If None, looks for HF_TOKEN env var or expects user to be logged in.
    """
    print(f"🚀 Starting download from {repo_id} to {local_dir}...")
    
    try:
        if token:
            login(token=token)
        
        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the dataset
        # allow_patterns=["*.h5"] ensures we only get the data files if that's what we want, 
        # but for a full backup/restore, we might want everything.
        # Using ignore_patterns to avoid downloading git internal files if any.
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            ignore_patterns=[".gitattributes", "README.md"],
            local_dir_use_symlinks=False # Important for actual file availability
        )
        
        print(f"✅ Successfully downloaded dataset to: {local_path}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Tip: Ensure you have access rights to the repository and a valid token if it's private.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Mobile-VLA dataset from Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g., 'minuum/mobile-vla-dataset')")
    parser.add_argument("--local_dir", type=str, default="./data/mobile_vla_dataset", help="Local directory to save the dataset")
    parser.add_argument("--token", type=str, help="Hugging Face API token (optional, can also set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    download_dataset(args.repo_id, args.local_dir, args.token)
