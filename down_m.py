from huggingface_hub import snapshot_download
from huggingface_hub import login
import dotenv
import os
dotenv.load_dotenv()
login(token="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")

repo_id = "openai/whisper-large-v3-turbo"
local_dir = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/models/whisper-large-v3-turbo"


def download_model_repo(repo_id, local_dir):
    print(f"Downloading repository {repo_id}...")
    
    # Download the complete repository
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir
        # allow_patterns=["DeepSeek-R1-Q6_K/*"] 
    )
    print(f"Repository downloaded to {local_dir}")

# Specify the model repo ID, save directory, and Huggingw Face token

# Download the repository
download_model_repo(repo_id, local_dir)