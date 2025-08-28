
### 1. Download model 
```bash
python /scratch/vladimir_albrekht/projects/__general_utils/hf/down_model.py
```
```python
from huggingface_hub import snapshot_download
from huggingface_hub import login
import dotenv
import os
dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))

repo_id = "Qwen/Qwen2.5-Omni-7B"
local_dir = "/scratch/vladimir_albrekht/projects/18_august_25_conaiki/qwen_omni/models/Qwen2.5-Omni-7B"


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
```

### 2. Conda

```bash
conda create -n conaiki_qwen_omni python=3.10 -y
conda activate conaiki_qwen_omni 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
pip install ipykernel
python -m ipykernel install --user --name conaiki_qwen_omni
pip install nvitop
pip install qwen-omni-utils
pip install librosa
pip install wandb
pip install deepspeed
pip install datasets
pip install matplotlib
pip install seaborn
```

### 3. Need to automate in pipeline.
```bash
cd conaiki
mkdir logs
```