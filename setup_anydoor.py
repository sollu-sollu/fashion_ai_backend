import os
import subprocess
import sys
from huggingface_hub import hf_hub_download

def setup_anydoor():
    print("=" * 50)
    print("ANYDOOR SETUP: Repository & Weights")
    print("=" * 50)
    
    # 1. Install Dependencies
    print("⏳ Installing AnyDoor dependencies (open-clip, xformers, etc.)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "open-clip-torch", "pytorch-lightning", "albumentations", "xformers"], check=False)

    # 2. Clone Repository if missing
    if not os.path.exists("./AnyDoor/cldm"): 
        print("⏳ AnyDoor repository missing. Cloning source code...")
        subprocess.run(["git", "clone", "https://github.com/ali-vilab/AnyDoor.git"], check=True)
    else:
        print("✅ AnyDoor repository source code present.")

    # 3. Ensure Weights directory exists
    os.makedirs("./AnyDoor/path", exist_ok=True)
    
    # 3. Download Model Weights
    print("⏳ Checking AnyDoor Pruned Model (4.57GB)...")
    anydoor_ckpt = hf_hub_download(
        repo_id="camenduru/AnyDoor",
        filename="epoch=1-step=8687-pruned.ckpt",
        local_dir="./AnyDoor/path"
    )
    
    # Rename for consistency
    target_ckpt = "./AnyDoor/path/epoch=1-step=8687.ckpt"
    if os.path.exists(anydoor_ckpt) and not os.path.exists(target_ckpt):
        try:
            os.rename(anydoor_ckpt, target_ckpt)
        except: pass
    
    print("⏳ Checking DINOv2 Pretrained Weights (1.2GB)...")
    dinov2_ckpt = hf_hub_download(
        repo_id="camenduru/AnyDoor",
        filename="dinov2_vitg14_pretrain.pth",
        local_dir="./AnyDoor/path"
    )

    print("\n✅ COMPLETE: AnyDoor is ready. Run 'python fash_backend.py' to start.")

if __name__ == "__main__":
    setup_anydoor()
