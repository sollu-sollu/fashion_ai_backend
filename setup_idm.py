import os
import subprocess
import sys
from huggingface_hub import hf_hub_download, snapshot_download

def setup_idm():
    # print("=" * 60)
    # print("🚀 IDM_VTON RECOVERY: Full Setup (20GB+)")
    # print("=" * 60)
    
    # # 1. Install specific dependencies for IDM
    # print("⏳ Installing IDM-specific dependencies...")
    # libs = [
    #     "diffusers==0.25.1", "transformers==4.38.1", "accelerate", 
    #     "huggingface_hub", "einops", "omegaconf", "tqdm", "ninja",
    #     "opencv-python", "albumentations"
    # ]
    # subprocess.run([sys.executable, "-m", "pip", "install"] + libs, check=False)
    
    # Detectron2 (DensePose requirement)
    print("⏳ Attempting to install Detectron2 (for DensePose)...")
    d2_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html"
    subprocess.run([sys.executable, "-m", "pip", "install", "detectron2", "-f", d2_url], check=False)

    # # 2. Clone IDM-VTON if missing
    # if not os.path.exists("IDM_VTON"):
    #     print("⏳ Cloning IDM_VTON repository...")
    #     subprocess.run(["git", "clone", "https://github.com/yisol/IDM-VTON.git"], check=True)
    # else:
    #     print("✅ IDM_VTON repository present.")

    # # 3. Download weights (The safe way using snapshot_download)
    # print("\n⏳ Downloading weights from yisol/IDM_VTON (High Accuracy Model)...")
    # print("   Note: This is ~20GB. It may take 10-30 mins depending on your instance bandwidth.")
    
    # try:
    #     # We download to a local folder to avoid HF cache issues on some instances
    #     model_path = snapshot_download(
    #         repo_id="yisol/IDM-VTON",
    #         local_dir="./IDM_VTON/weights",
    #         local_dir_use_symlinks=False,
    #         # We skip heavy unnecessary files if they exist (only need fp16 usually)
    #         allow_patterns=["*.json", "*.bin", "*.safetensors", "unet/*", "vae/*", "text_encoder*", "tokenizer*", "image_encoder/*", "scheduler/*"]
    #     )
    #     print(f"✅ Weights verified at {model_path}")
    # except Exception as e:
    #     print(f"❌ Weight download failed: {e}")
    #     print("💡 Try logging into huggingface: 'huggingface-cli login'")
    #     return

    # # 4. DensePose weights
    # os.makedirs("./IDM_VTON/ckpt/densepose", exist_ok=True)
    # if not os.path.exists("./IDM_VTON/ckpt/densepose/model_final_162be9.pkl"):
    #     print("⏳ Downloading DensePose weights...")
    #     # Direct link for DensePose R50 FPN s1x
    #     # (Standard URL provided in IDM-VTON readme)
    #     # Using hf_hub_download if mirrored
    #     try:
    #         hf_hub_download(
    #             repo_id="yisol/IDM-VTON",
    #             filename="ckpt/densepose/model_final_162be9.pkl",
    #             local_dir="./IDM_VTON"
    #         )
    #         print("✅ DensePose weights downloaded.")
    #     except:
    #         print("⚠️ Could not download DensePose via HF hub. You may need manual wget.")

    # # 5. AnyDoor (The user wants accessories in IDM too)
    # if not os.path.exists("setup_anydoor.py"):
    #      print("❌ setup_anydoor.py missing from root. Skipping accessory download.")
    # else:
    #      print("⏳ Running AnyDoor setup for accessories compatibility...")
    #      subprocess.run([sys.executable, "setup_anydoor.py"], check=False)

    # print("\n" + "=" * 60)
    # print("✅ IDM_VTON + ANYDOOR SETUP COMPLETE!")
    # print("   You can now try running 'python idm_backend.py'")
    # print("=" * 60)

if __name__ == "__main__":
    setup_idm()
