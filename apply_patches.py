import os
import subprocess
import sys

def install_packages():
    print("=" * 50)
    print("📦 Installing required PyPI dependencies...")
    packages = [
        "rembg",
        "transformers",
        "open-clip-torch",
        "pytorch-lightning",
        "albumentations",
        "omegaconf",
        "safetensors"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
    
    print("\n📦 Installing xformers specifically for PyTorch 2.5.1+cu121...")
    # Using the specific index for cu121 and exactly post3 to avoid torch downgrades
    subprocess.run([
        sys.executable, "-m", "pip", "install", "xformers==0.0.28.post3", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)

def patch_file(filepath, old_str, new_str, description):
    if not os.path.exists(filepath):
        print(f"⚠️  File not found, skipping: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # If the exact new string is already there, we assume it's patched
    if new_str in content:
        print(f"✅ Already patched: {description}")
        return
        
    # If the old string is found, replace it
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"🚀 Successfully patched: {description}")
    else:
        print(f"⚠️  Target string not found (might have been altered by another script), skipping: {description}")

def apply_all_patches():
    print("\n" + "=" * 50)
    print("🛠️  Applying Deep Source Code Patches...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # 1. IDM-VTON TabError Fix (Python 3.10+ strict indentation)
    # =========================================================================
    openpose_path = os.path.join(base_dir, "IDM_VTON", "preprocess", "openpose", "run_openpose.py")
    openpose_old = """\ttry:
\t\tpose_model = OpenPose(model_node)
\texcept Exception as e:
\t\tprint(f"Error loading OpenPose: {e}")
\t\tsys.exit(-1)"""
    openpose_new = """    try:
        pose_model = OpenPose(model_node)
    except Exception as e:
        print(f"Error loading OpenPose: {e}")
        sys.exit(-1)"""
    patch_file(openpose_path, openpose_old, openpose_new, "IDM-VTON TabError Fix")
    
    # =========================================================================
    # 2. AnyDoor DINOv2 Pathing Fix (Dynamic Absolute Path Resolver)
    # =========================================================================
    encoders_path = os.path.join(base_dir, "AnyDoor", "ldm", "modules", "encoders", "modules.py")
    encoders_old = """sys.path.append("./dinov2")
import hubconf
from omegaconf import OmegaConf
config_path = './configs/anydoor.yaml'
config = OmegaConf.load(config_path)
DINOv2_weight_path = config.model.params.cond_stage_config.weight"""
    encoders_new = """import os
import sys
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_anydoor_root = os.path.abspath(os.path.join(_current_file_dir, "..", "..", ".."))
sys.path.append(os.path.join(_anydoor_root, "dinov2"))
import hubconf
from omegaconf import OmegaConf
config_path = os.path.join(_anydoor_root, 'configs', 'anydoor.yaml')
config = OmegaConf.load(config_path)
DINOv2_weight_path = config.model.params.cond_stage_config.weight
if not os.path.isabs(DINOv2_weight_path):
    DINOv2_weight_path = os.path.join(_anydoor_root, DINOv2_weight_path)"""
    patch_file(encoders_path, encoders_old, encoders_new, "AnyDoor DINOv2 Absolute Path Fix")
    
    # =========================================================================
    # 3. AnyDoor PyTorch Lightning Distributed Fix (logger.py)
    # =========================================================================
    logger_path = os.path.join(base_dir, "AnyDoor", "cldm", "logger.py")
    logger_old = """from pytorch_lightning.utilities.distributed import rank_zero_only"""
    logger_new = """try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities.distributed import rank_zero_only"""
    patch_file(logger_path, logger_old, logger_new, "AnyDoor logger.py PyTorch Lightning Fallback")
    
    # =========================================================================
    # 4. AnyDoor PyTorch Lightning Distributed Fix (ddpm.py)
    # =========================================================================
    ddpm_path = os.path.join(base_dir, "AnyDoor", "ldm", "models", "diffusion", "ddpm.py")
    patch_file(ddpm_path, logger_old, logger_new, "AnyDoor ddpm.py PyTorch Lightning Fallback")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 FASHION AI BACKEND - AUTO PATCHER")
    print("=" * 50)
    
    try:
        install_packages()
        apply_all_patches()
        
        print("\n" + "=" * 50)
        print("🎉 All setup and patching complete! The backend is fully primed.")
        print("You can now safely run: python idm_backend.py")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
