## 0. Access & File Transfer
Before installing, you need to get your files onto the server.

### A. Connect via SSH
Open your local terminal (PowerShell or Bash) and run:
```bash
ssh ubuntu@<YOUR_INSTANCE_IP>
```
*(If you are on Lambda Labs, the user might be `ubuntu`; on some others, it's `root`.)*

### B. Transfer Backend Files
Run this **on your local computer** (where the code is) to send the files to the cloud:
```powershell
scp idm_backend.py cross_check.py check_env.py ubuntu@<YOUR_INSTANCE_IP>:~/
```

## 1. Environment Variable Setup
Before installing anything, ensure your system can find the pre-installed CUDA libraries. Add these to your `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
# Required for bitsandbytes on CUDA 12.x
echo 'export BNB_CUDA_VERSION="121"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Create a Clean Virtual Environment
Always use a venv to prevent library version conflicts.
```bash
python3 -m venv .vton_env
source .vton_env/bin/activate
```

Install the libraries in this specific order.

```bash
# If you chose CUDA 12.4, use index-url for cu124
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# B. Install High-Fidelity VTON Libraries
pip install diffusers transformers accelerate "numpy<2" flask pyngrok pillow ftfy regex tqdm einops omegaconf

# C. Install Windows-Friendly Libs (even on Linux, the wrapper helps)
pip install onnxruntime-gpu bitsandbytes
```

## 4. Environment Variables
Cloud systems often need explicit paths. Run these commands (or add them to your `~/.bashrc`):
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
export BNB_CUDA_VERSION="121"
```

## 5. Verify and Run
1.  **Check Health**:
    ```bash
    python check_env.py
    ```
    *Everything should show* ✅.

2.  **Start Backend**:
    ```bash
    python idm_backend.py
    ```

## 6. Troubleshooting
*   **"Access Denied" (HuggingFace)**: Download might require a login.
    ```bash
    pip install huggingface_hub
    huggingface-cli login
    ```
*   **ONNX GPU Error**: If `onnxruntime-gpu` fails to find your GPU, run:
    ```bash
    pip uninstall onnxruntime onnxruntime-gpu -y
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    ```
