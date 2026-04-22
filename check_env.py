import sys
import subprocess

def check_env():
    print("🔍 Environment Diagnostic Tool")
    print("-" * 30)
    print(f"Python: {sys.version}")
    
    packages = ["numpy", "torch", "torchvision", "clip", "diffusers", "transformers", "onnxruntime-gpu", "bitsandbytes"]
    
    print("-" * 30)
    import platform
    import os
    print(f"OS: {platform.system()} {platform.release()}")
    if platform.system() == "Linux":
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
        print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
        
        # Deep search for the elusive libcudart
        print("🔍 Searching for libcudart.so...")
        search_paths = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/opt/conda/lib"]
        for path in search_paths:
            if os.path.exists(path):
                libs = [f for f in os.listdir(path) if "libcudart.so" in f]
                if libs:
                    print(f"✅ Found in {path}: {libs}")

    for pkg in packages:
        try:
            # Fix: onnxruntime-gpu is installed via pip but imported as onnxruntime
            import_name = "onnxruntime" if pkg == "onnxruntime-gpu" else pkg.replace("-", "_")
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            
            if pkg == "numpy" and version.startswith("2."):
                print(f"❌ {pkg}: {version} (INCOMPATIBLE! Please downgrade to < 2.0)")
            elif pkg == "onnxruntime-gpu":
                providers = mod.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    print(f"✅ {pkg}: {version} (GPU Accelerated)")
                else:
                    print(f"⚠️ {pkg}: {version} (CPU ONLY! CUDA not found by ONNX)")
            elif pkg == "bitsandbytes":
                try:
                    import bitsandbytes as bnb
                    # Check if it can actually load the CUDA libs
                    if hasattr(bnb.nn, "Int8Params"):
                         print(f"✅ {pkg}: {version} (Library Loaded)")
                    else:
                         print(f"⚠️ {pkg}: {version} (Warning: CUDA functions missing)")
                except Exception as bnb_e:
                     print(f"❌ {pkg}: {version} (FAILED: {bnb_e})")
            else:
                print(f"✅ {pkg}: {version}")
        except ImportError:
            print(f"❌ {pkg}: NOT INSTALLED")
        except Exception as e:
            print(f"⚠️ {pkg}: ERROR ({e})")

    print("-" * 30)
    if platform.system() == "Linux":
        print("💡 CLOUD/LINUX FIX for bitsandbytes:")
        print("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64")
        print("pip install bitsandbytes --upgrade")
    else:
        print("💡 WINDOWS FIX for bitsandbytes:")
        print("pip uninstall bitsandbytes -y")
        print("pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui")

if __name__ == "__main__":
    check_env()
