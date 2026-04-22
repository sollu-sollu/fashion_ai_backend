import os
import subprocess
import sys
import torch
import numpy as np
import cv2
import io
import base64
import threading
import time
import gc
from PIL import Image, ImageDraw, ImageFilter
from flask import Flask, request, jsonify
from pyngrok import ngrok

# --- CONFIGURATION ---
REPO_NAME = "fashion"
NGROK_TOKEN = "36EnpN0f0tx9hre6EMpt1jRpHeb_38GYquy1KabEJGskmKeHL"

app = Flask(__name__)

# --- CLONE FASHN-VTON IF NEEDED ---
if not os.path.exists(REPO_NAME):
    print(f"📂 Cloning {REPO_NAME}...")
    subprocess.run(["git", "clone", "https://github.com/fashn-AI/fashn-vton-1.5.git", REPO_NAME], check=True)

if not os.path.exists(f"{REPO_NAME}/weights/model.safetensors"):
    print("⏳ Downloading FASHN weights (this may take a while)...")
    subprocess.run([sys.executable, f"{REPO_NAME}/scripts/download_weights.py", "--weights-dir", f"{REPO_NAME}/weights"], check=True)
    print("✅ Weights Downloaded!")

# --- CROSS CHECK ---
try:
    from cross_check import validate_gender_match
except ImportError:
    print("⚠️ cross_check.py not found. Validation disabled.")
    def validate_gender_match(*args, **kwargs):
        return True, "Validation disabled", "unknown", "unknown"

# =====================================================================
# ENGINE 1: FASHN-VTON — For Clothes (Tops, Bottoms, Dresses)
# =====================================================================

class FashnEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.ready = False

        try:
            print("=" * 50)
            print(f"🔧 ENGINE 1: Loading FASHN-VTON on {device.upper()}...")
            print("=" * 50)

            from fashion.src.fashn_vton import TryOnPipeline
            self.pipe = TryOnPipeline(f"{REPO_NAME}/weights", device=device)
            
            self.ready = True
            print("   ✅ FASHN-VTON Engine: READY (Clothes Mode)")

        except Exception as e:
            print(f"   ❌ FASHN-VTON Engine FAILED: {e}")
            import traceback; traceback.print_exc()
            self.pipe = None

    def __call__(self, person_image, garment_image, category="tops"):
        if not self.ready:
            return None

        print(f"   🎨 FASHN: Running try-on for '{category}'...")
        
        # FASHN optimal size is 768x1024
        person_r = person_image.convert("RGB").resize((768, 1024))
        garment_r = garment_image.convert("RGB").resize((768, 1024))

        result = self.pipe(
            person_image=person_r,
            garment_image=garment_r,
            category=category
        )

        if hasattr(result, 'images') and len(result.images) > 0:
            result_img = result.images[0]
        else:
            result_img = result

        print("   ✅ FASHN inference complete!")
        return result_img


# =====================================================================
# ENGINE 2: AnyDoor (Alibaba) — SOTA for Accessories (Shoes, Bags, Hats)
# =====================================================================
import sys
import os
import numpy as np
import cv2
from PIL import ImageDraw

class AnyDoorEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.ready = False

        try:
            print("=" * 50)
            print(f"🔧 ENGINE 2: Loading AnyDoor Accessory Engine on {device.upper()}...")
            print("=" * 50)

            # Ensure we use absolute path relative to THIS script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            anydoor_path = os.path.join(base_dir, 'AnyDoor')

            # Use our standalone wrapper
            from anydoor_wrapper import init_anydoor_model, inference_single_image
            
            # Initialize weights
            init_anydoor_model(anydoor_dir=anydoor_path, device=self.device)
            self.inference_fn = inference_single_image

            self.ready = True
            print("   ✅ AnyDoor Engine: READY (Zero-Shot Object Teleportation)")

        except Exception as e:
            print(f"   ❌ AnyDoor Engine FAILED: {e}")
            import traceback; traceback.print_exc()
            self.inference_fn = None

    def _make_bg_mask(self, w, h, category):
        # AnyDoor requires a bounding box mask of where to place the object
        mask = np.zeros((h, w), dtype=np.uint8)
        
        cat_lower = category.lower()
        if "shoe" in cat_lower or "foot" in cat_lower:
            # Bottom 20%
            y1, y2 = int(h * 0.8), int(h * 0.98)
            x1, x2 = int(w * 0.25), int(w * 0.75)
        elif "hat" in cat_lower or "head" in cat_lower:
            # Top 20%
            y1, y2 = int(h * 0.02), int(h * 0.2)
            x1, x2 = int(w * 0.25), int(w * 0.75)
        elif "bag" in cat_lower or "purse" in cat_lower:
            # Hip height, left side
            y1, y2 = int(h * 0.45), int(h * 0.8)
            x1, x2 = int(w * 0.05), int(w * 0.4)
        elif "jewelry" in cat_lower or "necklace" in cat_lower:
            # Chest area
            y1, y2 = int(h * 0.2), int(h * 0.4)
            x1, x2 = int(w * 0.35), int(w * 0.65)
        else:
            # Default generic accessory layout
            y1, y2 = int(h * 0.4), int(h * 0.8)
            x1, x2 = int(w * 0.1), int(w * 0.4)
            
        mask[y1:y2, x1:x2] = 255
        return mask

    def __call__(self, person_image, garment_image, category="shoes"):
        if not self.ready or not self.inference_fn:
            return None

        print(f"   🎨 AnyDoor: Teleporting '{category}' onto person...")

        # Resize to standard processing sizes while maintaining aspect ratio
        target_size = (768, 1024)
        
        p_img = person_image.convert("RGB")
        g_img = garment_image.convert("RGBA")
        
        # We need a reference mask (the accessory cut out from background). 
        # If the user uploaded a white background, we create a simple mask from non-white pixels
        garment_np = np.array(g_img)
        if garment_np.shape[2] == 4:
            # Use alpha channel if available
            ref_mask = (garment_np[:, :, 3] > 10).astype(np.uint8) * 255
        else:
            # Crude white-background removal
            gray = cv2.cvtColor(garment_np, cv2.COLOR_RGB2GRAY)
            _, ref_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        ref_image = cv2.cvtColor(garment_np[:, :, :3], cv2.COLOR_RGB2BGR)
        
        tar_image_pil = p_img.resize(target_size)
        tar_image = cv2.cvtColor(np.array(tar_image_pil), cv2.COLOR_RGB2BGR)
        
        h, w = tar_image.shape[:2]
        tar_mask = self._make_bg_mask(w, h, category)

        # Inference
        try:
            with torch.inference_mode():
                # AnyDoor expects masks as uint8 (0 or 1)
                ref_mask_bool = (ref_mask > 128).astype(np.uint8)
                tar_mask_bool = (tar_mask > 128).astype(np.uint8)
                
                # AnyDoor wrapper handles the heavy lifting
                gen_image = self.inference_fn(ref_image, ref_mask_bool, tar_image.copy(), tar_mask_bool)
                
            result_rgb = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
            print("   ✅ AnyDoor inference complete!")
            return Image.fromarray(result_rgb)
        except Exception as e:
            print(f"   ❌ AnyDoor processing error: {e}")
            return None


# =====================================================================
# DUAL ENGINE ROUTER — VRAM OPTIMIZED (LAZY LOADING)
# =====================================================================
import gc

class DualEngineVTON:
    def __init__(self, device="cuda"):
        self.device = device

        # Do NOT load models at startup to save VRAM.
        # We will load them lazily and exclusively when needed.
        self.fashn = None
        self.sdxl = None

        print("\n" + "=" * 50)
        print("📊 ROUTER STATUS: Dynamic VRAM Management Enabled")
        print("   Models will load dynamically based on category.")
        print("   Only ONE model will be in VRAM at a time to prevent OOM.")
        print("=" * 50)

    @property
    def ready(self):
        # The router is always "ready" to load a model dynamically
        return True

    def _unload_fashn(self):
        if self.fashn is not None:
            print("   🧹 Unloading FASHN-VTON from VRAM...")
            del self.fashn.pipe
            self.fashn = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _unload_anydoor(self):
        if self.sdxl is not None:
            print("   🧹 Unloading AnyDoor Accessory Engine from VRAM...")
            if hasattr(self.sdxl, 'model') and self.sdxl.model is not None:
                del self.sdxl.model
            self.sdxl = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_fashn(self):
        if self.fashn is None:
            # Ensure AnyDoor is unloaded first to free memory
            self._unload_anydoor()
            print("   🔄 Dynamically loading FASHN-VTON Engine...")
            self.fashn = FashnEngine(self.device)

    def _load_anydoor(self):
        if self.sdxl is None:
            # Ensure FASHN is unloaded first to free memory
            self._unload_fashn()
            print("   🔄 Dynamically loading AnyDoor Accessory Engine...")
            self.sdxl = AnyDoorEngine(self.device)

    def __call__(self, person_image, garment_image, category="tops"):
        is_accessory = any(x in category.lower() for x in ["shoe", "bag", "hat", "accessor", "jewel", "watch", "glasses"])

        # Dynamic Routing with Exclusive Loading
        if is_accessory:
            print(f"\n   🔀 Route: Accessory detected ('{category}'). Using Engine 2 (AnyDoor).")
            self._load_anydoor()
            if self.sdxl and self.sdxl.ready:
                return self.sdxl(person_image, garment_image, category)
        else:
            print(f"\n   🔀 Route: Clothing detected ('{category}'). Using Engine 1 (FASHN).")
            self._load_fashn()
            if self.fashn and self.fashn.ready:
                # FASHN expects specific categories, standardize them
                cat_lower = category.lower()
                fashn_cat = "tops"
                if "bottom" in cat_lower or "pant" in cat_lower or "skirt" in cat_lower:
                    fashn_cat = "bottoms"
                elif "dress" in cat_lower or "one-piece" in cat_lower:
                    fashn_cat = "dresses"
                
                return self.fashn(person_image, garment_image, fashn_cat)

        print("   ❌ Selected engine failed to initialize.")
        return person_image


# --- INITIALIZE PIPELINE ---
pipeline = None
try:
    print(f"✅ Numpy: {np.__version__}")
    pipeline = DualEngineVTON(device="cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"⚠️ Startup error: {e}")
    import traceback; traceback.print_exc()


# --- API HELPERS ---
def base64_to_image(b64):
    if "," in b64: b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def image_to_base64(img):
    if img is None: return ""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- API ENDPOINTS ---
@app.route("/api/tryon", methods=["POST"])
def tryon():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data"}), 400

        person_b64 = data.get("person")
        garment_b64 = data.get("garment")
        category = data.get("category", "tops")

        if not person_b64 or not garment_b64:
            return jsonify({"success": False, "error": "Missing person or garment image"}), 400

        print(f"\n📥 Request: category={category}")
        person_img = base64_to_image(person_b64)
        garment_img = base64_to_image(garment_b64)

        # Cross-check validation
        print("🔍 Running gender cross-check...")
        is_valid, msg, p_gen, g_gen = validate_gender_match(person_img, garment_img, category)
        if not is_valid:
            print(f"❌ Validation Failed: {msg}")
            return jsonify({
                "success": False, "error": "validation_failed",
                "message": f"Garment doesn't match person. ({msg})",
                "details": {"person": p_gen, "garment": g_gen},
            }), 400

        # Inference
        print("🚀 Starting inference...")
        if not pipeline or not pipeline.ready:
            return jsonify({"success": False, "error": "No engine available"}), 500

        result_img = pipeline(person_img, garment_img, category)
        
        if result_img is None:
            return jsonify({"success": False, "error": "Inference failed"}), 500
            
        result_b64 = image_to_base64(result_img)

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"📉 VRAM: {vram_mb:.0f}MB / {total_mb:.0f}MB")
            
        print("✅ Done!\n")

        return jsonify({"success": True, "result": f"data:image/jpeg;base64,{result_b64}"})

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    fash_ready = pipeline.fashn.ready if (pipeline and pipeline.fashn) else False
    sdxl_ready = pipeline.sdxl.ready if (pipeline and pipeline.sdxl) else False
    return jsonify({
        "status": "ok",
        "fashn_ready": fash_ready,
        "anydoor_ready": sdxl_ready,
        "vram_use": f"{torch.cuda.memory_allocated() / 1024**2:.0f}MB" if torch.cuda.is_available() else "cpu"
    })


# --- SERVER ---
def run_server():
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    if not NGROK_TOKEN or NGROK_TOKEN == "YOUR_TOKEN_HERE":
        print("❌ NGROK_TOKEN missing"); sys.exit(1)

    ngrok.set_auth_token(NGROK_TOKEN)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    url = ngrok.connect(5000).public_url
    print("=" * 60)
    print(f"🚀 FASH+SDXL DUAL API: {url}")
    print("=" * 60)
    print(f"   POST {url}/api/tryon")
    print(f"   GET  {url}/api/health")
    print("⏳ Running...\n")

    while True:
        time.sleep(10)