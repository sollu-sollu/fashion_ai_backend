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
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from flask import Flask, request, jsonify
from pyngrok import ngrok
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from typing import List

# --- CONFIGURATION ---
NGROK_TOKEN = "36EnpN0f0tx9hre6EMpt1jRpHeb_38GYquy1KabEJGskmKeHL"
IDM_REPO = "IDM_VTON"
# IDM_VTON model path: prefer HF cache (avoids 26GB duplicate), fallback to local
# If weights are in ~/.cache/huggingface/, the HF ID loads them instantly (no re-download)
IDM_MODEL_PATH = "yisol/IDM-VTON"

app = Flask(__name__)

# --- CLONE IDM-VTON REPO IF NEEDED ---
if not os.path.exists(IDM_REPO):
    print(f"📂 Cloning {IDM_REPO}...")
    subprocess.run(["git", "clone", "https://github.com/yisol/IDM-VTON.git"], check=True)

# Add IDM_VTON repo to path for custom module imports
sys.path.insert(0, os.path.join(os.getcwd(), IDM_REPO))

# --- CROSS CHECK ---
from cross_check import validate_gender_match


# =====================================================================
# ENGINE 1: IDM_VTON (SOTA) — For Clothes (Tops, Bottoms, Dresses)
# Requires: diffusers==0.25.1, huggingface_hub==0.21.4
# =====================================================================

class IDMVTONEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.ready = False

        try:
            print("=" * 50)
            print("🔧 ENGINE 1: Loading IDM_VTON (SOTA Clothes)...")
            print("=" * 50)

            # Import custom pipeline from cloned repo
            from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
            from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from src.unet_hacked_tryon import UNet2DConditionModel
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer,
            )
            from diffusers import DDPMScheduler, AutoencoderKL

            # Load preprocessing models FIRST to avoid meta tensor corruption from diffusers
            print("   ⏳ Loading Human Parsing + OpenPose...")
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose
            self.parsing_model = Parsing(0)
            self.openpose_model = OpenPose(0)

            # Load all model components
            print("   ⏳ Loading TryonNet UNet (13-channel)...")
            unet = UNet2DConditionModel.from_pretrained(
                IDM_MODEL_PATH, subfolder="unet", torch_dtype=torch.float16,
            )
            unet.requires_grad_(False)

            print("   ⏳ Loading GarmentNet UNet Encoder...")
            self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
                IDM_MODEL_PATH, subfolder="unet_encoder", torch_dtype=torch.float16,
            )
            self.UNet_Encoder.requires_grad_(False)

            print("   ⏳ Loading Tokenizers...")
            tokenizer_one = AutoTokenizer.from_pretrained(IDM_MODEL_PATH, subfolder="tokenizer", use_fast=False)
            tokenizer_two = AutoTokenizer.from_pretrained(IDM_MODEL_PATH, subfolder="tokenizer_2", use_fast=False)

            print("   ⏳ Loading Scheduler, Encoders, VAE...")
            noise_scheduler = DDPMScheduler.from_pretrained(IDM_MODEL_PATH, subfolder="scheduler")
            text_encoder_one = CLIPTextModel.from_pretrained(IDM_MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.float16)
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(IDM_MODEL_PATH, subfolder="text_encoder_2", torch_dtype=torch.float16)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(IDM_MODEL_PATH, subfolder="image_encoder", torch_dtype=torch.float16)
            vae = AutoencoderKL.from_pretrained(IDM_MODEL_PATH, subfolder="vae", torch_dtype=torch.float16)

            # Freeze all
            for m in [image_encoder, vae, unet, text_encoder_one, text_encoder_two]:
                m.requires_grad_(False)

            # Assemble the TryonPipeline using the robust path
            print("   ⏳ Assembling TryonPipeline...")
            self.pipe = TryonPipeline.from_pretrained(
                IDM_MODEL_PATH,
                unet=unet, vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
            )
            self.pipe.unet_encoder = self.UNet_Encoder

            self.tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            self.ready = True
            print("   ✅ IDM_VTON Engine: READY")

        except Exception as e:
            print(f"   ❌ IDM_VTON Engine FAILED: {e}")
            import traceback; traceback.print_exc()
            self.pipe = None
            self.ready = False

    def __call__(self, person_image, garment_image, category="tops", garment_desc=None):
        if not self.ready:
            return None

        device = self.device
        print(f"   🎨 IDM_VTON: Running for '{category}'...")

        self.openpose_model.preprocessor.body_estimation.model.to(device)
        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)

        garm_img = garment_image.convert("RGB").resize((768, 1024))
        human_img = person_image.convert("RGB").resize((768, 1024))

        # Auto-mask
        from IDM_VTON.gradio_demo.utils_mask import get_mask_location
        keypoints = self.openpose_model(human_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(human_img.resize((384, 512)))

        mask_type = "upper_body"
        if category in ("bottoms", "lower_body"):
            mask_type = "lower_body"
        elif category in ("dresses", "one-pieces", "full_body"):
            mask_type = "dresses"

        mask, _ = get_mask_location("hd", mask_type, model_parse, keypoints)
        mask = mask.resize((768, 1024))
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        # DensePose
        from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
        
        # Paths must be relative to IDM_VTON repo directory
        idm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IDM_REPO)
        
        # Force Python to load IDM-VTON's modified apply_net.py which does NOT require <input>
        gradio_demo_dir = os.path.join(idm_dir, "gradio_demo")
        if gradio_demo_dir not in sys.path:
            sys.path.insert(0, gradio_demo_dir)
        
        import apply_net
        import importlib
        importlib.reload(apply_net)  # ensure we have the correct one

        # Paths must be relative to IDM_VTON repo directory
        densepose_cfg = os.path.join(idm_dir, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
        densepose_ckpt = os.path.join(idm_dir, "ckpt", "densepose", "model_final_162be9.pkl")

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        args = apply_net.create_argument_parser().parse_args((
            "show", densepose_cfg,
            densepose_ckpt, "dp_segm",
            "-v", "--opts", "MODEL.DEVICE", "cuda",
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        # Inference
        if garment_desc is None:
            garment_desc = category
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                prompt = "model is wearing " + garment_desc
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                (prompt_embeds, negative_prompt_embeds,
                 pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                    prompt, num_images_per_prompt=1,
                    do_classifier_free_guidance=True, negative_prompt=negative_prompt,
                )

                prompt_c = ["a photo of " + garment_desc]
                (prompt_embeds_c, _, _, _) = self.pipe.encode_prompt(
                    prompt_c, num_images_per_prompt=1,
                    do_classifier_free_guidance=False, negative_prompt=[negative_prompt],
                )

                pose_tensor = self.tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(42)

                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=30, generator=generator, strength=1.0,
                    pose_img=pose_tensor,
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask, image=human_img,
                    height=1024, width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]

        print("   ✅ IDM_VTON inference complete!")
        return images[0]


# =====================================================================
# ENGINE 2: AnyDoor (Alibaba) — SOTA for Accessories 
# =====================================================================

class AnyDoorEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.ready = False
        try:
            print("=" * 50)
            print(f"🔧 ENGINE 2: (IDM-MODE) Loading AnyDoor...")
            print("=" * 50)
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            anydoor_path = os.path.join(base_dir, 'AnyDoor')

            # Standalone robust wrapper
            from anydoor_wrapper import init_anydoor_model, inference_single_image
            init_anydoor_model(anydoor_dir=anydoor_path, device=self.device)
            self.inference_fn = inference_single_image
            self.ready = True
            print("   ✅ AnyDoor Engine: READY")
        except Exception as e:
            print(f"   ❌ AnyDoor Engine FAILED: {e}")
            self.inference_fn = None

    def _make_bg_mask(self, w, h, category):
        mask = np.zeros((h, w), dtype=np.uint8)
        cat = category.lower()
        if "shoe" in cat or "foot" in cat:
            y1,y2,x1,x2 = int(h*0.8), int(h*0.98), int(w*0.25), int(w*0.75)
        elif "hat" in cat or "head" in cat:
            y1,y2,x1,x2 = int(h*0.02), int(h*0.18), int(w*0.25), int(w*0.75)
        elif "bag" in cat or "purse" in cat:
            y1,y2,x1,x2 = int(h*0.45), int(h*0.85), int(w*0.05), int(w*0.4)
        else:
            y1,y2,x1,x2 = int(h*0.4), int(h*0.8), int(w*0.1), int(w*0.4)
        mask[y1:y2, x1:x2] = 255
        return mask

    def __call__(self, person_image, garment_image, category="shoes", garment_desc=None):
        if not self.ready: return None
        try:
            print(f"   🎨 AnyDoor: Teleporting '{category}' onto person...")
            p_img = person_image.convert("RGB").resize((768, 1024))
            g_img = garment_image.convert("RGBA")
            garment_np = np.array(g_img)
            if garment_np.shape[2] == 4:
                ref_mask = (garment_np[:, :, 3] > 10).astype(np.uint8) * 255
            else:
                gray = cv2.cvtColor(garment_np[:,:,:3], cv2.COLOR_RGB2GRAY)
                _, ref_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

            ref_image = cv2.cvtColor(garment_np[:,:,:3], cv2.COLOR_RGB2BGR)
            tar_image = cv2.cvtColor(np.array(p_img), cv2.COLOR_RGB2BGR)
            h, w = tar_image.shape[:2]  # h=1024, w=768
            tar_mask = self._make_bg_mask(w, h, category)

            with torch.inference_mode():
                gen = self.inference_fn(
                    ref_image, (ref_mask > 128).astype(np.uint8),
                    tar_image, (tar_mask > 128).astype(np.uint8)
                )
            result_rgb = cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)
            print("   ✅ AnyDoor inference complete!")
            return Image.fromarray(result_rgb)
        except Exception as e:
            print(f"   ❌ AnyDoor processing error: {e}")
            import traceback; traceback.print_exc()
            return None


# =====================================================================
# DUAL ENGINE ROUTER
# =====================================================================

class DualEngineVTON:
    def __init__(self, device="cuda"):
        self.device = device
        self.idm = None
        self.anydoor = None
        self.blip_processor = None
        self.blip_model = None

        print("\n" + "=" * 50)
        print("📊 (IDM-MODE) Dynamic Router Status:")
        print("   Models will load exclusively when triggered.")
        print("=" * 50)

    @property
    def ready(self): return True

    def _unload_idm(self):
        if self.idm is not None:
            print("   🧹 Unloading IDM_VTON...")
            if hasattr(self.idm, 'pipe') and self.idm.pipe is not None:
                del self.idm.pipe
            if hasattr(self.idm, 'parsing_model'):
                del self.idm.parsing_model
            if hasattr(self.idm, 'openpose_model'):
                del self.idm.openpose_model
            self.idm = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _unload_anydoor(self):
        if self.anydoor is not None:
            print("   🧹 Unloading AnyDoor...")
            self.anydoor = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _generate_caption(self, garment_img, category):
        if self.blip_model is None:
            print("   🧠 Loading BLIP Captioner...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
            ).to(self.device)
            
        try:
            inputs = self.blip_processor(garment_img, return_tensors="pt").to(self.device, torch.float16)
            out = self.blip_model.generate(**inputs, max_new_tokens=20)
            desc = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"   📝 Auto-Caption: {desc}")
            return desc
        except Exception as e:
            print(f"   ❌ BLIP Failed: {e}")
            return category

    def _segment_garment(self, garment_img, category):
        w, h = garment_img.size
        # 1. Semantic Extraction
        if self.idm is not None and hasattr(self.idm, 'parsing_model'):
            try:
                parsed_image, _ = self.idm.parsing_model(garment_img.resize((384, 512)))
                parse_array = np.array(parsed_image)
                
                cat_lower = category.lower()
                if "top" in cat_lower or "upper" in cat_lower:
                    targets = [4]
                elif "bottom" in cat_lower or "lower" in cat_lower or "pant" in cat_lower:
                    targets = [6, 5]
                elif "dress" in cat_lower or "full" in cat_lower:
                    targets = [7, 4, 5, 6]
                else:
                    targets = [4, 5, 6, 7]
                    
                mask = np.isin(parse_array, targets).astype(np.uint8) * 255
                if np.sum(mask) > 1000:
                    print("   ✂️ Semantic extraction successful (found person).")
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    kernel = np.ones((5,5), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    mask = cv2.erode(mask, kernel, iterations=2)
                    
                    rgba = garment_img.convert("RGBA")
                    rgba.putalpha(Image.fromarray(mask).convert("L"))
                    bbox = rgba.getbbox()
                    if bbox: rgba = rgba.crop(bbox)
                    
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba)
                    return bg
            except Exception as e:
                print(f"   ❌ Semantic extraction failed: {e}")
                
        # 2. Rembg Fallback
        print("   ✂️ Falling back to rembg for garment segmentation...")
        try:
            from rembg import remove
            rgba = remove(garment_img.convert("RGBA"))
            bbox = rgba.getbbox()
            if bbox: rgba = rgba.crop(bbox)
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba)
            return bg
        except Exception as e:
            print(f"   ❌ rembg failed: {e}")
            return garment_img

    def __call__(self, person_image, garment_image, category="tops", garment_desc=None):
        is_accessory = any(x in category.lower() for x in ["shoe", "bag", "hat", "accessory", "accessories", "jewelry"])

        # Load engine first so we have access to parsing_model
        if is_accessory:
            if self.anydoor is None:
                self._unload_idm()
                self.anydoor = AnyDoorEngine(self.device)
        else:
            if self.idm is None:
                self._unload_anydoor()
                self.idm = IDMVTONEngine(self.device)

        print("   🔍 Enhancing garment image...")
        garment_image = self._segment_garment(garment_image, category)
        
        if garment_desc is None or garment_desc.strip().lower() == category.lower():
            garment_desc = self._generate_caption(garment_image, category)

        if is_accessory:
            return self.anydoor(person_image, garment_image, category, garment_desc)
        else:
            return self.idm(person_image, garment_image, category, garment_desc)


# --- INITIALIZE ---
pipeline = None
try:
    print(f"✅ Numpy: {np.__version__}")
    pipeline = DualEngineVTON(device="cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"⚠️ Startup error: {e}")
    import traceback; traceback.print_exc()


# --- HELPERS ---
def base64_to_image(b64):
    if "," in b64: b64 = b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return ImageOps.exif_transpose(img)

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- API ---
@app.route("/api/tryon", methods=["POST"])
def tryon():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data"}), 400

        person_b64 = data.get("person")
        garment_b64 = data.get("garment")
        category = data.get("category", "tops")
        garment_desc = data.get("garment_desc", category)

        if not person_b64 or not garment_b64:
            return jsonify({"success": False, "error": "Missing person or garment image"}), 400

        print(f"\n📥 Request: category={category}")
        person_img = base64_to_image(person_b64).convert("RGB").resize((768, 1024))
        garment_img = base64_to_image(garment_b64).convert("RGB").resize((768, 1024))

        # Gender cross-check
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
        print("🚀 [IDM-MODE] Starting inference...")
        if not pipeline or not pipeline.ready:
            return jsonify({"success": False, "error": "No engine available"}), 500

        result_img = pipeline(person_img, garment_img, category, garment_desc)

        if result_img is None:
            return jsonify({"success": False, "error": "Inference returned no result"}), 500

        result_b64 = image_to_base64(result_img)

        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"📉 VRAM: {vram_mb:.0f}MB")
        print("✅ Done!\n")

        return jsonify({"success": True, "result": f"data:image/jpeg;base64,{result_b64}"})

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    idm_ready = (pipeline.idm is not None and pipeline.idm.ready) if pipeline else False
    anydoor_ready = (pipeline.anydoor is not None and pipeline.anydoor.ready) if pipeline else False
    return jsonify({
        "status": "ok",
        "idm_vton": idm_ready,
        "anydoor": anydoor_ready,
        "vram_use": f"{torch.cuda.memory_allocated() / 1024**2:.0f}MB" if torch.cuda.is_available() else "cpu"
    })


# --- SERVER ---
def run_server():
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    if not NGROK_TOKEN:
        print("❌ NGROK_TOKEN missing"); sys.exit(1)

    ngrok.set_auth_token(NGROK_TOKEN)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    url = ngrok.connect(5000).public_url
    print("=" * 60)
    print(f"🚀 VTON API: {url}")
    print("=" * 60)
    print(f"   POST {url}/api/tryon")
    print(f"   GET  {url}/api/health")
    print("⏳ Running...\n")

    while True:
        time.sleep(10)
