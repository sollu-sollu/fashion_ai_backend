import os 
import subprocess
import sys

REPO_NAME = "fashion"

# Clone if not exists
if not os.path.exists(REPO_NAME):
    print(f"📂 Cloning {REPO_NAME}...")
    subprocess.run(["git", "clone", "https://github.com/fashn-AI/fashn-vton-1.5.git", REPO_NAME], check=True)

print(f"📍 Current Directory: {os.getcwd()}")
print("system executable", sys.executable)

if not os.path.exists("fashion/weights/model.safetensors"):
    print("⏳ Downloading weights (this may take a while)...")
    # Use sys.executable to ensure we use the same python interpreter
    subprocess.run([sys.executable, "fashion/scripts/download_weights.py", "--weights-dir", "fashion/weights"], check=True)
    print("✅ Weights Downloaded!")
else:
    print("✅ Weights already exist!")


import torch
from fashion.src.fashn_vton import TryOnPipeline
from PIL import Image
import io
import base64
from cross_check import validate_gender_match

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Loading on {device.upper()}...")

pipeline = TryOnPipeline("fashion/weights", device=device)
print("✅ Model Loaded!")


from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading

app = Flask(__name__)

# ⚠️ PUT YOUR NGROK TOKEN HERE ⚠️
NGROK_TOKEN = "36EnpN0f0tx9hre6EMpt1jRpHeb_38GYquy1KabEJGskmKeHL"

if not NGROK_TOKEN or NGROK_TOKEN == "YOUR_TOKEN_HERE":
    raise ValueError("❌ Please enter your Ngrok token!")

ngrok.set_auth_token(NGROK_TOKEN)

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/api/tryon', methods=['POST'])
def tryon():
    """
    Virtual Try-On API Endpoint
    
    Request JSON:
    {
        "person": "base64_image_string",
        "garment": "base64_image_string", 
        "category": "tops" | "bottoms" | "dresses"
    }
    
    Response JSON:
    {
        "success": true,
        "result": "base64_image_string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        person_b64 = data.get('person')
        garment_b64 = data.get('garment')
        category = data.get('category', 'tops')
        
        if not person_b64 or not garment_b64:
            return jsonify({"success": False, "error": "Missing person or garment image"}), 400
        
        # Convert base64 to images
        person_img = base64_to_image(person_b64).convert("RGB").resize((768, 1024))
        garment_img = base64_to_image(garment_b64).convert("RGB").resize((768, 1024))
        
        print(f"👕 Processing {category}...")
        
        # --- NEW: CROSS-CHECK VALIDATION ---
        print("🔍 Running gender cross-check...")
        is_valid, msg, p_gen, g_gen = validate_gender_match(person_img, garment_img, category)
        
        if not is_valid:
            print(f"❌ Validation Failed: {msg}")
            return jsonify({
                "success": False, 
                "error": "validation_failed",
                "message": f"Chosen garment doesn't match the selected person's category. ({msg})",
                "details": {"person": p_gen, "garment": g_gen}
            }), 400
        
        print("✅ Validation Passed!")
        # ----------------------------------
        
        # Run inference
        result = pipeline(
            person_image=person_img,
            garment_image=garment_img,
            category=category
        )
        
        # Extract image from PipelineOutput object
        if hasattr(result, 'images') and len(result.images) > 0:
            result_image = result.images[0]
        else:
            # Fallback if it returns the image directly or another format
            result_image = result
            
        result_b64 = image_to_base64(result_image)
        
        print("✅ Done!")
        
        if torch.cuda.is_available():
            print(f"📉 VRAM Usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")
        
        return jsonify({
            "success": True,
            "result": f"data:image/jpeg;base64,{result_b64}"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "device": device})

# Start server in background thread
def run_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)

server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# Create Ngrok tunnel
public_url = ngrok.connect(5000).public_url

print("=" * 60)
print(f"🚀 YOUR API URL: {public_url}")
print("=" * 60)
print("")
print("📝 API Endpoints:")
print(f"   POST {public_url}/api/tryon  - Run try-on")
print(f"   GET  {public_url}/api/health - Health check")
print("")
print("⏳ Keep this cell running...")

# Keep the cell alive
import time
while True:
    time.sleep(10)