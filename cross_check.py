import torch
import clip
from PIL import Image
import numpy as np
import sys

print(f"📦 Python version: {sys.version}")
try:
    print(f"📦 Numpy version: {np.__version__}")
except Exception as e:
    print(f"⚠️ Numpy check error: {e}")

class CrossCheckValidator:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🎨 Initializing CLIP on {self.device.upper()}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP Validator Ready with object & gender verification!")

    def classify_advanced(self, image, class_labels):
        """Classifies an image using ensembled prompts for each label."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        all_probs = {}
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            for label, prompts in class_labels.items():
                text_tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).mean(dim=-1)
                all_probs[label] = similarities.item()
                
        best_label = max(all_probs, key=all_probs.get)
        
        # Log probabilities
        total = sum(all_probs.values()) + 1e-9
        print(f"   [CLIP] Confidence: { {k: f'{(v/total)*100:.1f}%' for k,v in all_probs.items()} }")
        
        return best_label

    def validate_match(self, person_image, garment_image, category="tops"):
        try:
            category_lower = category.lower()

            # ==========================================
            # 1. OBJECT VERIFICATION (Is it really a shoe/bag/shirt?)
            # ==========================================
            print(f"   [CLIP] Verifying Object Type (Target: {category})...")
            
            # Define broad object categories
            object_classes = {
                "clothing": ["a photo of clothing, a shirt, pants, dress, jacket, apparel"],
                "shoe": ["a photo of a shoe, sneakers, footwear, boots, heels"],
                "bag": ["a photo of a bag, handbag, backpack, purse, luggage"],
                "hat": ["a photo of a hat, cap, beanie, headwear"],
                "invalid": ["a photo of an animal, dog, cat, car, food, landscape, random object"]
            }
            
            detected_object = self.classify_advanced(garment_image, object_classes)
            
            # If the image is completely invalid (like a dog)
            if detected_object == "invalid":
                return False, "Uploaded image does not appear to be a fashion item (detected random object/animal).", "unknown", detected_object

            # Validate that the selected category somewhat matches the visual item
            is_accessory_cat = any(x in category_lower for x in ["shoe", "bag", "hat", "accessor", "jewel", "watch", "glasses"])
            
            if is_accessory_cat:
                if detected_object == "clothing":
                    # They uploaded a shirt but selected "accessories"
                    return False, f"You selected '{category}' but the image looks like clothing.", "unknown", detected_object
                
                print(f"   [CLIP] Accessory visually verified ({detected_object}). Bypassing gender check.")
                return True, "Accessory visually verified", "neutral", detected_object

            else:
                if detected_object in ["shoe", "bag", "hat"]:
                    # They uploaded a shoe but selected "tops"
                    return False, f"You selected a clothing category ('{category}') but uploaded an accessory ({detected_object}).", "unknown", detected_object


            # ==========================================
            # 2. GENDER MATCHING (Only for Clothing)
            # ==========================================
            print(f"   [CLIP] Classifying Person Gender...")
            person_classes = {
                "male": ["a photo of a man", "a masculine face", "a male human", "short hair man"],
                "female": ["a photo of a woman", "a feminine face", "a female human", "a lady"]
            }
            person_gender = self.classify_advanced(person_image, person_classes)
            
            print("   [CLIP] Classifying Garment Gender...")
            if category_lower in ['dresses', 'one-pieces', 'skirt']:
                garment_gender = "female"
                print("   [CLIP] Auto-detected FEMALE for Dress/Skirt category")
            else:
                garment_classes = {
                    "male": ["menswear", "masculine clothing", "men's clothing", "boy's shirt", "suit"],
                    "female": ["womenswear", "feminine clothing", "women's clothing", "blouse", "women's top"]
                }
                garment_gender = self.classify_advanced(garment_image, garment_classes)
            
            print(f"🔍 Validation Result: PERSON={person_gender} | GARMENT={garment_gender}")
            
            if person_gender == garment_gender:
                return True, "Match success!", person_gender, garment_gender
            else:
                return False, f"Gender mismatch: Person is {person_gender} but clothing is {garment_gender}.", person_gender, garment_gender
                
        except Exception as e:
            print(f"❌ CLIP Validation Error: {str(e)}")
            import traceback; traceback.print_exc()
            return True, f"Validation skipped: {str(e)}", "unknown", "unknown"

# Singleton instance
_validator = None

def get_validator():
    global _validator
    if _validator is None:
        _validator = CrossCheckValidator()
    return _validator

def validate_gender_match(person_img, garment_img, category="tops"):
    """Entry point for fash_backend.py"""
    validator = get_validator()
    return validator.validate_match(person_img, garment_img, category)

