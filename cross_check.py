import torch
import clip
from PIL import Image
import io
import base64

class CrossCheckValidator:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🎨 Initializing CLIP on {self.device.upper()}...")
        # We use ViT-B/32 for a good balance of speed and accuracy
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define our text prompts for zero-shot classification - Using more descriptive, weighted terms
        self.gender_prompts = [
            "a photo of a man, masculine face, male person", 
            "a photo of a woman, feminine face, female person"
        ]
        self.garment_prompts = [
            "masculine clothing, menswear, clothing for men, suit, t-shirt for men", 
            "feminine clothing, womenswear, clothing for women, dress, blouse, skirt"
        ]
        
        print("✅ CLIP Validator Ready with enhanced prompts!")

    def classify_advanced(self, image, class_labels):
        """
        Classifies an image using ensembled prompts for each label.
        class_labels: dictionary where key is class name, value is list of prompts.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        all_probs = {}
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            for label, prompts in class_labels.items():
                text_tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Average similarity across all prompts for this label
                similarities = (image_features @ text_features.T).mean(dim=-1)
                all_probs[label] = similarities.item()
                
        # Result with highest average similarity
        best_label = max(all_probs, key=all_probs.get)
        
        # Normalize to look like probabilities (for logging)
        total = sum(all_probs.values())
        print(f"   [CLIP] Confidence: { {k: f'{(v/total)*100:.1f}%' for k,v in all_probs.items()} }")
        
        return best_label

    def validate_match(self, person_image, garment_image, category="tops"):
        """
        Validates if the garment matches the person's gender using ensembled prompts.
        """
        try:
            # 1. Classify Person
            print(f"   [CLIP] Classifying Person (Target: {category})...")
            person_classes = {
                "male": ["a photo of a man", "a masculine face", "a man with a beard", "a male human", "a portrait of a guy", "short hair man", "businessman"],
                "female": ["a photo of a woman", "a feminine face", "a woman with long hair", "a female human", "a lady", "woman wearing makeup", "fashion model woman"]
            }
            person_gender = self.classify_advanced(person_image, person_classes)
            
            # 2. Classify Garment
            print("   [CLIP] Classifying Garment...")
            
            # If it's a dress, it's overwhelmingly likely to be female
            if category == 'dresses' or category == 'one-pieces':
                garment_gender = "female"
                print("   [CLIP] Auto-detected FEMALE for Dress/One-piece category")
            else:
                garment_classes = {
                    "male": ["menswear", "masculine clothing", "men's shirt", "men's suit", "clothing for men"],
                    "female": ["womenswear", "feminine clothing", "woman's blouse", "female fashion", "feminine top", "clothing for women"]
                }
                garment_gender = self.classify_advanced(garment_image, garment_classes)
            
            print(f"🔍 Validation Result: PERSON={person_gender} | GARMENT={garment_gender}")
            
            if person_gender == garment_gender:
                return True, "Match success!", person_gender, garment_gender
            else:
                return False, f"Gender mismatch: Person is {person_gender} but garment is {garment_gender}.", person_gender, garment_gender
                
        except Exception as e:
            print(f"❌ CLIP Validation Error: {str(e)}")
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
