import cv2
import einops
import numpy as np
import torch
import random
import os
import sys

# Add AnyDoor to path so it can find its submodules like cldm
base_dir = os.path.dirname(os.path.abspath(__file__))
anydoor_root = os.path.join(base_dir, 'AnyDoor')
if anydoor_root not in sys.path:
    sys.path.append(anydoor_root)

# Standard AnyDoor Imports
try:
    import open_clip
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler
    from cldm.hack import disable_verbosity, enable_sliced_attention
    from datasets.data_utils import * 
    import albumentations as A
    from omegaconf import OmegaConf
except ImportError as e:
    print(f"\n❌ AnyDoor Wrapper: Dependency missing: {e}")
    print("👉 FIX: Run 'pip install open-clip-torch pytorch-lightning albumentations xformers'\n")
    # Don't raise here, allow init_anydoor_model to handle the missing imports if called

_ANYDOOR_MODEL = None
_DDIM_SAMPLER = None
_CONFIG_LOADED = False

def init_anydoor_model(anydoor_dir=None, device='cuda'):
    global _ANYDOOR_MODEL, _DDIM_SAMPLER, _CONFIG_LOADED
    if _CONFIG_LOADED:
        return
    
    if anydoor_dir is None:
        anydoor_dir = anydoor_root
        
    print(f"🔧 Wrapper: Initializing AnyDoor from {anydoor_dir}...")
    
    disable_verbosity()
    
    # Load Configs
    inference_yaml = os.path.join(anydoor_dir, 'configs', 'inference.yaml')
    config = OmegaConf.load(inference_yaml)
    
    # Weights Path
    model_ckpt = os.path.join(anydoor_dir, "path", "epoch=1-step=8687.ckpt")
    
    # Model Config
    model_config_yaml = os.path.join(anydoor_dir, config.config_file)
    model_config = OmegaConf.load(model_config_yaml)
    
    # Point DINOv2 to our local pth file
    model_config.model.params.cond_stage_config.weight = os.path.join(anydoor_dir, "path", "dinov2_vitg14_pretrain.pth")

    # Construct Model
    _ANYDOOR_MODEL = create_model(config_dict=model_config).cpu()
    _ANYDOOR_MODEL.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
    _ANYDOOR_MODEL = _ANYDOOR_MODEL.to(device)
    _DDIM_SAMPLER = DDIMSampler(_ANYDOOR_MODEL)
    
    _CONFIG_LOADED = True
    print("✅ AnyDoor Wrapper: Model Loaded Successfully!")

def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # From original AnyDoor repo code
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
    
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]

    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    ref_image_collage = sobel(masked_ref_image, ref_mask/255)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage
    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    return dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 

def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5
    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        return tar_image
    if W1 < W2:
        pad1 = int((W2 - W1) / 2); pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2); pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    tar_image[y1+m :y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
    return tar_image

def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    if not _CONFIG_LOADED:
        raise RuntimeError("AnyDoor model not initialized. Call init_anydoor_model() first.")
        
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    num_samples = 1
    
    control = torch.from_numpy(item['hint'].copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    clip_input = torch.from_numpy(item['ref'].copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    H,W = 512,512
    cond = {"c_concat": [control], "c_crossattn": [_ANYDOOR_MODEL.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": [control], "c_crossattn": [_ANYDOOR_MODEL.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    _ANYDOOR_MODEL.control_scales = [1.0] * 13
    samples, _ = _DDIM_SAMPLER.sample(20, num_samples, shape, cond, verbose=False, eta=0.0,
                                    unconditional_guidance_scale=guidance_scale,
                                    unconditional_conditioning=un_cond)

    x_samples = _ANYDOOR_MODEL.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
    pred = np.clip(x_samples[0], 0, 255)
    
    return crop_back(pred, tar_image, item['extra_sizes'], item['tar_box_yyxx_crop'])
