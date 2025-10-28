#!/usr/bin/env python3
"""
Model Setup Script for AR Furniture App
Downloads required models automatically during deployment
"""

import os
import torch
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def setup_sam_models():
    """Download SAM models if they don't exist"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # SAM ViT-H model
    sam_h_path = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_h_path):
        print("🔄 Downloading SAM ViT-H model...")
        sam_h_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        download_file(sam_h_url, sam_h_path)
        print(f"✅ SAM ViT-H model downloaded: {sam_h_path}")
    else:
        print("✅ SAM ViT-H model already exists")
    
    # SAM ViT-L model
    sam_l_path = os.path.join(models_dir, "sam_vit_l_0b3195.pth")
    if not os.path.exists(sam_l_path):
        print("🔄 Downloading SAM ViT-L model...")
        sam_l_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        download_file(sam_l_url, sam_l_path)
        print(f"✅ SAM ViT-L model downloaded: {sam_l_path}")
    else:
        print("✅ SAM ViT-L model already exists")

def setup_sam2_models():
    """Download SAM2 models if they don't exist"""
    sam2_h_path = "sam2_hiera_tiny.pt"
    if not os.path.exists(sam2_h_path):
        print("🔄 Downloading SAM2 model...")
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
        download_file(sam2_url, sam2_h_path)
        print(f"✅ SAM2 model downloaded: {sam2_h_path}")
    else:
        print("✅ SAM2 model already exists")

def setup_stable_diffusion():
    """Setup Stable Diffusion models (will be downloaded by diffusers automatically)"""
    print("✅ Stable Diffusion models will be downloaded automatically by diffusers library")
    print("   Model: runwayml/stable-diffusion-inpainting")

def main():
    """Main setup function"""
    print("🚀 Setting up AR Furniture App models...")
    print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        setup_sam_models()
        setup_sam2_models()
        setup_stable_diffusion()
        print("\n🎉 All models setup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during model setup: {e}")
        print("💡 Models will be downloaded automatically when first used")
        return False
    
    return True

if __name__ == "__main__":
    main()
