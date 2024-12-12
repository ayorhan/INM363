import os
import gdown
import zipfile
import requests
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse
import hashlib

def download_coco_images(num_images, output_dir, seed=42):
    """
    Download specific number of images from COCO dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # COCO API URL for 2017 validation set (smaller than training set)
    COCO_URL = "http://images.cocodataset.org/val2017"
    COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # First, download and cache annotation file if not present
    annotation_dir = Path("data/annotations")
    annotation_dir.mkdir(exist_ok=True)
    annotation_file = annotation_dir / "instances_val2017.json"
    
    if not annotation_file.exists():
        print("Downloading COCO annotations...")
        response = requests.get(COCO_ANNOTATIONS)
        zip_path = annotation_dir / "annotations.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(annotation_dir.parent)
        os.remove(zip_path)
    
    # Load annotations
    with open(annotation_file) as f:
        data = json.load(f)
    
    # Get consistent subset of images using seed
    import random
    random.seed(seed)
    selected_images = random.sample(data['images'], min(num_images, len(data['images'])))
    
    # Download images
    for img in tqdm(selected_images, desc="Downloading COCO images"):
        img_url = f"{COCO_URL}/{img['file_name']}"
        img_path = Path(output_dir) / img['file_name']
        
        if not img_path.exists():
            response = requests.get(img_url)
            with open(img_path, "wb") as f:
                f.write(response.content)

def download_wikiart_images(num_images, output_dir, seed=42):
    """
    Download style images from PyTorch's examples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List of style images from PyTorch's examples
    STYLE_IMAGES = [
        "candy.jpg", "mosaic.jpg", "rain-princess-cropped.jpg", "udnie.jpg",
        "seated-nude.jpg", "style1.jpg", "style2.jpg", "style3.jpg",
        "style4.jpg", "style5.jpg", "style6.jpg", "style7.jpg",
        "style8.jpg", "style9.jpg", "style10.jpg", "style11.jpg",
        "style12.jpg", "style13.jpg", "la_muse.jpg", "starry_night.jpg",
        "the_scream.jpg"
    ]
    
    BASE_URL = "https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images"
    
    # Get consistent subset using seed
    import random
    random.seed(seed)
    selected_images = random.sample(STYLE_IMAGES, min(num_images, len(STYLE_IMAGES)))
    
    # Download images
    for img_name in tqdm(selected_images, desc="Downloading style images"):
        img_url = f"{BASE_URL}/{img_name}"
        img_path = Path(output_dir) / img_name
        
        if not img_path.exists():
            try:
                response = requests.get(img_url)
                response.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {img_name}: {e}")
                continue

    # Return number of successfully downloaded images
    return len(list(Path(output_dir).glob("*.jpg")))

def verify_images(directory):
    """
    Verify all images in directory are valid
    Remove corrupted images
    """
    for img_path in Path(directory).glob("*"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except:
            print(f"Removing corrupted image: {img_path}")
            os.remove(img_path)

def main():
    parser = argparse.ArgumentParser(description='Download datasets for style transfer')
    parser.add_argument('--content-images', type=int, default=100,
                        help='Number of content images to download')
    parser.add_argument('--style-images', type=int, default=50,
                        help='Number of style images to download')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for consistent downloads')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup directories
    content_dir = Path(args.output_dir) / "coco"
    style_dir = Path(args.output_dir) / "wikiart"
    
    # Download datasets
    print(f"Downloading {args.content_images} content images...")
    download_coco_images(args.content_images, content_dir, args.seed)
    
    print(f"Downloading {args.style_images} style images...")
    download_wikiart_images(args.style_images, style_dir, args.seed)
    
    # Verify images
    print("Verifying downloaded images...")
    verify_images(content_dir)
    verify_images(style_dir)
    
    print(f"""
    Dataset download complete:
    - Content images: {len(list(content_dir.glob("*")))} images in {content_dir}
    - Style images: {len(list(style_dir.glob("*")))} images in {style_dir}
    """)

if __name__ == "__main__":
    main() 