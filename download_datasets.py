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
import fiftyone as fo
import fiftyone.zoo as foz
import random
import shutil

def download_coco_images(split="train", output_dir=None, num_images=None, seed=42):
    """
    Download images from COCO dataset
    Args:
        split: 'train' or 'val'
        output_dir: Output directory
        num_images: Number of images to download
        seed: Random seed for reproducibility
    """
    # Use FiftyOne to download a small subset directly
    import fiftyone as fo
    import fiftyone.zoo as foz
    
    # Create output directory
    output_dir = output_dir or f"data/coco/{split}"
    os.makedirs(output_dir, exist_ok=True)
    final_dir = Path(output_dir) / "images"
    final_dir.mkdir(exist_ok=True)

    print(f"Downloading {num_images} COCO {split} images...")
    
    # Download only the required number of images
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        max_samples=num_images,
        shuffle=True,
        seed=seed
    )
    
    # Export the images to our directory
    dataset.export(
        export_dir=str(final_dir),
        dataset_type=fo.types.ImageDirectory
    )
    
    return final_dir

def download_style_images_kaggle(num_images=1500, output_dir=None, seed=42):
    """
    Download style images from Kaggle Painter by Numbers dataset
    Args:
        num_images: Number of images to download
        output_dir: Output directory
        seed: Random seed for reproducibility
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError("Please install kaggle package: pip install kaggle")

    output_dir = output_dir or "data/style"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset from Kaggle (using dataset instead of competition)
    print("Downloading Painter by Numbers dataset from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'ikarus777/best-artworks-of-all-time',  # This is an alternative art dataset
        path=output_dir,
        unzip=True
    )
    
    # Process downloaded images
    print(f"Processing {num_images} images...")
    image_files = list(Path(output_dir).rglob('*.jpg'))
    
    # Randomly select images
    random.seed(seed)
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    # Move selected images to the root of output directory
    for img_path in tqdm(selected_images, desc="Processing images"):
        dest_path = Path(output_dir) / img_path.name
        if not dest_path.exists():
            shutil.copy2(img_path, dest_path)
    
    # Clean up subdirectories
    for item in Path(output_dir).iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    
    # Verify images
    verify_images(output_dir)
    
    return len(list(Path(output_dir).glob('*.jpg')))

def verify_images(directory):
    """Verify all images in directory are valid"""
    for img_path in Path(directory).glob("*"):
        try:
            with Image.open(img_path) as img:
                img.verify()
                # Also check minimum size
                if img.size[0] < 256 or img.size[1] < 256:
                    print(f"Removing small image: {img_path}")
                    os.remove(img_path)
        except:
            print(f"Removing corrupted image: {img_path}")
            os.remove(img_path)

def main():
    parser = argparse.ArgumentParser(description='Download datasets for style transfer')
    parser.add_argument('--content-images', type=int, default=20000,
                        help='Number of content images to download')
    parser.add_argument('--style-images', type=int, default=1500,
                        help='Number of style images to download')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for consistent downloads')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='Output directory')
    parser.add_argument('--download-train-content', action='store_true',
                      help='Download training content images from COCO')
    parser.add_argument('--download-train-style', action='store_true',
                      help='Download training style images from Kaggle')
    parser.add_argument('--download-val-content', action='store_true',
                      help='Download validation content images from COCO')
    parser.add_argument('--download-val-style', action='store_true',
                      help='Download validation style images from Kaggle')
    
    args = parser.parse_args()
    
    if args.download_train_content:
        print("Downloading training content images...")
        train_content_dir = download_coco_images("train", 
                                               f"{args.output_dir}/coco/train", 
                                               args.content_images, 
                                               args.seed)
    
    if args.download_train_style:
        print("Downloading training style images...")
        train_style_dir = download_style_images_kaggle(args.style_images,
                                                     f"{args.output_dir}/style/train",
                                                     args.seed)
    
    if args.download_val_content:
        print("Downloading validation content images...")
        val_content_dir = download_coco_images("validation",
                                             f"{args.output_dir}/coco/val",
                                             args.content_images // 10,
                                             args.seed)
    
    if args.download_val_style:
        print("Downloading validation style images...")
        val_style_dir = download_style_images_kaggle(args.style_images // 10,
                                                   f"{args.output_dir}/style/val",
                                                   args.seed)
    
    print("Download complete for selected datasets.")

if __name__ == "__main__":
    main() 