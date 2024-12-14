"""
Data loading utilities for style transfer models
"""

import os
from pathlib import Path
from typing import List, Dict, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.config import StyleTransferConfig
import random

class StyleTransferDataset(Dataset):
    """Base dataset class for style transfer"""
    def __init__(self, 
                 content_path: str,
                 style_path: str,
                 image_size: int = 256,
                 crop_size: int = None,
                 use_augmentation: bool = True,
                 max_content_size: int = None,
                 max_style_size: int = None):
        super().__init__()
        
        self.content_path = Path(content_path)
        self.style_path = Path(style_path)
        self.image_size = image_size
        self.crop_size = crop_size or image_size
        
        # Get image paths
        self.content_images = list(self.content_path.glob('*.jpg')) + \
                            list(self.content_path.glob('*.png'))
        self.style_images = list(self.style_path.glob('*.jpg')) + \
                           list(self.style_path.glob('*.png'))
        
        # Limit dataset sizes if specified
        if max_content_size:
            self.content_images = self.content_images[:max_content_size]
        if max_style_size:
            self.style_images = self.style_images[:max_style_size]
        
        # Set up transformations
        self.transform = self._setup_transforms(use_augmentation)
        
    def _get_image_paths(self, path: Path) -> List[Path]:
        """Get all image paths recursively"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        return [
            p for p in path.rglob("*")
            if p.suffix.lower() in valid_extensions
        ]

    def _setup_transforms(self, use_augmentation: bool) -> transforms.Compose:
        """Set up image transformations"""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ]
        
        if use_augmentation:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ] + transform_list
            
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.content_images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load content and style images
        content_img = Image.open(self.content_images[idx % len(self.content_images)]).convert('RGB')
        style_img = Image.open(self.style_images[idx % len(self.style_images)]).convert('RGB')
        
        # Apply transformations
        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)
        
        return {
            'content': content_tensor,
            'style': style_tensor,
            'content_path': str(self.content_images[idx % len(self.content_images)]),
            'style_path': str(self.style_images[idx % len(self.style_images)])
        }

class PairedDataset(Dataset):
    """Dataset class for paired images (like pix2pix)"""
    def __init__(self, paired_path: str, image_size: int = 256, crop_size: int = None, use_augmentation: bool = True):
        super().__init__()
        self.paired_path = Path(paired_path)
        self.image_size = image_size
        self.crop_size = crop_size or image_size
        self.image_paths = self._get_image_paths(self.paired_path)
        self.transform = self._setup_transforms(use_augmentation)
    
    def _get_image_paths(self, path: Path) -> List[Path]:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        return [p for p in path.rglob("*") if p.suffix.lower() in valid_extensions]
    
    def _setup_transforms(self, use_augmentation: bool) -> transforms.Compose:
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        if use_augmentation:
            transform_list = [transforms.RandomHorizontalFlip()] + transform_list
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # Assuming paired images are concatenated horizontally
        w, h = img.size
        content_img = img.crop((0, 0, w//2, h))
        style_img = img.crop((w//2, 0, w, h))
        
        return {
            'content': self.transform(content_img),
            'style': self.transform(style_img),
            'path': str(self.image_paths[idx])
        }

def create_dataloader(config: StyleTransferConfig, split: str):
    """Create appropriate dataloader based on configuration"""
    if getattr(config, 'model_type', None) in ['pix2pix']:
        dataset = PairedDataset(
            paired_path=config.dataset_path,
            image_size=config.data.image_size,
            crop_size=config.data.crop_size,
            use_augmentation=config.data.use_augmentation
        )
    else:
        # Set content and style sizes based on split
        content_size = config.data.train_content_size if split == 'train' else config.data.val_content_size
        style_size = config.data.train_style_size if split == 'train' else config.data.val_style_size
        
        # Adjust paths for split
        content_path = f"{config.data.content_path}/{split}/images"
        style_path = f"{config.data.style_path}/{split}"
        
        dataset = StyleTransferDataset(
            content_path=content_path,
            style_path=style_path,
            image_size=config.data.image_size,
            crop_size=config.data.crop_size,
            use_augmentation=config.data.use_augmentation and split == 'train',
            max_content_size=content_size,
            max_style_size=style_size
        )
    
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        pin_memory=True
    ) 