"""
Enhanced training script with validation and dataset management
"""

import os
# Suppress Intel MKL warnings
os.environ['MKL_DISABLE_FAST_MM'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# Specifically for Intel MKL warnings
warnings.filterwarnings('ignore', message='Intel MKL WARNING.*')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any
import datetime
import argparse

from utils.config import load_config, StyleTransferConfig
from utils.dataloader import StyleTransferDataset
from utils.losses import StyleTransferLoss
from utils.validation import Validator, ValidationCallback
from models import get_model
from evaluation_scripts.evaluate_model import evaluate_model
from download_datasets import download_coco_images, download_style_images_kaggle

def setup_logging(config: StyleTransferConfig) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(config, 'log_level', logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_datasets(config: StyleTransferConfig):
    """Setup dataset paths"""
    train_content_dir = Path(f"{config.data.content_path}/train/images")
    train_style_dir = Path(f"{config.data.style_path}/train")
    val_content_dir = Path(f"{config.data.content_path}/val/images")
    val_style_dir = Path(f"{config.data.style_path}/val")
    
    # Verify that all required directories exist
    required_dirs = [train_content_dir, train_style_dir, val_content_dir, val_style_dir]
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(
                f"Directory {dir_path} not found. Please run download_datasets.py first."
            )
        if not any(dir_path.iterdir()):
            raise ValueError(
                f"Directory {dir_path} is empty. Please run download_datasets.py first."
            )
    
    return train_content_dir, train_style_dir, val_content_dir, val_style_dir

def create_dataloaders(config: StyleTransferConfig, 
                      train_content_dir: Path,
                      train_style_dir: Path,
                      val_content_dir: Path,
                      val_style_dir: Path) -> tuple:
    """Create train and validation dataloaders"""
    # Debug logging
    logging.info(f"Creating datasets with paths:")
    logging.info(f"Train content: {train_content_dir} ({len(list(train_content_dir.glob('*.jpg')) + list(train_content_dir.glob('*.png')))} images)")
    logging.info(f"Train style: {train_style_dir} ({len(list(train_style_dir.glob('*.jpg')) + list(train_style_dir.glob('*.png')))} images)")
    logging.info(f"Val content: {val_content_dir} ({len(list(val_content_dir.glob('*.jpg')) + list(val_content_dir.glob('*.png')))} images)")
    logging.info(f"Val style: {val_style_dir} ({len(list(val_style_dir.glob('*.jpg')) + list(val_style_dir.glob('*.png')))} images)")
    
    # Create datasets
    train_dataset = StyleTransferDataset(
        content_path=train_content_dir,
        style_path=train_style_dir,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        use_augmentation=config.data.use_augmentation,
        max_content_size=config.data.train_content_size,
        max_style_size=config.data.train_style_size
    )
    
    logging.info(f"Created train dataset with {len(train_dataset.content_images)} content and {len(train_dataset.style_images)} style images")
    
    if len(train_dataset.content_images) == 0 or len(train_dataset.style_images) == 0:
        raise ValueError("Train dataset is empty! Please check your data paths and image files.")
    
    val_dataset = StyleTransferDataset(
        content_path=val_content_dir,
        style_path=val_style_dir,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        use_augmentation=False,
        max_content_size=config.data.val_content_size,
        max_style_size=config.data.val_style_size
    )
    
    logging.info(f"Created validation dataset with {len(val_dataset.content_images)} content and {len(val_dataset.style_images)} style images")
    
    if len(val_dataset.content_images) == 0 or len(val_dataset.style_images) == 0:
        raise ValueError("Validation dataset is empty! Please check your data paths and image files.")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    return train_loader, val_loader

def train(config_path: str):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Setup datasets and dataloaders
    logger.info("Setting up datasets...")
    train_content_dir, train_style_dir, val_content_dir, val_style_dir = setup_datasets(config)
    train_loader, val_loader = create_dataloaders(
        config, train_content_dir, train_style_dir, val_content_dir, val_style_dir
    )
    
    # Initialize wandb
    if getattr(config, 'use_wandb', False):
        wandb.init(
            project="style-transfer",
            config=config
        )
    
    # Initialize model
    model = get_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss functions
    loss_fn = StyleTransferLoss(config)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=config.training.min_lr
    )
    
    # Initialize validator
    validator = Validator(model, val_loader, loss_fn, config)
    validation_callback = ValidationCallback(
        validator,
        frequency=config.training.validation_interval
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(content=batch['content'], style=batch['style'])
                
                # Compute losses
                losses = loss_fn.compute_losses(outputs, batch)
                total_loss = sum(losses.values())
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping (if enabled)
                if hasattr(config.training, 'clip_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.training.clip_grad_norm
                    )
                
                optimizer.step()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    **{k: v.item() for k, v in losses.items()}
                })
                
                # Log to wandb
                if hasattr(config, 'use_wandb') and config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'total_loss': total_loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        **{k: v.item() for k, v in losses.items()}
                    })
                
                # Save checkpoints
                if batch_idx % config.training.save_interval == 0:
                    save_checkpoint(
                        model, optimizer, epoch, batch_idx,
                        total_loss.item(), config
                    )
        
        # Validation
        val_metrics = validation_callback(epoch, model)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if hasattr(config.training, 'early_stopping') and config.training.early_stopping:
            if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience = config.training.patience
            else:
                patience -= 1
                if patience <= 0:
                    logging.info("Early stopping triggered")
                    break

    # After training completes, run evaluation
    print("\nRunning post-training evaluation...")
    output_dir = Path(f'evaluation_results/{config.model.model_type}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_model(
        config_path=config_path,
        checkpoint_path=validation_callback.best_model_path,
        output_path=output_dir
    )

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   batch_idx: int,
                   loss: float,
                   config: StyleTransferConfig) -> None:
    """Save model checkpoint"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.model_type
    
    save_path = (Path(config.logging.save_dir) / 
                f"{model_name}_checkpoint_epoch{epoch}_iter{batch_idx}_{timestamp}.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train(args.config)