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
import torch.nn.functional as F

from utils.config import load_config, StyleTransferConfig
from utils.dataloader import StyleTransferDataset
from utils.losses import StyleTransferLoss, CycleGANLoss
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

def get_loss_function(config: StyleTransferConfig):
    """Get appropriate loss function based on model type"""
    model_type = config.model.model_type
    
    if model_type == 'cyclegan':
        return CycleGANLoss(config)
    else:  # johnson, adain
        return StyleTransferLoss(config)

def train_step(model, batch, loss_fn, optimizer, config):
    """Single training step with appropriate loss computation"""
    device = next(model.parameters()).device  # Get device from model
    model_type = config.model.model_type
    
    if model_type == 'cyclegan':
        # CycleGAN training step
        real_A = batch['content'].to(device)
        real_B = batch['style'].to(device)
        
        # Forward cycle
        fake_B = model(real_A, direction='AB')
        cycle_A = model(fake_B, direction='BA')
        identity_A = model(real_A, direction='BA')
        
        # Backward cycle
        fake_A = model(real_B, direction='BA')
        cycle_B = model(fake_A, direction='AB')
        identity_B = model(real_B, direction='AB')
        
        # Compute losses
        losses = loss_fn(real_A, real_B, fake_A, fake_B, 
                        cycle_A, cycle_B, identity_A, identity_B)
                        
    else:  # johnson, adain
        # Style transfer training step
        content = batch['content'].to(device)
        style = batch['style'].to(device)
        
        # Generate stylized image
        output = model(content)
        
        # Compute losses
        losses = loss_fn.compute_losses(output, batch)
    
    # Compute total loss
    total_loss = sum(losses.values())
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return losses

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
    loss_fn = get_loss_function(config)
    
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
        eta_min=getattr(config.training, 'min_lr', 1e-6)
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
                losses = train_step(model, batch, loss_fn, optimizer, config)
                total_loss = sum(losses.values())
                
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

def train_cyclegan(model, train_loader, val_loader, config, device, logger):
    # Initialize optimizers
    optimizer_G = torch.optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    optimizer_D = torch.optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )

    for epoch in range(config.training.num_epochs):
        model.train()
        running_losses = {
            'total_loss': 0.0,
            'G_AB': 0.0, 
            'G_BA': 0.0,
            'D_A': 0.0,
            'D_B': 0.0,
            'cycle_A': 0.0,
            'cycle_B': 0.0,
            'identity_A': 0.0,
            'identity_B': 0.0
        }
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                real_A = batch['content'].to(device)
                real_B = batch['style'].to(device)
                
                # Train Generators
                optimizer_G.zero_grad()
                
                # Forward cycle
                fake_B = model.G_AB(real_A)
                cycle_A = model.G_BA(fake_B)
                
                # Backward cycle
                fake_A = model.G_BA(real_B)
                cycle_B = model.G_AB(fake_A)
                
                # Identity loss
                identity_A = model.G_BA(real_A)
                identity_B = model.G_AB(real_B)
                
                # Calculate and log all losses
                loss_G, loss_dict = calculate_generator_loss(
                    model, real_A, real_B, fake_A, fake_B,
                    cycle_A, cycle_B, identity_A, identity_B,
                    config
                )
                
                loss_G.backward()
                optimizer_G.step()
                
                # Train Discriminators
                optimizer_D.zero_grad()
                loss_D, d_loss_dict = train_discriminator(
                    model, real_A, real_B, fake_A.detach(), fake_B.detach()
                )
                
                loss_D.backward()
                optimizer_D.step()
                
                # Update running losses
                running_losses['total_loss'] += (loss_G.item() + loss_D.item())
                for k, v in loss_dict.items():
                    running_losses[k] += v.item()
                for k, v in d_loss_dict.items():
                    running_losses[k] += v.item()
                
                # Update progress bar with running averages
                avg_losses = {k: v/(batch_idx + 1) for k, v in running_losses.items()}
                pbar.set_postfix(avg_losses)

        # Log epoch summary
        log_epoch_summary(logger, epoch, running_losses, len(train_loader))

def log_training_progress(logger, epoch, batch, losses, fake_A, fake_B, cycle_A, cycle_B, identity_A, identity_B):
    """Log training progress including images and metrics"""
    # Log losses
    for name, value in losses.items():
        logger.log({f'train/{name}_loss': value})
    
    # Log example images
    logger.log({
        'images/fake_A': wandb.Image(fake_A[0].cpu()),
        'images/fake_B': wandb.Image(fake_B[0].cpu()),
        'images/cycle_A': wandb.Image(cycle_A[0].cpu()),
        'images/cycle_B': wandb.Image(cycle_B[0].cpu()),
        'images/identity_A': wandb.Image(identity_A[0].cpu()),
        'images/identity_B': wandb.Image(identity_B[0].cpu())
    })

def log_epoch_summary(logger, epoch, losses, num_batches):
    """Log summary of epoch results"""
    avg_losses = {k: v/num_batches for k, v in losses.items()}
    
    logger.log({
        f'epoch/{k}_loss': v for k, v in avg_losses.items()
    })
    
    print(f"\nEpoch {epoch} Summary:")
    for k, v in avg_losses.items():
        print(f"{k}_loss: {v:.4f}")

def calculate_generator_loss(model, real_A, real_B, fake_A, fake_B, cycle_A, cycle_B, identity_A, identity_B, config):
    # Adversarial loss
    loss_G_AB = -torch.mean(model.D_B(fake_B))
    loss_G_BA = -torch.mean(model.D_A(fake_A))
    
    # Cycle consistency loss
    loss_cycle_A = F.l1_loss(cycle_A, real_A) * config.training.lambda_A
    loss_cycle_B = F.l1_loss(cycle_B, real_B) * config.training.lambda_B
    
    # Identity loss
    loss_identity_A = F.l1_loss(identity_A, real_A) * config.training.lambda_identity
    loss_identity_B = F.l1_loss(identity_B, real_B) * config.training.lambda_identity
    
    # Total generator loss
    loss_G = loss_G_AB + loss_G_BA + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B
    
    # Return losses dictionary for logging
    return loss_G, {
        'G_AB': loss_G_AB,
        'G_BA': loss_G_BA,
        'cycle_A': loss_cycle_A,
        'cycle_B': loss_cycle_B,
        'identity_A': loss_identity_A,
        'identity_B': loss_identity_B
    }

def train_discriminator(model, real_A, real_B, fake_A, fake_B):
    # Real loss
    loss_D_A_real = -torch.mean(model.D_A(real_A))
    loss_D_B_real = -torch.mean(model.D_B(real_B))
    
    # Fake loss
    loss_D_A_fake = torch.mean(model.D_A(fake_A))
    loss_D_B_fake = torch.mean(model.D_B(fake_B))
    
    # Total discriminator loss
    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
    loss_D = loss_D_A + loss_D_B
    
    return loss_D, {
        'D_A': loss_D_A,
        'D_B': loss_D_B
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train(args.config)