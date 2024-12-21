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
from collections import defaultdict
from torchvision.utils import make_grid, save_image
import torch.nn.utils as utils

from utils.config import load_config, StyleTransferConfig
from utils.dataloader import StyleTransferDataset
from utils.losses import StyleTransferLoss, CycleGANLoss
from utils.validation import Validator, ValidationCallback, validate
from models import get_model
from evaluation_scripts.evaluate_model import evaluate_model
from download_datasets import download_coco_images, download_style_images_kaggle
from utils.metrics import StyleTransferMetrics
from utils.logging_utils import setup_training_logger, log_training_step, log_validation_results, log_model_parameters

def setup_logging(config: StyleTransferConfig) -> None:
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Get timestamp for log file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'training_{timestamp}.log')
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will maintain console output
        ]
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
    """Single training step with enhanced logging"""
    device = next(model.parameters()).device
    model_type = config.model.model_type
    train_logger = logging.getLogger('training')
    
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
                        
        outputs = {'generated': fake_B}
    else:  # johnson, adain
        # Style transfer training step
        content = batch['content'].to(device)
        style = batch['style'].to(device)
        
        # Generate stylized image
        output = model(content)
        
        # Compute losses
        losses = loss_fn.compute_losses(output, batch)
        
        outputs = {'generated': output}
    
    # Compute total loss
    total_loss = sum(losses.values())
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Add tensor statistics logging
    if model_type == 'johnson':
        train_logger.debug(f"Content tensor range: [{batch['content'].min():.2f}, {batch['content'].max():.2f}]")
        train_logger.debug(f"Style tensor range: [{batch['style'].min():.2f}, {batch['style'].max():.2f}]")
        train_logger.debug(f"Output tensor range: [{outputs['generated'].min():.2f}, {outputs['generated'].max():.2f}]")
    
    return losses, outputs

def train(config_path: str):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Initialize single logger for entire training process
    train_logger = setup_training_logger()
    train_logger.info("Starting training process...")
    
    # Setup datasets and dataloaders
    train_logger.info("Setting up datasets...")
    train_content_dir, train_style_dir, val_content_dir, val_style_dir = setup_datasets(config)
    train_loader, val_loader = create_dataloaders(
        config, train_content_dir, train_style_dir, val_content_dir, val_style_dir
    )
    
    # Initialize wandb
    use_wandb = initialize_wandb(config)
    
    # Initialize model
    train_logger.info("Initializing model...")
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
    
    # Initialize metrics
    metrics = StyleTransferMetrics(device)
    
    # Create all required directories
    os.makedirs(config.logging.save_dir, exist_ok=True)
    os.makedirs(config.logging.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Start training
    train_logger.info("Starting CycleGAN training...")
    train_cyclegan(model, train_loader, val_loader, config, device, train_logger)
    
    # After training completes, run evaluation
    train_logger.info("\nRunning post-training evaluation...")
    output_dir = Path(f'evaluation_results/{config.model.model_type}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_model(
        config_path=config_path,
        checkpoint_path=validation_callback.best_model_path,
        output_path=output_dir
    )

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, config, val_loader, is_best=False):
    """Save model checkpoint and generate sample outputs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.model_type
    
    # Save checkpoint
    checkpoint_name = (f"{model_name}_checkpoint_epoch{epoch}_iter{batch_idx}_{timestamp}.pth")
    save_path = Path(config.logging.save_dir) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    
    # Generate and save sample outputs with config
    samples_output_dir = Path(config.logging.output_dir)
    save_checkpoint_samples(
        model, val_loader, epoch, samples_output_dir, 
        config.model.model_type, config
    )
    
    # Create/update progress visualization
    create_progress_visualization(samples_output_dir, model_name)
    
    # If this is the best model so far, save an additional copy
    if is_best:
        best_path = save_path.parent / f"{model_name}_best.pth"
        torch.save(checkpoint, best_path)

def train_cyclegan(model, train_loader, val_loader, config, device, logger):
    use_wandb = initialize_wandb(config)
    
    os.makedirs(config.logging.save_dir, exist_ok=True)
    os.makedirs(config.logging.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        model.train()
        running_losses = defaultdict(float)
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                real_A = batch['content']
                real_B = batch['style']
                
                # ---------------------------
                # Generator Forward & Backward
                # ---------------------------
                fake_B = model.G_AB(real_A)
                fake_A = model.G_BA(real_B)
                cycle_A = model.G_BA(fake_B)
                cycle_B = model.G_AB(fake_A)
                identity_A = model.G_BA(real_A)
                identity_B = model.G_AB(real_B)
                
                optimizer.zero_grad()
                g_loss, g_losses = calculate_generator_loss(
                    model, real_A, real_B, fake_A, fake_B, 
                    cycle_A, cycle_B, identity_A, identity_B, config
                )
                g_loss.backward()
                
                # Gradient clipping for generator
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # ---------------------------
                # Discriminator Forward & Backward
                # ---------------------------
                optimizer.zero_grad()
                d_loss, d_losses = train_discriminator(model, real_A, real_B, fake_A, fake_B)
                d_loss.backward()
                
                # Gradient clipping for discriminator
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update loss tracking
                for k, v in g_losses.items():
                    running_losses[k] += v.item() if torch.is_tensor(v) else v
                for k, v in d_losses.items():
                    running_losses[k] += v
                
                # Optional logging
                if batch_idx % config.logging.log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch+1}][{batch_idx}/{len(train_loader)}] "
                        f"G_AB: {g_losses['G_AB']:.4f}, G_BA: {g_losses['G_BA']:.4f}, "
                        f"D_A: {d_losses['D_A']:.4f}, D_B: {d_losses['D_B']:.4f}, "
                        f"Cycle: {(g_losses['cycle_A'] + g_losses['cycle_B']):.4f}, "
                        f"Identity: {(g_losses['identity_A'] + g_losses['identity_B']):.4f}"
                    )
                
                # Save samples periodically during training
                if batch_idx % config.logging.sample_interval == 0:
                    samples_output_dir = Path(config.logging.output_dir)
                    save_checkpoint_samples(
                        model, val_loader, epoch, 
                        samples_output_dir, 'cyclegan'
                    )
                
            # End of epoch => (2) Validation step
            if (epoch + 1) % config.training.validation_interval == 0:
                val_loss = run_cyclegan_validation(model, val_loader, config, device, logger)
                
                # Track best val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, 0, val_loss, config, val_loader, is_best=True)
                
        # We can also log epoch summary here
        if use_wandb:
            log_epoch_summary(wandb, epoch, running_losses, len(train_loader))
        
        # Save samples at end of epoch
        save_checkpoint_samples(
            model, val_loader, epoch, 
            Path(config.logging.output_dir), 'cyclegan'
        )
        create_progress_visualization(
            Path(config.logging.output_dir), 'cyclegan'
        )

def run_cyclegan_validation(model, val_loader, config, device, logger):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
            real_A = val_batch['content']
            real_B = val_batch['style']
            
            # Forward passes
            fake_B = model.G_AB(real_A)
            fake_A = model.G_BA(real_B)
            cycle_A = model.G_BA(fake_B)
            cycle_B = model.G_AB(fake_A)
            identity_A = model.G_BA(real_A)
            identity_B = model.G_AB(real_B)
            
            # Compute same generator losses used in training
            g_loss, g_losses = calculate_generator_loss(
                model, real_A, real_B, fake_A, fake_B, 
                cycle_A, cycle_B, identity_A, identity_B, config
            )
            total_loss += g_loss.item()
    
    model.train()
    avg_val_loss = total_loss / len(val_loader)
    logger.info(f"Validation loss: {avg_val_loss:.4f}")
    return avg_val_loss

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
    
    print(f"\nEpoch {epoch+1} Summary:")
    for k, v in avg_losses.items():
        print(f"{k}_loss: {v:.4f}")
    print()

def calculate_generator_loss(model, real_A, real_B, fake_A, fake_B, cycle_A, cycle_B, identity_A, identity_B, config):
    # Adversarial loss (LSGAN)
    loss_G_AB = torch.mean((model.D_B(fake_B) - 1)**2)
    loss_G_BA = torch.mean((model.D_A(fake_A) - 1)**2)
    
    # Cycle consistency loss
    loss_cycle_A = F.l1_loss(cycle_A, real_A) * config.training.lambda_A
    loss_cycle_B = F.l1_loss(cycle_B, real_B) * config.training.lambda_B
    
    # Identity loss (with reduced weight)
    loss_identity_A = F.l1_loss(identity_A, real_A) * config.training.lambda_identity
    loss_identity_B = F.l1_loss(identity_B, real_B) * config.training.lambda_identity
    
    # Total generator loss
    loss_G = loss_G_AB + loss_G_BA + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B
    
    return loss_G, {
        'G_AB': loss_G_AB,
        'G_BA': loss_G_BA,
        'cycle_A': loss_cycle_A,
        'cycle_B': loss_cycle_B,
        'identity_A': loss_identity_A,
        'identity_B': loss_identity_B
    }

def train_discriminator(model, real_A, real_B, fake_A, fake_B):
    """Train discriminator networks"""
    # Get discriminator outputs
    D_A_real = model.D_A(real_A)
    D_A_fake = model.D_A(fake_A.detach())
    D_B_real = model.D_B(real_B)
    D_B_fake = model.D_B(fake_B.detach())
    
    # Create labels matching the discriminator output shape
    real_label = torch.ones_like(D_A_real, device=real_A.device)
    fake_label = torch.zeros_like(D_A_fake, device=real_A.device)
    
    # Calculate losses using the correct shapes
    loss_D_A_real = torch.nn.MSELoss()(D_A_real, real_label)
    loss_D_A_fake = torch.nn.MSELoss()(D_A_fake, fake_label)
    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
    
    # Create labels for D_B with matching shape
    real_label_B = torch.ones_like(D_B_real, device=real_B.device)
    fake_label_B = torch.zeros_like(D_B_fake, device=real_B.device)
    
    loss_D_B_real = torch.nn.MSELoss()(D_B_real, real_label_B)
    loss_D_B_fake = torch.nn.MSELoss()(D_B_fake, fake_label_B)
    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
    
    loss_D = loss_D_A + loss_D_B
    
    return loss_D, {
        'D_A': loss_D_A.item(),
        'D_B': loss_D_B.item()
    }

def initialize_wandb(config: StyleTransferConfig) -> bool:
    """Initialize Weights & Biases logging"""
    if config.logging.use_wandb:
        try:
            wandb.init(
                project=config.logging.project_name,
                name=config.logging.run_name,
                config={
                    'model': config.model.__dict__,
                    'training': config.training.__dict__,
                    'data': config.data.__dict__,
                    'logging': config.logging.__dict__
                }
            )
            return True
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {str(e)}")
            return False
    return False

def log_metrics(metrics: Dict[str, float], step: int, prefix: str = '') -> None:
    """Log metrics to wandb if enabled"""
    if wandb.run is not None:
        wandb.log(
            {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()},
            step=step
        )

def setup_training_logger():
    """Setup training logger with a single file handler"""
    # Get current timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_sample_images(model, real_A, real_B, fake_A, fake_B, cycle_A, cycle_B, 
                      identity_A, identity_B, epoch, batch_idx, output_dir):
    """Save a grid of sample images"""
    # Create a grid of images
    images = torch.cat([
        real_A[:4], fake_B[:4], cycle_A[:4],
        real_B[:4], fake_A[:4], cycle_B[:4],
        real_A[:4], identity_A[:4],
        real_B[:4], identity_B[:4]
    ], dim=0)
    
    # Make grid
    grid = make_grid(images, nrow=4, normalize=True)
    
    # Save image
    save_path = Path(output_dir) / f'samples_epoch_{epoch}_batch_{batch_idx}.png'
    save_image(grid, save_path)

def monitor_losses(g_losses, d_losses, logger):
    """Monitor losses for instability"""
    # Check for extreme values
    for name, loss in {**g_losses, **d_losses}.items():
        value = loss.item() if torch.is_tensor(loss) else loss
        if abs(value) > 10.0:  # Threshold for extreme values
            logger.warning(f"High loss value detected - {name}: {value:.4f}")
        elif value != value:  # Check for NaN
            logger.error(f"NaN loss detected - {name}")

def save_checkpoint_samples(model, val_loader, epoch, output_dir, model_type='johnson'):
    """Generate and save sample outputs for the current checkpoint"""
    device = next(model.parameters()).device
    model.eval()
    
    # Create checkpoint-specific directory
    samples_dir = Path(output_dir) / 'training_progress' / f'epoch_{epoch}'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get a batch from validation loader
        batch = next(iter(val_loader))
        content = batch['content'].to(device)
        style = batch['style'].to(device)
        
        if model_type.lower() == 'cyclegan':
            # Generate CycleGAN translations
            fake_B = model(content, direction='AB')  # Content to style
            fake_A = model(fake_B, direction='BA')   # Back to content (cycle)
            identity_B = model(style, direction='AB') # Style identity
            
            # Create grid for each sample
            for i in range(min(4, content.size(0))):
                comparison = torch.cat([
                    content[i:i+1],     # Original content
                    style[i:i+1],       # Target style
                    fake_B[i:i+1],      # Stylized
                    fake_A[i:i+1],      # Reconstructed
                    identity_B[i:i+1]   # Style identity
                ], dim=2)
                
                save_image(
                    comparison,
                    samples_dir / f'sample_{i}_cyclegan.png',
                    normalize=True
                )
        else:
            # Generate Johnson model outputs
            output = model(content)
            
            # Create grid for each sample
            for i in range(min(4, content.size(0))):
                comparison = torch.cat([
                    content[i:i+1],    # Original content
                    style[i:i+1],      # Target style
                    output[i:i+1]      # Stylized output
                ], dim=2)
                
                save_image(
                    comparison,
                    samples_dir / f'sample_{i}_johnson.png',
                    normalize=True
                )
    
    model.train()

def create_progress_visualization(output_dir, model_type):
    """Create an HTML file showing training progression"""
    progress_dir = Path(output_dir) / 'training_progress'
    epochs = sorted([d for d in progress_dir.iterdir() if d.is_dir()], 
                   key=lambda x: int(x.name.split('_')[1]))
    
    html_content = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<style>',
        '.sample { margin-bottom: 40px; }',
        '.epoch { margin-bottom: 20px; padding: 20px; border: 1px solid #ccc; }',
        '.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }',
        'img { max-width: 100%; height: auto; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{model_type} Training Progress</h1>'
    ]
    
    for epoch_dir in epochs:
        epoch_num = epoch_dir.name.split('_')[1]
        html_content.append(f'<div class="epoch"><h2>Epoch {epoch_num}</h2><div class="grid">')
        
        # Add all sample comparisons for this epoch
        samples = sorted(epoch_dir.glob(f'sample_*_{model_type.lower()}.png'))
        for sample in samples:
            html_content.append(
                f'<div class="sample">'
                f'<img src="{sample.relative_to(progress_dir)}" />'
                f'<p>Sample {sample.stem.split("_")[1]}</p>'
                f'</div>'
            )
        html_content.append('</div></div>')
    
    html_content.extend(['</body>', '</html>'])
    
    # Save HTML file
    with open(progress_dir / f'{model_type.lower()}_progress.html', 'w') as f:
        f.write('\n'.join(html_content))

def train_johnson(model, train_loader, val_loader, config, device, logger):
    # Existing initialization code
    
    for epoch in range(config.training.num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # Existing training code
            
            # Save samples periodically
            if batch_idx % config.logging.sample_interval == 0:
                samples_output_dir = Path(config.logging.output_dir)
                save_checkpoint_samples(
                    model, val_loader, epoch, 
                    samples_output_dir, 'johnson'
                )
        
        # Save samples at end of epoch
        save_checkpoint_samples(
            model, val_loader, epoch, 
            Path(config.logging.output_dir), 'johnson'
        )
        create_progress_visualization(
            Path(config.logging.output_dir), 'johnson'
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train(args.config)