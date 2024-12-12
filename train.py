"""
Enhanced training script with validation
"""

import os
os.environ['MKL_DISABLE_FAST_MM'] = '1'

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

from utils.config import load_config, StyleTransferConfig
from utils.dataloader import create_dataloader
from utils.losses import StyleTransferLoss
from utils.validation import Validator, ValidationCallback
from models import get_model
from evaluation_scripts.evaluate_model import evaluate_model

def setup_logging(config: StyleTransferConfig) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(config, 'log_level', logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def train(config_path: str):
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    # Initialize wandb
    if getattr(config, 'use_wandb', False):
        wandb.init(
            project="style-transfer",
            config=config
        )
    
    # Create dataloaders
    train_dataloader = create_dataloader(config, 'train')
    val_dataloader = create_dataloader(config, 'val')
    
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
    validator = Validator(model, val_dataloader, loss_fn, config)
    validation_callback = ValidationCallback(
        validator,
        frequency=config.training.validation_interval
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}") as pbar:
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