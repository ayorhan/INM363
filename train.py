"""
Enhanced training script with validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any

from utils.config import load_config
from utils.dataloader import create_dataloader
from utils.losses import StyleTransferLoss
from utils.validation import Validator, ValidationCallback
from models import get_model

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=config.get('log_level', logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def train(config_path: str):
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)
    
    # Initialize wandb
    if config.get('use_wandb', False):
        wandb.init(
            project="style-transfer",
            config=config
        )
    
    # Create dataloaders
    train_dataloader = create_dataloader(config, 'train')
    val_dataloader = create_dataloader(config, 'val')
    
    # Initialize model
    model = get_model(config)
    model = model.to(config['device'])
    
    # Initialize loss functions
    loss_fn = StyleTransferLoss(config)
    
    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2'])
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['min_lr']
    )
    
    # Initialize validator
    validator = Validator(model, val_dataloader, loss_fn, config)
    val_callback = ValidationCallback(
        validator,
        frequency=config.get('validation_frequency', 1)
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = {k: v.to(config['device']) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Compute losses
                losses = loss_fn.compute_losses(outputs, batch)
                total_loss = sum(losses.values())
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if config.get('clip_grad_norm', False):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['max_grad_norm']
                    )
                
                optimizer.step()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    **{k: v.item() for k, v in losses.items()}
                })
                
                # Log to wandb
                if config.get('use_wandb', False):
                    wandb.log({
                        'epoch': epoch,
                        'total_loss': total_loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        **{k: v.item() for k, v in losses.items()}
                    })
                
                # Save checkpoints
                if batch_idx % config['save_interval'] == 0:
                    save_checkpoint(
                        model, optimizer, epoch, batch_idx,
                        total_loss.item(), config
                    )
        
        # Validation
        val_metrics = val_callback(epoch, model)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if config.get('early_stopping', False):
            if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience = config['patience']
            else:
                patience -= 1
                if patience <= 0:
                    logging.info("Early stopping triggered")
                    break

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   batch_idx: int,
                   loss: float,
                   config: Dict[str, Any]) -> None:
    """Save model checkpoint"""
    save_path = Path(config['checkpoint_dir']) / f"checkpoint_epoch{epoch}_iter{batch_idx}.pth"
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