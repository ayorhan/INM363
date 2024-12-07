"""
Validation utilities for style transfer models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import wandb
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from .metrics import compute_metrics
from .losses import StyleTransferLoss

class Validator:
    """Handles model validation and visualization"""
    def __init__(self, 
                 model: nn.Module,
                 val_dataloader: DataLoader,
                 loss_fn: StyleTransferLoss,
                 config: Dict[str, Any]):
        self.model = model
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.config = config
        self.device = config['device']
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Run validation loop
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {}
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute losses
                losses = self.loss_fn.compute_losses(outputs, batch)
                total_loss += sum(losses.values())
                
                # Compute metrics
                batch_metrics = compute_metrics(outputs, batch)
                all_metrics.append(batch_metrics)
                
                # Save validation images periodically
                if batch_idx % self.config.get('val_save_freq', 100) == 0:
                    self._save_validation_images(outputs, batch, epoch, batch_idx)
        
        # Aggregate metrics
        val_metrics['val_loss'] = total_loss / len(self.val_dataloader)
        for metric in all_metrics[0].keys():
            val_metrics[f'val_{metric}'] = np.mean([m[metric] for m in all_metrics])
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log(val_metrics, step=epoch)
        
        return val_metrics
    
    def _save_validation_images(self, 
                              outputs: Dict[str, torch.Tensor],
                              batch: Dict[str, torch.Tensor],
                              epoch: int,
                              batch_idx: int) -> None:
        """Save validation images"""
        # Create visualization grid
        vis_images = []
        
        # Add content images
        if 'content' in batch:
            vis_images.append(batch['content'])
        
        # Add style images
        if 'style' in batch:
            vis_images.append(batch['style'])
        
        # Add generated images
        if 'generated' in outputs:
            vis_images.append(outputs['generated'])
        
        # Add reconstructed images if available
        if 'reconstructed' in outputs:
            vis_images.append(outputs['reconstructed'])
        
        # Create grid
        grid = make_grid(torch.cat(vis_images, dim=0), 
                        nrow=len(vis_images),
                        normalize=True)
        
        # Save image
        save_path = self.output_dir / f'val_epoch{epoch}_batch{batch_idx}.png'
        save_image(grid, save_path)
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'validation_samples': wandb.Image(grid),
                'epoch': epoch
            })
    
    def compute_fid(self) -> float:
        """Compute FID score"""
        # Implement FID computation
        raise NotImplementedError("FID computation not implemented yet")

class ValidationCallback:
    """Callback for running validation during training"""
    def __init__(self, validator: Validator, frequency: int = 1):
        self.validator = validator
        self.frequency = frequency
        self.best_loss = float('inf')
        self.best_model_path = None
    
    def __call__(self, epoch: int, model: nn.Module) -> Dict[str, float]:
        """Run validation if needed"""
        if epoch % self.frequency == 0:
            metrics = self.validator.validate(epoch)
            
            # Save best model
            if metrics['val_loss'] < self.best_loss:
                self.best_loss = metrics['val_loss']
                self.best_model_path = self._save_best_model(model, epoch)
            
            return metrics
        return {}
    
    def _save_best_model(self, model: nn.Module, epoch: int) -> Path:
        """Save best model checkpoint"""
        save_path = Path(self.validator.config['checkpoint_dir']) / f'best_model_epoch{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': self.best_loss
        }, save_path)
        return save_path 