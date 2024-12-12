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
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import datetime

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
        self.dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Create output directories with default path
        model_name = getattr(self.config, 'model_name', 'model')
        self.output_dir = Path(f'outputs/validation/{model_name}')
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
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Validation")):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                content = batch['content']
                style = batch['style']
                outputs = self.model(content, style)
                
                # Compute losses
                losses = self.loss_fn.compute_losses(outputs, batch)
                total_loss += sum(losses.values())
                
                # Compute metrics
                batch_metrics = compute_metrics(outputs, batch)
                all_metrics.append(batch_metrics)
                
                # Save validation images periodically
                val_save_freq = getattr(self.config, 'val_save_freq', 100)
                if batch_idx % val_save_freq == 0:
                    self._save_validation_images(outputs, batch, epoch, batch_idx)
        
        # Aggregate metrics
        val_metrics['val_loss'] = total_loss / len(self.dataloader)
        for metric in all_metrics[0].keys():
            val_metrics[f'val_{metric}'] = np.mean([m[metric] for m in all_metrics])
        
        # Log to wandb
        if wandb.run is not None and getattr(self.config, 'use_wandb', False):
            wandb.log(val_metrics, step=epoch)
        
        return val_metrics
    
    def _save_validation_images(self, outputs, batch, epoch, batch_idx):
        """Save validation images with clear labels"""
        # Create visualization grid
        vis_images = []
        labels = []        
        # Add content and generated image pairs
        if isinstance(batch, dict) and 'content' in batch:
            for i in range(batch['content'].size(0)):  # Loop through batch
                vis_images.extend([
                    batch['content'][i],
                    outputs['generated'][i, :, :, :]
                ])
                labels.extend(['Content', 'Stylized'])
        
        # Convert all images to CPU at once for efficiency
        vis_images = [img.cpu() for img in vis_images]
        
        # Create grid with labels
        grid = make_grid(torch.stack(vis_images), 
                        nrow=2,  # 2 columns
                        normalize=True,
                        padding=5)
        
        # Add text labels using PIL
        grid_img = transforms.ToPILImage()(grid)
        draw = ImageDraw.Draw(grid_img)
        font = ImageFont.load_default()
        
        # Add labels above each column
        draw.text((grid.size(2)//4, 10), 'Content Images', fill='white', font=font)
        draw.text((3*grid.size(2)//4, 10), 'Stylized Images', fill='white', font=font)
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image with model name and timestamp
        save_path = (self.output_dir / 
                    f'{self.model_name}_epoch{epoch}_batch{batch_idx}_{timestamp}.png')
        grid_img.save(save_path)
        
        # Log to wandb
        if getattr(self.config, 'use_wandb', False):
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.validator.config.model.model_type
        save_path = (Path(self.validator.config.logging.save_dir) / 
                    f"{model_name}_best_model_epoch{epoch}_{timestamp}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': self.best_loss
        }, save_path)
        return save_path 