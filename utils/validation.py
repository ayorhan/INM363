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
from typing import Dict, Any, List, Tuple, Union
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import datetime
from torch.nn import functional as F
import logging

from .metrics import compute_metrics
from .losses import StyleTransferLoss
from .config import StyleTransferConfig
from utils.metrics import MetricsLogger
from models.CycleGAN import CycleGAN

class Validator:
    """Handles model validation and visualization"""
    def __init__(self, 
                 model: nn.Module,
                 val_dataloader: DataLoader,
                 loss_fn: StyleTransferLoss,
                 config: Union[Dict[str, Any], StyleTransferConfig]):
        self.model = model
        self.val_loader = val_dataloader
        self.loss_fn = loss_fn
        # Convert config to dict if it's not already
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = {
                'model': config.model.__dict__,
                'logging': config.logging.__dict__,
                'training': config.training.__dict__
            }
        self.device = next(model.parameters()).device
        self.best_model_path = None
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation loop"""
        try:
            self.model.eval()
            val_metrics = {}
            total_batches = len(self.val_loader)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validation epoch {epoch}")):
                    try:
                        # Move data to device
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        # Handle different model types
                        if isinstance(self.model, CycleGAN):
                            metrics = self._validate_cyclegan_batch(batch)
                        else:
                            metrics = self._validate_style_transfer_batch(batch)
                        
                        # Update metrics
                        for k, v in metrics.items():
                            if k != 'output':
                                val_metrics[k] = val_metrics.get(k, 0) + v.item()
                        
                        # Save validation images periodically
                        if batch_idx % 10 == 0:
                            self._save_validation_images(metrics['output'], batch, epoch, batch_idx)
                
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                        continue
                
                # Average metrics
                val_metrics = {k: v / total_batches for k, v in val_metrics.items()}
                val_metrics['val_loss'] = sum(v for k, v in val_metrics.items() if k != 'output')
                
                return val_metrics
                
        except Exception as e:
            self.logger.error(f"Validation failed for epoch {epoch}: {str(e)}")
            return {'val_loss': float('inf')}
    
    def _validate_cyclegan_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Handle CycleGAN validation for a single batch"""
        try:
            real_A = batch['content']
            real_B = batch['style']
            
            fake_B = self.model(real_A, direction='AB')
            cycle_A = self.model(fake_B, direction='BA')
            identity_A = self.model(real_A, direction='BA')
            
            fake_A = self.model(real_B, direction='BA')
            cycle_B = self.model(fake_A, direction='AB')
            identity_B = self.model(real_B, direction='AB')
            
            losses = self.loss_fn(real_A, real_B, fake_A, fake_B,
                                cycle_A, cycle_B, identity_A, identity_B)
            losses['output'] = fake_B
            return losses
            
        except Exception as e:
            self.logger.error(f"CycleGAN validation failed: {str(e)}")
            raise
    
    def _validate_style_transfer_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Handle style transfer validation for a single batch"""
        try:
            # Pass only content tensor to Johnson model
            output = self.model(batch['content'])
            losses = self.loss_fn.compute_losses(output, batch)
            
            # Compute additional metrics
            losses['content_similarity'] = self._compute_content_similarity(
                output, batch['content'])
            losses['style_similarity'] = self._compute_style_similarity(
                output, batch['style'])
            losses['output'] = output  # Store generated output for visualization
            return losses
        except Exception as e:
            self.logger.error(f"Style transfer validation failed: {str(e)}")
            raise
    
    def _save_model(self, epoch, val_loss):
        save_path = Path(self.config['logging']['save_dir']) / f"best_model_epoch_{epoch}_loss_{val_loss:.4f}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        return save_path
    
    def _save_validation_images(self, outputs, batch, epoch, batch_idx):
        """Save validation images with clear labels"""
        # Create model-specific output directory
        model_name = self.config.model.model_type if hasattr(self.config, 'model') else self.config['model']['model_type']
        output_dir = Path(self.config.logging.output_dir if hasattr(self.config, 'logging') 
                         else self.config['logging']['output_dir']) / 'validation' / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization grid
        vis_images = []
        labels = []        
        
        # Add content, style, and generated image triplets
        if isinstance(batch, dict) and 'content' in batch and 'style' in batch:
            # Convert outputs tensor to dictionary format
            if isinstance(outputs, torch.Tensor):
                generated = outputs
            else:
                generated = outputs.get('generated', outputs)
            
            content = batch['content']
            style = batch['style']
            
            # Take first n images from batch
            n = min(content.size(0), generated.size(0), style.size(0))
            
            for i in range(n):
                vis_images.extend([
                    content[i].cpu(),    # Content image
                    style[i].cpu(),      # Style image
                    generated[i].cpu()    # Generated image
                ])
                labels.extend(['Content', 'Style', 'Generated'])
        
        # Create grid with labels
        grid = make_grid(torch.stack(vis_images), 
                        nrow=3,  # 3 columns for content-style-generated triplets
                        normalize=True,
                        padding=5)
        
        # Add text labels using PIL
        grid_img = transforms.ToPILImage()(grid)
        draw = ImageDraw.Draw(grid_img)
        
        # Save with model name prefix and style indication
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_validation_epoch{epoch}_batch{batch_idx}_{timestamp}.png"
        save_path = output_dir / filename
        grid_img.save(save_path)
        
        # Log to wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.log({
                'validation_samples': wandb.Image(grid),
                'epoch': epoch
            })
        
        
    
    def compute_fid(self) -> float:
        """Compute FID score"""
        # Implement FID computation
        raise NotImplementedError("FID computation not implemented yet")
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def _compute_content_similarity(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Compute structural similarity between output and content image"""
        return F.mse_loss(output, target).item()
    
    def _compute_style_similarity(self, output: torch.Tensor, style: torch.Tensor) -> float:
        """Compute style similarity using Gram matrices"""
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, -1)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (c * h * w)
        
        return F.mse_loss(gram_matrix(output), gram_matrix(style)).item()

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
        model_name = self.validator.config['model']['model_type']
        save_path = (Path(self.validator.config['logging']['save_dir']) / 
                    f"{model_name}_best_model_epoch{epoch}_{timestamp}.pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': self.best_loss,
            'config': self.validator.config  # Save config for reproducibility
        }, save_path)
        return save_path

def validate(model, val_loader, metrics, config, device):
    model.eval()
    logger = MetricsLogger()
    
    # Convert ModelConfig to dict if needed
    if hasattr(config, '_asdict'):
        config = config._asdict()
    elif hasattr(config, '__dict__'):
        config = config.__dict__
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if isinstance(model, CycleGAN):
                    outputs = {'generated': model(batch['content'], direction='AB')}
                else:
                    outputs = model(batch)
                    if not isinstance(outputs, dict):
                        outputs = {'generated': outputs}
                
                batch_metrics = {
                    'content_loss': metrics.compute_content_loss(
                        outputs['generated'], batch['content']
                    ),
                    'style_loss': metrics.compute_style_loss(
                        outputs['generated'], batch['style']
                    )
                }
                
                logger.update(batch_metrics)
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue
    
    return logger.get_average()