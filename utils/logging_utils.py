import logging
from typing import Dict, Any
import torch
import wandb
import numpy as np
import os
import datetime

def setup_training_logger(log_dir: str = 'logs'):
    """Setup logger to write to both file and console"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This handles console output
        ]
    )
    
    return logging.getLogger('training')

def log_model_parameters(logger, model):
    """Log detailed model parameter statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("\nModel Architecture Statistics:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log parameter statistics by layer type
    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            if layer_type not in layer_stats:
                layer_stats[layer_type] = []
            layer_stats[layer_type].append({
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            })
    
    # Print layer-wise statistics
    logger.info("\nLayer-wise Statistics:")
    for layer_type, stats in layer_stats.items():
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        logger.info(f"{layer_type}:")
        logger.info(f"  Mean range: [{min(means):.4f}, {max(means):.4f}]")
        logger.info(f"  Std range: [{min(stds):.4f}, {max(stds):.4f}]")

def log_training_step(logger, model_type: str, epoch: int, batch_idx: int, 
                     loss_dict: Dict[str, float], outputs: Dict[str, torch.Tensor],
                     total_batches: int):
    """Enhanced training step logging"""
    if model_type == 'johnson':
        logger.info(
            f"Epoch [{epoch}][{batch_idx}/{total_batches}] "
            f"Content: {loss_dict.get('content', 0):.4f}, "
            f"Style: {loss_dict.get('style', 0):.4f}, "
            f"TV: {loss_dict.get('tv', 0):.4f}, "
            f"Output Range: [{outputs['generated'].min():.2f}, {outputs['generated'].max():.2f}]"
        )
    elif model_type == 'cyclegan':
        logger.info(
            f"Epoch [{epoch}][{batch_idx}/{total_batches}] "
            f"G_AB: {loss_dict.get('G_AB', 0):.4f}, "
            f"G_BA: {loss_dict.get('G_BA', 0):.4f}, "
            f"D_A: {loss_dict.get('D_A', 0):.4f}, "
            f"D_B: {loss_dict.get('D_B', 0):.4f}, "
            f"Cycle: {(loss_dict.get('cycle_A', 0) + loss_dict.get('cycle_B', 0))/2:.4f}, "
            f"Identity: {(loss_dict.get('identity_A', 0) + loss_dict.get('identity_B', 0))/2:.4f}"
        )

def log_validation_results(logger, model_type: str, epoch: int, metrics: Dict[str, float]):
    """Enhanced validation results logging"""
    logger.info(f"\nValidation Results - Epoch [{epoch}]")
    
    if model_type == 'johnson':
        logger.info(
            f"Content Loss: {metrics.get('content_loss', 0):.4f}\n"
            f"Style Loss: {metrics.get('style_loss', 0):.4f}\n"
            f"Total Loss: {metrics.get('content_loss', 0) + metrics.get('style_loss', 0):.4f}"
        )
    elif model_type == 'cyclegan':
        logger.info(
            f"Generator Loss: {metrics.get('generator_loss', 0):.4f}\n"
            f"Discriminator Loss: {metrics.get('discriminator_loss', 0):.4f}\n"
            f"Cycle Consistency: {metrics.get('cycle_consistency', 0):.4f}"
        )