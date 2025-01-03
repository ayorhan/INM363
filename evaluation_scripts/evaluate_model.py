"""
Example script for evaluating a trained style transfer model
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
from typing import Dict, Any
import datetime
import os

from utils.metrics import StyleTransferMetrics, MetricsLogger
from utils.dataloader import create_dataloader
from models import get_model
from models.CycleGAN import CycleGAN

def evaluate_model(config_path: str, checkpoint_path: str, output_path: str):
    """Evaluate a trained model using multiple metrics"""
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model-specific output directory
    model_output_dir = Path(output_path) / config['model']['model_type']
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'device' not in config:
        config['device'] = str(device)
    
    # Convert config to format expected by get_model and dataloader
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
            self.model_type = config_dict['model_type'].lower()

    class DataConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    class TrainingConfig:
        def __init__(self, config_dict):
            self.batch_size = config_dict.get('batch_size', 16)
            self.num_workers = config_dict.get('num_workers', 4)

    class Config:
        def __init__(self, model_config, data_config, training_config):
            self.model = model_config
            self.data = data_config
            self.training = training_config

        def get_model_config(self):
            return self.model.__dict__

    # Create config object with proper structure
    model_config = ModelConfig(config['model'])
    data_config = DataConfig(config['data'])
    training_config = TrainingConfig(config.get('training', {'batch_size': 16, 'num_workers': 4}))
    config_obj = Config(model_config, data_config, training_config)
    
    # Create model and load checkpoint
    model = get_model(config_obj)
    
    # Ensure checkpoint_path exists and is valid
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    # Create dataloader using the config object
    val_dataloader = create_dataloader(config_obj, 'val')
    
    # Initialize metrics
    metrics = StyleTransferMetrics(device=device)
    logger = MetricsLogger()
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate outputs - handle both model types
            if isinstance(model, CycleGAN):
                outputs = {'generated': model(batch['content'], direction='AB')}
            else:
                outputs = {'generated': model(batch['content'])}  # Pass only content tensor
            
            # Check and print value ranges before processing
            print(f"Content image range: [{batch['content'].min():.3f}, {batch['content'].max():.3f}]")
            print(f"Style image range: [{batch['style'].min():.3f}, {batch['style'].max():.3f}]")
            
            # Check generated image range
            print(f"Generated image range: [{outputs['generated'].min():.3f}, {outputs['generated'].max():.3f}]")
            
            # Ensure all images are in [0,1] range
            outputs['generated'] = torch.clamp(outputs['generated'], 0, 1)
            
            # Compute metrics
            batch_metrics = {
                'content_loss': metrics.compute_content_loss(
                    outputs['generated'], batch['content']
                ),
                'style_loss': metrics.compute_style_loss(
                    outputs['generated'], batch['style']
                ),
                'lpips': metrics.compute_lpips(
                    outputs['generated'], batch['content']
                ),
                'psnr': metrics.compute_psnr(
                    outputs['generated'], batch['content']
                ),
                'ssim': metrics.compute_ssim(
                    outputs['generated'], batch['content']
                )
            }
            
            # Update logger
            logger.update(batch_metrics)
    
    # Get average metrics
    final_metrics = logger.get_average()
    
    # Save results to model-specific directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['model_type']
    
    results_path = model_output_dir / f'{model_name}_evaluation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print("\nFinal Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save evaluation results')
    args = parser.parse_args()
    
    evaluate_model(args.config, args.checkpoint, args.output) 