"""
Example script for evaluating a trained style transfer model
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Dict, Any
import datetime

from utils.metrics import StyleTransferMetrics, MetricsLogger
from utils.dataloader import create_dataloader
from models import get_model

def evaluate_model(config_path: str, checkpoint_path: str, output_path: str):
    """Evaluate a trained model using multiple metrics"""
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Create model-specific output directory
    model_output_dir = Path(output_path) / config['model']['model_type']
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model and load checkpoint
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataloader
    val_dataloader = create_dataloader(config, 'val')
    
    # Initialize metrics
    metrics = StyleTransferMetrics(device=config['device'])
    logger = MetricsLogger()
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            # Move data to device
            batch = {k: v.to(config['device']) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate outputs
            outputs = model(batch)
            
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