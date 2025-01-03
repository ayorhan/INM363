"""
Example script for comparing multiple style transfer models
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from utils.metrics import StyleTransferMetrics, MetricsLogger
from utils.dataloader import create_dataloader
from models import get_model

def create_comparison_visualization(results: Dict, output_path: str) -> None:
    """Create heatmap visualization of model comparison results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(results).round(4)
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.4f')
    plt.title('Model Comparison - Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(config_paths: List[str], 
                  checkpoint_paths: List[str],
                  test_data_path: str,
                  output_path: str):
    """Compare multiple models using the same test dataset"""
    
    base_output_path = Path(output_path)
    
    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
            
        model_name = config['model']['model_type']
        model_output_dir = base_output_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nEvaluating {model_name}...")
        
        # Initialize metrics
        metrics = StyleTransferMetrics(device='cuda')
        results = {}
        
        # Load test dataset
        test_config = {
            'dataset_path': test_data_path,
            'batch_size': 4,
            'num_workers': 4,
            'device': 'cuda'
        }
        test_dataloader = create_dataloader(test_config, 'test')
        
        # Create model and load checkpoint
        model = get_model(config)
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize logger for this model
        logger = MetricsLogger()
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Testing {model_name}"):
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch)
                
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
                
                logger.update(batch_metrics)
        
        # Store results for this model
        results[model_name] = logger.get_average()
        
        # Save model-specific results
        results_path = model_output_dir / 'comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results[model_name], f, indent=4)
    
    # Save overall comparison
    comparison_path = base_output_path / 'model_comparison.csv'
    pd.DataFrame(results).round(4).to_csv(comparison_path)
    
    # Save visualization
    viz_path = base_output_path / 'comparison_heatmap.png'
    create_comparison_visualization(results, viz_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Paths to configuration files')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save comparison results')
    args = parser.parse_args()
    
    assert len(args.configs) == len(args.checkpoints), \
        "Number of configs must match number of checkpoints"
    
    compare_models(args.configs, args.checkpoints, 
                  args.test_data, args.output) 