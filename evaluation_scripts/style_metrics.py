"""
Example script for computing style-specific metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from utils.metrics import StyleTransferMetrics
from utils.dataloader import create_dataloader
from models import get_model

class StyleMetricsAnalyzer:
    """Analyzer for style-specific metrics"""
    def __init__(self, model: nn.Module, metrics: StyleTransferMetrics):
        self.model = model
        self.metrics = metrics
        
    def analyze_style_consistency(self, 
                                content_images: torch.Tensor,
                                style_images: torch.Tensor) -> Dict[str, float]:
        """Analyze consistency of style transfer across different content images"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            # Generate styled images
            outputs = []
            for content in content_images:
                content_batch = content.unsqueeze(0)
                output = self.model({'content': content_batch, 
                                   'style': style_images[0].unsqueeze(0)})
                outputs.append(output['generated'])
            
            outputs = torch.cat(outputs, dim=0)
            
            # Compute pairwise style consistency
            n = len(outputs)
            style_distances = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    style_loss = self.metrics.compute_style_loss(
                        outputs[i].unsqueeze(0),
                        outputs[j].unsqueeze(0)
                    )
                    style_distances.append(style_loss)
            
            return {
                'style_consistency_mean': np.mean(style_distances),
                'style_consistency_std': np.std(style_distances)
            }
    
    def analyze_content_preservation(self,
                                   content_images: torch.Tensor,
                                   style_images: torch.Tensor) -> Dict[str, float]:
        """Analyze content preservation across different styles"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for content in content_images:
                content_metrics = []
                content_batch = content.unsqueeze(0)
                
                for style in style_images:
                    style_batch = style.unsqueeze(0)
                    output = self.model({'content': content_batch, 
                                       'style': style_batch})
                    
                    metrics = {
                        'lpips': self.metrics.compute_lpips(
                            output['generated'],
                            content_batch
                        ),
                        'content_loss': self.metrics.compute_content_loss(
                            output['generated'],
                            content_batch
                        )
                    }
                    content_metrics.append(metrics)
                
                results.append({
                    'lpips_mean': np.mean([m['lpips'] for m in content_metrics]),
                    'lpips_std': np.std([m['lpips'] for m in content_metrics]),
                    'content_loss_mean': np.mean([m['content_loss'] for m in content_metrics]),
                    'content_loss_std': np.std([m['content_loss'] for m in content_metrics])
                })
        
        return {
            'lpips_mean': np.mean([r['lpips_mean'] for r in results]),
            'lpips_std': np.mean([r['lpips_std'] for r in results]),
            'content_loss_mean': np.mean([r['content_loss_mean'] for r in results]),
            'content_loss_std': np.mean([r['content_loss_std'] for r in results])
        }

def main(config_path: str, checkpoint_path: str, output_path: str):
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Initialize model and metrics
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    metrics = StyleTransferMetrics(device='cuda')
    analyzer = StyleMetricsAnalyzer(model, metrics)
    
    # Create dataloader
    test_dataloader = create_dataloader(config, 'test')
    
    # Get sample images
    content_images = []
    style_images = []
    
    for batch in test_dataloader:
        if len(content_images) < 10:  # Analyze with 10 images
            content_images.append(batch['content'])
            style_images.append(batch['style'])
        else:
            break
    
    content_images = torch.cat(content_images, dim=0)
    style_images = torch.cat(style_images, dim=0)
    
    # Run analysis
    style_consistency = analyzer.analyze_style_consistency(
        content_images, style_images
    )
    
    content_preservation = analyzer.analyze_content_preservation(
        content_images, style_images
    )
    
    # Combine results
    results = {
        'style_analysis': style_consistency,
        'content_analysis': content_preservation
    }
    
    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'style_analysis.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nStyle Analysis Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save analysis results')
    args = parser.parse_args()
    
    main(args.config, args.checkpoint, args.output) 