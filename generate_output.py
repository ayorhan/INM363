import os
import torch
from pathlib import Path
from tqdm import tqdm
import re
from models.JohnsonModel import JohnsonModel
from utils.dataloader import create_dataloader
from utils.config import Config, ModelConfig, DataConfig, TrainingConfig
import yaml
from utils.metrics import StyleTransferMetrics, MetricsLogger

def get_epoch_from_checkpoint(checkpoint_name):
    match = re.search(r'epoch(\d+)', checkpoint_name)
    return int(match.group(1)) if match else -1

def evaluate_checkpoints(config_path: str, checkpoints_dir: str, output_dir: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create config object
    model_config = ModelConfig(config['model'])
    data_config = DataConfig(config['data'])
    training_config = TrainingConfig(config.get('training', {}))
    config_obj = Config(model_config, data_config, training_config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and metrics
    model = JohnsonModel(config_obj)
    model = model.to(device)
    metrics = StyleTransferMetrics(device=device)
    
    # Get validation dataloader
    val_dataloader = create_dataloader(config_obj, 'val')
    
    # Get all checkpoints and sort by epoch
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    checkpoint_files.sort(key=get_epoch_from_checkpoint)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for checkpoint_file in tqdm(checkpoint_files, desc="Processing checkpoints"):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize logger for this checkpoint
        logger = MetricsLogger()
        
        # Evaluation loop
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = {'generated': model(batch['content'])}
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
                
                logger.update(batch_metrics)
        
        # Save results for this checkpoint
        epoch = get_epoch_from_checkpoint(checkpoint_file)
        results[epoch] = logger.get_average()
        
        # Save intermediate results
        torch.save(results, output_path / 'validation_results.pth')

if __name__ == "__main__":
    config_path = "configs/johnson_config.yaml"
    checkpoints_dir = "checkpoints/johnson"
    output_dir = "validation_results/johnson"
    
    evaluate_checkpoints(config_path, checkpoints_dir, output_dir)