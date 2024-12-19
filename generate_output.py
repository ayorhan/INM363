import os
import torch
from pathlib import Path
from tqdm import tqdm
import re
from models.JohnsonModel import JohnsonModel
from utils.dataloader import create_dataloader
from utils.config import StyleTransferConfig, ModelConfig, DataConfig, TrainingConfig
import yaml
from utils.metrics import StyleTransferMetrics, MetricsLogger
from torchvision.utils import save_image, make_grid
import datetime
import json
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_epoch_from_checkpoint(checkpoint_name):
    match = re.search(r'epoch(\d+)', checkpoint_name)
    return int(match.group(1)) if match else -1

def filter_checkpoints_by_epoch(checkpoint_files):
    """Select one checkpoint per epoch (the last one)"""
    epoch_to_checkpoint = {}
    for checkpoint in checkpoint_files:
        epoch = get_epoch_from_checkpoint(checkpoint)
        if epoch != -1:  # Valid epoch number found
            # Keep the last checkpoint for each epoch (based on filename sorting)
            if epoch not in epoch_to_checkpoint or checkpoint > epoch_to_checkpoint[epoch]:
                epoch_to_checkpoint[epoch] = checkpoint
    
    # Get the sorted list of checkpoints (one per epoch)
    selected_checkpoints = [epoch_to_checkpoint[epoch] for epoch in sorted(epoch_to_checkpoint.keys())]
    return selected_checkpoints

def save_validation_images(outputs, batch, epoch, batch_idx, output_dir):
    """Save validation images with content, style, and generated images side by side"""
    try:
        # Create visualization grid
        vis_images = []
        if isinstance(batch, dict) and 'content' in batch and 'style' in batch:
            content = batch['content']
            style = batch['style']
            generated = outputs['generated']
            n = min(content.size(0), style.size(0))
            
            for i in range(n):
                vis_images.extend([
                    content[i].cpu(),
                    style[i].cpu(),
                    generated[i].cpu()
                ])
        
        # Create and save grid
        if vis_images:
            grid = make_grid(torch.stack(vis_images), nrow=3, normalize=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_epoch{epoch}_batch{batch_idx}_{timestamp}.png"
            save_path = output_dir / filename
            save_image(grid, save_path)
            logging.debug(f"Saved grid image: {save_path}")
            
            # Also save individual images if needed
            for i in range(n):
                individual_dir = output_dir / f"epoch_{epoch}" / f"batch_{batch_idx}"
                individual_dir.mkdir(parents=True, exist_ok=True)
                
                # Save content image
                save_image(content[i].cpu(), 
                          individual_dir / f"content_{i}.png")
                # Save style image
                save_image(style[i].cpu(), 
                          individual_dir / f"style_{i}.png")
                # Save generated image
                save_image(generated[i].cpu(), 
                          individual_dir / f"generated_{i}.png")
            logging.debug(f"Saved individual images for batch {batch_idx}")

    except Exception as e:
        logging.error(f"Failed to save validation images: {str(e)}")

def evaluate_checkpoints(config_path: str, checkpoints_dir: str, output_dir: str):
    setup_logging()
    logging.info("Starting checkpoint evaluation")
    
    # Load config
    logging.info(f"Loading config from {config_path}")
    config = StyleTransferConfig(config_path)
    
    # Modify validation size for minimal but meaningful processing
    config.data.val_content_size = min(config.data.val_content_size, 40)   # 10 batches of 4 images
    config.data.val_style_size = min(config.data.val_style_size, 20)     # 5 different styles
    config.data.batch_size = 4  # Ensure batch size is 4
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Using {config.data.val_content_size} content images and {config.data.val_style_size} style images")
    
    # Convert ModelConfig to dictionary
    model_config = config.model.__dict__
    
    # Initialize model and metrics
    logging.info("Initializing model and metrics")
    model = JohnsonModel(model_config)
    model = model.to(device)
    metrics = StyleTransferMetrics(device=device)
    
    # Get validation dataloader
    logging.info("Creating validation dataloader")
    val_dataloader = create_dataloader(config, 'val')
    total_batches = len(val_dataloader)
    logging.info(f"Total validation batches: {total_batches}")
    
    # Get all checkpoints and filter to one per epoch
    all_checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    checkpoint_files = filter_checkpoints_by_epoch(all_checkpoint_files)
    total_checkpoints = len(checkpoint_files)
    logging.info(f"Found {len(all_checkpoint_files)} total checkpoints")
    logging.info(f"Selected {total_checkpoints} checkpoints (one per epoch) for evaluation")
    
    # Create output directories
    output_path = Path(output_dir)
    images_path = output_path / 'images'
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for checkpoint_idx, checkpoint_file in enumerate(checkpoint_files, 1):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
        logging.info(f"\nProcessing checkpoint {checkpoint_idx}/{total_checkpoints}: {checkpoint_file}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize logger for this checkpoint
        logger = MetricsLogger()
        
        # Get epoch number for this checkpoint
        epoch = get_epoch_from_checkpoint(checkpoint_file)
        
        # Create epoch-specific directory for images
        epoch_image_dir = images_path / f"epoch_{epoch}"
        epoch_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation loop
        batch_progress = tqdm(val_dataloader, 
                            desc=f"Evaluating epoch {epoch}",
                            total=total_batches,
                            leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_progress):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = {'generated': model(batch['content'])}
                outputs['generated'] = torch.clamp(outputs['generated'], 0, 1)
                
                # Save images every 5 batches (2 saves per checkpoint with 10 total batches)
                if batch_idx % 5 == 0:
                    save_validation_images(outputs, batch, epoch, batch_idx, epoch_image_dir)
                
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
                
                # Update progress bar with current metrics
                batch_progress.set_postfix({
                    k: f"{float(v):.4f}" for k, v in batch_metrics.items()
                })
        
        # Save results for this checkpoint
        epoch_metrics = logger.get_average()
        results[epoch] = {k: float(v) for k, v in epoch_metrics.items()}
        
        # Log average metrics for this epoch
        logging.info(f"Epoch {epoch} average metrics:")
        for metric_name, metric_value in results[epoch].items():
            logging.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Save intermediate results as JSON
        with open(output_path / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Saved results to {output_path / 'validation_results.json'}")
    
    logging.info("Evaluation complete!")

if __name__ == "__main__":
    config_path = "configs/johnson_config.yaml"
    checkpoints_dir = "checkpoints/johnson"
    output_dir = "validation_results/johnson"
    
    evaluate_checkpoints(config_path, checkpoints_dir, output_dir)